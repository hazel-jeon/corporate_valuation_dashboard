# valuation_model.py
# Altman Z-Score 계산 + ML 부도 예측 모델

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

os_import = True
try:
    import os
except:
    os_import = False

import os
os.makedirs("data/model", exist_ok=True)


# ─────────────────────────────────────────
# 1. Altman Z-Score 계산
# ─────────────────────────────────────────
def calculate_altman_zscore(df):
    """
    Altman Z-Score (1968) 계산
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

    X1 = 운전자본 / 총자산
    X2 = 이익잉여금 / 총자산  (→ ROE 근사치 사용)
    X3 = EBIT / 총자산
    X4 = 자기자본 / 총부채
    X5 = 매출액 / 총자산

    판정 기준:
    Z > 2.99  → 안전 (Safe Zone)
    1.81 < Z < 2.99 → 회색 (Grey Zone)
    Z < 1.81  → 위험 (Distress Zone)
    """
    print("Altman Z-Score 계산 중...")

    df = df.copy()

    # X2: 이익잉여금 없으므로 당기순이익/총자산으로 근사
    df["X2_순이익/총자산"] = (df["당기순이익"] / df["총자산"]).round(4)

    # Z-Score 계산
    df["Z_Score"] = (
        1.2 * df["X1_운전자본/총자산"] +
        1.4 * df["X2_순이익/총자산"] +
        3.3 * df["X3_EBIT/총자산"] +
        0.6 * df["X4_자기자본/총부채"] +
        1.0 * df["X5_매출액/총자산"]
    ).round(3)

    # 판정
    def classify_zone(z):
        if pd.isna(z):
            return "데이터 없음"
        elif z > 2.99:
            return "안전"
        elif z > 1.81:
            return "주의"
        else:
            return "위험"

    df["Z_판정"] = df["Z_Score"].apply(classify_zone)

    # 결과 출력
    latest = df[df["year"] == df["year"].max()][
        ["corp_name", "year", "Z_Score", "Z_판정"]
    ].sort_values("Z_Score", ascending=False)

    print("\n📋 최신 연도 Z-Score 순위:")
    print(latest.to_string(index=False))

    return df


# ─────────────────────────────────────────
# 2. 부도 위험 레이블 생성
# ─────────────────────────────────────────
def create_distress_labels(df):
    """
    실제 부도 데이터가 없으므로 재무 지표 기반으로
    부도 위험 레이블을 생성합니다.

    위험 기업 조건 (3개 이상 충족 시 위험=1):
    - Z-Score < 1.81
    - 부채비율 > 200%
    - 유동비율 < 100%
    - 영업이익률 < 0% (영업손실)
    - 영업활동현금흐름 < 0
    """
    print("\n부도 위험 레이블 생성 중...")

    df = df.copy()

    conditions = pd.DataFrame({
        "z_위험":     (df["Z_Score"] < 1.81).astype(int),
        "부채_위험":   (df["부채비율(%)"] > 200).astype(int),
        "유동성_위험": (df["유동비율(%)"] < 100).astype(int),
        "수익성_위험": (df["영업이익률(%)"] < 0).astype(int),
        "현금흐름_위험": (df["영업활동현금흐름"] < 0).astype(int),
    })

    df["위험_점수"] = conditions.sum(axis=1)
    df["부도위험"] = (df["위험_점수"] >= 3).astype(int)

    print(f"  위험 기업 비율: {df['부도위험'].mean()*100:.1f}%")
    print(f"  위험=1: {df['부도위험'].sum()}개, 안전=0: {(df['부도위험']==0).sum()}개")

    return df


# ─────────────────────────────────────────
# 3. ML 피처 준비
# ─────────────────────────────────────────
FEATURES = [
    "X1_운전자본/총자산",
    "X3_EBIT/총자산",
    "X4_자기자본/총부채",
    "X5_매출액/총자산",
    "X2_순이익/총자산",
    "부채비율(%)",
    "유동비율(%)",
    "영업이익률(%)",
    "ROE(%)",
    "ROA(%)",
    "매출성장률(%)",
    "영업이익성장률(%)",
    "연간변동성(%)",
]


def prepare_features(df):
    """
    ML 모델용 피처 행렬을 준비합니다.
    """
    df_ml = df.copy()

    # 무한대 값 처리
    df_ml = df_ml.replace([np.inf, -np.inf], np.nan)

    # 사용 가능한 피처만 선택
    available = [f for f in FEATURES if f in df_ml.columns]
    missing = [f for f in FEATURES if f not in df_ml.columns]
    if missing:
        print(f"  없는 피처: {missing}")

    df_ml = df_ml.dropna(subset=available + ["부도위험"])

    X = df_ml[available]
    y = df_ml["부도위험"]
    meta = df_ml[["corp_name", "year", "Z_Score", "Z_판정", "위험_점수"]]

    print(f"  ML 데이터: {len(X)}행, {len(available)}개 피처")
    return X, y, meta, available


# ─────────────────────────────────────────
# 4. ML 모델 학습
# ─────────────────────────────────────────
def train_models(X, y):
    """
    Random Forest + Gradient Boosting 모델을 학습합니다.
    """
    print("\n🤖 ML 모델 학습 중...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="f1")
        model.fit(X_scaled, y)
        results[name] = {
            "model": model,
            "f1_mean": scores.mean(),
            "f1_std": scores.std(),
        }
        print(f"  {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

    # 최고 성능 모델 선택
    best_name = max(results, key=lambda k: results[k]["f1_mean"])
    best_model = results[best_name]["model"]
    print(f"\n  최고 모델: {best_name}")

    return best_model, scaler, results


# ─────────────────────────────────────────
# 5. 피처 중요도 출력
# ─────────────────────────────────────────
def print_feature_importance(model, feature_names):
    print("\n피처 중요도 (상위 10):")
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    for _, row in importance.head(10).iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:<25} {bar} {row['importance']:.3f}")

    return importance


# ─────────────────────────────────────────
# 6. 기업별 종합 스코어카드 생성
# ─────────────────────────────────────────
def generate_scorecard(df, model, scaler, feature_names):
    """
    전체 기업의 최신 연도 기준 종합 스코어카드를 생성합니다.
    """
    print("\n📋 종합 스코어카드 생성 중...")

    latest_year = df["year"].max()
    latest = df[df["year"] == latest_year].copy()
    latest = latest.replace([np.inf, -np.inf], np.nan)
    latest = latest.dropna(subset=feature_names)

    if len(latest) == 0:
        print("  최신 연도 데이터 없음")
        return None

    X_latest = scaler.transform(latest[feature_names])
    latest["부도확률(%)"] = (model.predict_proba(X_latest)[:, 1] * 100).round(1)
    latest["ML_판정"] = latest["부도확률(%)"].apply(
        lambda x: "고위험" if x >= 50 else ("주의" if x >= 25 else "안전")
    )

    scorecard = latest[[
        "corp_name", "year",
        "Z_Score", "Z_판정",
        "부도확률(%)", "ML_판정",
        "영업이익률(%)", "ROE(%)", "부채비율(%)", "유동비율(%)",
        "매출성장률(%)", "영업이익성장률(%)"
    ]].sort_values("부도확률(%)", ascending=False)

    print("\n종합 스코어카드 (부도확률 높은 순):")
    print(scorecard.to_string(index=False))

    scorecard.to_csv("data/model/scorecard.csv", index=False, encoding="utf-8-sig")
    print("\n스코어카드 저장: data/model/scorecard.csv")

    return scorecard


# ─────────────────────────────────────────
# 7. 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("기업 가치평가 & 부도 위험 예측 모델")
    print("=" * 55)

    # 데이터 로드
    df = pd.read_csv("data/processed/master_dataset.csv")
    print(f"데이터 로드: {len(df)}행, {df['corp_name'].nunique()}개 기업")

    # Step 1: Altman Z-Score
    df = calculate_altman_zscore(df)

    # Step 2: 부도 레이블
    df = create_distress_labels(df)

    # Step 3: 피처 준비
    X, y, meta, feature_names = prepare_features(df)

    # Step 4: 모델 학습
    best_model, scaler, results = train_models(X, y)

    # Step 5: 피처 중요도
    importance = print_feature_importance(best_model, feature_names)

    # Step 6: 스코어카드
    scorecard = generate_scorecard(df, best_model, scaler, feature_names)

    # 전체 결과 저장
    df.to_csv("data/model/full_results.csv", index=False, encoding="utf-8-sig")
    print("\n전체 결과 저장: data/model/full_results.csv")

    print("\n모델 완료!")
    print("다음 단계: dashboard.py 실행하여 대시보드 구축")