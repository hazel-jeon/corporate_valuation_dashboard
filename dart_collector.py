# dart_collector.py
# DART OpenAPI를 통해 50개 기업의 재무제표 데이터를 수집합니다

import requests
import pandas as pd
import zipfile
import io
import os
import time
from config import DART_API_KEY, DART_BASE_URL, COMPANIES, START_YEAR, END_YEAR, FS_TYPE

# 저장 디렉토리 생성
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────
# 1. 기업 고유번호(corp_code) 전체 다운로드
# ─────────────────────────────────────────
def download_corp_codes():
    """
    DART 전체 기업 고유번호 XML 파일을 다운로드하고
    config의 COMPANIES와 매핑된 DataFrame을 반환합니다.
    """
    print("📥 기업 고유번호 다운로드 중...")
    url = f"{DART_BASE_URL}/corpCode.xml"
    params = {"crtfc_key": DART_API_KEY}

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"API 호출 실패: {response.status_code}")

    # ZIP 압축 해제
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("CORPCODE.xml") as f:
            corp_df = pd.read_xml(f)

    # stock_code 있는 상장사만 필터링
    corp_df = corp_df[corp_df["stock_code"].notna()].copy()
    corp_df["stock_code"] = corp_df["stock_code"].astype(str).str.zfill(6)

    print(f"✅ 전체 상장사 {len(corp_df)}개 로드 완료")
    corp_df.to_csv("data/raw/corp_codes.csv", index=False, encoding="utf-8-sig")
    return corp_df


def get_corp_code_map(corp_df):
    code_map = {}
    not_found = []

    for name, stock_code in COMPANIES.items():
        match = corp_df[corp_df["stock_code"] == stock_code]

        if not match.empty:
            code_map[name] = str(match.iloc[0]["corp_code"]).zfill(8)
            print(f"  ✅ {name} ({stock_code}) → {code_map[name]}")
        else:
            not_found.append(name)
            print(f"  ⚠️ {name}: 찾을 수 없음")

    print(f"\n✅ 매핑 완료: {len(code_map)}개 / 못 찾은 기업: {not_found}")
    return code_map


# ─────────────────────────────────────────
# 2. 단일 기업 재무제표 수집
# ─────────────────────────────────────────
def fetch_financial_statements(corp_code, corp_name, year):
    """
    특정 기업의 특정 연도 재무제표를 수집합니다.
    반환값: DataFrame (주요 재무 항목)
    """
    url = f"{DART_BASE_URL}/fnlttSinglAcntAll.json"
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": "11011",   # 11011: 사업보고서 (연간)
        "fs_div": FS_TYPE,
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data.get("status") != "000":
        print(f"  ⚠️ {corp_name} {year}: {data.get('message', '데이터 없음')}")
        return None

    df = pd.DataFrame(data["list"])
    df["corp_name"] = corp_name
    df["year"] = year
    return df


# ─────────────────────────────────────────
# 3. 전체 기업 재무제표 수집 루프
# ─────────────────────────────────────────
def collect_all_financials(code_map):
    """
    모든 기업의 연도별 재무제표를 수집하고 CSV로 저장합니다.
    """
    all_data = []
    total = len(code_map) * (END_YEAR - START_YEAR + 1)
    count = 0

    for corp_name, corp_code in code_map.items():
        for year in range(START_YEAR, END_YEAR + 1):
            count += 1
            print(f"[{count}/{total}] {corp_name} {year}년 수집 중...")

            df = fetch_financial_statements(corp_code, corp_name, year)
            if df is not None:
                all_data.append(df)

            time.sleep(0.3)  # API rate limit 방지

    if not all_data:
        print("❌ 수집된 데이터가 없습니다")
        return None

    result = pd.concat(all_data, ignore_index=True)
    result.to_csv("data/raw/financials_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 원본 재무제표 저장 완료: data/raw/financials_raw.csv ({len(result)}행)")
    return result


# ─────────────────────────────────────────
# 4. 핵심 재무 지표 추출 & 정제
# ─────────────────────────────────────────
# 분석에 필요한 DART 계정 항목명
KEY_ACCOUNTS = {
    # 손익계산서
    "ifrs-full_Revenue": "매출액",
    "ifrs-full_GrossProfit": "매출총이익",
    "dart_OperatingIncomeLoss": "영업이익",
    "ifrs-full_ProfitLoss": "당기순이익",
    "ifrs-full_ProfitLossAttributableToOwnersOfParent": "지배주주순이익",
    # 재무상태표
    "ifrs-full_Assets": "총자산",
    "ifrs-full_CurrentAssets": "유동자산",
    "ifrs-full_NoncurrentAssets": "비유동자산",
    "ifrs-full_Liabilities": "총부채",
    "ifrs-full_CurrentLiabilities": "유동부채",
    "ifrs-full_NoncurrentLiabilities": "비유동부채",
    "ifrs-full_Equity": "자기자본",
    "ifrs-full_IssuedCapital": "자본금",
    # 현금흐름표
    "ifrs-full_CashFlowsFromUsedInOperatingActivities": "영업활동현금흐름",
    "ifrs-full_CashFlowsFromUsedInInvestingActivities": "투자활동현금흐름",
    "ifrs-full_CashFlowsFromUsedInFinancingActivities": "재무활동현금흐름",
}


def extract_key_metrics(raw_df):
    """
    원본 재무제표에서 핵심 지표만 피벗하여 정제된 DataFrame을 반환합니다.
    """
    print("\n🔧 핵심 지표 추출 중...")

    # 필요한 계정만 필터링
    filtered = raw_df[raw_df["account_id"].isin(KEY_ACCOUNTS.keys())].copy()
    filtered["account_nm_kr"] = filtered["account_id"].map(KEY_ACCOUNTS)

    filtered = filtered.drop_duplicates(subset=["corp_name", "year", "account_nm_kr"])

    # 금액 컬럼 숫자 변환 (당기 기준)
    filtered["amount"] = (
        filtered["thstrm_amount"]
        .astype(str)
        .str.replace(",", "")
        .str.replace(" ", "")
        .replace("", "0")
        .pipe(pd.to_numeric, errors="coerce")
    )

    # 피벗: 행=기업+연도, 열=재무항목
    pivot = filtered.pivot_table(
        index=["corp_name", "year"],
        columns="account_nm_kr",
        values="amount",
        aggfunc="first"
    ).reset_index()

    pivot.columns.name = None

    # ─── 파생 지표 계산 ───
    # 수익성
    pivot["매출총이익률(%)"] = (pivot["매출총이익"] / pivot["매출액"] * 100).round(2)
    pivot["영업이익률(%)"] = (pivot["영업이익"] / pivot["매출액"] * 100).round(2)
    pivot["순이익률(%)"] = (pivot["당기순이익"] / pivot["매출액"] * 100).round(2)
    pivot["ROE(%)"] = (pivot["당기순이익"] / pivot["자기자본"] * 100).round(2)
    pivot["ROA(%)"] = (pivot["당기순이익"] / pivot["총자산"] * 100).round(2)

    # 안정성 (부도 예측 핵심 지표)
    pivot["부채비율(%)"] = (pivot["총부채"] / pivot["자기자본"] * 100).round(2)
    pivot["유동비율(%)"] = (pivot["유동자산"] / pivot["유동부채"] * 100).round(2)
    pivot["이자보상배율"] = (pivot["영업이익"] / pivot.get("이자비용", 1)).round(2)

    # Altman Z-Score 구성 요소 (간소화)
    pivot["X1_운전자본/총자산"] = ((pivot["유동자산"] - pivot["유동부채"]) / pivot["총자산"]).round(4)
    pivot["X3_EBIT/총자산"] = (pivot["영업이익"] / pivot["총자산"]).round(4)
    pivot["X4_자기자본/총부채"] = (pivot["자기자본"] / pivot["총부채"]).round(4)
    pivot["X5_매출액/총자산"] = (pivot["매출액"] / pivot["총자산"]).round(4)

    # 성장성
    pivot = pivot.sort_values(["corp_name", "year"])
    pivot["매출성장률(%)"] = pivot.groupby("corp_name")["매출액"].pct_change() * 100
    pivot["영업이익성장률(%)"] = pivot.groupby("corp_name")["영업이익"].pct_change() * 100

    pivot.to_csv("data/processed/financials_processed.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 정제 데이터 저장 완료: data/processed/financials_processed.csv ({len(pivot)}행)")
    return pivot


# ─────────────────────────────────────────
# 5. 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("📊 DART 재무제표 수집 시작")
    print("=" * 50)

    # Step 1: 기업 코드 다운로드 & 매핑
    corp_df = download_corp_codes()
    code_map = get_corp_code_map(corp_df)

    # Step 2: 재무제표 수집
    raw_df = collect_all_financials(code_map)

    # Step 3: 핵심 지표 추출
    if raw_df is not None:
        processed_df = extract_key_metrics(raw_df)
        print("\n📋 샘플 데이터 미리보기:")
        print(processed_df.head(3).to_string())

    print("\n🎉 데이터 수집 완료!")
    print("다음 단계: price_collector.py 실행하여 주가 데이터 수집")