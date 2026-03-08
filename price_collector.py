# price_collector.py
# yfinance를 통해 50개 기업의 주가 데이터를 수집합니다

import yfinance as yf
import pandas as pd
import os
import time
from config import START_YEAR, END_YEAR

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────
# Yahoo Finance 티커 매핑 (KRX: 종목코드.KS)
# ─────────────────────────────────────────
TICKER_MAP = {
    # 반도체/IT
    "삼성전자":         "005930.KS",
    "SK하이닉스":       "000660.KS",
    "삼성SDI":          "006400.KS",
    "LG전자":           "066570.KS",
    "삼성전기":         "009150.KS",
    "LG디스플레이":     "034220.KS",
    "DB하이텍":         "000990.KS",
    "리노공업":         "058470.KS",

    # 자동차/이차전지
    "현대차":           "005380.KS",
    "기아":             "000270.KS",
    "현대모비스":       "012330.KS",
    "LG에너지솔루션":   "373220.KS",
    "에코프로비엠":     "247540.KS",
    "포스코퓨처엠":     "003670.KS",

    # 금융
    "KB금융":           "105560.KS",
    "신한지주":         "055550.KS",
    "하나금융지주":     "086790.KS",
    "우리금융지주":     "316140.KS",
    "삼성생명":         "032830.KS",
    "삼성화재":         "000810.KS",
    "미래에셋증권":     "006800.KS",
    "한국투자증권":     "071050.KS",

    # 에너지/화학
    "POSCO홀딩스":      "005490.KS",
    "LG화학":           "051910.KS",
    "롯데케미칼":       "011170.KS",
    "SK이노베이션":     "096770.KS",
    "한화솔루션":       "009830.KS",
    "S-Oil":            "010950.KS",

    # 바이오/헬스케어
    "삼성바이오로직스": "207940.KS",
    "셀트리온":         "068270.KS",
    "유한양행":         "000100.KS",
    "한미약품":         "128940.KS",
    "종근당":           "185750.KS",
    "대웅제약":         "069620.KS",

    # 소비재/유통
    "삼성물산":         "028260.KS",
    "LG생활건강":       "051900.KS",
    "아모레퍼시픽":     "090430.KS",
    "CJ제일제당":       "097950.KS",
    "롯데쇼핑":         "023530.KS",
    "신세계":           "004170.KS",
    "BGF리테일":        "282330.KS",

    # 통신/미디어
    "SK텔레콤":         "017670.KS",
    "KT":               "030200.KS",
    "LG유플러스":       "032640.KS",
    "카카오":           "035720.KS",
    "NAVER":            "035420.KS",
    "크래프톤":         "259960.KS",

    # 건설
    "현대건설":         "000720.KS",
    "삼성엔지니어링":   "028050.KS",
    "GS건설":           "006360.KS",
    "대우건설":         "047040.KS",
}


# ─────────────────────────────────────────
# 1. 주가 데이터 수집
# ─────────────────────────────────────────
def fetch_stock_prices(ticker_map, start_year, end_year):
    start_date = f"{start_year}-01-01"
    end_date   = f"{end_year}-12-31"
    all_prices = []

    print(f"주가 데이터 수집 ({start_date} ~ {end_date})")
    print("=" * 50)

    for corp_name, ticker in ticker_map.items():
        try:
            print(f"  {corp_name} ({ticker}) 수집 중...")
            
            # 티커 1개씩 개별 다운로드
            df = yf.Ticker(ticker).history(start=start_date, end=end_date)

            if df.empty:
                print(f"  {corp_name}: 데이터 없음")
                continue

            df = df.reset_index()
            
            # 필요한 컬럼만 선택
            df = df[["Date", "Close", "High", "Low", "Open", "Volume"]].copy()
            df["corp_name"] = corp_name
            df["ticker"] = ticker
            df["Date"] = df["Date"].dt.tz_localize(None)  # timezone 제거
            
            all_prices.append(df)
            time.sleep(0.3)

        except Exception as e:
            print(f"  {corp_name} 오류: {e}")

    if not all_prices:
        print("수집된 주가 데이터가 없습니다")
        return None

    result = pd.concat(all_prices, ignore_index=True)
    result.to_csv("data/raw/prices_raw.csv", index=False, encoding="utf-8-sig")
    print(f"\n주가 원본 저장 완료: data/raw/prices_raw.csv ({len(result)}행)")
    return result


# ─────────────────────────────────────────
# 2. 연간 주가 지표 계산
# ─────────────────────────────────────────
def calculate_annual_price_metrics(price_df):
    """
    일별 주가에서 연간 지표를 계산합니다:
    - 연말 종가, 연평균 종가
    - 연간 수익률, 변동성
    - 52주 최고/최저
    """
    print("\n연간 주가 지표 계산 중...")

    df = price_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["Date"] = pd.to_datetime(df["Date"])
    df["year"] = df["Date"].dt.year
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    annual = df.groupby(["corp_name", "ticker", "year"]).agg(
        종가_연말=("Close", "last"),
        종가_연평균=("Close", "mean"),
        최고가=("High", "max"),
        최저가=("Low", "min"),
        거래량_평균=("Volume", "mean"),
    ).reset_index()

    # 연간 수익률
    annual = annual.sort_values(["corp_name", "year"])
    annual["연간수익률(%)"] = (
        annual.groupby("corp_name")["종가_연말"].pct_change() * 100
    ).round(2)

    # 연간 변동성 (일별 수익률의 표준편차 * √252)
    vol = (
        df.sort_values(["corp_name", "Date"])
        .groupby(["corp_name", "year"])["Close"]
        .apply(lambda x: x.pct_change().std() * (252 ** 0.5) * 100)
        .reset_index()
        .rename(columns={"Close": "연간변동성(%)"})
    )
    annual = annual.merge(vol, on=["corp_name", "year"], how="left")
    annual["연간변동성(%)"] = annual["연간변동성(%)"].round(2)

    annual.to_csv("data/processed/prices_annual.csv", index=False, encoding="utf-8-sig")
    print(f"연간 주가 지표 저장 완료: data/processed/prices_annual.csv ({len(annual)}행)")
    return annual


# ─────────────────────────────────────────
# 3. 재무 데이터와 주가 데이터 병합
# ─────────────────────────────────────────
def merge_financial_and_price(financial_path, price_path):
    """
    재무제표 데이터와 주가 데이터를 기업명+연도로 병합합니다.
    → 최종 분석용 마스터 데이터셋 생성
    """
    print("\n재무 + 주가 데이터 병합 중...")

    fin_df   = pd.read_csv(financial_path, encoding="utf-8-sig")
    price_df = pd.read_csv(price_path, encoding="utf-8-sig")

    master = fin_df.merge(price_df, on=["corp_name", "year"], how="left")

    # 시가총액 계산 (연말 종가 기준 — 발행주식수 필요 시 별도 수집)
    # PER, PBR 등 밸류에이션 멀티플은 대시보드 단계에서 추가

    master.to_csv("data/processed/master_dataset.csv", index=False, encoding="utf-8-sig")
    print(f"마스터 데이터셋 저장 완료: data/processed/master_dataset.csv ({len(master)}행)")
    print("\n컬럼 목록:")
    print(list(master.columns))
    return master


# ─────────────────────────────────────────
# 4. 메인 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("주가 데이터 수집 시작")
    print("=" * 50)

    # Step 1: 주가 수집
    price_raw = fetch_stock_prices(TICKER_MAP, START_YEAR, END_YEAR)

    # Step 2: 연간 지표 계산
    if price_raw is not None:
        price_annual = calculate_annual_price_metrics(price_raw)

        # Step 3: 재무 + 주가 병합
        fin_path   = "data/processed/financials_processed.csv"
        price_path = "data/processed/prices_annual.csv"

        if os.path.exists(fin_path):
            master = merge_financial_and_price(fin_path, price_path)
        else:
            print(f"\n재무제표 데이터가 없습니다. dart_collector.py를 먼저 실행하세요.")

    print("\n주가 데이터 수집 완료!")
    print("다음 단계: valuation_model.py 실행하여 밸류에이션 계산")