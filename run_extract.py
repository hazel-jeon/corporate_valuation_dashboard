# run_extract.py
import pandas as pd
from dart_collector import extract_key_metrics

raw_df = pd.read_csv("data/raw/financials_raw.csv")
print(f"원본 데이터: {len(raw_df)}행")
processed_df = extract_key_metrics(raw_df)
print(processed_df.head())