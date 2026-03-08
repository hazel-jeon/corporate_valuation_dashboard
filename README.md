# 📊 Corporate Valuation & Bankruptcy Risk Prediction Dashboard

> **Korean Listed Company Financial Analysis Platform**  
> An end-to-end financial analysis system covering 42 KOSPI-listed companies,  
> built on DART public filings and Yahoo Finance data.

---

## Project Overview

This project replicates a buy-side investment analysis workflow from scratch.  
It collects and processes public financial data, applies the Altman Z-Score model  
alongside machine learning classifiers, and delivers an interactive dashboard  
for assessing corporate financial health.

### Problem Statement

> *"Can we systematically identify bankruptcy risk from financial statement data alone?"*

Evaluating the financial soundness of investee companies is a core part of the investment process.  
However, manually analyzing dozens of firms across multiple years is time-consuming and error-prone.  
This project **automates and visualizes** that process end-to-end.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Data Collection | DART OpenAPI, yfinance |
| Data Processing | Python, pandas, numpy |
| Machine Learning | scikit-learn (Random Forest, Gradient Boosting) |
| Visualization | Streamlit, Plotly |
| Environment | python-dotenv |

---

## Project Structure

```
valuation/
├── config.py              # Company universe (42 firms) & API config
├── dart_collector.py      # DART financial statement collection & processing
├── price_collector.py     # Stock price collection & annual metrics
├── valuation_model.py     # Altman Z-Score + ML bankruptcy prediction
├── dashboard.py           # Streamlit dashboard (5 pages)
├── .env                   # API credentials (excluded from Git)
└── data/
    ├── raw/               # Raw collected data
    ├── processed/         # Cleaned master dataset
    └── model/             # Model outputs & scorecards
```

---

## Methodology

### 1. Altman Z-Score (1968)

A classic bankruptcy prediction model combining five financial ratios.

```
Z = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5
```

| Variable | Description | Proxy |
|----------|-------------|-------|
| X1 | Liquidity | Working Capital / Total Assets |
| X2 | Cumulative Profitability | Net Income / Total Assets |
| X3 | Operating Efficiency | EBIT / Total Assets |
| X4 | Financial Leverage | Equity / Total Liabilities |
| X5 | Asset Utilization | Revenue / Total Assets |

**Classification Thresholds**

| Z-Score | Zone |
|---------|------|
| Z > 2.99 | 🟢 Safe Zone |
| 1.81 < Z < 2.99 | 🟡 Grey Zone |
| Z < 1.81 | 🔴 Distress Zone |

### 2. ML Bankruptcy Prediction

Implemented to complement the Z-Score model's limitations.

- **Algorithms**: Random Forest, Gradient Boosting (best model selected by F1)
- **Features**: 13 financial indicators spanning profitability, stability, growth, and stock volatility
- **Validation**: Stratified 5-Fold Cross Validation
- **Labeling**: A firm is labeled as high-risk if it meets 3 or more of the following conditions:
  - Z-Score < 1.81
  - Debt-to-Equity > 200%
  - Current Ratio < 100%
  - Operating Margin < 0%
  - Operating Cash Flow < 0

### 3. Model Performance

| Model | F1 Score | AUC |
|-------|----------|-----|
| Random Forest | 0.400 ± 0.490 | **0.971 ± 0.042** |
| Gradient Boosting | 0.467 ± 0.400 | 0.886 ± 0.194 |

> The Random Forest classifier achieved an AUC of **0.971** in Stratified 5-Fold cross-validation,
> demonstrating strong discriminative power in identifying financially distressed firms.  
> The lower F1 score is expected given class imbalance (few distressed firms in the dataset),
> while AUC more accurately reflects the model's ranking ability across all thresholds.

---

## Data Coverage

- **Universe**: 42 KOSPI-listed companies (sector representatives)
- **Period**: FY2019 – FY2023 (5 years)
- **Financial Statements**: Consolidated (IFRS)
- **Sectors**: Semiconductors/IT · Automotive/Battery · Chemicals/Energy · Bio/Healthcare · Consumer/Retail · Telecom/Media · Construction

---

## Dashboard Features

| Page | Content |
|------|---------|
| 🏠 Overview | KPI summary, Z-Score distribution, profitability vs. stability bubble chart |
| 🏆 Z-Score Ranking | Full ranking table, Z-Score component decomposition chart |
| 📈 Financial Trends | Multi-company time series, 4-metric simultaneous comparison |
| 🗂️ Sector Comparison | Sector box plots, radar chart |
| 🔍 Company Scorecard | Individual company deep-dive, 5-year financial summary |

---

## Getting Started

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn streamlit plotly yfinance python-dotenv lxml requests
```

### 2. Set Up API Key

Obtain a free API key from [DART OpenAPI](https://opendart.fss.or.kr) and create a `.env` file:

```
DART_API_KEY=your_api_key_here
```

### 3. Collect Data

```bash
python dart_collector.py    # Financial statements (~10 min)
python price_collector.py   # Stock prices & merge
```

### 4. Run the Model

```bash
python valuation_model.py
```

### 5. Launch the Dashboard

```bash
streamlit run dashboard.py
```

---

## Limitations & Future Work

### Known Limitations

**1. Industry Bias in Altman Z-Score**  
The original model was designed for manufacturing SMEs in the 1960s.  
For large capital-intensive firms (e.g., semiconductors) or asset-light tech platforms,  
X4 (Equity/Total Liabilities) tends to be disproportionately large, inflating the Z-Score.  
This explains why fundamentally sound companies like SK Hynix and NAVER  
are classified in the distress zone despite strong underlying business performance.

**2. Synthetic Labeling**  
In the absence of actual bankruptcy event data for Korean listed companies,  
distress labels were generated using rule-based financial thresholds.  
Class imbalance (few distressed firms) causes the ML model to predict conservatively,
resulting in lower F1 scores despite high AUC.

**3. Financial Sector Exclusion**  
Banks, securities firms, and insurers were excluded due to structural differences  
in their financial statements. Analyzing these firms requires sector-specific metrics  
such as BIS capital ratios and net interest margins (NIM).

### Future Improvements

- Apply **Zmijewski Model** or **Ohlson O-Score** for multi-model comparison
- Implement **sector-specific thresholds** to account for industry characteristics
- Incorporate **actual delisting / workout event data** for supervised labeling
- Extend to **quarterly data** for higher-frequency monitoring
- Add a **DCF Valuation** module for intrinsic value estimation

---

## Key Insights

> **Samsung Electronics — Z-Score: 3.345**  
> The majority of the Z-Score (X4 contribution: 2.36) stems from its exceptionally low leverage,  
> with a debt-to-equity ratio of approximately 26%.  
> This suggests Samsung's financial safety is driven by its **balance sheet strength**  
> rather than operating profitability alone.

> **GS Engineering & Construction — Bankruptcy Probability: 100%**  
> Classified as high-risk by both the Z-Score (0.928) and the ML model.  
> A combination of rising leverage and deteriorating operating cash flow  
> points to structural financial stress, consistent with publicly reported challenges.

---

## References

- Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *Journal of Finance*, 23(4), 589–609.
- DART OpenAPI Documentation: https://opendart.fss.or.kr

---

## License

MIT License © 2026 [hazel-jeon](https://github.com/hazel-jeon)