# dashboard.py
# 기업 가치평가 & 부도 위험 예측 대시보드

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="기업 재무 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# 커스텀 CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans KR', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }

    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 4px;
    }

    .safe    { color: #059669; }
    .warning { color: #d97706; }
    .danger  { color: #dc2626; }

    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
        letter-spacing: -0.01em;
    }

    div[data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 데이터 로드
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/model/full_results.csv")
    sc = pd.read_csv("data/model/scorecard.csv")
    return df, sc

df, scorecard = load_data()

# 섹터 매핑
SECTOR_MAP = {
    "삼성전자": "반도체/IT", "SK하이닉스": "반도체/IT", "삼성SDI": "반도체/IT",
    "LG전자": "반도체/IT", "삼성전기": "반도체/IT", "LG디스플레이": "반도체/IT",
    "DB하이텍": "반도체/IT", "리노공업": "반도체/IT",
    "현대차": "자동차/이차전지", "기아": "자동차/이차전지", "현대모비스": "자동차/이차전지",
    "LG에너지솔루션": "자동차/이차전지", "에코프로비엠": "자동차/이차전지", "포스코퓨처엠": "자동차/이차전지",
    "POSCO홀딩스": "화학/에너지", "LG화학": "화학/에너지", "롯데케미칼": "화학/에너지",
    "SK이노베이션": "화학/에너지", "한화솔루션": "화학/에너지", "S-Oil": "화학/에너지",
    "삼성바이오로직스": "바이오/헬스케어", "셀트리온": "바이오/헬스케어", "유한양행": "바이오/헬스케어",
    "한미약품": "바이오/헬스케어", "종근당": "바이오/헬스케어", "대웅제약": "바이오/헬스케어",
    "삼성물산": "소비재/유통", "LG생활건강": "소비재/유통", "아모레퍼시픽": "소비재/유통",
    "CJ제일제당": "소비재/유통", "롯데쇼핑": "소비재/유통", "신세계": "소비재/유통", "BGF리테일": "소비재/유통",
    "SK텔레콤": "통신/미디어", "KT": "통신/미디어", "LG유플러스": "통신/미디어",
    "카카오": "통신/미디어", "NAVER": "통신/미디어", "크래프톤": "통신/미디어",
    "현대건설": "건설", "삼성E&A": "건설", "GS건설": "건설", "대우건설": "건설",
}
df["섹터"] = df["corp_name"].map(SECTOR_MAP).fillna("기타")
scorecard["섹터"] = scorecard["corp_name"].map(SECTOR_MAP).fillna("기타")

# ─────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 재무분석 대시보드")
    st.markdown("---")

    page = st.radio(
        "페이지 선택",
        ["🏠 Overview", "🏆 Z-Score 랭킹", "📈 재무지표 추이", "🗂️ 섹터 비교", "🔍 기업 스코어카드"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # 연도 필터
    years = sorted(df["year"].unique())
    selected_year = st.selectbox("기준 연도", years, index=len(years)-1)

    # 섹터 필터
    sectors = ["전체"] + sorted(df["섹터"].unique().tolist())
    selected_sector = st.selectbox("섹터 필터", sectors)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.75rem; color:#475569; line-height:1.8'>
    <b style='color:#64748b'>데이터 소스</b><br>
    DART OpenAPI<br>
    Yahoo Finance<br><br>
    <b style='color:#64748b'>분석 모델</b><br>
    Altman Z-Score (1968)<br>
    Random Forest / GBM<br><br>
    <b style='color:#64748b'>커버리지</b><br>
    42개 기업 · 5개년
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# 필터 적용
# ─────────────────────────────────────────
df_year = df[df["year"] == selected_year].copy()
if selected_sector != "전체":
    df_year = df_year[df_year["섹터"] == selected_sector]
    df_sector = df[df["섹터"] == selected_sector].copy()
else:
    df_sector = df.copy()


# ─────────────────────────────────────────
# 공통 색상
# ─────────────────────────────────────────
COLORS = {
    "safe":    "#059669",
    "warning": "#d97706",
    "danger":  "#dc2626",
    "primary": "#4f46e5",
    "bg":      "#ffffff",
    "grid":    "#e2e8f0",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(color="#475569", family="IBM Plex Sans KR"),
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1"),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#cbd5e1"),
        margin=dict(l=40, r=20, t=40, b=40),
    )
)

def apply_template(fig):
    fig.update_layout(**PLOTLY_TEMPLATE["layout"])
    return fig


# ─────────────────────────────────────────
# PAGE 1: Overview
# ─────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown(f"## 기업 재무분석 Overview — {selected_year}년")
    st.markdown("")

    # KPI 카드
    col1, col2, col3, col4 = st.columns(4)

    total = len(df_year)
    safe_n    = (df_year["Z_판정"].str.contains("안전", na=False)).sum()
    warning_n = (df_year["Z_판정"].str.contains("주의", na=False)).sum()
    danger_n  = (df_year["Z_판정"].str.contains("위험", na=False)).sum()

    with col1:
        st.metric("분석 기업", f"{total}개")
    with col2:
        st.metric("🟢 안전", f"{safe_n}개", f"{safe_n/total*100:.0f}%")
    with col3:
        st.metric("🟡 주의", f"{warning_n}개", f"{warning_n/total*100:.0f}%")
    with col4:
        st.metric("🔴 위험", f"{danger_n}개", f"{danger_n/total*100:.0f}%")

    st.markdown("---")
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<p class="section-title">Z-Score 분포</p>', unsafe_allow_html=True)
        df_plot = df_year.dropna(subset=["Z_Score"]).sort_values("Z_Score", ascending=True)

        def z_color(z):
            if z > 2.99: return COLORS["safe"]
            elif z > 1.81: return COLORS["warning"]
            else: return COLORS["danger"]

        colors = [z_color(z) for z in df_plot["Z_Score"]]

        fig = go.Figure(go.Bar(
            x=df_plot["Z_Score"],
            y=df_plot["corp_name"],
            orientation="h",
            marker_color=colors,
            text=df_plot["Z_Score"].round(2),
            textposition="outside",
            textfont=dict(size=10, color="#94a3b8"),
        ))
        fig.add_vline(x=2.99, line_dash="dash", line_color=COLORS["safe"],    annotation_text="안전(2.99)", annotation_font_color=COLORS["safe"])
        fig.add_vline(x=1.81, line_dash="dash", line_color=COLORS["warning"], annotation_text="위험(1.81)", annotation_font_color=COLORS["warning"])
        fig.update_layout(
            height=600,
            xaxis_title="Z-Score",
            yaxis_title="",
            showlegend=False,
            **PLOTLY_TEMPLATE["layout"]
        )
        st.plotly_chart(fig, width='stretch')

    with col_right:
        st.markdown('<p class="section-title">섹터별 평균 Z-Score</p>', unsafe_allow_html=True)
        sector_z = df_year.groupby("섹터")["Z_Score"].mean().sort_values(ascending=False).reset_index()
        fig2 = go.Figure(go.Bar(
            x=sector_z["Z_Score"].round(2),
            y=sector_z["섹터"],
            orientation="h",
            marker_color=COLORS["primary"],
            text=sector_z["Z_Score"].round(2),
            textposition="outside",
            textfont=dict(size=11),
        ))
        fig2.update_layout(
            height=300,
            showlegend=False,
            **PLOTLY_TEMPLATE["layout"]
        )
        st.plotly_chart(fig2, width='stretch')

        st.markdown('<p class="section-title">수익성 vs 안전성</p>', unsafe_allow_html=True)
        df_bubble = df_year.dropna(subset=["ROE(%)", "부채비율(%)", "Z_Score"])
        fig3 = px.scatter(
            df_bubble,
            x="부채비율(%)", y="ROE(%)",
            size=df_bubble["총자산"].clip(upper=df_bubble["총자산"].quantile(0.95)),
            color="Z_Score",
            color_continuous_scale=["#f87171", "#fbbf24", "#34d399"],
            hover_name="corp_name",
            labels={"부채비율(%)": "부채비율 (%)", "ROE(%)": "ROE (%)"},
        )
        fig3.update_layout(height=260, **PLOTLY_TEMPLATE["layout"])
        fig3.update_coloraxes(colorbar_title="Z-Score")
        st.plotly_chart(fig3, width='stretch')


# ─────────────────────────────────────────
# PAGE 2: 🏆 Z-Score 랭킹
# ─────────────────────────────────────────
elif page == "🏆 Z-Score 랭킹":
    st.write("페이지 로드됨")
    st.markdown(f"## 🏆 Z-Score 랭킹 — {selected_year}년")

    df_rank = df_year.dropna(subset=["Z_Score"]).sort_values("Z_Score", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    # 상위/하위 하이라이트
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Top 5 안전 기업")
        top5 = df_rank.head(5)[["corp_name", "섹터", "Z_Score", "Z_판정", "영업이익률(%)", "부채비율(%)"]]
        st.dataframe(top5, width='stretch', hide_index=False)
    with col2:
        st.markdown("#### 🔴 Bottom 5 위험 기업")
        bot5 = df_rank.tail(5)[["corp_name", "섹터", "Z_Score", "Z_판정", "영업이익률(%)", "부채비율(%)"]].iloc[::-1]
        st.dataframe(bot5, width='stretch', hide_index=False)

    st.markdown("---")
    st.markdown("#### 전체 랭킹")

    # 색상 매핑
    def highlight_z(val):
        if isinstance(val, str):
            if "안전" in val: return "color: #34d399"
            if "주의" in val: return "color: #fbbf24"
            if "위험" in val: return "color: #f87171"
        return ""

    display_cols = ["corp_name", "섹터", "Z_Score", "Z_판정", "부도확률(%)", "ML_판정",
                    "영업이익률(%)", "ROE(%)", "부채비율(%)", "유동비율(%)"]
    avail = [c for c in display_cols if c in df_rank.columns]
    styled = df_rank[avail].style.applymap(highlight_z, subset=["Z_판정"]).format({
        "Z_Score": "{:.3f}",
        "영업이익률(%)": "{:.1f}%",
        "ROE(%)": "{:.1f}%",
        "부채비율(%)": "{:.0f}%",
        "유동비율(%)": "{:.0f}%",
    })
    st.dataframe(styled, width='stretch')

    st.markdown("---")
    st.markdown("#### Z-Score 구성요소 분해")
    corp_select = st.selectbox("기업 선택", df_rank["corp_name"].tolist())
    row = df_rank[df_rank["corp_name"] == corp_select].iloc[0]

    components = {
        "X1 운전자본/총자산 (×1.2)": row.get("X1_운전자본/총자산", 0) * 1.2,
        "X2 순이익/총자산 (×1.4)":   row.get("X2_순이익/총자산", 0) * 1.4,
        "X3 EBIT/총자산 (×3.3)":     row.get("X3_EBIT/총자산", 0) * 3.3,
        "X4 자기자본/총부채 (×0.6)":  row.get("X4_자기자본/총부채", 0) * 0.6,
        "X5 매출액/총자산 (×1.0)":    row.get("X5_매출액/총자산", 0) * 1.0,
    }
    comp_df = pd.DataFrame({"구성요소": list(components.keys()), "기여값": list(components.values())})
    colors_comp = [COLORS["safe"] if v >= 0 else COLORS["danger"] for v in comp_df["기여값"]]
    fig = go.Figure(go.Bar(
        x=comp_df["기여값"], y=comp_df["구성요소"],
        orientation="h", marker_color=colors_comp,
        text=comp_df["기여값"].round(3), textposition="outside",
    ))
    fig.add_vline(x=0, line_color="#475569")
    fig.update_layout(
        title=f"{corp_select} Z-Score 구성 (합계: {row['Z_Score']:.3f})",
        height=300, **PLOTLY_TEMPLATE["layout"]
    )
    st.plotly_chart(fig, width='stretch')


# ─────────────────────────────────────────
# PAGE 3: 재무지표 추이
# ─────────────────────────────────────────
elif page == "📈 재무지표 추이":
    st.markdown("## 재무지표 시계열 추이")

    corps = sorted(df_sector["corp_name"].unique().tolist())
    selected_corps = st.multiselect("기업 선택 (최대 5개)", corps, default=corps[:3], max_selections=5)

    metric_options = {
        "영업이익률(%)": "영업이익률 (%)",
        "ROE(%)": "ROE (%)",
        "ROA(%)": "ROA (%)",
        "부채비율(%)": "부채비율 (%)",
        "유동비율(%)": "유동비율 (%)",
        "Z_Score": "Altman Z-Score",
        "매출성장률(%)": "매출 성장률 (%)",
        "영업이익성장률(%)": "영업이익 성장률 (%)",
        "연간변동성(%)": "주가 변동성 (%)",
    }
    selected_metric = st.selectbox("지표 선택", list(metric_options.keys()), format_func=lambda x: metric_options[x])

    if selected_corps:
        df_filtered = df_sector[df_sector["corp_name"].isin(selected_corps)]
        fig = go.Figure()
        palette = [COLORS["primary"], COLORS["safe"], COLORS["warning"], COLORS["danger"], "#a78bfa"]

        for i, corp in enumerate(selected_corps):
            corp_data = df_filtered[df_filtered["corp_name"] == corp].sort_values("year")
            fig.add_trace(go.Scatter(
                x=corp_data["year"],
                y=corp_data[selected_metric],
                name=corp,
                mode="lines+markers",
                line=dict(color=palette[i % len(palette)], width=2),
                marker=dict(size=7),
            ))

        # Z-Score 기준선
        if selected_metric == "Z_Score":
            fig.add_hline(y=2.99, line_dash="dash", line_color=COLORS["safe"],    annotation_text="안전 기준")
            fig.add_hline(y=1.81, line_dash="dash", line_color=COLORS["warning"], annotation_text="위험 기준")

        fig.update_layout(
            title=metric_options[selected_metric],
            xaxis=dict(
                tickvals=sorted(df_sector["year"].unique()),
                gridcolor="#1e2130",
                linecolor="#2e3250"
            ),
            height=420,
            legend=dict(orientation="h", y=-0.15),
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="#94a3b8", family="IBM Plex Sans KR"),
            yaxis=dict(gridcolor="#1e2130", linecolor="#2e3250"),
            margin=dict(l=40, r=20, t=40, b=40),
    )
        st.plotly_chart(fig, width='stretch')

    st.markdown("---")
    st.markdown("#### 4대 지표 동시 비교")

    if selected_corps:
        fig2 = make_subplots(rows=2, cols=2,
            subplot_titles=("영업이익률 (%)", "ROE (%)", "부채비율 (%)", "Z-Score"))
        metrics_4 = ["영업이익률(%)", "ROE(%)", "부채비율(%)", "Z_Score"]
        positions = [(1,1),(1,2),(2,1),(2,2)]

        for i, corp in enumerate(selected_corps):
            corp_data = df_sector[df_sector["corp_name"] == corp].sort_values("year")
            color = palette[i % len(palette)]
            for (row_n, col_n), metric in zip(positions, metrics_4):
                fig2.add_trace(go.Scatter(
                    x=corp_data["year"], y=corp_data[metric],
                    name=corp, showlegend=(row_n==1 and col_n==1),
                    line=dict(color=color, width=2),
                    mode="lines+markers",
                ), row=row_n, col=col_n)

        fig2.update_layout(height=500, **PLOTLY_TEMPLATE["layout"])
        fig2.update_annotations(font_color="#94a3b8")
        st.plotly_chart(fig2, width='stretch')


# ─────────────────────────────────────────
# PAGE 4: 🗂️ 섹터 비교
# ─────────────────────────────────────────
elif page == "🗂️ 섹터 비교":
    st.write("페이지 로드됨")
    st.markdown(f"## 섹터별 재무 비교 — {selected_year}년")

    metric_sel = st.selectbox("비교 지표", ["Z_Score", "영업이익률(%)", "ROE(%)", "부채비율(%)", "유동비율(%)"])

    sector_stats = df_year.groupby("섹터")[metric_sel].agg(["mean","median","min","max"]).round(2).reset_index()
    sector_stats.columns = ["섹터", "평균", "중앙값", "최소", "최대"]

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = go.Figure()
        for _, row in sector_stats.iterrows():
            fig.add_trace(go.Box(
                y=df_year[df_year["섹터"] == row["섹터"]][metric_sel].dropna(),
                name=row["섹터"],
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            ))
        fig.update_layout(
            title=f"섹터별 {metric_sel} 분포",
            height=420, showlegend=False,
            **PLOTLY_TEMPLATE["layout"]
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### 섹터 통계")
        st.dataframe(sector_stats.sort_values("평균", ascending=False), width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("#### 섹터별 Radar Chart")

    radar_metrics = ["영업이익률(%)", "ROE(%)", "유동비율(%)", "매출성장률(%)"]
    sector_radar = df_year.groupby("섹터")[radar_metrics].mean().reset_index()

    fig_radar = go.Figure()
    palette = [COLORS["primary"], COLORS["safe"], COLORS["warning"], COLORS["danger"], "#a78bfa", "#38bdf8", "#fb923c"]

    for i, row in sector_radar.iterrows():
        vals = [row[m] for m in radar_metrics]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=radar_metrics + [radar_metrics[0]],
            name=row["섹터"],
            line=dict(color=palette[i % len(palette)], width=2),
            fill="toself", opacity=0.15,
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(visible=True, gridcolor="#e2e8f0", color="#94a3b8"),
            angularaxis=dict(gridcolor="#e2e8f0", color="#475569"),
        ),
        height=420,
        **PLOTLY_TEMPLATE["layout"]
    )
    st.plotly_chart(fig_radar, width='stretch')


# ─────────────────────────────────────────
# PAGE 5: 개별 기업 스코어카드
# ─────────────────────────────────────────
elif page == "🔍 기업 스코어카드":
    st.write("페이지 로드됨")
    st.markdown("## 개별 기업 상세 스코어카드")

    corp = st.selectbox("기업 선택", sorted(df["corp_name"].unique()))
    corp_data = df[df["corp_name"] == corp].sort_values("year")
    latest = corp_data.iloc[-1]

    # 헤더
    z = latest.get("Z_Score", np.nan)
    z_color = "safe" if z > 2.99 else ("warning" if z > 1.81 else "danger")
    z_emoji = "🟢" if z > 2.99 else ("🟡" if z > 1.81 else "🔴")

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1e2130,#252840);
                border:1px solid #2e3250; border-radius:16px; padding:24px 32px; margin-bottom:24px'>
        <div style='font-size:0.75rem;color:#64748b;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px'>
            {latest.get('섹터','—')} · {int(latest['year'])}년 기준
        </div>
        <div style='font-size:2rem;font-weight:600;color:#e2e8f0;margin-bottom:4px'>{corp}</div>
        <div style='font-family:monospace;font-size:1.4rem' class='{z_color}'>
            {z_emoji} Z-Score: {z:.3f} &nbsp;|&nbsp; {latest.get('Z_판정','—')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # KPI 4개
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("영업이익률", f"{latest.get('영업이익률(%)', 0):.1f}%")
    with c2: st.metric("ROE", f"{latest.get('ROE(%)', 0):.1f}%")
    with c3: st.metric("부채비율", f"{latest.get('부채비율(%)', 0):.0f}%")
    with c4: st.metric("유동비율", f"{latest.get('유동비율(%)', 0):.0f}%")

    st.markdown("---")

    # 시계열 차트 2×2
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("매출액 (조원)", "영업이익률 (%)", "부채비율 (%)", "Z-Score 추이"))

    def tr(y_col, row, col, color=COLORS["primary"], scale=1e12):
        valid = corp_data.dropna(subset=[y_col])
        fig.add_trace(go.Scatter(
            x=valid["year"], y=valid[y_col] / scale if scale != 1 else valid[y_col],
            mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=8), showlegend=False,
        ), row=row, col=col)

    tr("매출액",      1, 1, COLORS["primary"])
    tr("영업이익률(%)", 1, 2, COLORS["safe"],    scale=1)
    tr("부채비율(%)",  2, 1, COLORS["warning"],  scale=1)
    tr("Z_Score",    2, 2, COLORS["danger"],   scale=1)

    fig.add_hline(y=2.99, line_dash="dash", line_color=COLORS["safe"],    row=2, col=2)
    fig.add_hline(y=1.81, line_dash="dash", line_color=COLORS["warning"], row=2, col=2)
    fig.update_layout(height=480, **PLOTLY_TEMPLATE["layout"])
    fig.update_annotations(font_color="#94a3b8")
    st.plotly_chart(fig, width='stretch')

    # 5년 재무 요약 테이블
    st.markdown("#### 5개년 재무 요약")
    summary_cols = ["year", "매출액", "영업이익", "당기순이익",
                    "영업이익률(%)", "ROE(%)", "부채비율(%)", "Z_Score"]
    avail = [c for c in summary_cols if c in corp_data.columns]
    summary = corp_data[avail].copy()

    # 금액 단위: 조원
    for col in ["매출액", "영업이익", "당기순이익"]:
        if col in summary.columns:
            summary[col] = (summary[col] / 1e12).round(2).astype(str) + "조"

    summary["year"] = summary["year"].astype(int)
    st.dataframe(summary.set_index("year"), width='stretch')