"""
app.py
다중 지표 앙상블(Multi-Indicator Ensemble) 모델 기반
기술적 분석 종목 스크리너 웹 애플리케이션
"""

import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import FinanceDataReader as fdr

from engine import run_screening_task
from indicators import calculate_technical_indicators
from excel_exporter import create_excel_bytes, create_csv_bytes, prepare_export_dataframe

# ---------------- 1. 페이지 환경 설정 ----------------
st.set_page_config(
    page_title="다중 지표 앙상블 스크리너",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- 2. 커스텀 CSS (00 Bookmarks 테마 일체화) ----------------
st.markdown("""
<style>
    /* 메인 배경 및 폰트 색상 */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }

    /* 메인 컨테이너 패딩 조절 */
    .main .block-container,
    [data-testid="stMainBlockContainer"] {
        padding-top: 2.0rem !important;
        padding-bottom: 3.5rem !important;
    }

    /* 사이드바 스타일링 */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }

    /* 사이드바 너비 편의 설정 */
    section[data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 320px !important;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        min-width: 0px !important;
        width: 0px !important;
    }

    /* =========================================================
       사이드바 접기(<<) 및 펼치기(>>) 버튼 항상 표시 및 시인성/대비 강화
       ========================================================= */
    /* 1. 사이드바가 열려 있을 때 접기 버튼 (<<) 상시 표시 */
    [data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        display: inline-flex !important;
    }
    
    [data-testid="stSidebarCollapseButton"] button {
        visibility: visible !important;
        opacity: 1 !important;
        background-color: #1e293b !important;       /* 진한 네이비 배경 */
        border: 1.5px solid #38bdf8 !important;     /* 선명한 스카이블루 테두리로 상자 명확화 */
        border-radius: 8px !important;
        width: 38px !important;
        height: 38px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 6px rgba(56, 189, 248, 0.2) !important;
        transition: all 0.2s ease !important;
    }
    
    /* 상자 내부의 << 아이콘(Material Icon span/svg/문자)을 순백색으로 강제하여 상자와 극명한 대비 구현 */
    [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="stSidebarCollapseButton"] span,
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
    }
    
    /* 호버(PC) 및 터치 시 반전 효과 */
    [data-testid="stSidebarCollapseButton"] button:hover {
        background-color: #38bdf8 !important;
        border-color: #38bdf8 !important;
    }
    [data-testid="stSidebarCollapseButton"] button:hover * {
        color: #0f172a !important;
        fill: #0f172a !important;
    }

    /* 2. 사이드바 헤더 영역 패딩 및 정렬 보정 */
    [data-testid="stSidebarHeader"] {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }

    /* 3. 사이드바가 닫혔을 때 다시 여는 버튼 (>>) 시인성 강화 */
    [data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] button {
        background-color: #1e293b !important;
        border: 1.5px solid #38bdf8 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4), 0 0 6px rgba(56, 189, 248, 0.2) !important;
    }
    
    [data-testid="stSidebarCollapsedControl"] button *,
    [data-testid="stSidebarCollapsedControl"] span,
    [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapsedControl"] svg {
        color: #38bdf8 !important;
        fill: #38bdf8 !important;
        opacity: 1 !important;
        visibility: visible !important;
        font-size: 1.35rem !important;
    }

    /* 헤더 및 타이틀 색상 (#8AB4F8) */
    h1, .app-main-title {
        color: #8AB4F8 !important;
        font-size: 2.0rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }

    h2, h3, h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }

    /* 모델 설명 카드 컴포넌트 */
    .ensemble-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }

    /* 가중치 배지 칩 (Pill Badges) */
    .weight-chip-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }

    .weight-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background-color: rgba(30, 41, 59, 0.85);
        color: #e2e8f0;
    }

    .chip-highlight {
        font-weight: 800;
        color: #38bdf8;
    }

    /* KPI 요약 카드 */
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
    }
    .kpi-label {
        font-size: 0.80rem;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #38bdf8;
    }

    /* 판정 기준 뱃지 */
    .grade-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Primary Button Styling (39 DividendStock 테마 통일) */
    .stButton button[kind="primary"],
    .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] button[kind="primary"] {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    .stButton button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background-color: #1d4ed8 !important;
        box-shadow: 0 0 10px rgba(37, 99, 235, 0.4) !important;
    }

    /* 다운로드 버튼 공통 통일 스타일 */
    div[data-testid="stDownloadButton"] > button,
    .stDownloadButton > button {
        background-color: #334155 !important;
        color: #f8fafc !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        line-height: 36px !important;
        padding: 0 16px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        transition: all 0.2s ease-in-out !important;
        box-sizing: border-box !important;
    }
    div[data-testid="stDownloadButton"] > button:hover,
    .stDownloadButton > button:hover {
        background-color: #475569 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
    }
    div[data-testid="stDownloadButton"] > button:active,
    .stDownloadButton > button:active {
        background-color: #1e293b !important;
        border-color: #0284c7 !important;
    }
    div[data-testid="stDownloadButton"] > button p,
    div[data-testid="stDownloadButton"] > button span,
    .stDownloadButton > button p,
    .stDownloadButton > button span {
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: inherit !important;
        line-height: inherit !important;
        margin: 0 !important;
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- 3. 세션 상태 초기화 및 스키마 검증 ----------------
SCHEMA_VERSION = 2
if st.session_state.get("schema_version") != SCHEMA_VERSION:
    st.session_state.screening_results = None
    st.session_state.schema_version = SCHEMA_VERSION

if "screening_results" not in st.session_state:
    st.session_state.screening_results = None
if "last_screened_market" not in st.session_state:
    st.session_state.last_screened_market = None
if "last_screened_scope" not in st.session_state:
    st.session_state.last_screened_scope = None
if "has_run_screening" not in st.session_state:
    st.session_state.has_run_screening = False


# ---------------- 4. 왼쪽 사이드바 (필터 및 스크리닝 실행) ----------------
with st.sidebar:
    st.markdown("<h2 style='color: #8AB4F8; font-size: 1.4rem; margin-bottom: 12px;'>⚙️ 스크리닝 설정</h2>", unsafe_allow_html=True)
    st.caption("시장 및 스크리닝 조건을 설정하고 발굴을 시작하세요.")
    
    st.markdown("---")

    # 1) 시장 선택 (KOSPI vs KOSDAQ)
    market_choice = st.radio(
        "🏛️ 시장 선택",
        options=["코스피 (KOSPI)", "코스닥 (KOSDAQ)"],
        index=0,
        help="스크리닝할 국내 주식 시장을 선택합니다."
    )
    market_code = "KOSPI" if "코스피" in market_choice else "KOSDAQ"

    st.markdown("---")
    st.subheader("🎯 앙상블 조건 필터")

    # 2) 대상 범위 선택 (속도 및 정밀도 조절)
    scope_options = {
        "시총 상위 300 (쾌속 모드 ~15초)": "top300",
        "시총 상위 500 (권장 모드 ~25초)": "top500",
        "시총 상위 1,000 (심층 모드 ~50초)": "top1000",
        "시장 전체 종목 (전체 모드)": "all"
    }
    scope_selected_label = st.selectbox(
        "🎯 스크리닝 대상 범위",
        options=list(scope_options.keys()),
        index=1,
        help="시가총액 상위 종목 위주로 분석하여 스크리닝 속도를 최적화합니다."
    )
    scope_code = scope_options[scope_selected_label]

    # 3) 최소 시가총액 필터
    marcap_filter_options = {
        "제한 없음 (전체)": 0,
        "500억원 이상": 500,
        "1,000억원 이상": 1000,
        "3,000억원 이상": 3000,
        "5,000억원 이상": 5000,
        "1조원 이상": 10000
    }
    marcap_label = st.selectbox(
        "💰 최소 시가총액",
        options=list(marcap_filter_options.keys()),
        index=2,
        help="극단적인 초소형/동전주를 배제하여 안정성을 높입니다."
    )
    min_marcap_val = marcap_filter_options[marcap_label]

    # 4) 종합 점수 필터 (기본: 매수 BUY 45점 이상)
    opinion_filter_options = {
        "매수 (BUY) 종목만 (종합 점수 ≥ 45) [권장]": 45,
        "비중확대 이상 (종합 점수 ≥ 15)": 15,
        "전체 종목 보기 (제한 없음)": -100
    }
    opinion_choice_label = st.selectbox(
        "📊 발굴 조건 (최소 점수)",
        options=list(opinion_filter_options.keys()),
        index=0,
        help="종합 점수 기준 충족 종목만 필터링합니다."
    )
    min_score_cutoff = opinion_filter_options[opinion_choice_label]

    st.markdown("---")

    # 5) 스크리닝 실행 버튼
    run_btn = st.button("🔍 스크리닝 시작", type="primary", use_container_width=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    st.info(
        "💡 **알림**: 멀티스레딩 엔진이 백그라운드에서 실시간 데이터를 수집 및 병렬 연산합니다. "
        "일반적으로 15~30초 내에 전 종목 앙상블 스크리닝이 완료됩니다."
    )


# ---------------- 5. 오른쪽 대시보드 - 타이틀 영역 ----------------
# 00 Bookmarks 스타일의 철학적 인용구 및 메인 타이틀
st.markdown(
    "<h1 style='text-align: center; font-size: 2.0rem; font-weight: 800; line-height: 1.35; margin: 0 0 8px 0; color: #8AB4F8 !important;'>"
    "다중 지표 앙상블(Multi-Indicator Ensemble) 종목 발굴 엔진"
    "</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align: center; font-size: 0.95rem; color: #cbd5e1; margin-bottom: 20px; line-height: 1.5;'>"
    "6대 핵심 기술적 지표의 가중치 앙상블 결합을 통해 시장 노이즈를 제거하고, 정량적 매수(BUY) 종목을 정밀 발굴합니다."
    "</div>",
    unsafe_allow_html=True
)

# 서브타이틀 바로 아래, 기술적 종합 점수 산출 기준 카드 바로 위 가로선
st.markdown("<hr style='border: 0; height: 1px; background-color: #334155; margin: 18px 0 22px 0;'>", unsafe_allow_html=True)

# 기술적 종합 점수 산출 기준 안내 카드 배너
st.markdown("""
<div class="ensemble-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc;">📐 기술적 종합 점수 산출 기준 (가중치 100%)</span>
        <span style="font-size: 0.80rem; color: #94a3b8;">종합 점수 범위: -100점 ~ +100점 | 매수(BUY) 판정 기준: <b>+45점 이상</b></span>
    </div>
    <div class="weight-chip-grid">
        <div class="weight-chip">📈 이동평균선 배열 <span class="chip-highlight">25% (25점)</span></div>
        <div class="weight-chip">📊 MACD 추세강도 <span class="chip-highlight">20% (20점)</span></div>
        <div class="weight-chip">⚡ RSI 모멘텀 <span class="chip-highlight">20% (20점)</span></div>
        <div class="weight-chip">🎯 볼린저 밴드 <span class="chip-highlight">15% (15점)</span></div>
        <div class="weight-chip">🌊 스토캐스틱 <span class="chip-highlight">10% (10점)</span></div>
        <div class="weight-chip">📢 거래량 수급 <span class="chip-highlight">10% (10점)</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------- 6. 스크리닝 실행 로직 ----------------
if run_btn:
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(current, total, name):
        ratio = min(1.0, current / total) if total > 0 else 0.0
        progress_bar.progress(ratio)
        status_text.markdown(f"⏳ **스크리닝 분석 진행 중...** ({current}/{total}) `{name}` 분석 완료")

    start_time = time.time()
    try:
        with st.spinner("코스피/코스닥 전 종목 데이터 병렬 수집 및 앙상블 분석 중..."):
            df_screened = run_screening_task(
                market=market_code,
                scope=scope_code,
                min_score=min_score_cutoff,
                min_marcap_eok=min_marcap_val,
                max_workers=24,
                progress_callback=update_progress
            )
            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            if df_screened is not None and not df_screened.empty:
                status_text.success(f"✅ 스크리닝 완료! ({len(df_screened)}개 종목 분석, 소요 시간: {elapsed:.1f}초)")
            else:
                status_text.warning(f"⚠️ 조건에 부합하는 종목이 없습니다. (소요 시간: {elapsed:.1f}초)")
            time.sleep(1.0)
            progress_bar.empty()
            status_text.empty()

            st.session_state.screening_results = df_screened
            st.session_state.last_screened_market = market_code
            st.session_state.last_screened_scope = scope_selected_label
            st.session_state.has_run_screening = True
    except Exception as e:
        st.error(f"스크리닝 도중 오류가 발생했습니다: {e}")


# ---------------- 7. 결과 대시보드 렌더링 ----------------
df_all_results = st.session_state.screening_results
has_run = st.session_state.get("has_run_screening", False)

if has_run and (df_all_results is None or df_all_results.empty):
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px; background-color: #1e293b; border: 1px solid #eab308; border-radius: 12px; margin-top: 20px;">
        <div style="font-size: 2.2rem; margin-bottom: 12px;">⚠️</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #facc15; margin-bottom: 8px;">
            스크리닝 대상 종목을 찾지 못했습니다
        </div>
        <div style="font-size: 0.90rem; color: #cbd5e1; max-width: 540px; margin: 0 auto; line-height: 1.6;">
            선택한 <b>시장</b> 또는 <b>최소 시가총액</b> 기준이 너무 높을 수 있습니다.<br>
            왼쪽 사이드바에서 <b>'최소 시가총액'</b>을 <b>'제한 없음 (전체)'</b> 또는 더 낮은 금액으로 조정한 후 다시 <b>[스크리닝 실행]</b>을 눌러보세요.
        </div>
    </div>
    """, unsafe_allow_html=True)
elif df_all_results is not None and not df_all_results.empty:
    # 데이터프레임 컬럼 스키마 무결성 보장 (구버전 캐시 호환)
    if "시가총액" not in df_all_results.columns:
        if "시가총액_원" in df_all_results.columns:
            df_all_results["시가총액"] = (pd.to_numeric(df_all_results["시가총액_원"], errors="coerce").fillna(0) / 100_000_000).round().astype(int)
        else:
            df_all_results["시가총액"] = 0

    if "시가총액_표시" not in df_all_results.columns:
        df_all_results["시가총액_표시"] = df_all_results["시가총액"].apply(lambda x: f"{x:,}억원")

    if "현재가" not in df_all_results.columns:
        df_all_results["현재가"] = 0

    if "거래량 수급" not in df_all_results.columns:
        df_all_results["거래량 수급"] = df_all_results.get("거래량", "")

    if "거래량" not in df_all_results.columns or not pd.api.types.is_numeric_dtype(df_all_results["거래량"]):
        df_all_results["거래량"] = 0

    # 필터 적용 (종합 점수 컷오프)
    df_filtered = df_all_results[df_all_results["종합 점수"] >= min_score_cutoff].copy()

    # 종합 점수 기준 내림차순 기본 정렬
    df_filtered = df_filtered.sort_values(by="종합 점수", ascending=False).reset_index(drop=True)

    # 1) 상단 통계 KPI 카드
    buy_count = len(df_filtered[df_filtered["종합 점수"] >= 45])
    total_screened = len(df_all_results)
    avg_score = df_filtered["종합 점수"].mean() if not df_filtered.empty else 0.0
    top_stock_name = df_filtered.iloc[0]["종목명"] if not df_filtered.empty else "-"
    top_stock_score = df_filtered.iloc[0]["종합 점수"] if not df_filtered.empty else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">발굴 종목 수 (선택 조건)</div>
            <div class="kpi-value">{len(df_filtered):,} <span style="font-size: 0.9rem; color: #94a3b8;">/ {total_screened:,}개</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">매수 (BUY) 등급 종목</div>
            <div class="kpi-value" style="color: #10b981;">{buy_count:,} <span style="font-size: 0.9rem; color: #94a3b8;">개</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">발굴 종목 평균 점수</div>
            <div class="kpi-value">{avg_score:.1f} <span style="font-size: 0.9rem; color: #94a3b8;">점</span></div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">최고 종합 점수 종목</div>
            <div class="kpi-value" style="color: #38bdf8;">{top_stock_name} <span style="font-size: 0.9rem; color: #94a3b8;">({top_stock_score:+d}점)</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # 2) 테이블 상단 툴바 (검색창 및 CSV/엑셀 다운로드 버튼: 좌측 CSV, 우측 엑셀)
    col_search, col_dl_csv, col_dl_excel = st.columns([2.5, 1, 1])
    with col_search:
        search_query = st.text_input("🔍 결과 내 종목 검색", placeholder="종목명 또는 종목코드 입력...", label_visibility="collapsed")
    
    # 검색어 필터링
    df_display = df_filtered.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        df_display = df_display[
            df_display["종목명"].str.lower().str.contains(q) |
            df_display["종목코드"].str.contains(q)
        ]

    # CSV 및 엑셀 다운로드 버튼 (좌측 CSV, 우측 엑셀 표준)
    excel_bytes = create_excel_bytes(df_display)
    csv_bytes = create_csv_bytes(df_display)
    now_str = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M")

    with col_dl_csv:
        st.download_button(
            label="📥 CSV 파일 다운로드",
            data=csv_bytes,
            file_name=f"MultiEnsemble_{st.session_state.last_screened_market}_{now_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl_excel:
        st.download_button(
            label="📥 엑셀 파일 다운로드",
            data=excel_bytes,
            file_name=f"MultiEnsemble_{st.session_state.last_screened_market}_{now_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    # 3) 엑셀 형식 데이터 영역 (헤더 규격)
    # 헤더: 종목명, 종목코드, 시장, 시가총액, 현재가, 거래량, 이동평균선 배열, MACD, RSI, 볼린저 밴드, 스토캐스틱, 거래량 수급, 종합 점수
    df_table = df_display.copy()

    # 이전 세션 캐시 및 컬럼 누락 방지 안전 처리
    if "시가총액" not in df_table.columns:
        if "시가총액_원" in df_table.columns:
            df_table["시가총액"] = (pd.to_numeric(df_table["시가총액_원"], errors="coerce").fillna(0) / 100_000_000).round().astype(int)
        else:
            df_table["시가총액"] = 0

    if "현재가" not in df_table.columns:
        df_table["현재가"] = 0

    if "거래량 수급" not in df_table.columns:
        df_table["거래량 수급"] = df_table.get("거래량", "")

    if "거래량" not in df_table.columns or not pd.api.types.is_numeric_dtype(df_table["거래량"]):
        df_table["거래량"] = 0

    display_cols = [
        "종목명",
        "종목코드",
        "시장",
        "시가총액",
        "현재가",
        "거래량",
        "이동평균선 배열",
        "MACD",
        "RSI",
        "볼린저 밴드",
        "스토캐스틱",
        "거래량 수급",
        "종합 점수"
    ]
    # 모든 display_cols가 존재하는지 확인 후 생성
    for col in display_cols:
        if col not in df_table.columns:
            df_table[col] = "-"

    df_table = df_table[display_cols].copy()

    # 컬럼 설정 (정렬, 너비, 서식)
    st.dataframe(
        df_table,
        use_container_width=True,
        hide_index=True,
        height=450,
        column_config={
            "종목명": st.column_config.TextColumn("종목명", width="medium"),
            "종목코드": st.column_config.TextColumn("종목코드", width="small"),
            "시장": st.column_config.TextColumn("시장", width="small"),
            "시가총액": st.column_config.NumberColumn(
                "시가총액",
                help="시가총액 (단위: 억원). 클릭하여 숫자 크기순으로 오름차순/내림차순 정렬할 수 있습니다.",
                format="%,d 억원",
                width="medium"
            ),
            "현재가": st.column_config.NumberColumn(
                "현재가",
                help="최신 종가 (단위: 원)",
                format="%,d 원",
                width="small"
            ),
            "거래량": st.column_config.NumberColumn(
                "거래량",
                help="당일 체결 거래량 (단위: 주)",
                format="%,d 주",
                width="medium"
            ),
            "이동평균선 배열": st.column_config.TextColumn("이동평균선 배열 (25%)", width="medium"),
            "MACD": st.column_config.TextColumn("MACD (20%)", width="medium"),
            "RSI": st.column_config.TextColumn("RSI (20%)", width="medium"),
            "볼린저 밴드": st.column_config.TextColumn("볼린저 밴드 (15%)", width="medium"),
            "스토캐스틱": st.column_config.TextColumn("스토캐스틱 (10%)", width="medium"),
            "거래량 수급": st.column_config.TextColumn("거래량 수급 (10%)", width="medium"),
            "종합 점수": st.column_config.ProgressColumn(
                "종합 점수",
                help="가중치 앙상블 종합 점수 (-100 ~ +100점)",
                format="%d점",
                min_value=-100,
                max_value=100,
                width="medium"
            )
        }
    )
    st.caption("💡 각 열 헤더를 클릭하여 오름차순/내림차순 정렬을 토글할 수 있습니다. 기본 정렬은 '종합 점수' 내림차순입니다.")

    st.markdown("---")

    # 4) 전문가 제안 시너지 기능: 발굴 종목 "원클릭 퀵 차트 뷰어 (Deep Dive)"
    with st.expander("🔍 발굴 종목 퀵 차트 뷰어 (캔들스틱 + 6대 지표 정밀 확인)", expanded=True):
        stock_options = [f"{row['종목명']} ({row['종목코드']}) - {row['종합 점수']}점 [{row['투자의견']}]" for _, row in df_display.iterrows()]
        
        if stock_options:
            selected_stock_label = st.selectbox("진단할 종목 선택", options=stock_options, index=0)
            # 코드 추출
            sel_code = selected_stock_label.split("(")[1].split(")")[0].strip()
            sel_name = selected_stock_label.split("(")[0].strip()

            # 차트와 우측 진단 카드가 나란히 2단으로 배치되도록 최적 비율 설정 (차트 70% : 카드 30%)
            col_chart_left, col_chart_right = st.columns([2.3, 1.0])

            # 주가 데이터 로드
            start_fetch = (pd.Timestamp.now() - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
            df_stock_raw = fdr.DataReader(sel_code, start_fetch)

            if df_stock_raw is not None and not df_stock_raw.empty:
                df_stock_ind = calculate_technical_indicators(df_stock_raw)

                # Plotly 서브플롯 차트 생성 (캔들 + 이평선 + 볼린저 / 거래량 / MACD / RSI)
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    row_heights=[0.5, 0.15, 0.18, 0.17]
                )

                # 1행: 캔들스틱 + SMA(20, 60, 120) + 볼린저 밴드
                fig.add_trace(
                    go.Candlestick(
                        x=df_stock_ind.index,
                        open=df_stock_ind['Open'],
                        high=df_stock_ind['High'],
                        low=df_stock_ind['Low'],
                        close=df_stock_ind['Close'],
                        name="주가 (OHLC)",
                        increasing_line_color='#ef4444',
                        decreasing_line_color='#3b82f6'
                    ), row=1, col=1
                )

                if "SMA_20" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["SMA_20"], name="SMA 20 (생명선)", line=dict(color="#f59e0b", width=1.5)), row=1, col=1)
                if "SMA_60" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["SMA_60"], name="SMA 60 (수급선)", line=dict(color="#10b981", width=1.5)), row=1, col=1)
                if "SMA_120" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["SMA_120"], name="SMA 120 (경기선)", line=dict(color="#8b5cf6", width=1.5)), row=1, col=1)
                if "BB_Upper" in df_stock_ind and "BB_Lower" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["BB_Upper"], name="볼린저 상단", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dash")), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["BB_Lower"], name="볼린저 하단", line=dict(color="rgba(148, 163, 184, 0.4)", dash="dash")), row=1, col=1)

                # 2행: 거래량
                vol_colors = ['#ef4444' if c >= o else '#3b82f6' for c, o in zip(df_stock_ind['Close'], df_stock_ind['Open'])]
                fig.add_trace(go.Bar(x=df_stock_ind.index, y=df_stock_ind['Volume'], name="거래량", marker_color=vol_colors), row=2, col=1)
                if "Vol_SMA20" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["Vol_SMA20"], name="거래량 20MA", line=dict(color="#fbbf24", width=1.2)), row=2, col=1)

                # 3행: MACD
                if "MACD" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["MACD"], name="MACD", line=dict(color="#38bdf8", width=1.5)), row=3, col=1)
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["MACD_Signal"], name="Signal", line=dict(color="#f43f5e", width=1.5)), row=3, col=1)
                    hist_colors = ['#ef4444' if h >= 0 else '#3b82f6' for h in df_stock_ind['MACD_Hist']]
                    fig.add_trace(go.Bar(x=df_stock_ind.index, y=df_stock_ind['MACD_Hist'], name="Hist", marker_color=hist_colors), row=3, col=1)

                # 4행: RSI
                if "RSI" in df_stock_ind:
                    fig.add_trace(go.Scatter(x=df_stock_ind.index, y=df_stock_ind["RSI"], name="RSI (14)", line=dict(color="#a855f7", width=1.5)), row=4, col=1)
                    fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=4, col=1)
                    fig.add_hline(y=30, line_dash="dot", line_color="#10b981", row=4, col=1)

                fig.update_layout(
                    template="plotly_dark",
                    height=650,
                    margin=dict(l=10, r=10, t=20, b=10),
                    paper_bgcolor='#1E293B',
                    plot_bgcolor='#0F172A',
                    font=dict(color='#cbd5e1'),
                    xaxis_rangeslider_visible=False,
                    showlegend=False
                )
                fig.update_xaxes(gridcolor='#334155')

                # 주말(토/일) 및 시장 휴장일(공휴일) 공백 제거 (봉이 끊기지 않고 연속 연결)
                if len(df_stock_ind) > 1:
                    all_b_days_29 = pd.date_range(start=df_stock_ind.index[0], end=df_stock_ind.index[-1], freq='B')
                    holidays_29 = [d.strftime("%Y-%m-%d") for d in all_b_days_29 if d not in df_stock_ind.index]
                    rbreaks_29 = [dict(bounds=["sat", "mon"])]
                    if holidays_29:
                        rbreaks_29.append(dict(values=holidays_29))
                    fig.update_xaxes(rangebreaks=rbreaks_29)

                # 왼쪽 Y축 4개 레이블 설정 (가격, 거래량, MACD, RSI)
                fig.update_yaxes(title_text="가격 (원)", title_font=dict(size=11, color='#94a3b8'), gridcolor='#334155', row=1, col=1)
                fig.update_yaxes(title_text="거래량", title_font=dict(size=11, color='#94a3b8'), gridcolor='#334155', row=2, col=1)
                fig.update_yaxes(title_text="MACD", title_font=dict(size=11, color='#94a3b8'), gridcolor='#334155', row=3, col=1)
                fig.update_yaxes(title_text="RSI", title_font=dict(size=11, color='#94a3b8'), gridcolor='#334155', row=4, col=1)

                with col_chart_left:
                    st.plotly_chart(fig, use_container_width=True)

                with col_chart_right:
                    # 선택 종목 세부 진단 결과 카드 (st.html 사용으로 HTML 코드 노출 원천 차단)
                    sel_row = df_display[df_display["종목코드"] == sel_code].iloc[0]
                    vol_diag = sel_row.get("거래량 수급", sel_row.get("거래량", ""))
                    
                    if "시가총액_표시" in sel_row and pd.notna(sel_row["시가총액_표시"]):
                        marcap_disp = str(sel_row["시가총액_표시"])
                    elif "시가총액" in sel_row and pd.notna(sel_row["시가총액"]):
                        marcap_disp = f"{int(sel_row['시가총액']):,}억원"
                    elif "시가총액_원" in sel_row and pd.notna(sel_row["시가총액_원"]):
                        marcap_disp = f"{int(round(sel_row['시가총액_원'] / 100_000_000)):,}억원"
                    else:
                        marcap_disp = "-"

                    card_html = f"""<div style="background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px; height: 100%; box-sizing: border-box;">
<div style="font-size: 1.15rem; font-weight: 800; color: #8AB4F8; margin-bottom: 2px;">{sel_row['종목명']} ({sel_row['종목코드']})</div>
<div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 12px;">{sel_row['시장']} | 시총: {marcap_disp}</div>
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; padding: 8px 10px; background-color: #0f172a; border-radius: 6px;">
<span style="font-weight: 700; color: #f8fafc;">종합 점수</span>
<span style="font-size: 1.3rem; font-weight: 800; color: #10b981;">{sel_row['종합 점수']:+d}점</span>
</div>
<div style="font-size: 0.85rem; font-weight: 700; color: #cbd5e1; margin-bottom: 6px;">6대 지표 진단 현황</div>
<ul style="padding-left: 16px; font-size: 0.82rem; color: #94a3b8; line-height: 1.6; margin-bottom: 12px;">
<li><b>이평선</b>: {sel_row['이동평균선 배열']}</li>
<li><b>MACD</b>: {sel_row['MACD']}</li>
<li><b>RSI</b>: {sel_row['RSI']}</li>
<li><b>볼린저</b>: {sel_row['볼린저 밴드']}</li>
<li><b>스토캐스틱</b>: {sel_row['스토캐스틱']}</li>
<li><b>거래량</b>: {vol_diag}</li>
</ul>
<div style="background-color: #0f172a; border-left: 3px solid #10b981; padding: 8px 10px; border-radius: 4px; font-size: 0.80rem; color: #cbd5e1;">
<b>투자 의견</b>: <span style="color: #10b981; font-weight: 700;">{sel_row['투자의견']}</span><br>
기술적 다중 지표가 매수 신호를 강하게 뒷받침하고 있습니다.
</div>
</div>"""
                    st.html(card_html)
            else:
                st.warning("선택한 종목의 차트 데이터를 불러오지 못했습니다.")
else:
    # 아직 스크리닝이 실행되지 않았을 때의 안내 화면
    st.markdown("""
    <div style="text-align: center; padding: 50px 20px; background-color: #1e293b; border: 1px dashed #334155; border-radius: 12px; margin-top: 20px;">
        <div style="font-size: 2.5rem; margin-bottom: 14px;">🔍</div>
        <div style="font-size: 1.25rem; font-weight: 700; color: #8AB4F8; margin-bottom: 8px;">
            스크리닝을 시작할 준비가 되었습니다
        </div>
        <div style="font-size: 0.90rem; color: #94a3b8; max-width: 580px; margin: 0 auto 20px auto; line-height: 1.6;">
            왼쪽 사이드바에서 시장(코스피 / 코스닥)과 대상 범위를 선택한 후, <b>[스크리닝 실행]</b> 버튼을 누르면
            6대 기술적 지표 앙상블 평가를 거쳐 <b>'매수(BUY)'</b>에 해당하는 유망 종목을 찾아냅니다.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- 8. 페이지 푸터 ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 8px;'>"
    "Technical Analysis Ensemble Screener | 다중 지표 앙상블 모델 종목 발굴 시스템"
    "</div>"
    "<div style='text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 4px; margin-bottom: 24px; line-height: 1.6;'>"
    "⚠️ 본 서비스에서 제공하는 모든 정보는 투자 참고용이며, 투자의 최종 결정과 책임은 투자자 본인에게 있습니다."
    "</div>",
    unsafe_allow_html=True
)
