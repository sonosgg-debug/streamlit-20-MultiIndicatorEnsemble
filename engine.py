"""
engine.py
코스피(KOSPI) 및 코스닥(KOSDAQ) 시장의 종목 목록을 추출하고,
멀티스레딩(ThreadPoolExecutor)을 활용하여 초고속으로 주가 데이터를 수집 및
다중 지표 앙상블 스코어를 정량 평가하는 스크리너 엔진 모듈
"""

import os
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import FinanceDataReader as fdr

from indicators import calculate_technical_indicators, evaluate_ensemble_score


def format_marcap(val) -> str:
    """
    시가총액(원 또는 백만원/억원 단위)을 'X조 Y,ZZZ억원' 또는 'Z,ZZZ억원' 문자열로 포맷팅합니다.
    FinanceDataReader의 Marcap은 보통 '원' 단위입니다.
    """
    if pd.isna(val) or val <= 0:
        return "-"
    
    # fdr Marcap이 '원' 단위인 경우 (1조 = 1,000,000,000,000)
    # 때에 따라 '억원' 단위로 들어올 수도 있으므로 자릿수로 판단
    if val >= 1_000_000_000:  # 원 단위 (예: 1조 = 1,000,000,000,000원)
        marcap_won = val
    else:  # 억원 단위로 들어온 경우
        marcap_won = val * 100_000_000

    eok = marcap_won / 100_000_000  # 억원
    if eok >= 10_000:
        jo = int(eok // 10_000)
        rem_eok = int(eok % 10_000)
        if rem_eok > 0:
            return f"{jo:,}조 {rem_eok:,}억원"
        return f"{jo:,}조원"
    else:
        return f"{int(eok):,}억원"


def is_valid_screening_stock(code: str, name: str) -> bool:
    """
    기술적 분석 스크리닝에 부적합한 노이즈 종목(우선주, 스팩, ETF/ETN, 정리매매 등)을 걸러냅니다.
    """
    code_str = str(code).strip()
    name_str = str(name).strip()

    # 1. 우선주 배제 (보통 끝자리 '0'이 아니거나 명칭에 '우' 포함)
    if not code_str.endswith('0'):
        return False
    if name_str.endswith('우') or ('우B' in name_str) or ('우C' in name_str):
        return False

    # 2. 스팩(SPAC) 배제
    if '스팩' in name_str:
        return False

    # 3. ETF / ETN 배제 (KODEX, TIGER, KBSTAR, ACE, SOL, HANARO 등)
    etf_prefixes = ['KODEX', 'TIGER', 'KBSTAR', 'ACE', 'SOL', 'HANARO', 'KOSEF', 'ARIRANG', 'PLUS', 'TIMEFOLIO']
    for prefix in etf_prefixes:
        if name_str.startswith(prefix):
            return False

    # 4. 리츠, 인프라투융자회사 등 일반 제조업/서비스업과 차트 속성이 다른 종목 배제
    if name_str.endswith('리츠') or '투융자' in name_str:
        return False

    return True


_CACHED_FALLBACK_MARCAP = None


def get_fallback_marcap_map() -> dict:
    """
    FinanceDataReader의 KRX 당일자 캐시 파일이 장중/장마감 전이라 시가총액(Marcap)이 결측치(NaN)인 경우,
    최근 10영업일을 역순으로 탐색하여 시가총액이 유효하게 존재하는 가장 최근 거래일의 데이터를 로드하고
    {종목코드: 시가총액} 매핑 딕셔너리를 반환합니다.
    """
    global _CACHED_FALLBACK_MARCAP
    if _CACHED_FALLBACK_MARCAP is not None and len(_CACHED_FALLBACK_MARCAP) > 0:
        return _CACHED_FALLBACK_MARCAP

    base_url = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/krx/"
    today = datetime.now()

    # 오늘부터 최근 10일간 역순 탐색
    for days_back in range(0, 11):
        target_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = f"{base_url}{target_date}.csv"
        try:
            df = pd.read_csv(url, dtype={"Code": str, "ISU_SRT_CD": str}, low_memory=False)
            code_col = "Code" if "Code" in df.columns else "ISU_SRT_CD"
            marcap_col = None
            for col in ["Marcap", "MKTCAP", "시가총액"]:
                if col in df.columns:
                    marcap_col = col
                    break

            if marcap_col and code_col in df.columns:
                valid_series = pd.to_numeric(df[marcap_col], errors="coerce")
                if valid_series.notna().sum() > 500:
                    df["clean_code"] = df[code_col].astype(str).str.zfill(6)
                    df["clean_marcap"] = valid_series.fillna(0)
                    marcap_dict = dict(zip(df["clean_code"], df["clean_marcap"]))
                    _CACHED_FALLBACK_MARCAP = marcap_dict
                    return _CACHED_FALLBACK_MARCAP
        except Exception:
            continue

    return {}


def get_market_universe(market: str = "KOSPI", scope: str = "top500", min_marcap_eok: int = 0) -> pd.DataFrame:
    """
    지정된 시장(KOSPI 또는 KOSDAQ)의 종목 목록을 로드하고 필터링 및 범위 지정을 수행합니다.
    시가총액 결측치 발생 시 최근 유효 거래일 데이터를 자동 보정합니다.
    """
    market_code = "KOSPI" if "KOSPI" in market.upper() or "코스피" in market else "KOSDAQ"
    
    try:
        df_listing = fdr.StockListing(market_code)
    except Exception as e:
        print(f"StockListing({market_code}) 로드 실패: {e}")
        # 폴백: KRX 전체에서 시장 추출
        df_all = fdr.StockListing("KRX")
        if "Market" in df_all.columns:
            df_listing = df_all[df_all["Market"].str.upper() == market_code].copy()
        elif "MarketId" in df_all.columns:
            target_id = "STK" if market_code == "KOSPI" else "KSQ"
            df_listing = df_all[df_all["MarketId"] == target_id].copy()
        else:
            df_listing = df_all.copy()

    # 필수 컬럼 검증 및 통일
    if "Code" not in df_listing.columns and "Symbol" in df_listing.columns:
        df_listing["Code"] = df_listing["Symbol"]
    if "Code" not in df_listing.columns and "ISU_SRT_CD" in df_listing.columns:
        df_listing["Code"] = df_listing["ISU_SRT_CD"]

    df_listing["Code"] = df_listing["Code"].astype(str).str.zfill(6)

    # 마켓 필터 재확인 (fdr 버전에 따라 전체가 들어올 수 있으므로)
    if "MarketId" in df_listing.columns:
        target_id = "STK" if market_code == "KOSPI" else "KSQ"
        if (df_listing["MarketId"] == target_id).any():
            df_listing = df_listing[df_listing["MarketId"] == target_id].copy()
    elif "Market" in df_listing.columns:
        if (df_listing["Market"].str.upper() == market_code).any():
            df_listing = df_listing[df_listing["Market"].str.upper() == market_code].copy()

    # 노이즈 종목 필터링
    valid_mask = df_listing.apply(lambda row: is_valid_screening_stock(row["Code"], row["Name"]), axis=1)
    df_filtered = df_listing[valid_mask].copy()

    # 시가총액 컬럼 확보
    marcap_col = None
    for col in ["Marcap", "MKTCAP", "시가총액", "MarketCap"]:
        if col in df_filtered.columns:
            marcap_col = col
            break

    if marcap_col:
        df_filtered["Marcap_Num"] = pd.to_numeric(df_filtered[marcap_col], errors="coerce").fillna(0)
    else:
        df_filtered["Marcap_Num"] = 0

    # 시가총액 결측치 또는 전체 0인 경우 최근 유효 거래일 기준 폴백 매핑 적용
    valid_marcap_count = (df_filtered["Marcap_Num"] > 0).sum()
    if valid_marcap_count < max(10, len(df_filtered) * 0.5):
        fallback_map = get_fallback_marcap_map()
        if fallback_map:
            fallback_series = df_filtered["Code"].map(fallback_map).fillna(0)
            df_filtered["Marcap_Num"] = np.where(
                df_filtered["Marcap_Num"] > 0,
                df_filtered["Marcap_Num"],
                fallback_series
            )

    # 시가총액 기준 내림차순 정렬
    df_sorted = df_filtered.sort_values(by="Marcap_Num", ascending=False).reset_index(drop=True)

    # 최소 시가총액 필터 (단위: 억원)
    # 만약 폴백 후에도 모든 시총이 0원인 극단적 상황인 경우, 전 종목이 날아가지 않도록 필터를 안전 우회
    has_valid_marcap = (df_sorted["Marcap_Num"] > 0).any()
    if min_marcap_eok > 0 and has_valid_marcap:
        min_won = min_marcap_eok * 100_000_000
        df_sorted = df_sorted[df_sorted["Marcap_Num"] >= min_won].reset_index(drop=True)

    # 대상 범위 지정 (scope)
    if scope == "top300":
        df_result = df_sorted.head(300)
    elif scope == "top500":
        df_result = df_sorted.head(500)
    elif scope == "top1000":
        df_result = df_sorted.head(1000)
    else:  # "all"
        df_result = df_sorted

    return df_result


def _process_single_stock(code: str, name: str, market: str, marcap: float, start_date: str) -> dict:
    """
    단일 종목의 일봉 데이터를 다운로드하고 6대 기술적 지표 및 앙상블 점수를 계산합니다.
    """
    try:
        df = fdr.DataReader(code, start_date)
        if df is None or len(df) < 35:
            return None

        # 장 개장 전(09:00 이전) 또는 미체결 더미 행(Volume=0) 방지 처리
        # 만약 당일 빈 봉(거래량 0)이 생성된 경우, 가장 최근 정규 거래일의 완료 봉을 기준으로 분석
        if df["Volume"].iloc[-1] == 0 and len(df) > 1:
            df = df.iloc[:-1]

        # 거래정지나 최근 5영업일 거래량 전무 종목 제외
        recent_vol = df["Volume"].iloc[-5:].sum()
        if pd.isna(recent_vol) or recent_vol <= 0:
            return None

        # 지표 계산
        df_ind = calculate_technical_indicators(df)
        res = evaluate_ensemble_score(df_ind)
        if res is None:
            return None

        marcap_won = marcap if marcap >= 1_000_000_000 else (marcap * 100_000_000)
        marcap_eok = int(round(marcap_won / 100_000_000))
        marcap_str = format_marcap(marcap)
        latest_vol = int(df["Volume"].iloc[-1]) if "Volume" in df and not df["Volume"].empty else 0
        current_close = int(round(res["close"]))

        return {
            "종목명": name,
            "종목코드": code,
            "시장": market,
            "시가총액": marcap_eok,
            "시가총액_표시": marcap_str,
            "시가총액_원": marcap_won,
            "현재가": current_close,
            "거래량": latest_vol,
            "이동평균선 배열": res["labels"]["ma"],
            "이동평균선 배열_점수": res["scores"]["ma"],
            "MACD": res["labels"]["macd"],
            "MACD_점수": res["scores"]["macd"],
            "RSI": res["labels"]["rsi"],
            "RSI_점수": res["scores"]["rsi"],
            "볼린저 밴드": res["labels"]["bb"],
            "볼린저 밴드_점수": res["scores"]["bb"],
            "스토캐스틱": res["labels"]["stoch"],
            "스토캐스틱_점수": res["scores"]["stoch"],
            "거래량 수급": res["labels"]["vol"],
            "거래량_점수": res["scores"]["vol"],
            "종합 점수": res["total_score"],
            "투자의견": res["opinion"],
            "의견코드": res["opinion_code"],
            "등락률": res["change_rate"]
        }
    except Exception:
        return None


def run_screening_task(
    market: str = "KOSPI",
    scope: str = "top500",
    min_score: int = 45,
    min_marcap_eok: int = 0,
    max_workers: int = 20,
    progress_callback = None
) -> pd.DataFrame:
    """
    멀티스레딩을 활용하여 선택된 시장의 종목들을 동시 스크리닝하고,
    종합 점수가 설정 기준(기본: 매수 45점 이상)에 해당하는 종목을 발굴합니다.
    """
    df_universe = get_market_universe(market, scope=scope, min_marcap_eok=min_marcap_eok)
    total_stocks = len(df_universe)

    if total_stocks == 0:
        return pd.DataFrame()

    # 약 1.5년 전 날짜부터 수집 (200일선 계산 충족)
    start_date = (datetime.now() - timedelta(days=450)).strftime("%Y-%m-%d")

    results = []
    completed_count = 0

    market_label = "KOSPI" if "KOSPI" in market.upper() or "코스피" in market else "KOSDAQ"

    # 멀티스레드 병렬 실행
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {
            executor.submit(
                _process_single_stock,
                row["Code"],
                row["Name"],
                market_label,
                row["Marcap_Num"],
                start_date
            ): (row["Code"], row["Name"])
            for _, row in df_universe.iterrows()
        }

        for future in as_completed(future_to_stock):
            code, name = future_to_stock[future]
            completed_count += 1

            if progress_callback:
                progress_callback(completed_count, total_stocks, name)

            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception:
                pass

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    # 종합 점수 기준 내림차순 정렬
    df_res = df_res.sort_values(by="종합 점수", ascending=False).reset_index(drop=True)

    return df_res

def get_latest_expected_trading_day(target_date: str = None) -> str:
    """
    가장 최근 거래 완료된 실제 영업일 YYYY-MM-DD 반환.
    - target_date가 전달된 경우: 해당 날짜 기준 (또는 직전 영업일)
    - target_date가 없는 경우: KST 기준 15:45 이전이거나 오늘이 주말/새벽이면 직전 마감 거래일 반환
    """
    from datetime import datetime, timezone, timedelta
    now_kst = datetime.now(timezone(timedelta(hours=9)))
    if target_date:
        try:
            clean_date = str(target_date).replace('-', '')
            dt = datetime.strptime(clean_date, "%Y%m%d").replace(tzinfo=timezone(timedelta(hours=9)))
        except Exception:
            dt = now_kst
    else:
        dt = now_kst

    # 평일 15:45 이후에만 당일 종가 확정
    if dt.weekday() < 5 and (dt.hour > 15 or (dt.hour == 15 and dt.minute >= 45)):
        return dt.strftime("%Y-%m-%d")

    # 장전, 새벽, 주말: 직전 마감 거래일 산출
    if dt.weekday() == 0:    # 월요일 장전 -> 지난주 금요일 (3일 전)
        days_back = 3
    elif dt.weekday() == 6:  # 일요일 -> 지난주 금요일 (2일 전)
        days_back = 2
    elif dt.weekday() == 5:  # 토요일 -> 지난주 금요일 (1일 전)
        days_back = 1
    else:                    # 화~금 장전/새벽 -> 전일 (1일 전)
        days_back = 1

    return (dt - timedelta(days=days_back)).strftime("%Y-%m-%d")
