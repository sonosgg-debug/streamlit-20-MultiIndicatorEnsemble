"""
indicators.py
다중 지표 앙상블(Multi-Indicator Ensemble) 모델을 적용한
6대 핵심 기술적 지표 계산 및 정량적 가중 평가 모듈

가중치 배분 기준 (총합 100%):
1. 이동평균선 배열: 25% (만점: 25점)
2. MACD: 20% (만점: 20점)
3. RSI: 20% (만점: 20점)
4. 볼린저 밴드: 15% (만점: 15점)
5. 스토캐스틱: 10% (만점: 10점)
6. 거래량: 10% (만점: 10점)
종합 점수 범위: -100점 ~ +100점 (매수 BUY 기준: +45점 이상)
"""

import pandas as pd
import numpy as np
import ta


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV 주가 데이터프레임에 6대 기술적 지표 컬럼을 추가합니다.
    최소 200봉 이상의 데이터가 권장됩니다.
    """
    if df is None or len(df) < 20:
        return df

    df = df.copy()

    # 1. 이동평균선 (SMA: 5, 20, 60, 120, 200)
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_60"] = df["Close"].rolling(window=60).mean()
    df["SMA_120"] = df["Close"].rolling(window=120).mean()
    df["SMA_200"] = df["Close"].rolling(window=200).mean()

    # 2. MACD (12, 26, 9)
    macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    # 3. RSI (14)
    rsi_indicator = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df["RSI"] = rsi_indicator.rsi()

    # 4. 볼린저 밴드 (20, 2)
    bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    df["BB_Percent"] = bb.bollinger_pband()
    df["BB_Width"] = bb.bollinger_wband()

    # 5. Stochastic Slow (%K: 14-3, %D: 3)
    stoch = ta.momentum.StochasticOscillator(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14,
        smooth_window=3
    )
    df["Stoch_K"] = stoch.stoch()
    df["Stoch_D"] = stoch.stoch_signal()

    # 6. 거래량: 20일 거래량 이동평균 및 변화율
    df["Vol_SMA20"] = df["Volume"].rolling(window=20).mean()
    vol_sma_safe = df["Vol_SMA20"].replace(0, np.nan)
    df["Vol_Ratio"] = (df["Volume"] / vol_sma_safe) * 100

    # 보조: ATR (14)
    atr = ta.volatility.AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14)
    df["ATR"] = atr.average_true_range()

    return df


def evaluate_ensemble_score(df: pd.DataFrame) -> dict:
    """
    최신 지표를 바탕으로 6대 지표별 점수와 가중 종합 점수(-100 ~ +100)를 산출합니다.
    """
    if df is None or len(df) < 5:
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    close = float(latest["Close"])
    prev_close = float(prev["Close"])
    open_price = float(latest.get("Open", close))

    # 1. 이동평균선 배열 점수 (가중치 25%, 배점 -25 ~ +25점)
    sma5 = float(latest["SMA_5"]) if pd.notna(latest.get("SMA_5")) else close
    sma20 = float(latest["SMA_20"]) if pd.notna(latest.get("SMA_20")) else close
    sma60 = float(latest["SMA_60"]) if pd.notna(latest.get("SMA_60")) else close
    sma120 = float(latest["SMA_120"]) if pd.notna(latest.get("SMA_120")) else None
    sma200 = float(latest["SMA_200"]) if pd.notna(latest.get("SMA_200")) else None

    ma_score = 0
    ma_status = "혼조세"

    # 배열 상태 판별
    if sma5 > sma20 and sma20 > sma60:
        if sma120 is not None and sma60 > sma120:
            ma_score += 15
            ma_status = "완전 정배열"
        else:
            ma_score += 10
            ma_status = "단기 정배열"
    elif sma5 < sma20 and sma20 < sma60:
        if sma120 is not None and sma60 < sma120:
            ma_score -= 15
            ma_status = "완전 역배열"
        else:
            ma_score -= 10
            ma_status = "단기 역배열"
    else:
        ma_status = "배열 혼조세"

    # 20일 생명선 상회 여부
    if close >= sma20:
        ma_score += 6
    else:
        ma_score -= 6

    # 200일 장기 추세선 상회 여부 (200일선 없으면 120일선 참조)
    ref_long_ma = sma200 if sma200 is not None else sma120
    if ref_long_ma is not None:
        if close >= ref_long_ma:
            ma_score += 4
        else:
            ma_score -= 4

    ma_score = int(np.clip(ma_score, -25, 25))
    ma_label = f"{ma_status} ({ma_score:+d})"

    # 2. MACD 점수 (가중치 20%, 배점 -20 ~ +20점)
    macd_val = float(latest["MACD"]) if pd.notna(latest.get("MACD")) else 0.0
    macd_sig = float(latest["MACD_Signal"]) if pd.notna(latest.get("MACD_Signal")) else 0.0
    macd_hist = float(latest["MACD_Hist"]) if pd.notna(latest.get("MACD_Hist")) else 0.0
    prev_macd_val = float(prev["MACD"]) if pd.notna(prev.get("MACD")) else 0.0
    prev_macd_sig = float(prev["MACD_Signal"]) if pd.notna(prev.get("MACD_Signal")) else 0.0
    prev_macd_hist = float(prev["MACD_Hist"]) if pd.notna(prev.get("MACD_Hist")) else 0.0

    macd_score = 0
    macd_status = "중립"

    # 시그널 교차 및 위치
    is_golden_cross = (prev_macd_val <= prev_macd_sig) and (macd_val > macd_sig)
    is_dead_cross = (prev_macd_val >= prev_macd_sig) and (macd_val < macd_sig)

    if is_golden_cross:
        macd_score += 10
        macd_status = "골든크로스"
    elif macd_val > macd_sig:
        macd_score += 8
        macd_status = "시그널 상회"
    elif is_dead_cross:
        macd_score -= 10
        macd_status = "데드크로스"
    else:
        macd_score -= 8
        macd_status = "시그널 하회"

    # 히스토그램 탄력
    if macd_hist > prev_macd_hist:
        macd_score += 3
        if macd_val > macd_sig:
            macd_status = "상승가속"
    else:
        macd_score -= 3
        if macd_val <= macd_sig:
            macd_status = "하락가속"

    # 0선 상회 여부 (대세 상승 사이클)
    if macd_val >= 0:
        macd_score += 7
    else:
        macd_score -= 7

    macd_score = int(np.clip(macd_score, -20, 20))
    macd_label = f"{macd_status} ({macd_score:+d})"

    # 3. RSI 점수 (가중치 20%, 배점 -20 ~ +20점)
    rsi_val = float(latest["RSI"]) if pd.notna(latest.get("RSI")) else 50.0
    prev_rsi = float(prev["RSI"]) if pd.notna(prev.get("RSI")) else 50.0

    rsi_score = 0
    rsi_status = "중립"

    if rsi_val <= 30:
        rsi_score += 20
        rsi_status = f"과매도반등 [{rsi_val:.1f}]"
    elif 50 <= rsi_val < 70:
        rsi_score += 15
        if rsi_val > prev_rsi:
            rsi_score += 5
            rsi_status = f"모멘텀확장 [{rsi_val:.1f}]"
        else:
            rsi_status = f"안정상승 [{rsi_val:.1f}]"
    elif 40 <= rsi_val < 50:
        if rsi_val > prev_rsi:
            rsi_score += 5
            rsi_status = f"반등시도 [{rsi_val:.1f}]"
        else:
            rsi_score -= 5
            rsi_status = f"약세전환 [{rsi_val:.1f}]"
    elif rsi_val >= 70:
        rsi_score -= 10
        rsi_status = f"과매수경계 [{rsi_val:.1f}]"
    else:  # 30 < rsi_val < 40
        rsi_score -= 15
        rsi_status = f"하락침체 [{rsi_val:.1f}]"

    rsi_score = int(np.clip(rsi_score, -20, 20))
    rsi_label = f"{rsi_status} ({rsi_score:+d})"

    # 4. 볼린저 밴드 점수 (가중치 15%, 배점 -15 ~ +15점)
    bb_pct = float(latest["BB_Percent"]) if pd.notna(latest.get("BB_Percent")) else 0.5
    bb_upper = float(latest["BB_Upper"]) if pd.notna(latest.get("BB_Upper")) else close
    bb_lower = float(latest["BB_Lower"]) if pd.notna(latest.get("BB_Lower")) else close

    bb_score = 0
    bb_status = "중립"

    if bb_pct < 0.0:
        bb_score += 12
        bb_status = f"하단이탈반등 [{bb_pct:.2f}]"
    elif 0.50 <= bb_pct <= 0.85:
        bb_score += 15
        bb_status = f"상승라이딩 [{bb_pct:.2f}]"
    elif 0.85 < bb_pct <= 1.0:
        bb_score += 8
        bb_status = f"상단근접 [{bb_pct:.2f}]"
    elif bb_pct > 1.0:
        bb_score -= 5
        bb_status = f"상단돌파과열 [{bb_pct:.2f}]"
    else:  # 0.0 <= bb_pct < 0.50
        bb_score -= 15
        bb_status = f"하단탐색 [{bb_pct:.2f}]"

    bb_score = int(np.clip(bb_score, -15, 15))
    bb_label = f"{bb_status} ({bb_score:+d})"

    # 5. 스토캐스틱 점수 (가중치 10%, 배점 -10 ~ +10점)
    stoch_k = float(latest["Stoch_K"]) if pd.notna(latest.get("Stoch_K")) else 50.0
    stoch_d = float(latest["Stoch_D"]) if pd.notna(latest.get("Stoch_D")) else 50.0
    prev_stoch_k = float(prev["Stoch_K"]) if pd.notna(prev.get("Stoch_K")) else 50.0
    prev_stoch_d = float(prev["Stoch_D"]) if pd.notna(prev.get("Stoch_D")) else 50.0

    stoch_score = 0
    stoch_status = "중립"

    is_stoch_gc = (prev_stoch_k <= prev_stoch_d) and (stoch_k > stoch_d)
    is_stoch_dc = (prev_stoch_k >= prev_stoch_d) and (stoch_k < stoch_d)

    if is_stoch_gc:
        stoch_score += 10
        stoch_status = f"골든크로스 [{stoch_k:.0f}]"
    elif stoch_k > stoch_d:
        stoch_score += 7
        stoch_status = f"상향유지 [{stoch_k:.0f}]"
    elif is_stoch_dc:
        stoch_score -= 10
        stoch_status = f"데드크로스 [{stoch_k:.0f}]"
    else:
        stoch_score -= 7
        stoch_status = f"하향유지 [{stoch_k:.0f}]"

    stoch_score = int(np.clip(stoch_score, -10, 10))
    stoch_label = f"{stoch_status} ({stoch_score:+d})"

    # 6. 거래량 점수 (가중치 10%, 배점 -10 ~ +10점)
    vol_ratio = float(latest["Vol_Ratio"]) if pd.notna(latest.get("Vol_Ratio")) else 100.0
    vol_score = 0
    vol_status = "보통"

    is_yangbong = (close > open_price) or (close > prev_close)

    if vol_ratio >= 150.0:
        if is_yangbong:
            vol_score += 10
            vol_status = f"대량수급유입 [{vol_ratio:.0f}%]"
        else:
            vol_score -= 10
            vol_status = f"대량매물출회 [{vol_ratio:.0f}%]"
    elif vol_ratio >= 90.0:
        if is_yangbong:
            vol_score += 6
            vol_status = f"양호한수급 [{vol_ratio:.0f}%]"
        else:
            vol_score -= 4
            vol_status = f"보통조정 [{vol_ratio:.0f}%]"
    else:
        vol_score -= 2
        vol_status = f"거래한산 [{vol_ratio:.0f}%]"

    vol_score = int(np.clip(vol_score, -10, 10))
    vol_label = f"{vol_status} ({vol_score:+d})"

    # 종합 점수 합산 (-100 ~ +100)
    total_score = ma_score + macd_score + rsi_score + bb_score + stoch_score + vol_score
    total_score = int(np.clip(total_score, -100, 100))

    # 5단계 투자 의견 판정
    if total_score >= 45:
        opinion = "매수 (BUY)"
        opinion_code = "BUY"
        badge_color = "#10B981"  # Emerald
    elif total_score >= 15:
        opinion = "비중확대 (OVERWEIGHT)"
        opinion_code = "OVERWEIGHT"
        badge_color = "#3B82F6"  # Blue
    elif total_score >= -14:
        opinion = "중립 (NEUTRAL)"
        opinion_code = "NEUTRAL"
        badge_color = "#94A3B8"  # Slate Gray
    elif total_score >= -44:
        opinion = "비중축소 (UNDERWEIGHT)"
        opinion_code = "UNDERWEIGHT"
        badge_color = "#F59E0B"  # Amber
    else:
        opinion = "매도 (SELL)"
        opinion_code = "SELL"
        badge_color = "#EF4444"  # Red

    return {
        "total_score": total_score,
        "opinion": opinion,
        "opinion_code": opinion_code,
        "badge_color": badge_color,
        "close": close,
        "change_rate": ((close - prev_close) / prev_close) * 100 if prev_close > 0 else 0.0,
        "scores": {
            "ma": ma_score,
            "macd": macd_score,
            "rsi": rsi_score,
            "bb": bb_score,
            "stoch": stoch_score,
            "vol": vol_score
        },
        "labels": {
            "ma": ma_label,
            "macd": macd_label,
            "rsi": rsi_label,
            "bb": bb_label,
            "stoch": stoch_label,
            "vol": vol_label
        },
        "raw_indicators": {
            "close": close,
            "sma20": sma20,
            "sma60": sma60,
            "rsi": rsi_val,
            "macd": macd_val,
            "macd_sig": macd_sig,
            "bb_pct": bb_pct,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
            "vol_ratio": vol_ratio
        }
    }
