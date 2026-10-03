"""
strategy.py - Systematic Mean-Reversion Strategy definitions and iterative models.

All indicators and signals are computed strictly on data up to and including day t close.
The target_position output at index t represents the portfolio positioning
for execution at day t+1 Open.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# TECHNICAL INDICATOR CALCULATORS (VECTORIZED, NO LOOKAHEAD)
# ---------------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """Compute Relative Strength Index (Wilder's RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    
    # Wilder's smoothing: ewm with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_zscore(series: pd.Series, lookback: int = 20) -> pd.Series:
    """Compute rolling z-score of price relative to rolling mean and std."""
    roll_mean = series.rolling(lookback).mean()
    roll_std = series.rolling(lookback).std()
    return (series - roll_mean) / roll_std.replace(0.0, np.nan)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands and %B."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0.0, np.nan)
    return sma, upper, lower, pct_b


# ---------------------------------------------------------
# 10 RESEARCH ITERATIONS
# ---------------------------------------------------------

def run_iteration_1_baseline(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, rsi_exit: float = 60.0) -> pd.DataFrame:
    """
    Iteration 1: Baseline Short-Term Mean Reversion (Larry Connors 2-period RSI)
    Hypothesis: Severe short-term pullbacks (2-day RSI < 10) represent temporary liquidity
    imbalances that quickly revert back to fair value.
    - Entry: RSI(2) < 10
    - Exit: RSI(2) > 60
    - No trend filter, no risk stop, 100% long/flat allocation.
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    signals["rsi"] = rsi
    
    pos = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        if pd.isna(r):
            target_pos.append(0.0)
            continue
        if pos == 0:
            if r < rsi_entry:
                pos = 1
        elif pos == 1:
            if r > rsi_exit:
                pos = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_2_trend_filter(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, rsi_exit: float = 60.0, sma_trend_period: int = 200) -> pd.DataFrame:
    """
    Iteration 2: Adding Macro Regime / Trend Filter (200-day SMA)
    Hypothesis: Mean reversion fails during secular bear markets (e.g. 2022) because
    oversold conditions trend persistently downward ('falling knife' phenomenon).
    - Entry: RSI(2) < 10 AND Close > SMA(200)
    - Exit: RSI(2) > 60 OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    sma200 = close.rolling(sma_trend_period).mean()
    
    pos = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s200 = sma200.iloc[i]
        if pd.isna(r) or pd.isna(s200):
            target_pos.append(0.0)
            continue
        trend_bull = c > s200
        if pos == 0:
            if r < rsi_entry and trend_bull:
                pos = 1
        elif pos == 1:
            if r > rsi_exit or not trend_bull:
                pos = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_3_moving_average_exit(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200) -> pd.DataFrame:
    """
    Iteration 3: Mean Touch Exit (5-day SMA)
    Hypothesis: RSI exit is a momentum indicator that can lag during a fast snapback.
    Exiting as soon as price touches the 5-day SMA directly captures the core mean-reversion move
    and minimizes exposure to subsequent reversals.
    - Entry: RSI(2) < 10 AND Close > SMA(200)
    - Exit: Close > SMA(5) OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    
    pos = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(s_exit):
            target_pos.append(0.0)
            continue
        trend_bull = c > s_trend
        if pos == 0:
            if r < rsi_entry and trend_bull:
                pos = 1
        elif pos == 1:
            if c > s_exit or not trend_bull:
                pos = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_4_time_based_stop(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5) -> pd.DataFrame:
    """
    Iteration 4: Adding Time-Based Exit (Max 5 Days)
    Hypothesis: Mean-reversion edge decays rapidly after entry. If price has not bounced
    within 5 trading days, the hypothesis has failed and holding further ties up capital in a dead trade.
    - Entry: RSI(2) < 10 AND Close > SMA(200)
    - Exit: Close > SMA(5) OR holding_bars >= 5 OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    
    pos = 0
    bars = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(s_exit):
            target_pos.append(0.0)
            continue
        trend_bull = c > s_trend
        if pos == 0:
            if r < rsi_entry and trend_bull:
                pos = 1
                bars = 0
        elif pos == 1:
            bars += 1
            if c > s_exit or bars >= max_holding_days or not trend_bull:
                pos = 0
                bars = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_5_zscore_entry(df: pd.DataFrame, z_lookback: int = 20, z_threshold: float = -1.5, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5) -> pd.DataFrame:
    """
    Iteration 5: Statistical Mean-Reversion via Rolling Z-Score
    Hypothesis: Replacing the heuristic RSI indicator with a statistical price z-score
    (price relative to 20-day mean normalized by rolling standard deviation) tests
    whether pure standard-deviation stretch provides superior timing.
    - Entry: ZScore(20) < -1.5 AND Close > SMA(200)
    - Exit: Close > SMA(5) OR holding_bars >= 5 OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    zscore = compute_zscore(close, lookback=z_lookback)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    
    pos = 0
    bars = 0
    target_pos = []
    for i in range(len(df)):
        z = zscore.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        if pd.isna(z) or pd.isna(s_trend) or pd.isna(s_exit):
            target_pos.append(0.0)
            continue
        trend_bull = c > s_trend
        if pos == 0:
            if z < z_threshold and trend_bull:
                pos = 1
                bars = 0
        elif pos == 1:
            bars += 1
            if c > s_exit or bars >= max_holding_days or not trend_bull:
                pos = 0
                bars = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_6_volatility_adaptive_entry(df: pd.DataFrame, rsi_period: int = 2, sma_trend_period: int = 200, exit_sma_period: int = 5, max_holding_days: int = 5) -> pd.DataFrame:
    """
    Iteration 6: Volatility-Adaptive Entry Threshold
    Hypothesis: In high-volatility environments (realized vol > 20%), small pullbacks are normal noise;
    only extreme flushes (RSI < 6) represent true capitulation. In low-volatility regimes (vol <= 20%),
    mild pullbacks (RSI < 12) are already significant opportunities.
    - Entry: RSI(2) < adaptive_threshold AND Close > SMA(200)
    - Exit: Close > SMA(5) OR holding_bars >= 5 OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    ret = close.pct_change()
    ann_vol_20d = ret.rolling(20).std() * np.sqrt(252)
    
    pos = 0
    bars = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        v = ann_vol_20d.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(s_exit) or pd.isna(v):
            target_pos.append(0.0)
            continue
        thresh = 6.0 if v > 0.20 else 12.0
        trend_bull = c > s_trend
        if pos == 0:
            if r < thresh and trend_bull:
                pos = 1
                bars = 0
        elif pos == 1:
            bars += 1
            if c > s_exit or bars >= max_holding_days or not trend_bull:
                pos = 0
                bars = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_7_catastrophic_atr_stop(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5, atr_mult: float = 3.5) -> pd.DataFrame:
    """
    Iteration 7: Catastrophic Tail-Risk ATR Stop Loss
    Hypothesis: Tight stops destroy mean-reversion edges (getting whipsawed at local bottoms).
    However, a wide circuit breaker (3.5 * ATR from entry) protects the portfolio against
    black-swan gap downs and structural breakdowns without interfering with normal bounces.
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    high = df["Adj_High"]
    low = df["Adj_Low"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    atr = compute_atr(high, low, close, period=14)
    
    pos = 0
    bars = 0
    entry_p = 0.0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        a = atr.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(s_exit) or pd.isna(a):
            target_pos.append(0.0)
            continue
        trend_bull = c > s_trend
        if pos == 0:
            if r < rsi_entry and trend_bull:
                pos = 1
                bars = 0
                entry_p = c
        elif pos == 1:
            bars += 1
            stop_hit = (c < entry_p - atr_mult * a)
            if c > s_exit or bars >= max_holding_days or stop_hit or not trend_bull:
                pos = 0
                bars = 0
                entry_p = 0.0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_8_volume_exhaustion(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5, vol_ratio: float = 1.0) -> pd.DataFrame:
    """
    Iteration 8: Volume Exhaustion Filter
    Hypothesis: High-conviction mean-reversion bottoms exhibit seller exhaustion and liquidity flushes
    where volume is at least equal to or greater than its 20-day average.
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    volume = df["Volume"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_exit = close.rolling(exit_sma_period).mean()
    vol_sma = volume.rolling(20).mean()
    
    pos = 0
    bars = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        v = volume.iloc[i]
        vm = vol_sma.iloc[i]
        s_trend = sma_trend.iloc[i]
        s_exit = sma_exit.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(s_exit) or pd.isna(vm):
            target_pos.append(0.0)
            continue
        trend_bull = c > s_trend
        vol_confirmed = (v >= vol_ratio * vm)
        if pos == 0:
            if r < rsi_entry and trend_bull and vol_confirmed:
                pos = 1
                bars = 0
        elif pos == 1:
            bars += 1
            if c > s_exit or bars >= max_holding_days or not trend_bull:
                pos = 0
                bars = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_9_trend_slope_filter(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5, slope_lookback: int = 20) -> pd.DataFrame:
    """
    Iteration 9: 200-day Moving Average Slope Filter
    Hypothesis: Merely being above the 200-day SMA is insufficient if the 200-day SMA itself is rolling over.
    Requiring the 200-day SMA to be flat or rising (SMA200[t] >= SMA200[t-20]) filters out late-stage bull traps.
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_slope = sma_trend - sma_trend.shift(slope_lookback)
    sma_exit = close.rolling(exit_sma_period).mean()
    
    pos = 0
    bars = 0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        slp = sma_slope.iloc[i]
        s_exit = sma_exit.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(slp) or pd.isna(s_exit):
            target_pos.append(0.0)
            continue
        quality_trend = (c > s_trend) and (slp >= 0.0)
        if pos == 0:
            if r < rsi_entry and quality_trend:
                pos = 1
                bars = 0
        elif pos == 1:
            bars += 1
            if c > s_exit or bars >= max_holding_days or not (c > s_trend):
                pos = 0
                bars = 0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals


def run_iteration_10_refined_champion(df: pd.DataFrame, rsi_period: int = 2, rsi_entry: float = 10.0, exit_sma_period: int = 5, sma_trend_period: int = 200, max_holding_days: int = 5, atr_mult: float = 3.5, slope_lookback: int = 20) -> pd.DataFrame:
    """
    Iteration 10: Refined Production Ensemble Champion
    Synthesizes the optimal validated components:
    - Entry: RSI(2) < 10 AND Close > SMA(200) AND SMA(200) Slope(20) >= 0
    - Exit: Close > SMA(5) OR holding_bars >= 5 OR Close < Entry - 3.5*ATR OR Close < SMA(200)
    """
    signals = pd.DataFrame(index=df.index)
    close = df["Adj_Close"]
    high = df["Adj_High"]
    low = df["Adj_Low"]
    rsi = compute_rsi(close, period=rsi_period)
    sma_trend = close.rolling(sma_trend_period).mean()
    sma_slope = sma_trend - sma_trend.shift(slope_lookback)
    sma_exit = close.rolling(exit_sma_period).mean()
    atr = compute_atr(high, low, close, period=14)
    
    pos = 0
    bars = 0
    entry_p = 0.0
    target_pos = []
    for i in range(len(df)):
        r = rsi.iloc[i]
        c = close.iloc[i]
        s_trend = sma_trend.iloc[i]
        slp = sma_slope.iloc[i]
        s_exit = sma_exit.iloc[i]
        a = atr.iloc[i]
        if pd.isna(r) or pd.isna(s_trend) or pd.isna(slp) or pd.isna(s_exit) or pd.isna(a):
            target_pos.append(0.0)
            continue
        quality_trend = (c > s_trend) and (slp >= 0.0)
        if pos == 0:
            if r < rsi_entry and quality_trend:
                pos = 1
                bars = 0
                entry_p = c
        elif pos == 1:
            bars += 1
            stop_hit = (c < entry_p - atr_mult * a)
            if c > s_exit or bars >= max_holding_days or stop_hit or not (c > s_trend):
                pos = 0
                bars = 0
                entry_p = 0.0
        target_pos.append(float(pos))
        
    signals["target_position"] = target_pos
    return signals
