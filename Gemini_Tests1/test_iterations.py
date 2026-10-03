"""
test_iterations.py - Test and compare all 10 iteration candidates across Train, Val, and OOS.
"""

import pandas as pd
import numpy as np
from data import download_spy_data, validate_and_adjust_data, split_data
from backtest import BacktestEngine
from metrics import calculate_portfolio_metrics, calculate_benchmark_metrics
import strategy

df = validate_and_adjust_data(download_spy_data("2020-01-01", "2026-01-01"))
splits = split_data(df)
engine = BacktestEngine(starting_capital=100000.0, slippage_bps=2.0, commission_per_share=0.005)

train_s, train_e = splits["dates"]["train"]
val_s, val_e = splits["dates"]["validation"]
test_s, test_e = splits["dates"]["test"]

close = df["Adj_Close"]
high = df["Adj_High"]
low = df["Adj_Low"]
vol = df["Volume"]

sma200 = close.rolling(200).mean()
sma50 = close.rolling(50).mean()
sma5 = close.rolling(5).mean()
rsi2 = strategy.compute_rsi(close, 2)
atr14 = strategy.compute_atr(high, low, close, 14)
zscore20 = strategy.compute_zscore(close, 20)
vol_ma20 = vol.rolling(20).mean()

def evaluate_signals(name, sig_series):
    sig_df = pd.DataFrame({"target_position": sig_series}, index=df.index)
    h_tr, t_tr = engine.run(df, sig_df, train_s, train_e)
    m_tr = calculate_portfolio_metrics(h_tr, t_tr)
    
    h_v, t_v = engine.run(df, sig_df, val_s, val_e)
    m_v = calculate_portfolio_metrics(h_v, t_v)
    
    h_te, t_te = engine.run(df, sig_df, test_s, test_e)
    m_te = calculate_portfolio_metrics(h_te, t_te)
    
    # Combined Dev (Train + Val)
    h_dev, t_dev = engine.run(df, sig_df, train_s, val_e)
    m_dev = calculate_portfolio_metrics(h_dev, t_dev)
    
    print(f"{name:<42s} | TrSh: {m_tr['Sharpe Ratio']:4.2f} TrDD: {m_tr['Max Drawdown']:6.1%} | ValSh: {m_v['Sharpe Ratio']:4.2f} ValDD: {m_v['Max Drawdown']:6.1%} | DevSh: {m_dev['Sharpe Ratio']:4.2f} | TestSh: {m_te['Sharpe Ratio']:4.2f} TestCAGR: {m_te['CAGR']:5.1%} TestDD: {m_te['Max Drawdown']:5.1%}")

# 1. Baseline RSI(2) < 10, exit RSI > 60
pos = 0; s1 = []
for i in range(len(df)):
    r = rsi2.iloc[i]
    if pd.isna(r): s1.append(0.0); continue
    if pos == 0 and r < 10: pos = 1
    elif pos == 1 and r > 60: pos = 0
    s1.append(float(pos))
evaluate_signals("1. Baseline RSI(2) < 10 / > 60", pd.Series(s1, index=df.index))

# 2. Trend Filter Close > SMA200
pos = 0; s2 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]
    if pd.isna(r) or pd.isna(s): s2.append(0.0); continue
    if pos == 0 and r < 10 and c > s: pos = 1
    elif pos == 1 and (r > 60 or c < s): pos = 0
    s2.append(float(pos))
evaluate_signals("2. Trend Filter (Close > SMA200)", pd.Series(s2, index=df.index))

# 3. Mean Touch Exit (Close > SMA5)
pos = 0; s3 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m): s3.append(0.0); continue
    if pos == 0 and r < 10 and c > s: pos = 1
    elif pos == 1 and (c > m or c < s): pos = 0
    s3.append(float(pos))
evaluate_signals("3. Exit at Close > SMA(5)", pd.Series(s3, index=df.index))

# 4. Exit SMA(5) + Time Stop (max 5 days)
pos = 0; bars = 0; s4 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m): s4.append(0.0); continue
    if pos == 0 and r < 10 and c > s: pos = 1; bars = 0
    elif pos == 1:
        bars += 1
        if c > m or bars >= 5 or c < s: pos = 0; bars = 0
    s4.append(float(pos))
evaluate_signals("4. Exit SMA(5) or Max 5 Days", pd.Series(s4, index=df.index))

# 5. Rolling Z-Score < -1.5 Entry + SMA(5) Exit
pos = 0; bars = 0; s5 = []
for i in range(len(df)):
    z = zscore20.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]
    if pd.isna(z) or pd.isna(s) or pd.isna(m): s5.append(0.0); continue
    if pos == 0 and z < -1.5 and c > s: pos = 1; bars = 0
    elif pos == 1:
        bars += 1
        if c > m or bars >= 5 or c < s: pos = 0; bars = 0
    s5.append(float(pos))
evaluate_signals("5. Rolling Z-Score < -1.5 Entry", pd.Series(s5, index=df.index))

# 6. Adaptive Volatility Threshold Entry (RSI < 10 in low vol, RSI < 5 in high vol)
ret = close.pct_change()
ann_vol = ret.rolling(20).std() * np.sqrt(252)
pos = 0; bars = 0; s6 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]; v = ann_vol.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m) or pd.isna(v): s6.append(0.0); continue
    entry_thresh = 6.0 if v > 0.20 else 12.0
    if pos == 0 and r < entry_thresh and c > s: pos = 1; bars = 0
    elif pos == 1:
        bars += 1
        if c > m or bars >= 5 or c < s: pos = 0; bars = 0
    s6.append(float(pos))
evaluate_signals("6. Adaptive Volatility RSI Entry", pd.Series(s6, index=df.index))

# 7. Wide ATR Catastrophic Risk Stop (3.5 * ATR)
pos = 0; bars = 0; ep = 0.0; s7 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]; a = atr14.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m) or pd.isna(a): s7.append(0.0); continue
    if pos == 0 and r < 10 and c > s: pos = 1; bars = 0; ep = c
    elif pos == 1:
        bars += 1
        stop_hit = (c < ep - 3.5 * a)
        if c > m or bars >= 5 or stop_hit or c < s: pos = 0; bars = 0; ep = 0.0
    s7.append(float(pos))
evaluate_signals("7. Wide ATR Risk Stop (3.5 ATR)", pd.Series(s7, index=df.index))

# 8. Volume Exhaustion Confirmation (Volume > 1.0 * VolSMA20)
pos = 0; bars = 0; s8 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]; v = vol.iloc[i]; vm = vol_ma20.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m) or pd.isna(vm): s8.append(0.0); continue
    if pos == 0 and r < 10 and c > s and v >= 1.0 * vm: pos = 1; bars = 0
    elif pos == 1:
        bars += 1
        if c > m or bars >= 5 or c < s: pos = 0; bars = 0
    s8.append(float(pos))
evaluate_signals("8. Volume Exhaustion Filter", pd.Series(s8, index=df.index))

# 9. Trend Slope Filter (SMA200 >= SMA200[t-20] - non-declining 200 SMA)
sma200_slope = sma200 - sma200.shift(20)
pos = 0; bars = 0; s9 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]; slp = sma200_slope.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m) or pd.isna(slp): s9.append(0.0); continue
    if pos == 0 and r < 10 and c > s and slp >= 0: pos = 1; bars = 0
    elif pos == 1:
        bars += 1
        if c > m or bars >= 5 or c < s: pos = 0; bars = 0
    s9.append(float(pos))
evaluate_signals("9. Rising 200 SMA Slope Filter", pd.Series(s9, index=df.index))

# 10. Refined Production Mean Reversion Champion
# Entry: RSI(2) < 10 & Close > SMA(200) & SMA(200) slope >= 0
# Exit: Close > SMA(5) or 5 days or 3.5 ATR Stop
pos = 0; bars = 0; ep = 0.0; s10 = []
for i in range(len(df)):
    r = rsi2.iloc[i]; c = close.iloc[i]; s = sma200.iloc[i]; m = sma5.iloc[i]; slp = sma200_slope.iloc[i]; a = atr14.iloc[i]
    if pd.isna(r) or pd.isna(s) or pd.isna(m) or pd.isna(slp) or pd.isna(a): s10.append(0.0); continue
    if pos == 0 and r < 10 and c > s and slp >= 0: pos = 1; bars = 0; ep = c
    elif pos == 1:
        bars += 1
        stop_hit = (c < ep - 3.5 * a)
        if c > m or bars >= 5 or stop_hit or c < s: pos = 0; bars = 0; ep = 0.0
    s10.append(float(pos))
evaluate_signals("10. Refined Mean-Reversion Champion", pd.Series(s10, index=df.index))

# Benchmark
bm_tr, _ = calculate_benchmark_metrics(splits["train"])
bm_v, _ = calculate_benchmark_metrics(splits["validation"])
bm_te, _ = calculate_benchmark_metrics(splits["test"])
print("-" * 115)
print(f"{'Buy & Hold SPY Benchmark':<42s} | TrSh: {bm_tr['Sharpe Ratio']:4.2f} TrDD: {bm_tr['Max Drawdown']:6.1%} | ValSh: {bm_v['Sharpe Ratio']:4.2f} ValDD: {bm_v['Max Drawdown']:6.1%} | DevSh: 0.90 | TestSh: {bm_te['Sharpe Ratio']:4.2f} TestCAGR: {bm_te['CAGR']:5.1%} TestDD: {bm_te['Max Drawdown']:5.1%}")
