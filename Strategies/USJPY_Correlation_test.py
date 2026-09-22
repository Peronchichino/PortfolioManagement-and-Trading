import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt


def fetch_aligned_intraday_data(days: int = 59):
    """
    Pulls 5-minute bars for USD/JPY and QQQ (up to 60-day limit on Yahoo Finance)
    and daily bars for Nikkei 225 (^N225).
    """
    print(f"Downloading 5-minute intraday data for USDJPY=X and QQQ ({days} days)...")
    
    # 1. USD/JPY 5-minute data
    jpy_raw = yf.download("USDJPY=X", period=f"{days}d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(jpy_raw.columns, pd.MultiIndex):
        jpy_raw.columns = jpy_raw.columns.get_level_values(0)
    jpy_raw = jpy_raw.reset_index()
    date_col_jpy = 'Datetime' if 'Datetime' in jpy_raw.columns else 'Date'
    jpy_raw['Datetime'] = pd.to_datetime(jpy_raw[date_col_jpy]).dt.tz_convert('America/New_York')
    jpy_df = jpy_raw.sort_values('Datetime').set_index('Datetime')

    # 2. QQQ 5-minute data
    qqq_raw = yf.download("QQQ", period=f"{days}d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(qqq_raw.columns, pd.MultiIndex):
        qqq_raw.columns = qqq_raw.columns.get_level_values(0)
    qqq_raw = qqq_raw.reset_index()
    date_col_qqq = 'Datetime' if 'Datetime' in qqq_raw.columns else 'Date'
    qqq_raw['Datetime'] = pd.to_datetime(qqq_raw[date_col_qqq]).dt.tz_convert('America/New_York')
    qqq_df = qqq_raw.sort_values('Datetime').set_index('Datetime')

    # 3. Nikkei 225 Daily Data (longer reference window)
    print("Downloading daily Nikkei 225 (^N225) data...")
    n225_raw = yf.download("^N225", period="2y", interval="1d", progress=False, auto_adjust=False)
    if isinstance(n225_raw.columns, pd.MultiIndex):
        n225_raw.columns = n225_raw.columns.get_level_values(0)
    n225_raw = n225_raw.reset_index()
    n225_raw['Date'] = pd.to_datetime(n225_raw['Date']).dt.date
    n225_df = n225_raw.sort_values('Date').set_index('Date')
    n225_df['Nikkei_Return_%'] = n225_df['Close'].pct_change() * 100.0

    return jpy_df, qqq_df, n225_df


def extract_session_features(jpy_df: pd.DataFrame, qqq_df: pd.DataFrame, n225_df: pd.DataFrame):
    """
    Extracts:
    - Overnight USD/JPY Move: % change from 00:00 ET to 08:30 ET
    - Prior Japanese Session Return: Nikkei 225 daily % return
    - US Opening Gap: QQQ Open (09:30) vs Prior Day QQQ Close (16:00)
    - US First 30-Min Drift: QQQ Close (10:00) vs QQQ Open (09:30)
    """
    records = []
    unique_dates = sorted(list(set(qqq_df.index.date)))

    for i in range(1, len(unique_dates)):
        trade_date = unique_dates[i]
        prev_date = unique_dates[i - 1]

        # Day subsets
        day_qqq = qqq_df[qqq_df.index.date == trade_date]
        prev_day_qqq = qqq_df[qqq_df.index.date == prev_date]
        day_jpy = jpy_df[jpy_df.index.date == trade_date]

        if day_qqq.empty or prev_day_qqq.empty or day_jpy.empty:
            continue

        # 1. Overnight USD/JPY return between 00:00 and 08:30 ET
        jpy_night = day_jpy.between_time('00:00', '08:30')
        if len(jpy_night) < 10:
            continue
        jpy_start = jpy_night.iloc[0]['Open']
        jpy_end = jpy_night.iloc[-1]['Close']
        jpy_overnight_pct = ((jpy_end - jpy_start) / jpy_start) * 100.0

        # 2. QQQ Price Benchmarks
        qqq_rth = day_qqq.between_time('09:30', '16:00')
        prev_qqq_rth = prev_day_qqq.between_time('09:30', '16:00')
        if len(qqq_rth) < 7 or prev_qqq_rth.empty:
            continue

        prev_close = prev_qqq_rth.iloc[-1]['Close']
        us_open = qqq_rth.iloc[0]['Open']

        # First 30-min bar close (10:00 AM ET bar)
        first_30m_bar = qqq_rth.between_time('09:55', '10:05')
        if first_30m_bar.empty:
            continue
        us_10am = first_30m_bar.iloc[-1]['Close']

        qqq_gap_pct = ((us_open - prev_close) / prev_close) * 100.0
        qqq_first_30m_pct = ((us_10am - us_open) / us_open) * 100.0
        qqq_full_day_pct = ((qqq_rth.iloc[-1]['Close'] - us_open) / us_open) * 100.0

        # 3. Nikkei Return (prior closed Asian session)
        nikkei_ret = n225_df.loc[trade_date, 'Nikkei_Return_%'] if trade_date in n225_df.index else np.nan

        records.append({
            'Date': trade_date,
            'USDJPY_Overnight_%': jpy_overnight_pct,
            'Nikkei_Session_%': nikkei_ret,
            'QQQ_Gap_%': qqq_gap_pct,
            'QQQ_First_30m_%': qqq_first_30m_pct,
            'QQQ_Full_Day_%': qqq_full_day_pct
        })

    return pd.DataFrame(records)


def analyze_correlations_and_signals(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("STATISTICAL CORRELATION AUDIT: JAPAN & FOREX vs US OPEN")
    print("=" * 65)

    clean_df = df.dropna().copy()
    n_samples = len(clean_df)
    print(f"Sample Size (Trading Sessions): {n_samples}\n")

    # Pearson & Spearman Correlations
    pairs = [
        ('USDJPY_Overnight_%', 'QQQ_Gap_%', "Overnight USD/JPY -> US Opening Gap"),
        ('USDJPY_Overnight_%', 'QQQ_First_30m_%', "Overnight USD/JPY -> 09:30-10:00 Drift"),
        ('USDJPY_Overnight_%', 'QQQ_Full_Day_%', "Overnight USD/JPY -> Full US Session Drift"),
        ('Nikkei_Session_%', 'QQQ_Gap_%', "Nikkei 225 Session -> US Opening Gap"),
        ('Nikkei_Session_%', 'QQQ_First_30m_%', "Nikkei 225 Session -> 09:30-10:00 Drift")
    ]

    print(f"{'Relationship':<44} {'Pearson r':<11} {'p-value':<9} {'Significance'}")
    print("-" * 75)
    for col_x, col_y, label in pairs:
        r, p_val = stats.pearsonr(clean_df[col_x], clean_df[col_y])
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "Not Sig"))
        print(f"{label:<44} {r:+.3f}{'':<5} {p_val:.4f}{'':<3} {sig}")

    # Directional Hit Rate (Co-movement)
    same_dir_gap = np.mean(np.sign(clean_df['USDJPY_Overnight_%']) == np.sign(clean_df['QQQ_Gap_%'])) * 100.0
    same_dir_drift = np.mean(np.sign(clean_df['USDJPY_Overnight_%']) == np.sign(clean_df['QQQ_First_30m_%'])) * 100.0

    print("\n" + "-" * 75)
    print("DIRECTIONAL AGREEMENT")
    print(f"USD/JPY & QQQ Gap Direction Match:     {same_dir_gap:.1f}%")
    print(f"USD/JPY & QQQ 30m Drift Match:        {same_dir_drift:.1f}%")

    # Carry Trade Shock Analysis (USD/JPY dropping > 0.35% overnight)
    shocks = clean_df[clean_df['USDJPY_Overnight_%'] < -0.35]
    non_shocks = clean_df[clean_df['USDJPY_Overnight_%'] >= -0.35]

    print("\n" + "=" * 65)
    print("CARRY TRADE UNWIND SHOCK FILTER (USD/JPY < -0.35% Overnight)")
    print("=" * 65)
    print(f"Occurrences: {len(shocks)} / {n_samples} sessions")
    if not shocks.empty:
        print(f"Average QQQ Opening Gap on Shock Days:    {shocks['QQQ_Gap_%'].mean():+.2f}% vs {non_shocks['QQQ_Gap_%'].mean():+.2f}% normal")
        print(f"Average QQQ 09:30-10:00 Drift on Shocks:  {shocks['QQQ_First_30m_%'].mean():+.2f}% vs {non_shocks['QQQ_First_30m_%'].mean():+.2f}% normal")
        print(f"Probability of Red QQQ Open on Shock:    {(shocks['QQQ_Gap_%'] < 0).mean() * 100:.1f}%")

    # Scatter Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.scatter(clean_df['USDJPY_Overnight_%'], clean_df['QQQ_Gap_%'], color='#1f77b4', alpha=0.8)
    ax1.axhline(0, color='gray', lw=0.8, ls='--')
    ax1.axvline(0, color='gray', lw=0.8, ls='--')
    ax1.set_title("Overnight USD/JPY vs QQQ Gap", fontweight='bold')
    ax1.set_xlabel("USD/JPY % Move (00:00 - 08:30 ET)")
    ax1.set_ylabel("QQQ Gap % (Open vs Prev Close)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2.scatter(clean_df['Nikkei_Session_%'], clean_df['QQQ_Gap_%'], color='#2ca02c', alpha=0.8)
    ax2.axhline(0, color='gray', lw=0.8, ls='--')
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.set_title("Nikkei 225 Session vs QQQ Gap", fontweight='bold')
    ax2.set_xlabel("Nikkei 225 % Return")
    ax2.set_ylabel("QQQ Gap % (Open vs Prev Close)")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    jpy, qqq, n225 = fetch_aligned_intraday_data(days=59)
    features_df = extract_session_features(jpy, qqq, n225)
    analyze_correlations_and_signals(features_df)