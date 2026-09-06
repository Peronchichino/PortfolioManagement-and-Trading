import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def build_hybrid_strategy_returns(symbol: str = "SPY", start_date: str = "2000-01-01", max_holding_days: int = 15):
    # Pull extra history so SMA_200 is warm on start_date
    df = yf.download(symbol, period="max", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date').set_index('Date')
    df['DayOfWeek'] = df.index.day_name()

    # Technical Indicators
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()

    # Track Hougaard targets
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week
    df['Hougaard_Sweep_Target'] = np.nan

    # Mid-Week Target (Wed High < Mon High -> Thu target is Wed Low)
    for (year, week), week_df in df.groupby(['Year', 'Week']):
        day_map = {row['DayOfWeek']: row for _, row in week_df.iterrows()}
        if 'Monday' in day_map and 'Wednesday' in day_map and 'Thursday' in day_map:
            if day_map['Wednesday']['High'] < day_map['Monday']['High']:
                df.loc[day_map['Thursday'].name, 'Hougaard_Sweep_Target'] = day_map['Wednesday']['Low']

    # Weekend Target (Fri High < Thu High -> Next session target is Fri Low)
    for i in range(1, len(df) - 1):
        if df.iloc[i]['DayOfWeek'] == 'Friday' and df.iloc[i - 1]['DayOfWeek'] == 'Thursday':
            if df.iloc[i]['High'] < df.iloc[i - 1]['High']:
                df.iloc[i + 1, df.columns.get_loc('Hougaard_Sweep_Target')] = df.iloc[i]['Low']

    # Filter to start_date after indicator burn-in
    df = df.loc[start_date:].copy()
    df.dropna(subset=['SMA_200', 'SMA_5'], inplace=True)

    # Strategy Signals
    df['Sweep_Occurred'] = (df['Low'] <= df['Hougaard_Sweep_Target'])
    df['Bull_Regime'] = df['Close'] > df['SMA_200']
    df['Under_Mean'] = df['Close'] < df['SMA_5']

    df['Buy_Signal'] = df['Bull_Regime'] & df['Sweep_Occurred'] & df['Under_Mean']
    df['Exit_Signal'] = df['Close'] > df['SMA_5']

    in_position = False
    entry_idx = 0
    daily_positions = np.zeros(len(df))

    buy_signals = df['Buy_Signal'].values
    exit_signals = df['Exit_Signal'].values

    for i in range(len(df) - 1):
        if in_position:
            days_held = i - entry_idx
            if exit_signals[i] or days_held >= max_holding_days:
                in_position = False
            else:
                daily_positions[i] = 1.0

        if not in_position and buy_signals[i]:
            in_position = True
            entry_idx = i
            daily_positions[i] = 1.0

    df['Position'] = daily_positions

    # Market returns vs Strategy returns
    df['Market_Return'] = df['Close'].pct_change().fillna(0.0)
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0.0) * df['Market_Return']

    df['Cum_Market'] = (1 + df['Market_Return']).cumprod()
    df['Cum_Strategy'] = (1 + df['Strategy_Return']).cumprod()

    return df


def calculate_metrics(df: pd.DataFrame):
    years = (df.index[-1] - df.index[0]).days / 365.25

    strat_cagr = ((df['Cum_Strategy'].iloc[-1]) ** (1 / years) - 1) * 100
    bnh_cagr = ((df['Cum_Market'].iloc[-1]) ** (1 / years) - 1) * 100

    strat_dd = ((df['Cum_Strategy'] - df['Cum_Strategy'].cummax()) / df['Cum_Strategy'].cummax()).min() * 100
    bnh_dd = ((df['Cum_Market'] - df['Cum_Market'].cummax()) / df['Cum_Market'].cummax()).min() * 100

    strat_ann_vol = (df['Strategy_Return'].std() * np.sqrt(252)) * 100
    bnh_ann_vol = (df['Market_Return'].std() * np.sqrt(252)) * 100

    strat_sharpe = (strat_cagr / strat_ann_vol) if strat_ann_vol > 0 else 0
    bnh_sharpe = (bnh_cagr / bnh_ann_vol) if bnh_ann_vol > 0 else 0
    exposure = (df['Position'] > 0).mean() * 100

    return {
        'Strat_CAGR': strat_cagr,
        'BnH_CAGR': bnh_cagr,
        'Strat_MaxDD': strat_dd,
        'BnH_MaxDD': bnh_dd,
        'Strat_Vol': strat_ann_vol,
        'BnH_Vol': bnh_ann_vol,
        'Strat_Sharpe': strat_sharpe,
        'BnH_Sharpe': bnh_sharpe,
        'Exposure': exposure
    }


if __name__ == "__main__":
    data = build_hybrid_strategy_returns("SPY", start_date="2000-01-01")
    m = calculate_metrics(data)

    print("=" * 60)
    print("HYBRID HOUGAARD SWEEP vs. SPY BUY & HOLD (2000–PRESENT)")
    print("=" * 60)
    print(f"{'Metric':<28} {'Hybrid Sweep':<16} {'SPY Buy & Hold'}")
    print("-" * 60)
    print(f"{'CAGR (Annual Return)':<28} {m['Strat_CAGR']:+.2f}%{'':<9} {m['BnH_CAGR']:+.2f}%")
    print(f"{'Max Drawdown':<28} {m['Strat_MaxDD']:.2f}%{'':<9} {m['BnH_MaxDD']:.2f}%")
    print(f"{'Annualized Volatility':<28} {m['Strat_Vol']:.2f}%{'':<10} {m['BnH_Vol']:.2f}%")
    print(f"{'Return / Max DD (Calmar)':<28} {abs(m['Strat_CAGR']/m['Strat_MaxDD']):.2f}{'':<12} {abs(m['BnH_CAGR']/m['BnH_MaxDD']):.2f}")
    print(f"{'Market Exposure Time':<28} {m['Exposure']:.1f}%{'':<10} 100.0%")
    print("=" * 60)

    # Plot Comparison
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Cum_Strategy'], label="Hybrid Sweep Strategy", color='#1f77b4', lw=2)
    plt.plot(data.index, data['Cum_Market'], label="SPY Buy & Hold", color='#7f7f7f', lw=1.2, ls='--')
    plt.yscale('log')
    plt.title("Cumulative Performance (Log Scale): Hybrid Sweep vs. SPY Buy & Hold", fontsize=12, fontweight='bold')
    plt.ylabel("Growth of $1.00 (Log Scale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()