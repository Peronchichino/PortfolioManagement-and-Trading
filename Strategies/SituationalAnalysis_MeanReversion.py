import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DEFAULT_ASSETS = {
    'DAX_Index': '^GDAXI',
    'FTSE_100': '^FTSE',
    'Dow_Jones': '^DJI',
    'S&P_500': '^GSPC',
    'NASDAQ_100': '^NDX',
    'SPY_ETF': 'SPY',
    'QQQ_ETF': 'QQQ'
}


def calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def build_strategy_dataset(symbol: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    df = yf.download(symbol, start=start_date, progress=False, auto_adjust=False)
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date').set_index('Date')
    df['DayOfWeek'] = df.index.day_name()

    # Trend and mean indicators
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['RSI_2'] = calculate_rsi(df['Close'], period=2)

    # Hougaard Signal Tracking
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week

    df['Hougaard_Sweep_Target'] = np.nan

    # 1. Flag Mid-Week Target (Wed High < Mon High -> Thu target is Wed Low)
    for (year, week), week_df in df.groupby(['Year', 'Week']):
        day_map = {row['DayOfWeek']: row for _, row in week_df.iterrows()}
        if 'Monday' in day_map and 'Wednesday' in day_map and 'Thursday' in day_map:
            if day_map['Wednesday']['High'] < day_map['Monday']['High']:
                df.loc[day_map['Thursday'].name, 'Hougaard_Sweep_Target'] = day_map['Wednesday']['Low']

    # 2. Flag Weekend Target (Fri High < Thu High -> Next day target is Fri Low)
    for i in range(1, len(df) - 1):
        if df.iloc[i]['DayOfWeek'] == 'Friday' and df.iloc[i - 1]['DayOfWeek'] == 'Thursday':
            if df.iloc[i]['High'] < df.iloc[i - 1]['High']:
                target_low = df.iloc[i]['Low']
                df.iloc[i + 1, df.columns.get_loc('Hougaard_Sweep_Target')] = target_low

    df.dropna(subset=['SMA_200', 'SMA_5'], inplace=True)
    return df


def backtest_hybrid_sweep(df: pd.DataFrame, max_holding_days: int = 15):
    """
    Mean Reversion using Hougaard Liquidity Sweep as trigger:
    - Macro Condition: Close > SMA_200
    - Filter Condition: Low <= Hougaard_Sweep_Target (target low visited / stops swept)
    - Mean Reversion Trigger: Close < SMA_5 (or RSI_2 < 25)
    - Buy: Next day Open
    - Exit: Close > SMA_5 or holding period reached
    """
    df = df.copy()

    # Sweep occurred and price was suppressed below SMA 5
    df['Sweep_Occurred'] = (df['Low'] <= df['Hougaard_Sweep_Target'])
    df['Bull_Regime'] = df['Close'] > df['SMA_200']
    df['Under_Mean'] = df['Close'] < df['SMA_5']

    df['Buy_Signal'] = df['Bull_Regime'] & df['Sweep_Occurred'] & df['Under_Mean']
    df['Exit_Signal'] = df['Close'] > df['SMA_5']

    trades = []
    in_position = False
    entry_price = 0.0
    entry_idx = 0

    dates = df.index
    opens = df['Open'].values
    buy_signals = df['Buy_Signal'].values
    exit_signals = df['Exit_Signal'].values

    for i in range(len(df) - 1):
        if in_position:
            days_held = i - entry_idx
            if exit_signals[i] or days_held >= max_holding_days:
                exit_price = opens[i + 1]
                pnl = ((exit_price - entry_price) / entry_price) * 100
                trades.append({
                    'Entry_Date': dates[entry_idx],
                    'Exit_Date': dates[i + 1],
                    'PnL': pnl,
                    'Days_Held': days_held + 1
                })
                in_position = False

        if not in_position and buy_signals[i]:
            in_position = True
            entry_price = opens[i + 1]
            entry_idx = i

    trades_df = pd.DataFrame(trades)
    return trades_df


def print_comparison_results(asset_name: str, trades_df: pd.DataFrame):
    print(f"\n==================== {asset_name} Hybrid Sweep Results ====================")
    if trades_df.empty:
        print("No trades generated.")
        return

    total = len(trades_df)
    wins = (trades_df['PnL'] > 0).sum()
    win_rate = (wins / total) * 100
    avg_return = trades_df['PnL'].mean()
    profit_factor = (
        trades_df.loc[trades_df['PnL'] > 0, 'PnL'].sum() /
        abs(trades_df.loc[trades_df['PnL'] < 0, 'PnL'].sum())
    ) if (trades_df['PnL'] < 0).sum() > 0 else np.inf

    print(f"Total Completed Trades:  {total}")
    print(f"Win Rate:                {win_rate:.2f}%")
    print(f"Profit Factor:           {profit_factor:.2f}")
    print(f"Average Return / Trade:  {avg_return:+.2f}%")
    print(f"Average Holding Period:  {trades_df['Days_Held'].mean():.1f} days")


def run():
    print("Testing Hybrid Hougaard Liquidity Sweep + SMA Mean Reversion...")
    for name, sym in DEFAULT_ASSETS.items():
        df = build_strategy_dataset(sym, start_date="2000-01-01")
        if df.empty:
            continue
        trades = backtest_hybrid_sweep(df)
        print_comparison_results(name, trades)


if __name__ == "__main__":
    run()