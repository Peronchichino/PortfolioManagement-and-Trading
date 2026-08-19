import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def load_local_csv(file_name_or_path: str) -> pd.DataFrame:
    possible_paths = [
        file_name_or_path,
        os.path.join("../datasets/indexes_etfs/", file_name_or_path),
        os.path.join("../datasets/indexes_etfs/", f"{file_name_or_path}.csv"),
        os.path.join("../datasets/", file_name_or_path),
        os.path.join("../datasets/", f"{file_name_or_path}.csv")
    ]

    target_path = None
    for p in possible_paths:
        if os.path.exists(p):
            target_path = p
            break

    if not target_path:
        raise FileNotFoundError(f"File {file_name_or_path} not found in any of the expected paths.")

    print(f"Loading data from: {target_path}")
    df = pd.read_csv(target_path)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').set_index('Date')

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing from the data.")

    return df[required_cols].dropna().copy()

def run_local_mean_reversion_backtest(asset_name: str, start_date: str = "2000-01-01"):
    print(f"\nRunning mean reversion backtest for {asset_name} starting from {start_date} {'='*20}")

    df = load_local_csv(asset_name)

    df = df.loc[start_date:].copy()
    if df.empty:
        print(f"No data available for {asset_name} starting from {start_date}.")
        return

    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['RSI_2'] = calculate_rsi(df['Close'], period=2)

    df.dropna(inplace=True)

    df['Buy_Signal'] = (df['Close'] > df['SMA_200']) & (df['RSI_2'] < 10)
    df['Sell_Signal'] = (df['Close'] > df['SMA_200']) & (df['RSI_2'] > 90)
    df['Exit_Signal'] = df['Close'] > df['SMA_5']

    in_position = False
    entry_price = 0.0
    entry_idx = 0
    max_holding_period = 20
    trades = []
    daily_positions = np.zeros(len(df))

    dates = df.index
    opens = df['Open'].values
    buy_signals = df['Buy_Signal'].values
    sell_signals = df['Sell_Signal'].values
    exit_signals = df['Exit_Signal'].values

    for i in range(len(df) - 1):
        if in_position:
            days_held = i - entry_idx
            if exit_signals[i] or days_held >= max_holding_period or sell_signals[i]:
                exit_price = opens[i +1]
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'Entry_Date': dates[entry_idx].strftime('%Y-%m-%d'),
                    'Exit_Date': dates[i + 1].strftime('%Y-%m-%d'),
                    'Entry_Price': entry_price,
                    'Exit_Price': exit_price,
                    'PnL': pnl * 100,
                    'Days_Held': days_held + 1
                })
                in_position = False
            else:
                daily_positions[i] = 1.0

        if not in_position and buy_signals[i]:
            in_position = True
            entry_price = opens[i + 1]
            entry_idx = i
            daily_positions[i] = 1.0

    df['Position'] = daily_positions

    #metrics
    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0) * df['Market_Return']

    df['Cumulative_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        print("No trades were executed during the backtest period.")
        return

    total_trades = len(trades_df)
    win_rate = (trades_df['PnL'] > 0).mean() * 100
    avg_pnl = trades_df['PnL'].mean()
    losing_sum = abs(trades_df.loc[trades_df['PnL'] < 0, 'PnL'].sum())
    profit_factor = trades_df.loc[trades_df['PnL'] > 0, 'PnL'].sum() / losing_sum if losing_sum != 0 else np.inf

    strat_cagr = (df['Cumulative_Strategy_Return'].iloc[-1]) ** (1 / ((df.index[-1] - df.index[0]).days / 365.25)) - 1
    bench_cagr = (df['Cumulative_Market_Return'].iloc[-1]) ** (1 / ((df.index[-1] - df.index[0]).days / 365.25)) - 1

    strat_peaks = df['Cumulative_Strategy_Return'].cummax()
    strat_dd = float(((df['Cumulative_Strategy_Return'] - strat_peaks) / strat_peaks).min()) * 100
    bench_peaks = df['Cumulative_Market_Return'].cummax()
    bench_dd = float(((df['Cumulative_Market_Return'] - bench_peaks) / bench_peaks).min()) * 100

    print(f"Total Trades:           {total_trades}")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Profit Factor:          {profit_factor:.2f}")
    print(f"Avg Trade Return:       {avg_pnl:.2f}%")
    holding_col = 'Days_Held' if 'Days_Held' in trades_df.columns else ('Holding_Days' if 'Holding_Days' in trades_df.columns else None)
    if holding_col:
        print(f"Avg Holding Period:     {trades_df[holding_col].mean():.1f} trading days")
    else:
        print("Avg Holding Period:     N/A")
    print(f"Strategy CAGR:          {strat_cagr * 100:.2f}%")
    print(f"Buy & Hold CAGR:        {bench_cagr * 100:.2f}%")
    print(f"Strategy Max Drawdown:  {strat_dd:.2f}%")
    print(f"Buy & Hold Max Drawdown:{bench_dd:.2f}%")
    print(f"Market Exposure:        {(df['Position'] > 0).mean() * 100:.1f}%")

    #visual
    plt.figure(figsize=(13, 6))
    plt.plot(df.index, df['Cumulative_Strategy_Return'], label=f"RSI(2) Strategy ({asset_name})", color='#2ca02c', lw=1.8)
    plt.plot(df.index, df['Cumulative_Market_Return'], label=f"Buy & Hold ({asset_name})", color='#7f7f7f', lw=1.2, ls='--')
    plt.title(f"{asset_name} Local CSV Backtest ({start_date[:4]}–Present)", fontsize=13, fontweight='bold')
    plt.ylabel("Growth of $1.00")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_local_mean_reversion_backtest("SPY_500", start_date="2000-01-01")
    run_local_mean_reversion_backtest("QQQ_NASDAQ100", start_date="2000-01-01")
