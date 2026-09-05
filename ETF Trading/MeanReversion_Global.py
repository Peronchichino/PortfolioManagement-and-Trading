import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

INDEX_ETFS = {
    'SPY_500': 'SPY',
    'QQQ_NASDAQ100': 'QQQ',
    'DIA_DOWJONES': 'DIA',
    'IWM_Russell2000': 'IWM',
    'FTSE_100': 'ISF.L',
    'DAX': 'DAX.DE',
    'CAC_40': 'PX1.PA',
    'Nikkei_225': '^N225',
    'Hang_Seng': '^HSI',
    'ShanghaiComposite': '000001.SS',
    'MSCI_World': 'URTH',
    'EEM_MSCI_Emerging_Markets': 'EEM',
    'IWF_Russell1000Growth': 'IWF',
    'IWD_Russell1000Value': 'IWD',
    'EFA_Europe': 'IEV',
    'EFA_DevelopedMarkets': 'EFA'
}


def calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_yfinance_data(symbol: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval="1d", auto_adjust=False)

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.sort_values(by='Date').set_index('Date')

        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return pd.DataFrame()

        return df[required_cols].dropna().copy()
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def run_etf_strategy(name: str, df: pd.DataFrame, max_holding_period: int = 10):
    if len(df) < 201:
        print(f"Insufficient history for {name}. Skipping.")
        return None, None

    df = df.copy()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['RSI_2'] = calculate_rsi(df['Close'], period=2)

    df.dropna(inplace=True)
    if df.empty:
        return None, None

    df['Buy_Signal'] = (df['Close'] > df['SMA_200']) & (df['RSI_2'] < 10)
    df['Sell_Signal'] = (df['Close'] > df['SMA_200']) & (df['RSI_2'] > 90)
    df['Exit_Signal'] = df['Close'] > df['SMA_5']

    in_position = False
    entry_price = 0.0
    entry_idx = 0
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
                exit_price = opens[i + 1]
                pnl = (exit_price - entry_price) / entry_price
                trades.append({
                    'Asset': name,
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
    df['Market_Return'] = df['Close'].pct_change().fillna(0)
    df['Strategy_Return'] = df['Position'].shift(1).fillna(0) * df['Market_Return']

    df['Cumulative_Market_Return'] = (1 + df['Market_Return']).cumprod()
    df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()

    trades_df = pd.DataFrame(trades)
    return df, trades_df


def print_performance_metrics(name: str, df: pd.DataFrame, trades_df: pd.DataFrame):
    print(f"\n{name} Performance Summary {'='*30}")
    if trades_df is None or trades_df.empty:
        print("No trades executed.")
        return

    total_trades = len(trades_df)
    win_rate = (trades_df['PnL'] > 0).mean() * 100
    avg_pnl = trades_df['PnL'].mean()
    losing_sum = abs(trades_df.loc[trades_df['PnL'] < 0, 'PnL'].sum())
    profit_factor = trades_df.loc[trades_df['PnL'] > 0, 'PnL'].sum() / losing_sum if losing_sum != 0 else np.inf

    years = (df.index[-1] - df.index[0]).days / 365.25
    strat_cagr = (df['Cumulative_Strategy_Return'].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    bench_cagr = (df['Cumulative_Market_Return'].iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    strat_peaks = df['Cumulative_Strategy_Return'].cummax()
    strat_dd = float(((df['Cumulative_Strategy_Return'] - strat_peaks) / strat_peaks).min()) * 100
    bench_peaks = df['Cumulative_Market_Return'].cummax()
    bench_dd = float(((df['Cumulative_Market_Return'] - bench_peaks) / bench_peaks).min()) * 100

    print(f"Total Trades:           {total_trades}")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Profit Factor:          {profit_factor:.2f}")
    print(f"Avg Trade Return:       {avg_pnl:.2f}%")
    print(f"Avg Holding Period:     {trades_df['Days_Held'].mean():.1f} trading days")
    print(f"Strategy CAGR:          {strat_cagr * 100:.2f}%")
    print(f"Buy & Hold CAGR:        {bench_cagr * 100:.2f}%")
    print(f"Strategy Max Drawdown:  {strat_dd:.2f}%")
    print(f"Buy & Hold Max Drawdown:{bench_dd:.2f}%")
    print(f"Market Exposure:        {(df['Position'] > 0).mean() * 100:.1f}%")


def run_global_mean_reversion_portfolio(start_date: str = "2000-01-01"):
    all_trades = []
    strat_returns_dict = {}
    bench_returns_dict = {}

    print(f"Fetching data and backtesting {len(INDEX_ETFS)} ETFs simultaneously...")

    for name, symbol in INDEX_ETFS.items():
        print(f"Processing {name} ({symbol})...")
        raw_df = fetch_yfinance_data(symbol, start_date=start_date)
        if raw_df.empty:
            print(f"No price data retrieved for {symbol}. Skipping.")
            continue

        processed_df, trades_df = run_etf_strategy(name, raw_df)
        if processed_df is None:
            continue

        print_performance_metrics(name, processed_df, trades_df)

        if trades_df is not None and not trades_df.empty:
            all_trades.append(trades_df)

        strat_returns_dict[name] = processed_df['Strategy_Return']
        bench_returns_dict[name] = processed_df['Market_Return']

    if not strat_returns_dict:
        print("No valid asset returns available to construct the portfolio.")
        return

    # Simultaneous multi-asset portfolio simulation
    strat_returns_df = pd.DataFrame(strat_returns_dict).fillna(0.0)
    bench_returns_df = pd.DataFrame(bench_returns_dict).fillna(0.0)

    # Equal weighting across all active assets
    portfolio_strat_return = strat_returns_df.mean(axis=1)
    portfolio_bench_return = bench_returns_df.mean(axis=1)

    cum_strat = (1 + portfolio_strat_return).cumprod()
    cum_bench = (1 + portfolio_bench_return).cumprod()

    # Portfolio metrics
    total_combined_trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    years = (cum_strat.index[-1] - cum_strat.index[0]).days / 365.25

    port_strat_cagr = (cum_strat.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0
    port_bench_cagr = (cum_bench.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    strat_peaks = cum_strat.cummax()
    port_strat_dd = float(((cum_strat - strat_peaks) / strat_peaks).min()) * 100
    bench_peaks = cum_bench.cummax()
    port_bench_dd = float(((cum_bench - bench_peaks) / bench_peaks).min()) * 100

    print("\n" + "=" * 50)
    print("GLOBAL MULTI-ASSET PORTFOLIO RESULTS")
    print("=" * 50)
    print(f"Active Assets Traded:   {len(strat_returns_dict)}")
    print(f"Total Portfolio Trades: {len(total_combined_trades)}")
    if not total_combined_trades.empty:
        win_rate = (total_combined_trades['PnL'] > 0).mean() * 100
        print(f"Overall Win Rate:       {win_rate:.2f}%")
        print(f"Average Trade Return:   {total_combined_trades['PnL'].mean():.2f}%")
    print(f"Portfolio Strat CAGR:   {port_strat_cagr * 100:.2f}%")
    print(f"Benchmark Portfolio CAGR:{port_bench_cagr * 100:.2f}%")
    print(f"Portfolio Max Drawdown: {port_strat_dd:.2f}%")
    print(f"Benchmark Max Drawdown: {port_bench_dd:.2f}%")

    # Plot simultaneous portfolio equity curve
    plt.figure(figsize=(14, 7))
    plt.plot(cum_strat.index, cum_strat, label="Global Mean Reversion Portfolio (RSI 2 / SMA 200)", color='#1f77b4', lw=2)
    plt.plot(cum_bench.index, cum_bench, label="Equal-Weight Global Benchmark (Buy & Hold)", color='#7f7f7f', lw=1.2, ls='--')
    plt.title("Simultaneous Multi-ETF Mean Reversion Strategy Performance", fontsize=14, fontweight='bold')
    plt.ylabel("Growth of $1.00")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_global_mean_reversion_portfolio(start_date="2000-01-01")