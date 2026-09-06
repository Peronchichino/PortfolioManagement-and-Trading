import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_5m_nasdaq_data(ticker_symbol: str = "QQQ") -> pd.DataFrame:
    """Fetch 60 days of 5-minute intraday data from Yahoo Finance."""
    print(f"Downloading 5-minute intraday data for {ticker_symbol} (last 60 days max)...")
    df = yf.download(ticker_symbol, period="60d", interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No intraday data returned for {ticker_symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Standardize time index to US/Eastern
    df = df.reset_index()
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df['Datetime'] = pd.to_datetime(df[date_col]).dt.tz_convert('America/New_York')
    df = df.sort_values('Datetime').set_index('Datetime')

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    return df[required_cols].dropna().copy()


def calculate_indicators(df: pd.DataFrame, ema_period: int = 12, atr_period: int = 14) -> pd.DataFrame:
    df = df.copy()
    # Continuous EMA across 5-minute bars
    df['EMA_12'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    # True Range and ATR
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = df['TR'].rolling(window=atr_period).mean()

    return df


def backtest_opening_ema_strategy(
    ticker: str = "QQQ",
    atr_multiplier: float = 2.0,
    risk_per_trade_pct: float = 0.01,  # 1% account risk per day
    slippage_commission_pct: float = 0.0003  # 3 bps friction per round trip
):
    raw_df = fetch_5m_nasdaq_data(ticker)
    df = calculate_indicators(raw_df)

    # Filter strictly to Regular Market Hours (09:30 to 16:00 ET)
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    df = df.between_time('09:30', '16:00').copy()

    initial_capital = 100_000.0
    equity = initial_capital
    daily_equity_curve = []
    trades = []

    unique_days = df['Date'].unique()

    for current_day in unique_days:
        day_df = df[df['Date'] == current_day]
        if len(day_df) < 5:
            continue

        # Bar 1: 09:30 - 09:35 ET
        first_bar = day_df.iloc[0]
        if pd.isna(first_bar['EMA_12']) or pd.isna(first_bar['ATR']):
            continue

        direction = 1 if first_bar['Close'] > first_bar['EMA_12'] else -1
        entry_bar = day_df.iloc[1]
        entry_price = entry_bar['Open']
        entry_time = entry_bar.name

        atr_val = first_bar['ATR']
        stop_distance = atr_multiplier * atr_val
        if stop_distance <= 0:
            continue

        # Position sizing: Fixed 1% dollar risk
        dollar_risk = equity * risk_per_trade_pct
        shares = dollar_risk / stop_distance
        position_notional = shares * entry_price

        # Track trailing stop
        exit_price = None
        exit_time = None
        exit_reason = None

        if direction == 1:
            trailing_stop = entry_price - stop_distance
            for idx in range(1, len(day_df)):
                bar = day_df.iloc[idx]
                # Update highest high and raise trailing stop
                trail_candidate = bar['High'] - stop_distance
                if trail_candidate > trailing_stop:
                    trailing_stop = trail_candidate

                # Check if stop is hit during this bar
                if bar['Low'] <= trailing_stop:
                    exit_price = min(bar['Open'], trailing_stop) if bar['Open'] < trailing_stop else trailing_stop
                    exit_time = bar.name
                    exit_reason = "Trailing Stop"
                    break
        else:
            trailing_stop = entry_price + stop_distance
            for idx in range(1, len(day_df)):
                bar = day_df.iloc[idx]
                # Update lowest low and lower trailing stop
                trail_candidate = bar['Low'] + stop_distance
                if trail_candidate < trailing_stop:
                    trailing_stop = trail_candidate

                # Check if stop is hit during this bar
                if bar['High'] >= trailing_stop:
                    exit_price = max(bar['Open'], trailing_stop) if bar['Open'] > trailing_stop else trailing_stop
                    exit_time = bar.name
                    exit_reason = "Trailing Stop"
                    break

        # If not stopped out, exit on the closing bell (15:55-16:00 ET bar)
        if exit_price is None:
            last_bar = day_df.iloc[-1]
            exit_price = last_bar['Close']
            exit_time = last_bar.name
            exit_reason = "Market Close"

        # Apply friction
        net_entry = entry_price * (1 + slippage_commission_pct if direction == 1 else 1 - slippage_commission_pct)
        net_exit = exit_price * (1 - slippage_commission_pct if direction == 1 else 1 + slippage_commission_pct)

        trade_pct = ((net_exit - net_entry) / net_entry) * direction
        dollar_pnl = shares * (net_exit - net_entry) if direction == 1 else shares * (net_entry - net_exit)

        equity += dollar_pnl
        daily_equity_curve.append({'Date': current_day, 'Equity': equity})

        trades.append({
            'Date': current_day,
            'Direction': 'LONG' if direction == 1 else 'SHORT',
            'Entry_Time': entry_time.strftime('%H:%M'),
            'Exit_Time': exit_time.strftime('%H:%M'),
            'Entry_Price': round(entry_price, 2),
            'Exit_Price': round(exit_price, 2),
            'Return_%': round(trade_pct * 100, 2),
            'Dollar_PnL': round(dollar_pnl, 2),
            'Exit_Reason': exit_reason,
            'Ending_Equity': round(equity, 2)
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(daily_equity_curve).set_index('Date')
    return trades_df, equity_df


def print_backtest_audit(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float = 100_000.0):
    print("\n" + "=" * 60)
    print("5-MINUTE OPENING CANDLE + 12 EMA BACKTEST RESULTS")
    print("=" * 60)

    if trades_df.empty:
        print("No trades generated.")
        return

    total_trades = len(trades_df)
    wins = trades_df[trades_df['Dollar_PnL'] > 0]
    losses = trades_df[trades_df['Dollar_PnL'] <= 0]

    win_rate = (len(wins) / total_trades) * 100
    gross_profits = wins['Dollar_PnL'].sum()
    gross_losses = abs(losses['Dollar_PnL'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf

    final_equity = equity_df['Equity'].iloc[-1]
    net_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

    peak = equity_df['Equity'].cummax()
    max_dd = ((equity_df['Equity'] - peak) / peak).min() * 100

    long_trades = trades_df[trades_df['Direction'] == 'LONG']
    short_trades = trades_df[trades_df['Direction'] == 'SHORT']

    print(f"Total Trading Days Tested: {total_trades}")
    print(f"Win Rate:                  {win_rate:.2f}%")
    print(f"Profit Factor:             {profit_factor:.2f}")
    print(f"Total Net Return:          {net_return_pct:+.2f}%")
    print(f"Max Strategy Drawdown:     {max_dd:.2f}%")
    print(f"Average Trade Return:      {trades_df['Return_%'].mean():+.2f}%")
    print(f"Long Trades Win Rate:      {(len(long_trades[long_trades['Dollar_PnL'] > 0]) / max(len(long_trades), 1)) * 100:.2f}% ({len(long_trades)} trades)")
    print(f"Short Trades Win Rate:     {(len(short_trades[short_trades['Dollar_PnL'] > 0]) / max(len(short_trades), 1)) * 100:.2f}% ({len(short_trades)} trades)")
    print("=" * 60)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(equity_df.index, equity_df['Equity'], label="Strategy Account Equity ($)", color='#1f77b4', lw=2)
    plt.title("5-Min Opening EMA + Volatility Trailing Stop Equity Curve", fontsize=12, fontweight='bold')
    plt.ylabel("Account Equity ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    trades, equity = backtest_opening_ema_strategy("QQQ", atr_multiplier=2.0, risk_per_trade_pct=0.01)
    print_backtest_audit(trades, equity)