import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


def fetch_5m_data(symbol: str = "QQQ") -> pd.DataFrame:
    """Fetch 60 days of 5-minute data from Yahoo Finance."""
    print(f"Downloading 5-minute intraday bars for {symbol}...")
    df = yf.download(symbol, period="60d", interval="5m", progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No intraday data returned for {symbol}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df['Datetime'] = pd.to_datetime(df[date_col]).dt.tz_convert('America/New_York')
    df = df.sort_values('Datetime').set_index('Datetime')

    # Regular Trading Hours (09:30 to 16:00 ET)
    df = df.between_time('09:30', '16:00').copy()
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    return df[['Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Time']].dropna().copy()


def calculate_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate cumulative VWAP and rolling standard deviation bands reset daily at 09:30 AM."""
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['PV'] = df['Typical_Price'] * df['Volume']

    # Group by trading session
    df['Cum_PV'] = df.groupby('Date')['PV'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cum_Vol']

    # Standard deviation around session VWAP
    df['VWAP_Diff_Sq'] = ((df['Typical_Price'] - df['VWAP']) ** 2) * df['Volume']
    df['Cum_VWAP_Diff_Sq'] = df.groupby('Date')['VWAP_Diff_Sq'].cumsum()
    df['VWAP_Std'] = np.sqrt(df['Cum_VWAP_Diff_Sq'] / df['Cum_Vol'])

    df['VWAP_Upper'] = df['VWAP'] + (2.0 * df['VWAP_Std'])
    df['VWAP_Lower'] = df['VWAP'] - (2.0 * df['VWAP_Std'])
    return df


def backtest_30m_orb(df: pd.DataFrame, risk_per_day: float = 0.01):
    """
    30-Minute Opening Range Breakout:
    - Formation Window: 09:30 - 10:00 AM ET (first six 5-min bars)
    - Trigger: 5-minute bar closes outside the 30m High/Low
    - Risk & Stop: Fixed stop at the 30-minute midpoint
    - Exit: Target 2.0x R/R, stop at range midpoint, or exit at market close
    """
    initial_equity = 100_000.0
    equity = initial_equity
    equity_curve = []
    trades = []

    for date, day_df in df.groupby('Date'):
        if len(day_df) < 12:
            continue

        orb_bars = day_df.between_time('09:30', '10:00')
        post_orb_bars = day_df.between_time('10:05', '15:55')

        if len(orb_bars) < 6 or post_orb_bars.empty:
            continue

        orb_high = orb_bars['High'].max()
        orb_low = orb_bars['Low'].min()
        orb_mid = (orb_high + orb_low) / 2.0

        direction = 0
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        entry_idx = -1

        # Check for breakout post 10:00 AM
        for i in range(len(post_orb_bars)):
            bar = post_orb_bars.iloc[i]
            if bar['Close'] > orb_high:
                direction = 1
                entry_price = bar['Close']
                stop_price = orb_mid
                risk = entry_price - stop_price
                target_price = entry_price + (2.0 * risk)
                entry_idx = i
                break
            elif bar['Close'] < orb_low:
                direction = -1
                entry_price = bar['Close']
                stop_price = orb_mid
                risk = stop_price - entry_price
                target_price = entry_price - (2.0 * risk)
                entry_idx = i
                break

        if direction == 0 or (entry_price - stop_price) == 0:
            equity_curve.append({'Date': date, 'Equity': equity})
            continue

        # Position Sizing based on 1% equity risk
        dollar_risk = equity * risk_per_day
        shares = dollar_risk / abs(entry_price - stop_price)

        exit_price = None
        exit_reason = None

        # Manage active position
        for j in range(entry_idx + 1, len(post_orb_bars)):
            bar = post_orb_bars.iloc[j]
            if direction == 1:
                if bar['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss (Midpoint)"
                    break
                elif bar['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = "Target (2.0 R/R)"
                    break
            else:
                if bar['High'] >= stop_price:
                    exit_price = stop_price
                    exit_reason = "Stop Loss (Midpoint)"
                    break
                elif bar['Low'] <= target_price:
                    exit_price = target_price
                    exit_reason = "Target (2.0 R/R)"
                    break

        if exit_price is None:
            exit_price = post_orb_bars.iloc[-1]['Close']
            exit_reason = "Market Close"

        pnl = shares * (exit_price - entry_price) * direction
        equity += pnl
        equity_curve.append({'Date': date, 'Equity': equity})

        trades.append({
            'Date': date,
            'Model': '30m ORB',
            'Direction': 'LONG' if direction == 1 else 'SHORT',
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'PnL': pnl,
            'Return_%': ((exit_price - entry_price) / entry_price) * direction * 100,
            'Exit_Reason': exit_reason
        })

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('Date')


def backtest_vwap_mean_reversion(df: pd.DataFrame, risk_per_day: float = 0.01):
    """
    Session-Anchored VWAP Band Reversion:
    - Allowed Entry: 10:00 AM - 14:30 PM (avoids opening whip and close decay)
    - Trigger: Close pierces ±2.0 VWAP Std Dev Band
    - Take Profit: Return to central VWAP line
    - Stop Loss: 1.0x band width outside the band
    """
    df = calculate_session_vwap(df)
    initial_equity = 100_000.0
    equity = initial_equity
    equity_curve = []
    trades = []

    for date, day_df in df.groupby('Date'):
        if len(day_df) < 15:
            continue

        active_window = day_df.between_time('10:00', '15:45')
        if active_window.empty:
            continue

        in_pos = False
        direction = 0
        entry_price = 0.0
        stop_price = 0.0
        shares = 0.0
        exit_price = None
        exit_reason = None

        for i in range(len(active_window)):
            bar = active_window.iloc[i]
            vwap = bar['VWAP']
            upper = bar['VWAP_Upper']
            lower = bar['VWAP_Lower']
            std = bar['VWAP_Std']

            if not in_pos:
                # Pierced upper band -> Short back to VWAP
                if bar['Close'] >= upper and std > 0:
                    direction = -1
                    entry_price = bar['Close']
                    stop_price = upper + (1.0 * std)
                    risk = stop_price - entry_price
                    shares = (equity * risk_per_day) / risk
                    in_pos = True
                # Pierced lower band -> Long back to VWAP
                elif bar['Close'] <= lower and std > 0:
                    direction = 1
                    entry_price = bar['Close']
                    stop_price = lower - (1.0 * std)
                    risk = entry_price - stop_price
                    shares = (equity * risk_per_day) / risk
                    in_pos = True
            else:
                # Manage active trade
                if direction == 1:
                    if bar['High'] >= vwap:
                        exit_price = vwap
                        exit_reason = "Mean Reversion (VWAP Tagged)"
                        break
                    elif bar['Low'] <= stop_price:
                        exit_price = stop_price
                        exit_reason = "Stop Loss"
                        break
                else:
                    if bar['Low'] <= vwap:
                        exit_price = vwap
                        exit_reason = "Mean Reversion (VWAP Tagged)"
                        break
                    elif bar['High'] >= stop_price:
                        exit_price = stop_price
                        exit_reason = "Stop Loss"
                        break

        if in_pos:
            if exit_price is None:
                exit_price = active_window.iloc[-1]['Close']
                exit_reason = "Session Close"

            pnl = shares * (exit_price - entry_price) * direction
            equity += pnl
            trades.append({
                'Date': date,
                'Model': 'VWAP Reversion',
                'Direction': 'LONG' if direction == 1 else 'SHORT',
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'PnL': pnl,
                'Return_%': ((exit_price - entry_price) / entry_price) * direction * 100,
                'Exit_Reason': exit_reason
            })

        equity_curve.append({'Date': date, 'Equity': equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('Date')


def print_model_audit(name: str, trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float = 100_000.0):
    print(f"\n{'='*20} {name} {'='*20}")
    if trades_df.empty:
        print("No completed trades.")
        return

    total = len(trades_df)
    wins = trades_df[trades_df['PnL'] > 0]
    losses = trades_df[trades_df['PnL'] <= 0]
    win_rate = (len(wins) / total) * 100

    gross_profit = wins['PnL'].sum()
    gross_loss = abs(losses['PnL'].sum())
    pf = (gross_profit / gross_loss) if gross_loss > 0 else np.inf

    net_return = ((equity_df['Equity'].iloc[-1] - initial_capital) / initial_capital) * 100
    peak = equity_df['Equity'].cummax()
    max_dd = ((equity_df['Equity'] - peak) / peak).min() * 100

    print(f"Total Trades Taken:       {total}")
    print(f"Win Rate:                 {win_rate:.2f}%")
    print(f"Profit Factor:            {pf:.2f}")
    print(f"Total Strategy Return:    {net_return:+.2f}%")
    print(f"Max Strategy Drawdown:    {max_dd:.2f}%")
    print(f"Average Return / Trade:   {trades_df['Return_%'].mean():+.2f}%")
    print(f"Top Exit Reason:          {trades_df['Exit_Reason'].value_counts().index[0]} ({trades_df['Exit_Reason'].value_counts().iloc[0]} trades)")


def run_full_comparison():
    df = fetch_5m_data("QQQ")

    orb_trades, orb_equity = backtest_30m_orb(df)
    vwap_trades, vwap_equity = backtest_vwap_mean_reversion(df)

    print("\n" + "=" * 65)
    print("INSTITUTIONAL INTRADAY SETUPS AUDIT (60 DAYS 5-MIN QQQ)")
    print("=" * 65)

    print_model_audit("30-Minute Opening Range Breakout (ORB)", orb_trades, orb_equity)
    print_model_audit("Session-Anchored VWAP 2.0-StdDev Mean Reversion", vwap_trades, vwap_equity)

    # Comparative Plot
    plt.figure(figsize=(12, 6))
    if not orb_equity.empty:
        plt.plot(orb_equity.index, orb_equity['Equity'], label="30-Min ORB (Midpoint Stop, 2R Target)", color='#2ca02c', lw=2)
    if not vwap_equity.empty:
        plt.plot(vwap_equity.index, vwap_equity['Equity'], label="VWAP 2σ Mean Reversion", color='#1f77b4', lw=2)

    plt.axhline(100_000, color='gray', linestyle='--', alpha=0.6, label="Starting Capital ($100k)")
    plt.title("Institutional Intraday Setups on QQQ (5-Minute Bars)", fontsize=13, fontweight='bold')
    plt.ylabel("Portfolio Equity ($)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_full_comparison()