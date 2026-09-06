import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def fetch_intraday_and_vix(symbol: str = "QQQ") -> pd.DataFrame:
    print(f"Downloading 5-minute {symbol} and daily ^VIX data...")
    # 5-minute intraday price data
    df = yf.download(symbol, period="60d", interval="5m", progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df['Datetime'] = pd.to_datetime(df[date_col]).dt.tz_convert('America/New_York')
    df = df.sort_values('Datetime').set_index('Datetime')
    df = df.between_time('09:30', '16:00').copy()
    df['Date'] = df.index.date

    # Pull daily VIX to compute Rule of 16 expected move
    vix = yf.download("^VIX", period="90d", interval="1d", progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix.reset_index()
    vix['Date'] = pd.to_datetime(vix['Date']).dt.date
    vix['Expected_Move_Pct'] = (vix['Close'] / 16.0) / 100.0  # VIX / 16 as decimal

    vix_map = vix.set_index('Date')['Expected_Move_Pct'].to_dict()
    df['VIX_Daily_Move_Pct'] = df['Date'].map(vix_map).fillna(0.01)  # Default to 1% if missing
    return df


def calculate_session_vwap(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3.0
    df['PV'] = df['Typical_Price'] * df['Volume']

    df['Cum_PV'] = df.groupby('Date')['PV'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_PV'] / df['Cum_Vol']
    return df


def backtest_vwap_pullback_vix_targets(df: pd.DataFrame, risk_per_day: float = 0.01):
    """
    Nicholas Crown Hybrid:
    - Bias: Price must establish above VWAP between 09:30 and 10:15
    - Trigger: First pullback to touch VWAP from above, bouncing and closing back > VWAP
    - Target: Daily Open + Rule of 16 Implied Daily Move (VIX / 16)
    - Stop Loss: Invalidation below VWAP (0.3% buffer)
    """
    df = calculate_session_vwap(df)
    initial_equity = 100_000.0
    equity = initial_equity
    trades = []
    equity_curve = []

    for date, day_df in df.groupby('Date'):
        if len(day_df) < 20:
            continue

        day_open = day_df.iloc[0]['Open']
        exp_move_pct = day_df.iloc[0]['VIX_Daily_Move_Pct']
        target_price = day_open * (1.0 + exp_move_pct)

        in_position = False
        pulled_back = False
        established_above = False
        entry_price = 0.0
        stop_price = 0.0
        shares = 0.0
        exit_price = None
        exit_reason = None

        # Scan from 10:00 AM onwards (allowing morning auction to set the benchmark)
        trading_bars = day_df.between_time('10:00', '15:50')

        for i in range(len(trading_bars)):
            bar = trading_bars.iloc[i]
            vwap = bar['VWAP']

            if not in_position:
                # 1. Check if buyers established clear control above VWAP
                if bar['Close'] > vwap * 1.0015:
                    established_above = True

                # 2. First pullback: touches VWAP and holds
                if established_above and bar['Low'] <= vwap and bar['Close'] >= vwap:
                    entry_price = bar['Close']
                    stop_price = vwap * 0.997  # 30 bps structural buffer under VWAP
                    risk = entry_price - stop_price
                    if risk <= 0:
                        continue

                    shares = (equity * risk_per_day) / risk
                    in_position = True
            else:
                # Target hit: Rule of 16 Implied Upper Boundary
                if bar['High'] >= target_price:
                    exit_price = target_price
                    exit_reason = "VIX Expected Move (+1σ)"
                    break
                # Stopped out: Price broke structurally below VWAP
                elif bar['Low'] <= stop_price:
                    exit_price = stop_price
                    exit_reason = "VWAP Invalidation Stop"
                    break

        if in_position:
            if exit_price is None:
                exit_price = trading_bars.iloc[-1]['Close']
                exit_reason = "Session Close"

            pnl = shares * (exit_price - entry_price)
            equity += pnl
            trades.append({
                'Date': date,
                'Entry_Price': entry_price,
                'Exit_Price': exit_price,
                'PnL': pnl,
                'Return_%': ((exit_price - entry_price) / entry_price) * 100,
                'Exit_Reason': exit_reason
            })

        equity_curve.append({'Date': date, 'Equity': equity})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve).set_index('Date')


if __name__ == "__main__":
    df = fetch_intraday_and_vix("QQQ")
    trades, equity = backtest_vwap_pullback_vix_targets(df)

    print("\n" + "=" * 60)
    print("NICHOLAS CROWN HYBRID: VWAP PULLBACK + VIX RULE OF 16")
    print("=" * 60)
    if not trades.empty:
        total = len(trades)
        wins = trades[trades['PnL'] > 0]
        losses = trades[trades['PnL'] <= 0]
        pf = (wins['PnL'].sum() / abs(losses['PnL'].sum())) if len(losses) > 0 else np.inf
        net_ret = ((equity['Equity'].iloc[-1] - 100_000) / 100_000) * 100

        print(f"Total Completed Trades:   {total}")
        print(f"Win Rate:                 {(len(wins)/total)*100:.2f}%")
        print(f"Profit Factor:            {pf:.2f}")
        print(f"Total Strategy Return:    {net_ret:+.2f}%")
        print(f"Average Return / Trade:   {trades['Return_%'].mean():+.2f}%")
        print("\nExit Breakdown:")
        print(trades['Exit_Reason'].value_counts())
    else:
        print("No valid trades triggered.")