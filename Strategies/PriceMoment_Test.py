import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Download adjusted monthly price data
tickers = ["MP", "NVO", "CVX", "AMD", "JPM"]
raw_data = yf.download(tickers, start="2020-01-01", interval="1mo", auto_adjust=True)["Close"]
raw_data = raw_data.dropna()

# 2. Compute 1-month forward returns and 12-1 momentum signal
# Standard monthly return: R(t) = P(t) / P(t-1) - 1
monthly_returns = raw_data.pct_change()

# R_cum: 12-month formation skipping the most recent 1 month (t-13 to t-1)
formation_window = 12
skip_window = 1

# Price 1 month ago divided by price 13 months ago - 1
momentum_score = raw_data.shift(skip_window) / raw_data.shift(skip_window + formation_window) - 1
momentum_score = momentum_score.dropna()

# Align monthly returns with signal dates
aligned_returns = monthly_returns.loc[momentum_score.index]

# 3. Strategy Simulation
dollar_neutral_returns = []
long_only_returns = []
benchmark_equal_weight = aligned_returns.mean(axis=1)

for date, scores in momentum_score.iterrows():
    # Rank assets: highest score = top winner, lowest score = bottom loser
    ranked = scores.sort_values(ascending=False)
    winner = ranked.index[0]
    loser = ranked.index[-1]
    
    # Next period return for winner and loser
    r_winner = aligned_returns.loc[date, winner]
    r_loser = aligned_returns.loc[date, loser]
    
    # Dollar-neutral (Long Winner / Short Loser)
    dollar_neutral_returns.append(r_winner - r_loser)
    
    # Long-only top winner
    long_only_returns.append(r_winner)

# 4. Assemble Performance DataFrame
df_perf = pd.DataFrame({
    "Long/Short (12-1)": dollar_neutral_returns,
    "Long-Only Winner": long_only_returns,
    "Equal-Weight Universe": benchmark_equal_weight
}, index=momentum_score.index)

# Compute cumulative equity curves
cumulative_growth = (1 + df_perf).cumprod()

# 5. Summary Metrics Function
def calc_metrics(returns_series):
    ann_return = returns_series.mean() * 12
    ann_vol = returns_series.std() * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    cum = (1 + returns_series).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    return pd.Series({
        "Ann. Return": f"{ann_return * 100:.2f}%",
        "Ann. Volatility": f"{ann_vol * 100:.2f}%",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd * 100:.2f}%"
    })

performance_summary = df_perf.apply(calc_metrics)
print(performance_summary)