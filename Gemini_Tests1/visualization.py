"""
visualization.py - Publication-quality charts and diagnostics for systematic quantitative trading.

Generates:
1. Strategy vs. SPY equity curves
2. Strategy vs. SPY drawdown curves
3. Rolling Sharpe ratio comparison (6-month / 126-day)
4. Monthly returns comparison heatmap / table
5. Distribution of trade returns (KDE + histogram)
6. Parameter sensitivity heatmaps
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_equity_and_drawdowns(
    strat_history: pd.DataFrame,
    bm_history: pd.DataFrame,
    save_path: str = "charts/equity_drawdown.png",
    title_suffix: str = ""
):
    """Plot Strategy vs. Buy & Hold SPY Equity Curve and Drawdown Curves."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.3, 1.2]})
    
    # 1. Equity curves
    ax1.plot(strat_history.index, strat_history["portfolio_value"], label="Mean-Reversion Strategy", color="#1f77b4", linewidth=2.0)
    ax1.plot(bm_history.index, bm_history["portfolio_value"], label="Buy & Hold SPY", color="#ff7f0e", linestyle="--", linewidth=1.6, alpha=0.9)
    ax1.set_title(f"Cumulative Portfolio Value ($100k Starting Capital) {title_suffix}", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter("${x:,.0f}")
    
    # Shading periods
    oos_start = pd.to_datetime("2025-01-01")
    if strat_history.index[-1] > oos_start:
        ax1.axvspan(oos_start, strat_history.index[-1], color="#2ca02c", alpha=0.1, label="Out-of-Sample Period (2025)")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Drawdowns
    ax2.plot(strat_history.index, strat_history["drawdown"] * 100, label="Strategy Drawdown", color="#1f77b4", linewidth=1.3)
    ax2.fill_between(strat_history.index, strat_history["drawdown"] * 100, 0, color="#1f77b4", alpha=0.25)
    ax2.plot(bm_history.index, bm_history["drawdown"] * 100, label="SPY Drawdown", color="#ff7f0e", linestyle="--", linewidth=1.3, alpha=0.8)
    ax2.fill_between(bm_history.index, bm_history["drawdown"] * 100, 0, color="#ff7f0e", alpha=0.15)
    
    if strat_history.index[-1] > oos_start:
        ax2.axvspan(oos_start, strat_history.index[-1], color="#2ca02c", alpha=0.1)
        
    ax2.set_title("Underwater Drawdown Profile", fontsize=11, fontweight="bold", pad=6)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.yaxis.set_major_formatter("{x:.1f}%")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_rolling_sharpe(
    strat_history: pd.DataFrame,
    bm_history: pd.DataFrame,
    window: int = 126,  # ~6 months
    save_path: str = "charts/rolling_sharpe.png"
):
    """Plot Rolling 6-Month (126-day) Annualized Sharpe Ratio."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    s_ret = strat_history["daily_return"]
    b_ret = bm_history["daily_return"]
    
    s_roll_mean = s_ret.rolling(window).mean()
    s_roll_std = s_ret.rolling(window).std()
    s_roll_sharpe = (s_roll_mean / s_roll_std * np.sqrt(252)).dropna()
    
    b_roll_mean = b_ret.rolling(window).mean()
    b_roll_std = b_ret.rolling(window).std()
    b_roll_sharpe = (b_roll_mean / b_roll_std * np.sqrt(252)).dropna()
    
    plt.figure(figsize=(12, 5))
    plt.plot(s_roll_sharpe.index, s_roll_sharpe, label=f"Strategy ({window}d Rolling Sharpe)", color="#1f77b4", linewidth=1.6)
    plt.plot(b_roll_sharpe.index, b_roll_sharpe, label=f"SPY ({window}d Rolling Sharpe)", color="#ff7f0e", linestyle="--", linewidth=1.4, alpha=0.85)
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    
    oos_start = pd.to_datetime("2025-01-01")
    if s_roll_sharpe.index[-1] > oos_start:
        plt.axvspan(oos_start, s_roll_sharpe.index[-1], color="#2ca02c", alpha=0.1, label="Out-of-Sample (2025)")
        
    plt.title(f"Rolling {window}-Day (~6 Month) Annualized Sharpe Ratio", fontsize=13, fontweight="bold", pad=10)
    plt.ylabel("Sharpe Ratio", fontsize=11)
    plt.xlabel("Date", fontsize=11)
    plt.legend(loc="upper left", frameon=True, framealpha=0.9)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_monthly_returns(
    strat_history: pd.DataFrame,
    bm_history: pd.DataFrame,
    save_path: str = "charts/monthly_returns.png"
):
    """Plot Monthly Returns Heatmaps side by side."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def get_monthly_table(hist):
        m_rets = hist["portfolio_value"].resample("ME").last().pct_change().dropna()
        df_m = pd.DataFrame({"Return": m_rets})
        df_m["Year"] = df_m.index.year
        df_m["Month"] = df_m.index.strftime("%b")
        pivot = df_m.pivot(index="Year", columns="Month", values="Return")
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        existing_months = [m for m in month_order if m in pivot.columns]
        return pivot[existing_months] * 100.0

    s_monthly = get_monthly_table(strat_history)
    b_monthly = get_monthly_table(bm_history)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    
    sns.heatmap(s_monthly, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False, ax=ax1, linewidths=0.5)
    ax1.set_title("Strategy Monthly Returns (%)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Year")
    ax1.set_xlabel("")
    
    sns.heatmap(b_monthly, annot=True, fmt=".1f", cmap="RdYlGn", center=0, cbar=False, ax=ax2, linewidths=0.5)
    ax2.set_title("Buy & Hold SPY Monthly Returns (%)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Year")
    ax2.set_xlabel("Month")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trade_distribution(
    trades: list,
    save_path: str = "charts/trade_distribution.png"
):
    """Plot histogram and KDE of trade returns."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not trades or len(trades) == 0:
        return
        
    df_t = pd.DataFrame(trades)
    returns = df_t["pnl_pct"] * 100.0
    
    plt.figure(figsize=(9, 5))
    sns.histplot(returns, kde=True, bins=20, color="#1f77b4", edgecolor="black", alpha=0.6)
    
    mean_ret = returns.mean()
    median_ret = returns.median()
    win_rate = (returns > 0).mean() * 100.0
    
    plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(mean_ret, color="red", linestyle="-", linewidth=1.5, label=f"Mean: {mean_ret:+.2f}%")
    plt.axvline(median_ret, color="green", linestyle=":", linewidth=1.5, label=f"Median: {median_ret:+.2f}%")
    
    plt.title(f"Trade Return Distribution ({len(df_t)} Trades, Win Rate: {win_rate:.1f}%)", fontsize=13, fontweight="bold", pad=10)
    plt.xlabel("Trade Return (%)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_parameter_sensitivity_heatmap(
    sens_df: pd.DataFrame,
    x_param: str = "rsi_entry",
    y_param: str = "exit_sma",
    metric: str = "sharpe",
    save_path: str = "charts/parameter_sensitivity.png"
):
    """Plot Parameter Sensitivity Heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if sens_df.empty:
        return
        
    pivot = sens_df.groupby([y_param, x_param])[metric].mean().unstack()
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", linewidths=0.5, cbar_kws={"label": metric.upper()})
    plt.title(f"Parameter Sensitivity: Development Sharpe Ratio vs ({x_param} and {y_param})", fontsize=12, fontweight="bold", pad=10)
    plt.xlabel(f"{x_param.replace('_', ' ').title()}", fontsize=11)
    plt.ylabel(f"{y_param.replace('_', ' ').title()}", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
