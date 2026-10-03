"""
metrics.py - Quantitative performance metrics and trade analysis.
Calculates risk-adjusted returns, drawdowns, and trade-level statistics
for both active strategies and buy-and-hold benchmarks.
"""

import numpy as np
import pandas as pd


def calculate_portfolio_metrics(
    portfolio_history: pd.DataFrame,
    trade_log: list,
    starting_capital: float = 100000.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> dict:
    """
    Compute comprehensive portfolio-level and trade-level performance metrics.
    
    portfolio_history must contain:
    - 'portfolio_value'
    - 'daily_return'
    - 'position' (1 if long, 0 if flat)
    - 'drawdown'
    """
    if portfolio_history.empty:
        return {}
        
    p_vals = portfolio_history["portfolio_value"].values
    d_rets = portfolio_history["daily_return"].values
    positions = portfolio_history["position"].values
    
    n_days = len(p_vals)
    if n_days <= 1:
        return {}
        
    start_val = starting_capital
    final_val = p_vals[-1]
    
    # 1. Total Return & CAGR
    total_return = (final_val - start_val) / start_val
    n_years = n_days / annualization_factor
    if n_years > 0 and final_val > 0:
        cagr = (final_val / start_val) ** (1.0 / n_years) - 1.0
    else:
        cagr = 0.0
        
    # 2. Volatility (Annualized)
    daily_vol = np.std(d_rets, ddof=1) if len(d_rets) > 1 else 0.0
    ann_vol = daily_vol * np.sqrt(annualization_factor)
    
    # 3. Sharpe Ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess_rets = d_rets - rf_daily
    excess_mean = np.mean(excess_rets)
    sharpe_ratio = (excess_mean / daily_vol * np.sqrt(annualization_factor)) if daily_vol > 1e-8 else 0.0
    
    # 4. Sortino Ratio (Downside deviation relative to rf)
    downside_rets = d_rets[d_rets < rf_daily] - rf_daily
    if len(downside_rets) > 0:
        downside_std = np.sqrt(np.mean(downside_rets ** 2))
        sortino_ratio = (excess_mean / downside_std * np.sqrt(annualization_factor)) if downside_std > 1e-8 else 0.0
    else:
        sortino_ratio = np.nan
        
    # 5. Drawdowns
    running_max = np.maximum.accumulate(p_vals)
    drawdowns = (p_vals - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) # negative number, e.g. -0.15
    
    # 6. Calmar Ratio
    calmar_ratio = (cagr / abs(max_drawdown)) if abs(max_drawdown) > 1e-6 else 0.0
    
    # 7. Exposure (% of days invested in market)
    exposure = float(np.mean(positions > 0))
    
    # 8. Trade Statistics
    if trade_log and len(trade_log) > 0:
        trade_df = pd.DataFrame(trade_log)
        n_trades = len(trade_df)
        trade_returns = trade_df["pnl_pct"].values
        trade_pnls = trade_df["pnl"].values
        
        wins = trade_df[trade_df["pnl"] > 0]
        losses = trade_df[trade_df["pnl"] < 0]
        
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_trade_return = float(np.mean(trade_returns)) if n_trades > 0 else 0.0
        
        gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-6 else (np.inf if gross_profit > 0 else 0.0)
        
        best_trade = float(np.max(trade_returns)) if n_trades > 0 else 0.0
        worst_trade = float(np.min(trade_returns)) if n_trades > 0 else 0.0
        
        avg_holding_period = float(trade_df["holding_days"].mean()) if "holding_days" in trade_df.columns else 0.0
    else:
        n_trades = 0
        win_rate = 0.0
        avg_trade_return = 0.0
        profit_factor = 0.0
        best_trade = 0.0
        worst_trade = 0.0
        avg_holding_period = 0.0
        
    metrics = {
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "Exposure": exposure,
        "Total Trades": n_trades,
        "Win Rate": win_rate,
        "Average Trade Return": avg_trade_return,
        "Profit Factor": profit_factor,
        "Best Trade": best_trade,
        "Worst Trade": worst_trade,
        "Average Holding Period": avg_holding_period,
        "Final Portfolio Value": final_val,
        "Starting Capital": start_val
    }
    return metrics


def calculate_benchmark_metrics(
    df: pd.DataFrame,
    starting_capital: float = 100000.0,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252
) -> tuple[dict, pd.DataFrame]:
    """
    Compute Buy-and-Hold benchmark performance metrics for SPY over the exact same period.
    The benchmark buys at the Open of the first bar with starting_capital,
    holds continuously, and values daily at Adj_Close.
    """
    if df.empty:
        return {}, pd.DataFrame()
        
    # Buy at Open of first day
    entry_price = df["Adj_Open"].iloc[0]
    shares = starting_capital / entry_price
    
    # Daily valuation
    history = pd.DataFrame(index=df.index)
    history["portfolio_value"] = shares * df["Adj_Close"]
    history["daily_return"] = history["portfolio_value"].pct_change().fillna(0.0)
    history["position"] = 1.0
    
    # Running max & drawdown
    running_max = history["portfolio_value"].cummax()
    history["drawdown"] = (history["portfolio_value"] - running_max) / running_max
    
    # Build a single buy-and-hold trade for trade statistics
    exit_price = df["Adj_Close"].iloc[-1]
    pnl = (exit_price - entry_price) * shares
    pnl_pct = (exit_price - entry_price) / entry_price
    trade_log = [{
        "entry_date": df.index[0],
        "exit_date": df.index[-1],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "shares": shares,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "holding_days": len(df)
    }]
    
    metrics = calculate_portfolio_metrics(
        history,
        trade_log,
        starting_capital=starting_capital,
        risk_free_rate=risk_free_rate,
        annualization_factor=annualization_factor
    )
    
    return metrics, history


def format_metrics_table(strategy_metrics: dict, benchmark_metrics: dict, title: str = "Performance Comparison") -> str:
    """
    Format side-by-side performance metrics into a clean Markdown table.
    """
    lines = [
        f"### {title}",
        "| Metric | Strategy | Buy & Hold SPY | Difference / Edge |",
        "| :--- | :---: | :---: | :---: |"
    ]
    
    metric_formats = [
        ("Total Return", "{:.2%}", "{:.2%}", "diff_pct"),
        ("CAGR", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Annualized Volatility", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Sharpe Ratio", "{:.2f}", "{:.2f}", "diff_raw"),
        ("Sortino Ratio", "{:.2f}", "{:.2f}", "diff_raw"),
        ("Max Drawdown", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Calmar Ratio", "{:.2f}", "{:.2f}", "diff_raw"),
        ("Market Exposure", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Total Trades", "{:d}", "{:d}", "diff_int"),
        ("Win Rate", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Average Trade Return", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Profit Factor", "{:.2f}", "{:.2f}", "diff_raw"),
        ("Best Trade", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Worst Trade", "{:.2%}", "{:.2%}", "diff_pct"),
        ("Avg Holding Period (Days)", "{:.1f}", "{:.1f}", "diff_raw"),
        ("Final Portfolio Value", "${:,.2f}", "${:,.2f}", "diff_currency")
    ]
    
    key_mapping = {
        "Market Exposure": "Exposure",
        "Avg Holding Period (Days)": "Average Holding Period"
    }
    
    for label, strat_fmt, bm_fmt, diff_type in metric_formats:
        key = key_mapping.get(label, label)
        strat_v = strategy_metrics.get(key, np.nan)
        bm_v = benchmark_metrics.get(key, np.nan)
        
        # Format strategy
        if pd.isna(strat_v) or np.isinf(strat_v):
            strat_str = "N/A"
        elif "d" in strat_fmt:
            strat_str = strat_fmt.format(int(strat_v))
        else:
            strat_str = strat_fmt.format(strat_v)
            
        # Format benchmark
        if pd.isna(bm_v) or np.isinf(bm_v):
            bm_str = "N/A"
        elif "d" in bm_fmt:
            bm_str = bm_fmt.format(int(bm_v))
        else:
            bm_str = bm_fmt.format(bm_v)
            
        # Calculate diff
        diff_str = "-"
        if not (pd.isna(strat_v) or pd.isna(bm_v) or np.isinf(strat_v) or np.isinf(bm_v)):
            diff = strat_v - bm_v
            if diff_type == "diff_pct":
                diff_str = f"{diff:+.2%}"
            elif diff_type == "diff_raw":
                diff_str = f"{diff:+.2f}"
            elif diff_type == "diff_int":
                diff_str = f"{int(diff):+d}"
            elif diff_type == "diff_currency":
                diff_str = f"${diff:+,.2f}"
                
        lines.append(f"| {label} | {strat_str} | {bm_str} | {diff_str} |")
        
    return "\n".join(lines)
