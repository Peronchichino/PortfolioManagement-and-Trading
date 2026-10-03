"""
optimization.py - Parameter sensitivity, robustness checks, and market regime analysis.
"""

import numpy as np
import pandas as pd
from backtest import BacktestEngine
from metrics import calculate_portfolio_metrics, calculate_benchmark_metrics
import strategy


def run_parameter_sensitivity(
    df: pd.DataFrame,
    eval_start: str = "2021-01-04",
    eval_end: str = "2024-12-31",
    rsi_thresholds: list = [6.0, 8.0, 10.0, 12.0, 15.0],
    exit_smas: list = [3, 5, 7, 10],
    max_holds: list = [3, 4, 5, 6, 7]
) -> pd.DataFrame:
    """
    Evaluate parameter sensitivity across a grid of reasonable values on Train+Validation data.
    """
    engine = BacktestEngine(starting_capital=100000.0, slippage_bps=2.0, commission_per_share=0.005)
    results = []
    
    for r_entry in rsi_thresholds:
        for ex_sma in exit_smas:
            for m_hold in max_holds:
                sig = strategy.run_iteration_4_time_based_stop(
                    df,
                    rsi_period=2,
                    rsi_entry=r_entry,
                    exit_sma_period=ex_sma,
                    sma_trend_period=200,
                    max_holding_days=m_hold
                )
                hist, trades = engine.run(df, sig, eval_start_date=eval_start, eval_end_date=eval_end)
                m = calculate_portfolio_metrics(hist, trades)
                
                if m:
                    results.append({
                        "rsi_entry": r_entry,
                        "exit_sma": ex_sma,
                        "max_holding": m_hold,
                        "total_return": m["Total Return"],
                        "cagr": m["CAGR"],
                        "sharpe": m["Sharpe Ratio"],
                        "max_dd": m["Max Drawdown"],
                        "win_rate": m["Win Rate"],
                        "profit_factor": m["Profit Factor"],
                        "trades": m["Total Trades"]
                    })
                        
    return pd.DataFrame(results)


def run_transaction_cost_sensitivity(
    df: pd.DataFrame,
    signals_func,
    eval_start: str = "2021-01-04",
    eval_end: str = "2025-12-31"
) -> pd.DataFrame:
    """
    Test performance under varying transaction costs and slippage conditions.
    """
    cost_scenarios = [
        {"name": "Zero Cost (Theoretical)", "slippage_bps": 0.0, "commission": 0.0},
        {"name": "Baseline (Realistic SPY: 2 bps, $0.005/sh)", "slippage_bps": 2.0, "commission": 0.005},
        {"name": "Conservative (Moderate Cost: 5 bps, $0.01/sh)", "slippage_bps": 5.0, "commission": 0.01},
        {"name": "Stress Test (High Friction: 10 bps, $0.02/sh)", "slippage_bps": 10.0, "commission": 0.02},
    ]
    
    signals = signals_func(df)
    results = []
    
    for sc in cost_scenarios:
        engine = BacktestEngine(
            starting_capital=100000.0,
            slippage_bps=sc["slippage_bps"],
            commission_per_share=sc["commission"]
        )
        hist, trades = engine.run(df, signals, eval_start_date=eval_start, eval_end_date=eval_end)
        m = calculate_portfolio_metrics(hist, trades)
        
        results.append({
            "Scenario": sc["name"],
            "Slippage (bps)": sc["slippage_bps"],
            "Commission ($/sh)": sc["commission"],
            "CAGR": m.get("CAGR", 0.0),
            "Sharpe Ratio": m.get("Sharpe Ratio", 0.0),
            "Max Drawdown": m.get("Max Drawdown", 0.0),
            "Win Rate": m.get("Win Rate", 0.0),
            "Profit Factor": m.get("Profit Factor", 0.0),
            "Total Trades": m.get("Total Trades", 0),
            "Final Value": m.get("Final Portfolio Value", 0.0)
        })
        
    return pd.DataFrame(results)


def run_regime_analysis(
    df: pd.DataFrame,
    portfolio_history: pd.DataFrame,
    benchmark_history: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze strategy vs benchmark performance across distinct market regimes:
    1. Macro Trend: Bull (Close > 200 SMA) vs Bear (Close <= 200 SMA)
    2. Volatility: High Volatility (20-day realized vol > 18%) vs Low Volatility (<= 18%)
    """
    close = df["Adj_Close"]
    sma200 = close.rolling(200).mean()
    ret = close.pct_change()
    ann_vol_20d = ret.rolling(20).std() * np.sqrt(252)
    
    common_idx = portfolio_history.index.intersection(df.index)
    strat_rets = portfolio_history.loc[common_idx, "daily_return"]
    bm_rets = benchmark_history.loc[common_idx, "daily_return"]
    
    bull_mask = (close.loc[common_idx] > sma200.loc[common_idx]).values
    bear_mask = ~bull_mask
    high_vol_mask = (ann_vol_20d.loc[common_idx] > 0.18).values
    low_vol_mask = ~high_vol_mask
    
    def calc_regime_stats(name: str, mask: np.ndarray):
        s_r = strat_rets.iloc[mask]
        b_r = bm_rets.iloc[mask]
        n_days = len(s_r)
        if n_days == 0:
            return {}
            
        s_cagr = (1.0 + np.mean(s_r)) ** 252 - 1.0
        b_cagr = (1.0 + np.mean(b_r)) ** 252 - 1.0
        
        s_vol = np.std(s_r, ddof=1) * np.sqrt(252) if len(s_r) > 1 else 0.0
        b_vol = np.std(b_r, ddof=1) * np.sqrt(252) if len(b_r) > 1 else 0.0
        
        s_sharpe = (s_cagr / s_vol) if s_vol > 0 else 0.0
        b_sharpe = (b_cagr / b_vol) if b_vol > 0 else 0.0
        
        return {
            "Regime": name,
            "Days": n_days,
            "Strategy Ann Return": s_cagr,
            "SPY Ann Return": b_cagr,
            "Strategy Ann Vol": s_vol,
            "SPY Ann Vol": b_vol,
            "Strategy Sharpe": s_sharpe,
            "SPY Sharpe": b_sharpe,
        }
        
    records = [
        calc_regime_stats("Bull Market (Close > 200 SMA)", bull_mask),
        calc_regime_stats("Bear Market (Close <= 200 SMA)", bear_mask),
        calc_regime_stats("High Volatility (Realized Vol > 18%)", high_vol_mask),
        calc_regime_stats("Low Volatility (Realized Vol <= 18%)", low_vol_mask),
    ]
    return pd.DataFrame(records)
