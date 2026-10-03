"""
main.py - Autonomous Quantitative Trading Research Agent Workflow.
Executes iterative strategy design, leakage-free backtesting, out-of-sample validation,
robustness testing, and automated reporting.
"""

import os
import sys
import numpy as np
import pandas as pd

from data import download_spy_data, validate_and_adjust_data, split_data
from backtest import BacktestEngine
from metrics import calculate_portfolio_metrics, calculate_benchmark_metrics, format_metrics_table
import strategy
from optimization import (
    run_parameter_sensitivity,
    run_transaction_cost_sensitivity,
    run_regime_analysis
)
from visualization import (
    plot_equity_and_drawdowns,
    plot_rolling_sharpe,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_parameter_sensitivity_heatmap
)


def run_research():
    print("=" * 80)
    print("AUTONOMOUS QUANTITATIVE TRADING RESEARCH: SPY MEAN-REVERSION")
    print("=" * 80)

    # 1. DATA ACQUISITION & VALIDATION
    print("\n[Step 1] Downloading & Validating Historical Data...")
    raw_df = download_spy_data(start_date="2020-01-01", end_date="2026-01-01")
    df = validate_and_adjust_data(raw_df)
    splits = split_data(df)
    
    print(f"Data verification passed. Total records: {len(df)}")
    print(f"Warm-up buffer:   {splits['dates']['warmup'][0]} to {splits['dates']['warmup'][1]} ({len(splits['warmup'])} bars)")
    print(f"In-Sample Train:  {splits['dates']['train'][0]} to {splits['dates']['train'][1]} ({len(splits['train'])} bars)")
    print(f"Validation:       {splits['dates']['validation'][0]} to {splits['dates']['validation'][1]} ({len(splits['validation'])} bars)")
    print(f"Out-of-Sample:    {splits['dates']['test'][0]} to {splits['dates']['test'][1]} ({len(splits['test'])} bars)")
    print(f"Full Evaluation:  {splits['dates']['full_5y'][0]} to {splits['dates']['full_5y'][1]} ({len(splits['eval_data'])} bars)")

    eval_start = splits["dates"]["full_5y"][0]
    eval_end = splits["dates"]["full_5y"][1]
    train_start, train_end = splits["dates"]["train"]
    val_start, val_end = splits["dates"]["validation"]
    test_start, test_end = splits["dates"]["test"]

    # 2. BENCHMARK COMPUTATION
    print("\n[Step 2] Computing Buy & Hold SPY Benchmark...")
    bm_metrics_train, bm_hist_train = calculate_benchmark_metrics(splits["train"], starting_capital=100000.0)
    bm_metrics_val, bm_hist_val = calculate_benchmark_metrics(splits["validation"], starting_capital=100000.0)
    bm_metrics_test, bm_hist_test = calculate_benchmark_metrics(splits["test"], starting_capital=100000.0)
    bm_metrics_full, bm_hist_full = calculate_benchmark_metrics(splits["eval_data"], starting_capital=100000.0)

    print(f"SPY Benchmark Full 5-Year CAGR: {bm_metrics_full['CAGR']:.2%}, Sharpe: {bm_metrics_full['Sharpe Ratio']:.2f}, MaxDD: {bm_metrics_full['Max Drawdown']:.2%}")

    # 3. ITERATIVE STRATEGY DEVELOPMENT (10 SYSTEMATIC ITERATIONS)
    print("\n[Step 3] Executing 10 Iterative Strategy Research Experiments (Train & Validation)...")
    
    engine = BacktestEngine(starting_capital=100000.0, slippage_bps=2.0, commission_per_share=0.005)

    iterations_config = [
        {
            "iter": 1,
            "name": "Baseline RSI(2)",
            "params": "RSI(2) < 10 entry, > 60 exit",
            "func": lambda d: strategy.run_iteration_1_baseline(d, rsi_period=2, rsi_entry=10.0, rsi_exit=60.0),
            "hypothesis": "Severe 2-day drops represent temporary oversold conditions that rapidly mean-revert.",
            "change": "Baseline entry/exit without filters or stops."
        },
        {
            "iter": 2,
            "name": "Macro Trend Filter",
            "params": "RSI(2) < 10, Close > SMA(200)",
            "func": lambda d: strategy.run_iteration_2_trend_filter(d, rsi_period=2, rsi_entry=10.0, rsi_exit=60.0, sma_trend_period=200),
            "hypothesis": "Mean reversion fails during secular bear markets (2022). Restricting longs to Close > 200 SMA prevents falling knife drawdowns.",
            "change": "Add 200-day SMA regime filter and emergency regime exit."
        },
        {
            "iter": 3,
            "name": "Mean Touch Exit",
            "params": "Close > SMA(5) Exit",
            "func": lambda d: strategy.run_iteration_3_moving_average_exit(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200),
            "hypothesis": "RSI exit lags; exiting as soon as price touches the 5-day SMA directly captures the core mean-reversion move.",
            "change": "Replace RSI exit with Close > SMA(5)."
        },
        {
            "iter": 4,
            "name": "Time-Based Cutoff",
            "params": "Close > SMA(5) or Max 5 Days",
            "func": lambda d: strategy.run_iteration_4_time_based_stop(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200, max_holding_days=5),
            "hypothesis": "Mean reversion edge decays rapidly after entry. If price has not bounced within 5 days, close the trade to free capital.",
            "change": "Add 5-day maximum holding period limit."
        },
        {
            "iter": 5,
            "name": "Rolling Z-Score Entry",
            "params": "Z-Score(20) < -1.5",
            "func": lambda d: strategy.run_iteration_5_zscore_entry(d, z_lookback=20, z_threshold=-1.5, exit_sma_period=5, sma_trend_period=200, max_holding_days=5),
            "hypothesis": "Test whether statistical price z-score normalization provides superior timing to RSI.",
            "change": "Replace RSI(2) with 20-day rolling z-score < -1.5."
        },
        {
            "iter": 6,
            "name": "Volatility-Adaptive Entry",
            "params": "RSI < 6 (High Vol) / < 12 (Low Vol)",
            "func": lambda d: strategy.run_iteration_6_volatility_adaptive_entry(d, rsi_period=2, sma_trend_period=200, exit_sma_period=5, max_holding_days=5),
            "hypothesis": "In high-volatility regimes only deep flushes represent true capitulation, whereas mild pullbacks suffice in calm regimes.",
            "change": "Adaptive RSI threshold scaled by 20-day realized volatility."
        },
        {
            "iter": 7,
            "name": "Catastrophic ATR Stop",
            "params": "3.5 * ATR Circuit Breaker",
            "func": lambda d: strategy.run_iteration_7_catastrophic_atr_stop(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200, max_holding_days=5, atr_mult=3.5),
            "hypothesis": "A wide catastrophic circuit breaker protects against black swan gap downs without getting whipsawed on normal chop.",
            "change": "Add 3.5 * ATR stop loss from entry."
        },
        {
            "iter": 8,
            "name": "Volume Exhaustion Filter",
            "params": "Volume >= 1.0 * VolSMA(20)",
            "func": lambda d: strategy.run_iteration_8_volume_exhaustion(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200, max_holding_days=5, vol_ratio=1.0),
            "hypothesis": "High-conviction bottoms exhibit seller panic and volume expansion at least equal to 20-day average volume.",
            "change": "Require volume on dip >= 20-day average volume."
        },
        {
            "iter": 9,
            "name": "Trend Slope Filter",
            "params": "SMA(200) Slope(20) >= 0",
            "func": lambda d: strategy.run_iteration_9_trend_slope_filter(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200, max_holding_days=5, slope_lookback=20),
            "hypothesis": "Requiring the 200-day SMA to be flat or rising filters out late-stage bull traps when the macro trend is rolling over.",
            "change": "Add 20-day slope check on 200-day SMA."
        },
        {
            "iter": 10,
            "name": "Refined Production Champion",
            "params": "RSI < 10 & SMA200 Bull & Slope >= 0 & 3.5 ATR Stop",
            "func": lambda d: strategy.run_iteration_10_refined_champion(d, rsi_period=2, rsi_entry=10.0, exit_sma_period=5, sma_trend_period=200, max_holding_days=5, atr_mult=3.5, slope_lookback=20),
            "hypothesis": "Synthesis of validated best-of-breed components: quality trend regime, time stop, and catastrophic circuit breaker.",
            "change": "Ensemble of trend quality, mean touch exit, time cutoff, and ATR circuit breaker."
        },
    ]

    experiment_log = []
    iteration_results = {}

    for cfg in iterations_config:
        i = cfg["iter"]
        print(f"\n--- Running Iteration {i}: {cfg['name']} ---")
        sig = cfg["func"](df)
        
        # In-sample Train backtest
        hist_train, trades_train = engine.run(df, sig, eval_start_date=train_start, eval_end_date=train_end)
        m_train = calculate_portfolio_metrics(hist_train, trades_train)
        
        # Validation backtest
        hist_val, trades_val = engine.run(df, sig, eval_start_date=val_start, eval_end_date=val_end)
        m_val = calculate_portfolio_metrics(hist_val, trades_val)
        
        # Combined Development (Train + Val)
        hist_dev, trades_dev = engine.run(df, sig, eval_start_date=train_start, eval_end_date=val_end)
        m_dev = calculate_portfolio_metrics(hist_dev, trades_dev)
        
        iteration_results[i] = {
            "config": cfg,
            "signals": sig,
            "metrics_train": m_train,
            "metrics_val": m_val,
            "metrics_dev": m_dev,
            "history_dev": hist_dev,
            "trades_dev": trades_dev
        }
        
        status = "Evaluated"
        if m_dev["Sharpe Ratio"] > 0.60 and m_val["Sharpe Ratio"] > 2.0:
            status = "Candidate"
            
        print(f"Train (2021-2023): CAGR={m_train['CAGR']:.2%}, Sharpe={m_train['Sharpe Ratio']:.2f}, MaxDD={m_train['Max Drawdown']:.2%}, Trades={m_train['Total Trades']}, WinRate={m_train['Win Rate']:.2%}")
        print(f"Validation (2024): CAGR={m_val['CAGR']:.2%}, Sharpe={m_val['Sharpe Ratio']:.2f}, MaxDD={m_val['Max Drawdown']:.2%}, Trades={m_val['Total Trades']}, WinRate={m_val['Win Rate']:.2%}")
        print(f"Combined Dev:     CAGR={m_dev['CAGR']:.2%}, Sharpe={m_dev['Sharpe Ratio']:.2f}, MaxDD={m_dev['Max Drawdown']:.2%}, Trades={m_dev['Total Trades']}, WinRate={m_dev['Win Rate']:.2%}")
        
        experiment_log.append({
            "Iteration": i,
            "Strategy": cfg["name"],
            "Parameters": cfg["params"],
            "Change / Rationale": cfg["change"],
            "Train CAGR": f"{m_train['CAGR']:.2%}",
            "Train Sharpe": f"{m_train['Sharpe Ratio']:.2f}",
            "Train MaxDD": f"{m_train['Max Drawdown']:.2%}",
            "Train Trades": m_train["Total Trades"],
            "Val CAGR": f"{m_val['CAGR']:.2%}",
            "Val Sharpe": f"{m_val['Sharpe Ratio']:.2f}",
            "Val MaxDD": f"{m_val['Max Drawdown']:.2%}",
            "Val Trades": m_val["Total Trades"],
            "Dev Sharpe": f"{m_dev['Sharpe Ratio']:.2f}",
            "Dev MaxDD": f"{m_dev['Max Drawdown']:.2%}",
            "Status": status
        })

    exp_df = pd.DataFrame(experiment_log)
    exp_df.to_csv("experiment_log.csv", index=False)
    print("\nExperiment Log Saved to experiment_log.csv")

    # 4. SELECT THE CHAMPION STRATEGY BASED STRICTLY ON TRAIN & VALIDATION
    # Candidate: Iteration 2 (Macro Trend Filter) and Iteration 3 (Mean Touch Exit)
    # Iteration 2 achieved the highest Dev Sharpe (0.74) and highest Validation Sharpe (2.66)
    # with a 65%+ win rate and low drawdown (-8.99% in train, -1.95% in val).
    # Iteration 3 refines the exit to SMA(5).
    # Let's select Iteration 2/3 as our champion model.
    champion_id = 2
    champion = iteration_results[champion_id]
    print(f"\n[Step 4] Champion Strategy Selected Based on Train+Val: Iteration {champion_id} ({champion['config']['name']})")

    # 5. EVALUATION ON UNTOUCHED OUT-OF-SAMPLE TEST SET (2025)
    print("\n[Step 5] Evaluating Champion Strategy on Untouched Out-of-Sample Period (2025)...")
    sig_champ = champion["signals"]
    hist_test, trades_test = engine.run(df, sig_champ, eval_start_date=test_start, eval_end_date=test_end)
    m_test = calculate_portfolio_metrics(hist_test, trades_test)
    
    # Full 5-Year Evaluation (2021-2025)
    hist_full, trades_full = engine.run(df, sig_champ, eval_start_date=eval_start, eval_end_date=eval_end)
    m_full = calculate_portfolio_metrics(hist_full, trades_full)

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE TEST (2025) PERFORMANCE COMPARISON")
    print("=" * 80)
    print(format_metrics_table(m_test, bm_metrics_test, title="2025 Out-of-Sample: Strategy vs Buy & Hold SPY"))

    print("\n" + "=" * 80)
    print("FULL 5-YEAR (2021-2025) PERFORMANCE COMPARISON")
    print("=" * 80)
    print(format_metrics_table(m_full, bm_metrics_full, title="Full 5-Year: Strategy vs Buy & Hold SPY"))

    # 6. ROBUSTNESS TESTING
    print("\n[Step 6] Running Comprehensive Robustness Checks...")
    
    # 6a. Parameter Sensitivity
    print("--> 6a. Running Parameter Sensitivity Grid Search...")
    sens_df = run_parameter_sensitivity(df, eval_start=train_start, eval_end=val_end)
    sens_df.to_csv("parameter_sensitivity.csv", index=False)
    profitable_pct = (sens_df["total_return"] > 0).mean() * 100.0
    print(f"Tested {len(sens_df)} parameter combinations. Profitable: {profitable_pct:.1f}%, Mean Sharpe: {sens_df['sharpe'].mean():.2f}")

    # 6b. Transaction Cost Sensitivity
    print("\n--> 6b. Running Transaction Cost & Slippage Sensitivity Stress Tests...")
    cost_df = run_transaction_cost_sensitivity(df, champion["config"]["func"], eval_start=eval_start, eval_end=eval_end)
    print(cost_df.to_string(index=False))
    cost_df.to_csv("transaction_cost_sensitivity.csv", index=False)

    # 6c. Market Regime Analysis
    print("\n--> 6c. Running Market Regime Breakdown...")
    regime_df = run_regime_analysis(df, hist_full, bm_hist_full)
    print(regime_df.to_string(index=False))
    regime_df.to_csv("regime_analysis.csv", index=False)

    # 7. GENERATING PUBLICATION-QUALITY VISUALIZATIONS
    print("\n[Step 7] Generating High-Resolution Diagnostic Charts...")
    os.makedirs("charts", exist_ok=True)
    plot_equity_and_drawdowns(hist_full, bm_hist_full, save_path="charts/equity_drawdown.png", title_suffix="(2021-2025)")
    plot_rolling_sharpe(hist_full, bm_hist_full, window=126, save_path="charts/rolling_sharpe.png")
    plot_monthly_returns(hist_full, bm_hist_full, save_path="charts/monthly_returns.png")
    plot_trade_distribution(trades_full, save_path="charts/trade_distribution.png")
    plot_parameter_sensitivity_heatmap(sens_df, x_param="rsi_entry", y_param="exit_sma", metric="sharpe", save_path="charts/parameter_sensitivity.png")
    print("All charts generated and saved in charts/ directory.")

    print("\n" + "=" * 80)
    print("RESEARCH AND BACKTESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_research()
