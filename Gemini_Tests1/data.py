"""
data.py - Data acquisition, validation, adjustment, and splitting for SPY Mean Reversion Research.
"""

import os
import yfinance as yf
import pandas as pd
import numpy as np


def download_spy_data(start_date: str = "2020-01-01", end_date: str = "2026-01-01") -> pd.DataFrame:
    """
    Download SPY daily data from yfinance.
    We download starting in 2020 to provide a 1-year warm-up period for rolling indicators
    (e.g., 200-day moving average) so that 2021-01-01 to 2025-12-31 has complete indicator values.
    """
    ticker = yf.Ticker("SPY")
    # Download raw unadjusted and adjusted historical bars
    df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
    
    if df.empty:
        raise ValueError(f"No data returned for SPY between {start_date} and {end_date}")
    
    # Clean index
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    
    # Required columns check
    req_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in req_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
            
    # Keep only required columns
    df = df[req_cols].copy()
    
    return df


def validate_and_adjust_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data cleanliness and compute adjusted prices.
    
    Adjusted price formulation:
    adj_factor = Adj Close / Close
    Adj Open = Open * adj_factor
    Adj High = High * adj_factor
    Adj Low = Low * adj_factor
    
    This ensures that OHLC bar spreads, high-low ranges, and overnight gaps
    are fully consistent with dividend distributions and splits without artificial jumps.
    """
    # 1. Check chronological ordering
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        
    # 2. Check and remove duplicate dates
    duplicates = df.index.duplicated(keep="first")
    if duplicates.any():
        print(f"Warning: Found {duplicates.sum()} duplicate dates. Removing duplicates.")
        df = df[~duplicates]
        
    # 3. Check for missing values
    null_counts = df.isnull().sum()
    if null_counts.any():
        print("Warning: Missing values detected:")
        print(null_counts[null_counts > 0])
        # Forward fill then backward fill if any
        df = df.ffill().bfill()
        
    # 4. Check for invalid prices (non-positive)
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if (df[col] <= 0).any():
            raise ValueError(f"Non-positive values found in column {col}")
            
    # 5. Compute Adjusted OHLC
    adj_factor = df["Adj Close"] / df["Close"]
    df["Adj_Open"] = df["Open"] * adj_factor
    df["Adj_High"] = df["High"] * adj_factor
    df["Adj_Low"] = df["Low"] * adj_factor
    df["Adj_Close"] = df["Adj Close"]  # Alias
    
    # Ensure High is >= Low, Open, Close
    df["Adj_High"] = df[["Adj_High", "Adj_Open", "Adj_Close"]].max(axis=1)
    df["Adj_Low"] = df[["Adj_Low", "Adj_Open", "Adj_Close"]].min(axis=1)
    
    return df


def split_data(
    df: pd.DataFrame,
    train_end: str = "2023-12-31",
    val_end: str = "2024-12-31",
    test_end: str = "2025-12-31",
    backtest_start: str = "2021-01-01"
) -> dict:
    """
    Split data into In-Sample (Train), Validation, and Out-of-Sample (Test) periods.
    The backtest strictly begins on backtest_start (2021-01-01), with data prior to that
    used exclusively as an indicator warm-up buffer (e.g. 200-day SMA).
    
    Periods:
    - Warm-up: 2020-01-02 to 2020-12-31 (Indicator initialization)
    - Train (In-Sample): 2021-01-01 to 2023-12-31 (3 complete years: 2021 bull, 2022 bear, 2023 recovery)
    - Validation: 2024-01-01 to 2024-12-31 (1 complete year: 2024 bull trend)
    - Test (Out-of-Sample): 2025-01-01 to 2025-12-31 (1 complete year: strictly held out for final test)
    """
    df_eval = df.loc[df.index >= backtest_start].copy()
    
    train_mask = (df.index >= backtest_start) & (df.index <= train_end)
    val_mask = (df.index > train_end) & (df.index <= val_end)
    test_mask = (df.index > val_end) & (df.index <= test_end)
    
    splits = {
        "full_data": df,
        "eval_data": df_eval.loc[df_eval.index <= test_end],
        "train": df.loc[train_mask],
        "validation": df.loc[val_mask],
        "test": df.loc[test_mask],
        "warmup": df.loc[df.index < backtest_start],
        "dates": {
            "warmup": (df.index[0].strftime("%Y-%m-%d"), df.loc[df.index < backtest_start].index[-1].strftime("%Y-%m-%d")),
            "train": (df.loc[train_mask].index[0].strftime("%Y-%m-%d"), df.loc[train_mask].index[-1].strftime("%Y-%m-%d")),
            "validation": (df.loc[val_mask].index[0].strftime("%Y-%m-%d"), df.loc[val_mask].index[-1].strftime("%Y-%m-%d")),
            "test": (df.loc[test_mask].index[0].strftime("%Y-%m-%d"), df.loc[test_mask].index[-1].strftime("%Y-%m-%d")),
            "full_5y": (df_eval.loc[df_eval.index <= test_end].index[0].strftime("%Y-%m-%d"), df_eval.loc[df_eval.index <= test_end].index[-1].strftime("%Y-%m-%d"))
        }
    }
    return splits


if __name__ == "__main__":
    print("Testing data downloading and validation pipeline...")
    raw_df = download_spy_data("2020-01-01", "2026-01-01")
    clean_df = validate_and_adjust_data(raw_df)
    splits = split_data(clean_df)
    
    print("\nData Validation Successful!")
    print(f"Total rows downloaded: {len(clean_df)}")
    print(f"Warm-up period: {splits['dates']['warmup']} ({len(splits['warmup'])} bars)")
    print(f"Train period (In-Sample): {splits['dates']['train']} ({len(splits['train'])} bars)")
    print(f"Validation period: {splits['dates']['validation']} ({len(splits['validation'])} bars)")
    print(f"Test period (Out-of-Sample): {splits['dates']['test']} ({len(splits['test'])} bars)")
    print(f"Full 5-Year Evaluation: {splits['dates']['full_5y']} ({len(splits['eval_data'])} bars)")
