import numpy as np
import pandas as pd
import yfinance as yf

# Assets frequently traded by Tom Hougaard (or proxy ETFs)
DEFAULT_ASSETS = {
    'DAX_Index': '^GDAXI',
    'FTSE_100': '^FTSE',
    'Dow_Jones': '^DJI',
    'S&P_500': '^GSPC',
    'NASDAQ_100': '^NDX',
    'SPY_ETF': 'SPY',
    'QQQ_ETF': 'QQQ'
}


def fetch_clean_data(symbol: str, start_date: str = "2000-01-01") -> pd.DataFrame:
    """Fetch and align daily OHLC bars."""
    df = yf.download(symbol, start=start_date, progress=False)
    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns if present (common in newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values('Date').set_index('Date')
    df['DayOfWeek'] = df.index.day_name()
    return df[['Open', 'High', 'Low', 'Close', 'DayOfWeek']].dropna()


def backtest_hougaard_rules(df: pd.DataFrame, asset_name: str, touch_threshold_pct: float = 0.0):
    """
    Tests both hypotheses:
    Rule A: Wednesday High < Monday High -> Thursday visits Wednesday Low
    Rule B: Friday High < Thursday High -> Next Session (Monday) visits Friday Low
    
    A 'visit' occurs if Target_Day_Low <= Reference_Low * (1 + touch_threshold_pct / 100).
    Default threshold is 0.0% (must hit or breach the exact low).
    """
    if len(df) < 50:
        return

    # Add forward-shifted and backward-shifted lookup columns
    # We group by weekly cycles or walk row by row
    records_wed_thu = []
    records_fri_next = []

    # Map each week to its calendar days
    df['Year'] = df.index.year
    df['Week'] = df.index.isocalendar().week

    # Iterate week by week to maintain strict calendar integrity
    for (year, week), week_df in df.groupby(['Year', 'Week']):
        day_map = {row['DayOfWeek']: row for _, row in week_df.iterrows()}

        # -------------------------------------------------------------
        # Hypothesis 2: Wed High < Mon High -> Low of Wed visited on Thu
        # -------------------------------------------------------------
        if 'Monday' in day_map and 'Wednesday' in day_map and 'Thursday' in day_map:
            mon = day_map['Monday']
            wed = day_map['Wednesday']
            thu = day_map['Thursday']

            condition_met = wed['High'] < mon['High']
            if condition_met:
                ref_target_low = wed['Low']
                # Target visited if Thursday touches or breaches Wednesday's low
                hit = thu['Low'] <= ref_target_low * (1 + touch_threshold_pct / 100)
                slippage_or_distance = ((thu['Low'] - ref_target_low) / ref_target_low) * 100

                records_wed_thu.append({
                    'Date': thu.name.strftime('%Y-%m-%d'),
                    'Asset': asset_name,
                    'Cond_High_Diff_Pct': ((wed['High'] - mon['High']) / mon['High']) * 100,
                    'Hit': hit,
                    'Target_Low': ref_target_low,
                    'Actual_Low': thu['Low'],
                    'Distance_Pct': slippage_or_distance
                })

    # -------------------------------------------------------------
    # Hypothesis 1: Fri High < Thu High -> Fri Low visited next trading day
    # -------------------------------------------------------------
    for i in range(len(df) - 1):
        curr_bar = df.iloc[i]
        next_bar = df.iloc[i + 1]

        # Ensure current bar is Friday and previous bar was Thursday
        if curr_bar['DayOfWeek'] == 'Friday' and i > 0:
            prev_bar = df.iloc[i - 1]
            if prev_bar['DayOfWeek'] == 'Thursday':
                condition_met = curr_bar['High'] < prev_bar['High']
                if condition_met:
                    ref_target_low = curr_bar['Low']
                    hit = next_bar['Low'] <= ref_target_low * (1 + touch_threshold_pct / 100)
                    slippage_or_distance = ((next_bar['Low'] - ref_target_low) / ref_target_low) * 100

                    records_fri_next.append({
                        'Date': next_bar.name.strftime('%Y-%m-%d'),
                        'Next_Day': next_bar['DayOfWeek'],
                        'Asset': asset_name,
                        'Hit': hit,
                        'Target_Low': ref_target_low,
                        'Actual_Low': next_bar['Low'],
                        'Distance_Pct': slippage_or_distance
                    })

    df_wed_thu = pd.DataFrame(records_wed_thu)
    df_fri_next = pd.DataFrame(records_fri_next)

    return df_wed_thu, df_fri_next


def print_rule_summary(title: str, results_df: pd.DataFrame):
    print(f"\n--- {title} ---")
    if results_df.empty:
        print("No triggering instances found.")
        return

    total = len(results_df)
    hits = results_df['Hit'].sum()
    hit_rate = (hits / total) * 100

    print(f"Total Occurrences:       {total}")
    print(f"Successful Targets:      {hits}")
    print(f"Hit Rate (Accuracy):     {hit_rate:.2f}%")

    # Sample streaks of 25 (to evaluate the 24/25 claim)
    hits_series = results_df['Hit'].astype(int)
    rolling_25 = hits_series.rolling(25).sum()
    max_in_25 = rolling_25.max()
    print(f"Max Hits in any 25-run:  {int(max_in_25) if not np.isnan(max_in_25) else 'N/A'} / 25")


def run_full_suite():
    print("=" * 70)
    print("EMPIRICAL TEST: TOM HOUGAARD SITUATIONAL ANALYSIS RULES")
    print("Period: 2000 to Present")
    print("=" * 70)

    for name, sym in DEFAULT_ASSETS.items():
        print(f"\n==================== {name} ({sym}) ====================")
        df = fetch_clean_data(sym)
        if df.empty:
            print("Failed to download data.")
            continue

        res_wed_thu, res_fri_next = backtest_hougaard_rules(df, name)

        print_rule_summary(
            "Rule 1: If Wed High < Mon High -> Thu visits Wed Low",
            res_wed_thu
        )
        print_rule_summary(
            "Rule 2: If Fri High < Thu High -> Next Session visits Fri Low",
            res_fri_next
        )


if __name__ == "__main__":
    run_full_suite()