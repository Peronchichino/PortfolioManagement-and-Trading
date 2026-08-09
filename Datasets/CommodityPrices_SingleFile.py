import os
import yfinance as yf
import pandas as pd

TICKERS = {
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Copper': 'HG=F',
    'Zinc': 'ZNC=F',
    'Aluminum': 'ALI=F',
    'Platinum': 'PL=F',
    'Palladium': 'PA=F',
    'Iron Ore Proxy': 'VALE',
    'Uranium Proxy': 'URA',
    'Zinc': 'ZNC=F',
    'Brent Crude Oil': 'CL=F',
    'WTI Crude Oil': 'BZ=F'
}

CSV_LONG_FILE = "../datasets/commodities_master_long.csv"
CSV_WIDE_FILE = "../datasets/commodities_daily_wide.csv"

def fetch_yfinance_data(period="7d"):
    ticket_symbols = list(TICKERS.values())
    df = yf.download(ticket_symbols, period=period, interval="1d", group_by='ticker')

    records = []
    for name, sym in TICKERS.items():
        if sym in df.columns.levels[0]:
            sub_df = df[sym].dropna(subset=['Close']).copy()
            sub_df.reset_index(inplace=True)

            for _, row in sub_df.iterrows():
                records.append({
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'Commodity': name,
                    'Ticker': sym,
                    'Open': round(float(row['Open']), 4),
                    'High': round(float(row['High']), 4),
                    'Low': round(float(row['Low']), 4),
                    'Close': round(float(row['Close']), 4),
                    'Volume': int(row['Volume']) if pd.notnull(row['Volume']) else 0
                })

            return pd.DataFrame(records)

def sync_csv_dataset():
    if os.path.exists(CSV_LONG_FILE):
        print("Existing CSV file found. Loading data...")
        existing_df = pd.read_csv(CSV_LONG_FILE)
        print(f"Fetching 7-day rolling window for recent updates...")
        new_df = fetch_yfinance_data(period="7d")
    else:
        print("No existing CSV file found. Fetching full dataset...")
        existing_df = pd.DataFrame()
        new_df = fetch_yfinance_data(period="max")

    if new_df.empty and existing_df.empty:
        print("No data fetched. Exiting.")
        return

    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
    combined_df = combined_df.sort_values(by=['Date', 'Commodity'], ascending=[True, True])
    combined_df.to_csv(CSV_LONG_FILE, index=False)
    print(f"Updated CSV file saved as {CSV_LONG_FILE}.")

    wide_df = combined_df.pivot(index='Date', columns='Commodity', values='Close')
    wide_df = wide_df.sort_index(ascending=True)
    wide_df.to_csv(CSV_WIDE_FILE)
    print(f"Wide format CSV file saved as {CSV_WIDE_FILE}.")

if __name__ == "__main__":
    sync_csv_dataset()

