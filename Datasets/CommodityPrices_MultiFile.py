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


DATA_DIR = "../datasets"

def fetch_single_commodity(symbol, period="7d"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()

        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_cols]

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].round(4)
        df['Volume'] = df['Volume'].fillna(0).astype(int)

        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


def sync_csv_dataset():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing commodity prices dataset to {DATA_DIR}...")

    for name, symbol in TICKERS.items():
        file_path = os.path.join(DATA_DIR, f"{name}.csv")

        if os.path.exists(file_path):
            print(f"File {file_path} found, pulling 7 day rolling update...")
            existing_df = pd.read_csv(file_path)
            new_df = fetch_single_commodity(symbol, period="7d")
        else:
            print(f"File {file_path} not found, fetching full dataset...")
            existing_df = pd.DataFrame()
            new_df = fetch_single_commodity(symbol, period="max")
        if new_df.empty and existing_df.empty:
            print(f"No data available for {name}. Skipping...")
            continue

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        combined_df['Date'] = combined_df['Date'].astype(str)
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')

        combined_df = combined_df.sort_values(by='Date', ascending=True)

        combined_df.to_csv(file_path, index=False)
        print(f"Data for {name} synced successfully to {file_path}.")

    print("All commodity prices datasets synced successfully.")


if __name__ == "__main__":
    sync_csv_dataset()

