import os
import pandas as pd
import yfinance as yf

INDEX_ETFS = {
    'SPY_500': 'SPY',
    'QQQ_NASDAQ100': 'QQQ',
    'DIA_DOWJONES': 'DIA',
    'IWM_Russell2000': 'IWM',
    'FTSE_100': 'ISF.L',
    'DAX': 'DAX.DE',
    'CAC_40': 'PX1.PA',
    'Nikkei_225': '^N225',
    'Hang_Seng': '^HSI',
    'ShanghaiComposite': '000001.SS',
    'MSCI_World': 'URTH',
    'EEM_MSCI_Emerging_Markets': 'EEM',
    'TLT_20YearTreasury': 'TLT',
    'IEF_7-10YearTreasury': 'IEF',
    'IWF_Russell1000Growth': 'IWF',
    'IWD_Russell1000Value': 'IWD',
    'EFA_Europe': 'IEV',
    'EFA_DevelopedMarkets': 'EFA'
}

DATA_DIR = "../datasets/indexes_etfs"

def fetch_etf_history(symbol, period="max"):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1d")

        if df.empty:
            return pd.DataFrame()

        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_cols].copy()

        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df[col].round(4)
        df['Volume'] = df['Volume'].fillna(0).astype(int)

        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def sync_indices():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Syncing index/ETF datasets to {DATA_DIR}...")

    for name, symbol in INDEX_ETFS.items():
        file_path = os.path.join(DATA_DIR, f"{name}.csv")

        if os.path.exists(file_path):
            print(f"File {file_path} found, pulling 7 day rolling update...")
            existing_df = pd.read_csv(file_path)
            new_df = fetch_etf_history(symbol, period="7d")
        else:
            print(f"File {file_path} not found, fetching full dataset...")
            existing_df = pd.DataFrame()
            new_df = fetch_etf_history(symbol, period="max")

        if new_df.empty and existing_df.empty:
            print(f"No data available for {name}. Skipping...")
            continue

        combined_df = pd.concat([existing_df, new_df], ignore_index=True)

        combined_df['Date'] = combined_df['Date'].astype(str)
        combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')

        combined_df = combined_df.sort_values(by='Date', ascending=True)

        combined_df.to_csv(file_path, index=False)
        print(f"Data for {name} synced successfully to {file_path}.")

if __name__ == "__main__":
    sync_indices()
