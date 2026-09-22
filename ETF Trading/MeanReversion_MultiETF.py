#idea: Mean Reversion: Multi-sector + Multi-Market:
#- same as regular mean reversion SMA5 and SMA200 calc
#- instead of just looking at SPY or VOO, look at all markets and trade mutliple sectors
#- would need comprehensive list of various ETFs
#- max hold period should be variable and not set to specific length

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt

INDEX_ETFS = {
    'SPY_500': 'SPY',
    'QQQ_NASDAQ100': 'QQQ',
    'DIA_DOWJONES': 'DIA',
    'IWM_Russell2000': 'IWM',
    'FTSE_100': 'ISF.L',
    'DAX_Index': '^GDAXI',          # Fixed from DAX.DE (or use 'EXS1.DE' for ETF)
    'CAC_40_Index': '^FCHI',        # Fixed from PX1.PA (or use 'CAC.PA' for ETF)
    'Nikkei_225': '^N225',
    'Hang_Seng': '^HSI',
    'ShanghaiComposite': '000001.SS',
    'MSCI_World': 'URTH',
    'EEM_MSCI_Emerging_Markets': 'EEM',
    'IWF_Russell1000Growth': 'IWF',
    'IWD_Russell1000Value': 'IWD',
    'EFA_Europe': 'IEV',
    'EFA_DevelopedMarkets': 'EFA'
}

METALS_ETFS = {
    'VanEck Vectors Gold Miners ETF': 'GDX',
    'iShares MSCI Global Metals & Mining Producers ETF': 'PICK',
    'State Street Gold and Silver ETF': 'GLTR',
    'State Street Materials ETF': 'XLB',
    'VanEck junior Gold Miners ETF': 'GDXJ',
    'Global X Copper Miners ETF': 'COPX',
    'Northern Trust Morningstar Global Upstream Natural Resources ETF': 'GUNR',
    'Global X Uranium ETF': 'URA',
    'State Street SPDR Metals & Mining ETF': 'XME'
}

INDUSTRIALS_ETFS = {
    'Industrial Select Sector SPDR Fund': 'XLI',
    'First Trust Industrials/Producer Durables AlphaDEX Fund': 'FXR',
    'iShares Global Infrastructure ETF': 'IGF',
    'iShares U.S. Industrials ETF': 'IYJ',
    'Vanguard Industrials ETF': 'VIS'
}

TECH_ETFS = {
    'Vanguard IT ETF': 'VGT',
    'Technology Select Sector SPDR Fund': 'XLK',
    'VanEck Semiconductor ETF': 'SMH',
    'State Street Semiconductor ETF': 'XSD',
    'iShares U.S. Technology ETF': 'IYW',
    'iShares Global Tech ETF': 'IXN',
    'iShares Semiconductor ETF': 'SOXX',
    'Fidelity MSCI Information Technology Index ETF': 'FTEC',
    'State Street Communications ETF': 'XLC',
    'iShares Expanded Tech-Software ETF': 'IGV'
}

CONSUMERS_ETFS = {
    'State Street Consumer Discretionary ETF': 'XLY',
    'State Street Consumer Staples ETF': 'XLP',
    'Vanguard Consumer Staples ETF': 'VDC',
    'Vanguard Consumer Discretionary ETF': 'VCR'
}

HEALTHCARE_ETFS = {
    'Health Care Select Sector SPDR Fund': 'XLV',
    'Vanguard Healthcare ETF': 'VHT',
    'iShares biotechnology ETF': 'IBB',
    'State Street Biotechnology ETF': 'XBI',
    'iShares U.S. Healthcare ETF': 'IYH'
}

FINANCIALS_ETFS = {
    'State Street Financial ETF': 'XLF',
    'Vanguard Financials ETF': 'VFH',
    'Invesco KBW Bank ETF': 'KBWB',
    'SPDR S&P Bank ETF': 'KBE',
    'iShares MSCI Europe Financials ETF': 'EUFN'
}

UTILITY_ETFS = {
    'State Street Utilities ETF': 'XLU',
    'Vanguard Utilities ETF': 'VPU'
}

ENERGY_ETFS = {
    'State Street Energy ETF': 'XLE',
    'iShares U.S. Oil & Gas Exploration & Production ETF': 'IEO',
    'Vanguard Energy ETF': 'VDE',
    'State Street SPDR Oil & Gas Exploration & Production ETF': 'XOP'   
}

#TODO: add more sectors and ETFs to the above dictionaries, then create a function to fetch data for all of them and calculate mean reversion signals based on SMA5 and SMA200.

def calculate_rsi(series: pd.Series, period: int = 2) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI) for a given price series.
    
    :param series: A pandas Series of prices.
    :param period: The number of periods to use for the RSI calculation (default is 2).
    :return: A pandas Series containing the RSI values.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def fetch_yfinance_data(symbol: str, start_date: str = "2010-01-01") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, interval="1d", auto_adjust=True)
        if df.empty or len(df) < 205:
            print(f"Warning: Not enough data for {symbol}. Skipping.")
            return pd.DataFrame()  # Return an empty DataFrame if not enough data
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone information
        df = df.sort_values(by='Date').set_index('Date')
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        return df[required_cols].dropna().copy()  # Ensure we return a copy to avoid SettingWithCopyWarning
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


def run_ranked_portfolio(universe: dict, max_positions: int = 10, start_date: str = "2015-01-01"):
    data = {}
    print(f"Loading and validating data for {len(universe)} tickers...")
    for name, sym in universe.items():
        df = fetch_yfinance_data(sym, start_date=start_date)
        if len(df) > 205:
            # Drop any duplicate index dates
            df = df[~df.index.duplicated(keep='first')].copy()
            df['SMA5'] = df['Close'].rolling(5).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['RSI2'] = calculate_rsi(df['Close'], 2)
            df['ATR14'] = calculate_atr(df, 14)
            data[name] = df.dropna()

    if not data:
        print("No valid ETF data retrieved.")
        return

    #base market calendar on a benchmark (e.g. SPY) or union of major dates
    primary_ticker = 'SPY_500' if 'SPY_500' in data else list(data.keys())[0]
    calendar = data[primary_ticker].index
    print(f"Simulating across {len(calendar)} sessions using {primary_ticker} calendar...")

    # Portfolio state
    cash = 10000.0
    start_cash = cash
    portfolio_history = []
    positions = {}  # ticker: {'units': float, 'entry_price': float, 'stop_price': float, 'entry_date': date}
    trades = []
    slot_size = 1.0 / max_positions

    for idx in range(len(calendar) - 1):
        current_date = calendar[idx]
        next_date = calendar[idx + 1]

        tickers_to_close = []
        for ticker, pos in list(positions.items()):
            df = data[ticker]
            if current_date not in df.index or next_date not in df.index:
                continue

            # Force scalar extraction
            curr_close = float(df.loc[current_date, 'Close'])
            curr_sma5 = float(df.loc[current_date, 'SMA5'])
            curr_rsi2 = float(df.loc[current_date, 'RSI2'])
            next_open = float(df.loc[next_date, 'Open'])

            reversion_exit = (curr_close > curr_sma5) or (curr_rsi2 > 90.0)
            stop_exit = curr_close < pos['stop_price']

            if reversion_exit or stop_exit:
                trade_pnl = (next_open - pos['entry_price']) * pos['units']
                pct_return = (next_open - pos['entry_price']) / pos['entry_price']
                cash += pos['units'] * next_open

                trades.append({
                    'Ticker': ticker,
                    'Entry_Date': pos['entry_date'],
                    'Exit_Date': next_date,
                    'PnL_%': pct_return * 100,
                    'Reason': 'Reversion' if reversion_exit else 'ATR_Stop'
                })
                tickers_to_close.append(ticker)

        for ticker in tickers_to_close:
            del positions[ticker]

        open_slots = max_positions - len(positions)
        if open_slots > 0:
            candidates = []
            for ticker, df in data.items():
                if ticker not in positions:
                    if current_date in df.index and next_date in df.index:
                        c_close = float(df.loc[current_date, 'Close'])
                        c_sma200 = float(df.loc[current_date, 'SMA200'])
                        c_rsi2 = float(df.loc[current_date, 'RSI2'])
                        c_atr14 = float(df.loc[current_date, 'ATR14'])

                        # Buy trigger condition
                        if (c_close > c_sma200) and (c_rsi2 < 10.0):
                            candidates.append((ticker, c_rsi2, c_atr14))

            # Sort safely by scalar RSI float
            candidates.sort(key=lambda x: x[1])

            # Current total equity for position sizing
            current_portfolio_val = cash
            for t, p in positions.items():
                if current_date in data[t].index:
                    current_portfolio_val += float(data[t].loc[current_date, 'Close']) * p['units']

            target_pos_size = current_portfolio_val * slot_size

            # Enter top candidates
            for ticker, rsi, atr in candidates[:open_slots]:
                next_open = float(data[ticker].loc[next_date, 'Open'])
                alloc_capital = min(cash, target_pos_size)

                if alloc_capital > 500:
                    units = alloc_capital / next_open
                    cash -= units * next_open
                    positions[ticker] = {
                        'units': units,
                        'entry_price': next_open,
                        'stop_price': next_open - (3.0 * atr),
                        'entry_date': next_date
                    }

        end_of_day_equity = cash
        for t, p in positions.items():
            if next_date in data[t].index:
                end_of_day_equity += float(data[t].loc[next_date, 'Close']) * p['units']
            else:
                end_of_day_equity += p['entry_price'] * p['units']

        portfolio_history.append({'Date': next_date, 'Equity': end_of_day_equity})

    # Performance Analytics
    equity_df = pd.DataFrame(portfolio_history).set_index('Date')
    trades_df = pd.DataFrame(trades)

    years = (equity_df.index[-1] - equity_df.index[0]).days / 365.25
    cagr = ((equity_df['Equity'].iloc[-1] / start_cash) ** (1 / years) - 1) * 100

    print("\n" + "=" * 55)
    print("RANKED PORTFOLIO BACKTEST SUMMARY")
    print("=" * 55)
    print(f"Total Filtered Trades:    {len(trades_df)}")
    if not trades_df.empty:
        print(f"Win Rate:                 {(trades_df['PnL_%'] > 0).mean() * 100:.2f}%")
        print(f"Average PnL per Trade:    {trades_df['PnL_%'].mean():.2f}%")
        days = (pd.to_datetime(trades_df['Exit_Date']) - pd.to_datetime(trades_df['Entry_Date'])).dt.days
        print(f"Average Holding Days:     {days.mean():.2f} (calendar days)")
    print(f"Compounded CAGR:          {cagr:.2f}%")
    print(f"Final Portfolio Value:    ${equity_df['Equity'].iloc[-1]:,.2f}")
    print(f"Start Cash:               ${start_cash:,.2f}")

    # Plot
    plt.figure(figsize=(13, 6))
    plt.plot(equity_df.index, equity_df['Equity'] / start_cash, label=f"Ranked Portfolio ({max_positions} Slots)", color='#1f77b4', lw=2)
    plt.title(f"Ranked Mean Reversion Equity Curve ({max_positions} Concentrated Slots)", fontsize=13, fontweight='bold')
    plt.ylabel("Portfolio Multiplier ($1 Base)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    combined_universe = {**INDEX_ETFS, **METALS_ETFS, **INDUSTRIALS_ETFS, **TECH_ETFS, **CONSUMERS_ETFS, **HEALTHCARE_ETFS, **FINANCIALS_ETFS, **UTILITY_ETFS, **ENERGY_ETFS}
    #run_multi_sector_test(combined_universe, starte_date="2015-01-01")
    
    run_ranked_portfolio(combined_universe, max_positions=10, start_date="2015-01-01")