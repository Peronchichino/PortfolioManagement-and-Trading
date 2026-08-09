import os
import yfinance as yf
import pandas as pd
import glob
import matplotlib.pyplot as plt

#taken from localized commodity price datasets

DATA_DIR = "../datasets"

def load_all_commodities():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in '{DATA_DIR}'. Make sure your sync script has run.")
        return None

    df_dict = {}
    for file in csv_files:
        commodity_name = os.path.basename(file).replace(".csv", "")
        
        df = pd.read_csv(file)
        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df_dict[commodity_name] = df.set_index('Date')['Close']

    combined_df = pd.DataFrame(df_dict).sort_index()
    return combined_df


def plot_charts():
    combined_df = load_all_commodities()
    if combined_df is None or combined_df.empty:
        return

    # ------------------------------------------------------------------
    # CHART 1: Grid of Subplots (Actual Nominal Prices)
    # ------------------------------------------------------------------
    num_commodities = len(combined_df.columns)
    cols = 3
    rows = (num_commodities + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows), sharex=False)
    axes = axes.flatten()

    for i, col_name in enumerate(combined_df.columns):
        ax = axes[i]
        series = combined_df[col_name].dropna()
        
        ax.plot(series.index, series.values, color='#1f77b4', linewidth=1.2)
        ax.set_title(f"{col_name} (Price)", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='x', rotation=30)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Commodity Price Histories (Individual Scales)", fontsize=16, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # CHART 2: Normalized Growth / Indexed Return (Base = 100)
    # ------------------------------------------------------------------
    recent_df = combined_df.loc['2015-01-01':].ffill().dropna()
    
    if not recent_df.empty:
        normalized_df = (recent_df / recent_df.iloc[0]) * 100

        plt.figure(figsize=(14, 7))
        for col in normalized_df.columns:
            plt.plot(normalized_df.index, normalized_df[col], label=col, linewidth=1.5)

        plt.title("Relative Performance Comparison (Rebased to 100 in 2015)", fontsize=14, fontweight='bold')
        plt.xlabel("Date")
        plt.ylabel("Indexed Price (Base = 100)")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    plot_charts()
