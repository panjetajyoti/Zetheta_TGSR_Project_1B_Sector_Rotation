import yfinance as yf
import pandas as pd
import numpy as np

print("="*60)
print("PROJECT 1B: TEMPORAL GRAPH SECTOR ROTATION (TGSR)")
print("Step 1: Downloading 10+ Years NSE Sectoral & Macro Data")
print("="*60)

# Key NSE Sectoral Tickers + Benchmark + Macro proxies
sector_tickers = {
    'BANK': '^NSEBANK',
    'IT': '^CNXIT',
    'AUTO': '^CNXAUTO',
    'PHARMA': '^CNXPHARMA',
    'FMCG': '^CNXFMCG',
    'METAL': '^CNXMETAL',
    'REALTY': '^CNXREALTY',
    'ENERGY': '^CNXENERGY',
    'NIFTY50': '^NSEI',
    'INDIAVIX': '^INDIAVIX'
}

start_date = "2012-01-01"
end_date = "2025-12-31"

data_dict = {}

for name, ticker in sector_tickers.items():
    print(f"Fetching {name} ({ticker})...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if 'Close' in df.columns:
        data_dict[name] = df['Close']
    elif 'Adj Close' in df.columns:
        data_dict[name] = df['Adj Close']

master_df = pd.DataFrame(data_dict)
master_df = master_df.ffill().bfill().dropna()

# Compute Daily Log Returns
returns_df = np.log(master_df / master_df.shift(1)).dropna()

# Compute Rolling 21-Day Relative Strength vs Nifty 50
rs_df = pd.DataFrame(index=returns_df.index)
for col in ['BANK', 'IT', 'AUTO', 'PHARMA', 'FMCG', 'METAL', 'REALTY', 'ENERGY']:
    rs_df[f'{col}_RS_21d'] = returns_df[col].rolling(21).mean() - returns_df['NIFTY50'].rolling(21).mean()

rs_df = rs_df.dropna()

# Save Datasets
master_df.to_csv("nse_sector_prices.csv")
returns_df.to_csv("nse_sector_returns.csv")
rs_df.to_csv("nse_sector_relative_strength.csv")

print("\nData Ingestion Completed Successfully!")
print(f"Total Trading Days Processed: {len(master_df)}")
print("Files Saved: 'nse_sector_prices.csv', 'nse_sector_returns.csv', 'nse_sector_relative_strength.csv'")