import pandas as pd
import time
from vnstock import Quote
from datetime import datetime

# 1. Configuration
# Top 5 Insurance stocks on HOSE
tickers = ['BIC', 'BMI', 'BVH', 'MIG', 'PGI']
start_date = '2014-11-26'
end_date = '2025-11-23'
output_file = 'data/raw/insurance_stocks_weekly_10y.csv'

all_data = [] # List to store dataframes

print(f"Fetching data from {start_date} to {end_date}...\n")

# 2. Loop through each ticker
for ticker in tickers:
    try:
        print(f"Processing {ticker}...")
        
        # Initialize Quote
        quote = Quote(symbol=ticker, source='VCI')
        
        # Get Data
        df = quote.history(start=start_date, end=end_date, interval='1W')
        
        # Basic Validation
        if df is None or df.empty:
            print(f"⚠️ No data found for {ticker}")
            continue

        # --- CLEANING STEPS (From our previous discussion) ---
        
        # A. Ensure datetime
        df['time'] = pd.to_datetime(df['time'])
        
        # B. Filter strict start date (Fixing the VCI 'batch' issue)
        df = df[df['time'] >= pd.to_datetime(start_date)]
        
        # C. Select columns & Create Copy (Fixing SettingWithCopyWarning)
        df_clean = df[['time', 'close', 'volume']].copy()
        
        # D. Add Ticker Name (Crucial for a single CSV file)
        df_clean['ticker'] = ticker
        
        # Append to list
        all_data.append(df_clean)
        
        # Sleep briefly to be nice to the API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {str(e)}")

# 3. Merge and Save
if all_data:
    # Combine all individual dataframes into one big table
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns to look nice: time | ticker | close | volume
    final_df = final_df[['time', 'ticker', 'close', 'volume']]
    
    # Save to CSV
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Success! Saved {len(final_df)} rows to '{output_file}'")
    print(final_df.head())
    print(final_df.tail())
else:
    print("\n❌ No data collected.")