import os
import pandas as pd

# === Split each ticker into its own cleaned CSV for training ===

# 1. Load raw weekly insurance stocks data
raw_path = "data/raw/insurance_stocks_weekly_10y.csv"
df_raw = pd.read_csv(raw_path)

# 2. Convert time to datetime and pivot to time × ticker matrix
df_raw["time"] = pd.to_datetime(df_raw["time"])
df_wide = (
    df_raw
    .pivot(index="time", columns="ticker", values="close")
    .sort_index()
)

# 3. Prepare output directory
output_dir = "../data/processed"
os.makedirs(output_dir, exist_ok=True)

# 4. Clean and save core tickers (BIC, BMI, BVH, PGI) over full history
core_tickers = ["BIC", "BMI", "BVH", "PGI"]
core_filled = (
    df_wide[core_tickers]
    .interpolate(method="linear")
    .ffill()
    .bfill()
)

for ticker in core_tickers:
    series = core_filled[[ticker]].copy()
    series = series.rename(columns={ticker: "close"})
    out = series.reset_index()  # columns: time, close
    out_path = os.path.join(output_dir, f"{ticker}_weekly_clean.csv")
    out.to_csv(out_path, index=False)
    print(f"Saved {ticker} to: {out_path}")

# 5. Clean and save MIG separately: start from first valid date (active period only)
mig_series = df_wide["MIG"].copy()
first_valid_mig = mig_series.first_valid_index()
print(f"First valid MIG date (raw): {first_valid_mig}")

if first_valid_mig is not None:
    # Keep only active MIG period (drop inactive rows before listing)
    mig_active = mig_series.loc[first_valid_mig:]

    # Interpolate + ffill + bfill within active period only
    mig_active_filled = (
        mig_active
        .interpolate(method="linear")
        .ffill()
        .bfill()
    )

    mig_out = mig_active_filled.reset_index()  # columns: time, MIG
    mig_out = mig_out.rename(columns={"MIG": "close"})

    mig_path = os.path.join(output_dir, "MIG_weekly_clean.csv")
    mig_out.to_csv(mig_path, index=False)
    print(f"Saved MIG to: {mig_path}")
else:
    print("No valid MIG data found in raw dataset.")
