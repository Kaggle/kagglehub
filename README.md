import pandas as pd
import numpy as np

# ========== USER CONFIG ===========
# Input CSV filepath (5-min OHLCV) with columns: DateTime, Open, High, Low, Close
input_csv = "XAUUSD_5min.csv"
# Output Excel file
output_excel = "XAUUSD_breakout_analysis.xlsx"

# Timezone / time shifts if needed
# We assume DateTime is in IST or convert to IST if in UTC

# Constants
london_start = "12:30"  # IST
london_end = "21:30"    # IST
check_start = "20:00"
check_end = "22:00"
min_rrr = 2.0  # minimum reward/risk ratio

# ===================================

# Load data
df = pd.read_csv(input_csv, parse_dates=["DateTime"])
# If timezone conversion needed, apply here

# Create additional columns for date, time
df["Date"] = df["DateTime"].dt.date
df["Time"] = df["DateTime"].dt.time.astype(str)

# Function to get London session high & low for each date
london = df.set_index("DateTime").between_time(london_start, london_end).copy()
# Group by date
session = london.groupby(london.index.date).agg({"High":"max", "Low":"min"}).rename_axis("Date").reset_index()

# Merge session high/low back to df
df = df.merge(session, left_on="Date", right_on="Date", how="left", suffixes=("","_London"))

# Filter rows in check window (next day 20:00-22:00)
check = df.set_index("DateTime").between_time(check_start, check_end).reset_index()

results = []
monthly_counts = {}

for idx, row in check.iterrows():
    date = row["Date"]
    dt = row["DateTime"]
    price = row["Close"]
    high_l = row["High_London"]
    low_l = row["Low_London"]
    # if no session high/low, skip
    if pd.isna(high_l) or pd.isna(low_l):
        continue

    # Determine breakout or breakdown
    trade_type = None
    # risk = difference from session boundary to opposite side? (we approximate with symmetric risk)
    if price > high_l:
        # potential Buy — risk = price − high_l, target = risk * min_rrr
        # We need to check if price moves enough before reverse — for simplicity skip deeper check
        trade_type = "Buy"
    elif price < low_l:
        trade_type = "Sell"
    else:
        continue

    # Check trap logic: if immediate reversal within next few bars — we skip if trap
    # Define trap window, e.g. next 3 bars (15 min)
    future = df[df["DateTime"] > dt].head(3)
    reversed = False
    for _, fr in future.iterrows():
        if trade_type == "Buy" and fr["Low"] < high_l:
            reversed = True
        if trade_type == "Sell" and fr["High"] > low_l:
            reversed = True
    if reversed:
        # count as trap
        results.append({
            "Date": date,
            "Time": dt.time().strftime("%H:%M"),
            "Trade Type": "Trap",
            "Monthly Trades": None,
        })
        # increment trap monthly count
        m = dt.month
        monthly_counts.setdefault((dt.year, m), {"trades":0, "traps":0})
        monthly_counts[(dt.year, m)]["traps"] += 1
        continue

    # Valid trade
    results.append({
        "Date": date,
        "Time": dt.time().strftime("%H:%M"),
        "Trade Type": trade_type,
        "Monthly Trades": None,
    })
    m = dt.month
    monthly_counts.setdefault((dt.year, m), {"trades":0, "traps":0})
    monthly_counts[(dt.year, m)]["trades"] += 1

# Now fill monthly trades into results
for res in results:
    d = res["Date"]
    # find month-year
    # Note: res["Time"] is string, so need to find dt in original df to get year
    # Simplify: match by date in original df
    # This part can be improved
    mon = pd.to_datetime(d).month
    yr = pd.to_datetime(d).year
    res["Monthly Trades"] = monthly_counts.get((yr, mon), {}).get("trades", 0)

# Convert to DataFrame
res_df = pd.DataFrame(results, columns=["Date","Time","Trade Type","Monthly Trades"])

# Save to Excel
res_df.to_excel(output_excel, index=False)
print("Generated:", output_excel)
