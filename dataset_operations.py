
import pandas as pd

# Load the original rainfall dataset
df = pd.read_csv("data/rainfall_statewise_daily_imd.csv")

# TASK 1: Filter rows where Daily Actual rainfall is greater than 20
filtered_df = df[df["Daily Actual"] > 20]

print("Filtered records: Daily Actual > 20")
print(filtered_df[["State", "Date", "Daily Actual"]].head(10))
print("\nNumber of matching records:", len(filtered_df))


# TASK 2: Calculate average Daily Actual rainfall for each state
# Exclude the national summary row
state_rows = df[df["State"] != "COUNTRY : INDIA"]

average_rainfall = state_rows.groupby("State")["Daily Actual"].mean()

print("\nAverage daily rainfall by state:")
print(average_rainfall.head(10))


# TASK 3: Merge rainfall data with region information
# Create a small lookup table for three matching states
region_data = {
    "State": [
        "ANDAMAN & NICOBAR (UT)",
        "ANDHRA PRADESH",
        "ARUNACHAL PRADESH"
    ],
    "Region": ["Islands", "South", "Northeast"]
}

region_df = pd.DataFrame(region_data)

# Keep all rainfall records and add Region where a match exists
merged_df = pd.merge(df, region_df, on="State", how="left")

print("\nMerged rainfall data:")
print(merged_df.head())


# TASK 4: Create a Month column from the Date column
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month

print("\nDate, Daily Actual, and Month:")
print(df[["Date", "Daily Actual", "Month"]].head())


# Create a pivot table of average rainfall by state and month
# Exclude the national summary row
state_rows = df[df["State"] != "COUNTRY : INDIA"]

rainfall_pivot = state_rows.pivot_table(
    index="State",
    columns="Month",
    values="Daily Actual",
    aggfunc="mean"
)

print("\nAverage rainfall by state and month:")
print(rainfall_pivot.head(10))
