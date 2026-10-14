#Task 3:Cleaning        
                      
import pandas as pd

file_path = "data/rainfall_statewise_daily_imd.csv"
df = pd.read_csv(file_path)

df.columns = (
    df.columns
    .str.replace("\n", " ", regex=False)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

df = df.rename(columns={
    "Cumulative Departue Per": "Cumulative Departure Per",
    "Monthly Acutual": "Monthly Actual"
})

text_columns = df.select_dtypes(include=["object", "str"]).columns

for column in text_columns:
    df[column] = df[column].str.strip()

df["Date"] = pd.to_datetime(
    df["Date"],
    format="%Y-%m-%d",
    errors="coerce"
)

percentage_columns = [
    "Daily Departure Per",
    "Weekly Departure Per",
    "Cumulative Departure Per",
    "Monthly Departure Per"
]

for column in percentage_columns:
    df[column] = pd.to_numeric(
        df[column].str.replace("%", "", regex=False),
        errors="coerce"
    )

duplicate_count = df.duplicated().sum()

print("Duplicate rows:", duplicate_count)
print("Dataset shape after cleaning:", df.shape)
print("Missing values after cleaning:")
print(df.isna().sum())

df.to_csv("data/rainfall_cleaned.csv", index=False)
print("\nCleaned dataset saved to data/rainfall_cleaned.csv")
