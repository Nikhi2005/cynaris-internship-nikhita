
import pandas as pd

# Load the original rainfall dataset
df = pd.read_csv("data/rainfall_statewise_daily_imd.csv")

# Display the number of rows and columns
print("Shape:", df.shape)

# Display all column names
print("\nColumns:", df.columns.tolist())

# Display the first 10 rows
print("\nFirst 10 rows:")
print(df.head(10))

# Display the data type of each column
print("\nData types:")
print(df.dtypes)

# Count missing values in each column
print("\nMissing values:")
print(df.isnull().sum())

# Display the first 10 rows with missing Daily Category
print("\nRows with missing Daily Category:")
print(df[df["Daily Category"].isna()].head(10))
