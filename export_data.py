
import pandas as pd
import os

# Load the cleaned rainfall dataset
df = pd.read_csv("data/rainfall_cleaned.csv")

# Export the data to CSV
df.to_csv("data/rainfall_export.csv", index=False)
print("Cleaned data exported to CSV successfully!")

# Export the same data to Parquet
df.to_parquet("data/rainfall_export.parquet", index=False)
print("Cleaned data exported to Parquet successfully!")

# Get the size of each file in bytes
csv_size = os.path.getsize("data/rainfall_export.csv")
parquet_size = os.path.getsize("data/rainfall_export.parquet")

# Display the file sizes
print("\nFile Size Comparison:")
print("CSV size:", csv_size, "Bytes")
print("Parquet size:", parquet_size, "Bytes")

# Identify the smaller file
if csv_size < parquet_size:
    print("CSV is smaller.")
elif parquet_size < csv_size:
    print("Parquet is smaller.")
else:
    print("Both files have the same size.")
