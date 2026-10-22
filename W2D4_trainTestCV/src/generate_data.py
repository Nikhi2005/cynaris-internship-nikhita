
# W2D4: Generate a synthetic dataset

import numpy as np
import pandas as pd

# Generate 100 sample records
np.random.seed(42)

n = 100

# Create three input features
age = np.random.randint(20, 65, n)
income = np.random.exponential(40000, n)
score = np.random.uniform(40, 100, n).round(1)

# Add extreme income values (outliers)
income[0] = 500000
income[1] = 750000
income[2] = 1200000

# Create the target value for prediction
noise = np.random.normal(0, 5, n)
target = 0.3 * age + 0.0005 * income + 2 * score + noise

# Store everything in a DataFrame
df = pd.DataFrame({
    "age": age,
    "income": income,
    "score": score,
    "target": target
})

# Save the dataset
import os
os.makedirs("W2D4_trainTestCV/data", exist_ok=True)

df.to_csv("W2D4_trainTestCV/data/synthetic_data.csv", index=False)

# Display the results
print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nLargest income values:", df["income"].nlargest(3).tolist())

print("\nDataset saved successfully!")
