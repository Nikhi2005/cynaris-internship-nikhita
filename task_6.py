
# Task 6
# NumPy Statistics using the Iris Dataset

import numpy as np
import pandas as pd


df = pd.read_csv("data/iris/iris.data", header=None)

df.columns = ["sepal_length","sepal_width","petal_length","petal_width","species"]

df = df.dropna(how="all")


print("First five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

data = df.iloc[:, 0:4].to_numpy()

mean_values = np.mean(data, axis=0)
print("\nMean:", mean_values)

std_values = np.std(data, axis=0)
print("Standard deviation:", std_values)

correlation = np.corrcoef(data, rowvar=False)
print("Correlation matrix:\n", correlation)
