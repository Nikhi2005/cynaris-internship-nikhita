"""
W2D1: Feature Engineering and Encoding

Purpose:
This program demonstrates three scaling techniques:
1. StandardScaler
2. MinMaxScaler
3. RobustScaler

It scales numerical data and saves plots to compare
the original and scaled data.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt



marks = pd.DataFrame({
    "Study_hours": [2, 5, 8],
    "Exam_marks": [40, 70, 90]
})
print("Original data:")
print(marks)

scaler = StandardScaler()
scaled_marks = scaler.fit_transform(marks)
print("Scaled data:")
print(scaled_marks)

scaler2 = MinMaxScaler()
scaled_marks2 = scaler2.fit_transform(marks)
print("MinMax Scaled data:")
print(scaled_marks2)

scaler3 = RobustScaler()
scaled_marks3 = scaler3.fit_transform(marks)
print("Robust Scaled data:")
print(scaled_marks3)


plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.hist(marks["Exam_marks"])
plt.title("Original Marks")

plt.subplot(1, 2, 2)
plt.hist(scaled_marks[:, 1])
plt.title("Standard Scaled Marks")

plt.hist(scaled_marks2[:, 1])
plt.title("MinMax Scaled Marks")

plt.hist(scaled_marks3[:, 1])
plt.title("Robust Scaled Marks")

plt.tight_layout()
plt.savefig(
    r"D:\Cynaris\W2D1_featureEng\scaler_std.png"
    r"D:\Cynaris\W2D1_featureEng\scaler_minmax.png"
    r"D:\Cynaris\W2D1_featureEng\scaler_robust.png"
)
plt.show()