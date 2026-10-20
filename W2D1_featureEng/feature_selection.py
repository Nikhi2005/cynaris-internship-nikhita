"""
W2D1: Feature Engineering and Encoding

Purpose:
This program demonstrates feature selection using SelectKBest.

It selects the top five features from a sample student
dataset and displays the selected feature names.
"""

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

data = {
    "Study_hours": [2, 5, 8, 3, 7, 4],
    "Midterm": [40, 70, 90, 50, 85, 60],
    "Assignments": [3, 7, 9, 4, 8, 6],
    "Attendance": [60, 80, 95, 65, 90, 75],
    "Practice_tests": [2, 6, 9, 3, 8, 5],
    "Grade": ["C", "B", "A", "C", "A", "B"]
}

df = pd.DataFrame(data)
print("Student dataset:")
print(df)

X = df.drop("Grade", axis=1)
y = df["Grade"]

print("\nInput features:")
print(X)

print("\nTarget grades:")
print(y)

selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

print("\nSelected features:")
print(selected_features)

with open("W2D1_featureEng/top5_features.md", "w") as file:
    file.write("# Selected Features\n\n")

    for feature in selected_features:
        file.write(f"- {feature}\n")

print("Feature names saved successfully!")