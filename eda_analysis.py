import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Task 1:

df = pd.read_csv('D:/Cynaris/W1D4_eda/rainfall_statewise_daily_imd.csv')
print("First 5 rows are :",df.head())
print("Dataset shape :",df.shape)
print("Statistical Summary :",df.describe())
print("Data information :",df.info())
print("Missing values :",df.isnull().sum())
print("Duplicate Rows :",df.duplicated().sum())

#Task 2 :

numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns].hist(
    figsize=(15, 12),
    bins=20,
    edgecolor="black"
)
plt.suptitle("Distribution of Numerical Columns", fontsize=16)
plt.tight_layout()
plt.savefig("numeric_distributions.png", dpi=150)
plt.show()


correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap of Rainfall Variables")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150)
plt.show()


top_states = df["State"].value_counts().head(10)
plt.figure(figsize=(10, 6))
sns.barplot(
    x=top_states.values,
    y=top_states.index,
    color="steelblue"
)
plt.title("Top 10 States by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("State")
plt.tight_layout()
plt.savefig("top10_state_counts.png", dpi=150)
plt.show()
