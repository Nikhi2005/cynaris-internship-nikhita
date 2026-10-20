import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df = pd.DataFrame({
    'age': [20, 25, np.nan, 30, 35],
    'income': [20000, np.nan, 40000, 50000, 60000],
    'score': [60, 70, 80, np.nan, 90]
})

print("Original Dataset:")
print(df)

print("\nMissing Values Before Imputation:")
print(df.isnull().sum())

imputer = KNNImputer(n_neighbors=2)

df[['age', 'income', 'score']] = imputer.fit_transform(
    df[['age', 'income', 'score']]
)

print("\nDataset After KNN Imputation:")
print(df)

print("\nMissing Values After KNN Imputation:")
print(df.isnull().sum())