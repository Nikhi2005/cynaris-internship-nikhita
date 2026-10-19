"""
W2D1: Feature Engineering and Encoding

Purpose:
This program demonstrates three encoding techniques:
1. LabelEncoder
2. OneHotEncoder
3. OrdinalEncoder

It applies these techniques to a sample student dataset
and saves the encoded data into CSV files.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

data = {
    "Name": ["Asha", "Ravi", "Priya", "Kiran"],
    "Grade": ["A", "B", "A", "C"]
}

df = pd.DataFrame(data)
print(df)

encoder = LabelEncoder()
df["Grade_encoded"] = encoder.fit_transform(df["Grade"])
print(df)

encoder2 = OneHotEncoder(sparse_output=False)
grade_encoded = encoder2.fit_transform(df[["Grade"]])
print(grade_encoded)

encoder3 = OrdinalEncoder(categories=[["A", "B", "C"]])
df["Grade_ordinal"] = encoder3.fit_transform(df[["Grade"]])
print(df)

encoder2 = OneHotEncoder(sparse_output=False)
grade_encoded = encoder2.fit_transform(df[["Grade"]])
df_onehot = pd.DataFrame(
    grade_encoded,
    columns=encoder2.get_feature_names_out(["Grade"])
)
print(df_onehot)

df.to_csv("W2D1_featureEng/df_label.csv", index=False)
df_onehot.to_csv("W2D1_featureEng/df_onehot.csv", index=False)
df.to_csv("W2D1_featureEng/df_ordinal.csv", index=False)
print("CSV files saved successfully!")