
# W2D4: Train/Test Split and Feature Scaling

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Load the dataset
df = pd.read_csv("data/synthetic_data.csv")

# Separate input features and target
X = df[["age", "income", "score"]]
y = df["target"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# Create the three scalers
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

# Fit each scaler on training data and transform both sets
for name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n--- {name} ---")
    print("Training data (first 3 rows):")
    print(pd.DataFrame(
        X_train_scaled, columns=X.columns
    ).head(3).round(2))

    print("Testing data (first 3 rows):")
    print(pd.DataFrame(
        X_test_scaled, columns=X.columns
    ).head(3).round(2))
