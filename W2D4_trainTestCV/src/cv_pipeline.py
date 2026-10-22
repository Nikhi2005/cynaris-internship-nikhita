
# W2D4: Cross-Validation with Feature Scaling

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("data/synthetic_data.csv")

# Separate features and target
X = df[["age", "income", "score"]]
y = df["target"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create five cross-validation folds
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Select the scalers to compare
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

# Evaluate each scaler with Linear Regression
for name, scaler in scalers.items():

    # Combine scaling and regression into one pipeline
    model = make_pipeline(scaler, LinearRegression())

    # Evaluate the model using five-fold cross-validation
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="r2"
    )

    print(f"\n--- {name} ---")
    print("Cross-validation scores:", scores.round(3))
    print("Average R² score:", round(scores.mean(), 3))