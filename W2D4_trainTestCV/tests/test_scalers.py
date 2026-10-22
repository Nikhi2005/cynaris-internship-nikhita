
# W2D4: Tests for feature scaling and cross-validation

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


# Load the dataset
df = pd.read_csv("data/synthetic_data.csv")

X = df[["age", "income", "score"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


def test_standard_scaler():
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X_train)

    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-6)


def test_minmax_scaler():
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X_train)

    assert np.all(scaled >= 0)
    assert np.all(scaled <= 1)


def test_robust_scaler():
    scaler = RobustScaler()
    scaled = scaler.fit_transform(X_train)

    assert np.allclose(np.median(scaled, axis=0), 0, atol=1e-6)


def test_cross_validation():
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    model = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring="r2"
    )

    assert len(scores) == 5
    assert np.all(np.isfinite(scores))
