
from pathlib import Path
import sys

import pytest
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Allow the test file to import our main program.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from wine_hyperparameter_tuning import make_pipeline


# Load a small, ready-made dataset for testing.
wine = load_wine()
X = wine.data
y = wine.target


def test_pipeline_can_fit_and_predict():
    """Check that the pipeline can train and make predictions."""
    model = make_pipeline(KNeighborsClassifier(n_neighbors=5))

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset(set(y))


def test_all_seven_knn_values_are_valid():
    """Check that all seven planned K values can be used."""
    k_values = [1, 3, 5, 7, 11, 15, 21]

    for k in k_values:
        model = make_pipeline(KNeighborsClassifier(n_neighbors=k))
        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == len(y)


def test_gridsearch_finds_best_parameters():
    """Check that GridSearchCV can find Random Forest settings."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(random_state=42)),
    ])

    parameter_grid = {
        "clf__n_estimators": [50, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42,
        ),
        scoring="accuracy",
        n_jobs=-1,
    )

    search.fit(X, y)

    assert search.best_params_ is not None
    assert "clf__n_estimators" in search.best_params_
    assert "clf__max_depth" in search.best_params_
    assert "clf__min_samples_split" in search.best_params_
