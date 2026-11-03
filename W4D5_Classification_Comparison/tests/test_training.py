
from pathlib import Path

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"


def test_dataset_split():
    dataset = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=42,
        stratify=dataset.target,
    )

    assert len(X_train) == 455
    assert len(X_test) == 114
    assert len(y_train) == 455
    assert len(y_test) == 114


def test_models_can_train_and_predict():
    dataset = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.20,
        random_state=42,
        stratify=dataset.target,
    )

    models = [
        Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=5000)),
        ]),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    ]

    for model in models:
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)
        assert set(predictions).issubset({0, 1})


def test_saved_models_exist_and_load():
    model_files = [
        MODEL_DIR / "logistic_regression.joblib",
        MODEL_DIR / "random_forest.joblib",
    ]

    for model_file in model_files:
        assert model_file.exists()

        loaded_model = joblib.load(model_file)

        assert hasattr(loaded_model, "predict")
