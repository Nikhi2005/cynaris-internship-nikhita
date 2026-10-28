
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save plots without opening a window.
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier


# Keep generated files inside this day's outputs folder.
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def evaluate_model(name, model, X_train, X_test, y_train, y_test, X, y):
    """Train a model and return its evaluation metrics."""
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    cv_accuracy = cross_val_score(model, X, y, cv=5).mean()

    results = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "cv_accuracy": float(cv_accuracy),
    }

    print(
        f"{name:28s} "
        f"Train={train_accuracy:.4f}  "
        f"Test={test_accuracy:.4f}  "
        f"CV-5={cv_accuracy:.4f}"
    )
    return results


def main():
    # 1. Generate the same type of dataset as the supplied playground.
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=RANDOM_STATE,
    )

    # 2. Split data for model training and final testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Configure local MLflow experiment tracking.
    mlflow.set_experiment("Cynaris-W3D4-Tree-Ensemble")

    # Run cross-validation and record metrics for each experiment.
    experiments = [
        (
            "Decision Tree baseline",
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {},
        ),
        (
            "Random Forest baseline",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                oob_score=True,
                n_jobs=-1,
            ),
            {"n_estimators": 100, "oob_score": True},
        ),
        (
            "Random Forest 500 trees",
            RandomForestClassifier(
                n_estimators=500,
                random_state=RANDOM_STATE,
                oob_score=True,
                n_jobs=-1,
            ),
            {"n_estimators": 500, "oob_score": True},
        ),
        (
            "Random Forest depth 5",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=RANDOM_STATE,
                oob_score=True,
                n_jobs=-1,
            ),
            {"n_estimators": 100, "max_depth": 5, "oob_score": True},
        ),
    ]

    fitted_forest = None
    forest_results = None

    print("\n=== W3D4: Decision Tree vs Random Forest ===")
    print(f"{'Experiment':28s} Train       Test        CV-5")

    for name, model, parameters in experiments:
        with mlflow.start_run(run_name=name):
            results = evaluate_model(
                name, model, X_train, X_test, y_train, y_test, X, y
            )

            mlflow.log_params(parameters)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_metrics(results)

            if isinstance(model, RandomForestClassifier):
                oob_score = float(model.oob_score_)
                print(f"  OOB score: {oob_score:.4f}")
                mlflow.log_metric("oob_score", oob_score)

                if name == "Random Forest baseline":
                    fitted_forest = model
                    forest_results = results

    # 4. Save feature importance from the baseline Random Forest.
    if fitted_forest is not None:
        importances = fitted_forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 5))
        plt.bar(
            [f"feature_{i}" for i in indices],
            importances[indices],
        )
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(rotation=90)
        plt.tight_layout()

        plot_path = OUTPUT_DIR / "rf_feature_importance.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()

        with mlflow.start_run(run_name="Feature importance artifact"):
            mlflow.log_artifact(str(plot_path))

        print(f"\nFeature-importance plot saved to: {plot_path}")

        print("\nTop 10 feature importances:")
        for index in indices[:10]:
            print(f"feature_{index}: {importances[index]:.4f}")

        print("\nBaseline Random Forest OOB score:",
              f"{fitted_forest.oob_score_:.4f}")

    print("\nW3D4 experiments completed.")
    print("MLflow experiment: Cynaris-W3D4-Tree-Ensemble")


if __name__ == "__main__":
    main()
