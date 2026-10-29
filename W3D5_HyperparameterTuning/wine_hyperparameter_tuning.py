
"""W3D5: Hyperparameter Tuning with GridSearchCV.

Compares six classifiers on the Wine dataset, tests KNN and SVM
hyperparameters, and tunes a Random Forest using GridSearchCV.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def make_pipeline(classifier):
    """Scale features inside each training fold, then fit the classifier."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier),
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model and calculate test and cross-validation metrics."""
    cv_scores = cross_val_score(
        model, X_train, y_train, cv=CV, scoring="accuracy"
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    result = {
        "model": name,
        "test_accuracy": accuracy_score(y_test, predictions),
        "macro_f1": f1_score(y_test, predictions, average="macro"),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }
    return result


def main():
    # 1. Load the Wine dataset.
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target

    print(
        f"Wine dataset: {X.shape[0]} samples, "
        f"{X.shape[1]} features, {len(np.unique(y))} classes"
    )
    print(f"Class distribution: {np.bincount(y)}")

    # 2. Split the data before fitting any preprocessing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 3. Compare the six classifiers supplied in the lesson.
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "SVM RBF": SVC(kernel="rbf", C=10, gamma="scale"),
        "KNN k=5": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
    }

    results = []

    print("\n=== Six Classifier Comparison ===")
    for name, classifier in classifiers.items():
        model = make_pipeline(classifier)
        result = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        results.append(result)
        print(
            f"{name:22s} "
            f"Test={result['test_accuracy']:.4f}  "
            f"Macro-F1={result['macro_f1']:.4f}  "
            f"CV={result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
        )

    # 4. Evaluate all seven K values for KNN.
    print("\n=== KNN: Effect of K ===")
    knn_results = []

    for k in [1, 3, 5, 7, 11, 15, 21]:
        model = make_pipeline(KNeighborsClassifier(n_neighbors=k))
        result = evaluate_model(
            f"KNN k={k}", model, X_train, X_test, y_train, y_test
        )
        knn_results.append(result)
        print(
            f"K={k:2d}  Test={result['test_accuracy']:.4f}  "
            f"CV={result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
        )

    # 5. Complete the two SVM experiments from the lesson.
    print("\n=== SVM Experiments ===")
    svm_experiments = {
        "SVM RBF C=100": SVC(
            kernel="rbf", C=100, gamma="scale"
        ),
        "SVM Linear": SVC(kernel="linear", C=1.0),
    }

    svm_results = []
    for name, classifier in svm_experiments.items():
        model = make_pipeline(classifier)
        result = evaluate_model(
            name, model, X_train, X_test, y_train, y_test
        )
        svm_results.append(result)
        print(
            f"{name:22s} "
            f"Test={result['test_accuracy']:.4f}  "
            f"Macro-F1={result['macro_f1']:.4f}  "
            f"CV={result['cv_mean']:.4f} ± {result['cv_std']:.4f}"
        )

    # 6. GridSearchCV: 12 parameter combinations, each with 5 folds.
    print("\n=== Random Forest GridSearchCV ===")
    rf_pipeline = make_pipeline(
        RandomForestClassifier(random_state=RANDOM_STATE)
    )
    param_grid = {
        "clf__n_estimators": [50, 200],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    }

    search = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=CV,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_predictions = best_model.predict(X_test)
    best_cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=CV, scoring="accuracy"
    )

    grid_result = {
        "model": "Random Forest GridSearch best",
        "test_accuracy": accuracy_score(y_test, best_predictions),
        "macro_f1": f1_score(y_test, best_predictions, average="macro"),
        "cv_mean": best_cv_scores.mean(),
        "cv_std": best_cv_scores.std(),
    }
    results.append(grid_result)

    print("Best parameters:", search.best_params_)
    print(f"Best GridSearch CV accuracy: {search.best_score_:.4f}")
    print(
        f"Best model test accuracy: {grid_result['test_accuracy']:.4f}"
    )
    print(f"Best model macro-F1: {grid_result['macro_f1']:.4f}")

    # 7. Save all main model results as a CSV.
    results.extend(knn_results)
    results.extend(svm_results)
    results_df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "wine_tuning_results.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV saved to: {csv_path}")

    # 8. Log compact experiment details to local MLflow tracking.
    mlflow.set_experiment("Cynaris-W3D5-Hyperparameter-Tuning")

    with mlflow.start_run(run_name="Wine classifier comparison"):
        mlflow.set_tag("lesson", "W3D5")
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("dataset", "Wine")
        mlflow.log_param("baseline_classifier_count", len(classifiers))
        mlflow.log_metrics({
            "grid_best_cv_accuracy": float(search.best_score_),
            "grid_test_accuracy": float(grid_result["test_accuracy"]),
            "grid_macro_f1": float(grid_result["macro_f1"]),
        })
        mlflow.log_params({
            f"best_{key}": str(value)
            for key, value in search.best_params_.items()
        })
        mlflow.log_artifact(str(csv_path))

    print("\nW3D5 experiments completed.")
    print("MLflow experiment: Cynaris-W3D5-Hyperparameter-Tuning")


if __name__ == "__main__":
    main()
