
# ============================================================
# W4D1: Model Evaluation Metrics
# Cynaris Solutions - AI/ML Internship
#
# Topics:
# 1. Dataset preparation
# 2. Stratified 5-Fold Cross-Validation
# 3. Accuracy, Precision, Recall, F1-Score and ROC-AUC
# 4. Model comparison
# 5. Learning curve and overfitting analysis
# ============================================================

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# 1. LOAD AND PREPARE DATASET
# ------------------------------------------------------------

def load_and_prepare_data():
    """
    Load the Breast Cancer dataset.

    Original labels:
        0 = Malignant
        1 = Benign

    Updated labels:
        1 = Malignant (positive class)
        0 = Benign (negative class)
    """

    data = load_breast_cancer()

    X = data.data
    original_y = data.target

    # Make malignant cases the positive class
    y = (original_y == 0).astype(int)

    return X, y


# ------------------------------------------------------------
# 2. CREATE STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

def create_stratified_folds():
    """
    Create five folds while preserving class proportions.
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    return cv


# ------------------------------------------------------------
# 3. DISPLAY DATASET AND FOLD INFORMATION
# ------------------------------------------------------------

def display_dataset_information(X, y, cv):

    print("===== DATASET INFORMATION =====")
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    print("\n===== CLASS DISTRIBUTION =====")
    print("Malignant (1):", np.sum(y == 1))
    print("Benign (0):", np.sum(y == 0))

    print("\nTarget labels:", np.unique(y))

    print("\n===== STRATIFIED 5-FOLD DISTRIBUTION =====")

    for fold_number, (train_index, test_index) in enumerate(
        cv.split(X, y), start=1
    ):
        y_test = y[test_index]

        malignant_count = np.sum(y_test == 1)
        benign_count = np.sum(y_test == 0)

        print(
            f"Fold {fold_number}: "
            f"Malignant = {malignant_count}, "
            f"Benign = {benign_count}, "
            f"Total = {len(test_index)}"
        )


# ------------------------------------------------------------
# 4. DEFINE MODELS
# ------------------------------------------------------------

def get_models():

    models = [
        (
            "Logistic Regression (C=0.01)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=0.01,
                    max_iter=2000
                ))
            ])
        ),

        (
            "Logistic Regression (C=1)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=1.0,
                    max_iter=2000
                ))
            ])
        ),

        (
            "Logistic Regression (C=100)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=100,
                    max_iter=2000
                ))
            ])
        ),

        (
            "Decision Tree (depth=2)",
            DecisionTreeClassifier(
                max_depth=2,
                random_state=42
            )
        ),

        (
            "Decision Tree (unlimited)",
            DecisionTreeClassifier(
                max_depth=None,
                random_state=42
            )
        ),

        (
            "Random Forest (100 trees)",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )
    ]

    return models


# ------------------------------------------------------------
# 5. EVALUATE MODELS USING FIVE METRICS
# ------------------------------------------------------------

def evaluate_models(models, X, y, cv):

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    print("\n===== 5-FOLD MODEL EVALUATION =====")

    for model_name, model in models:

        results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        print(f"\nModel: {model_name}")

        print(
            f"{'Metric':<12}"
            f"{'Mean':>10}"
            f"{'Std':>10}"
        )

        print("-" * 32)

        for metric in scoring:

            scores = results[f"test_{metric}"]

            print(
                f"{metric:<12}"
                f"{scores.mean():>10.4f}"
                f"{scores.std():>10.4f}"
            )


# ------------------------------------------------------------
# 6. LEARNING CURVE - RANDOM FOREST
# ------------------------------------------------------------

def analyze_learning_curve(X, y, cv):

    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    train_sizes, train_scores, validation_scores = learning_curve(
        rf,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 8),
        scoring="accuracy",
        n_jobs=-1
    )

    train_means = train_scores.mean(axis=1)
    validation_means = validation_scores.mean(axis=1)

    print("\n===== LEARNING CURVE: RANDOM FOREST =====")

    print(
        f"{'Train Size':>12}"
        f"{'Train Acc':>12}"
        f"{'CV Acc':>12}"
        f"{'Gap':>10}"
    )

    print("-" * 46)

    for size, train_acc, cv_acc in zip(
        train_sizes,
        train_means,
        validation_means
    ):

        gap = train_acc - cv_acc

        print(
            f"{size:12d}"
            f"{train_acc:12.4f}"
            f"{cv_acc:12.4f}"
            f"{gap:10.4f}"
        )

    print("\n===== LEARNING CURVE INTERPRETATION =====")
    print("Small train-validation gap: better generalisation.")
    print("Large gap: possible overfitting.")
    print("Both scores low: possible underfitting.")


# ------------------------------------------------------------
# 7. MAIN PROGRAM
# ------------------------------------------------------------

if __name__ == "__main__":

    # Load and prepare the dataset
    X, y = load_and_prepare_data()

    # Create cross-validation strategy
    cv = create_stratified_folds()

    # Display dataset and fold information
    display_dataset_information(X, y, cv)

    # Define models for comparison
    models = get_models()

    # Evaluate all models
    evaluate_models(models, X, y, cv)

    # Analyze learning curve
    analyze_learning_curve(X, y, cv)

    print("\n===== W4D1 EXECUTION COMPLETED =====")
