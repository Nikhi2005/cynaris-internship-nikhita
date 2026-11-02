
"""
W4D2: Bias-Variance Tradeoff & Regularisation
Practical: Hyperparameter Tuning using GridSearchCV
           and RandomizedSearchCV.

Dataset: Wine Recognition Dataset
Model: Random Forest Classifier
"""

import time
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# 1. LOAD THE WINE DATASET
# --------------------------------------------------

print("\n========== WINE DATASET ==========")

wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("Dataset shape:", X.shape)
print("Target classes:", wine.target_names)
print("\nFirst five rows:")
print(X.head())


# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------

print("\n========== TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 3. BASELINE RANDOM FOREST MODEL
# --------------------------------------------------

print("\n========== BASELINE MODEL ==========")

baseline_model = RandomForestClassifier(
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print("Baseline Test Accuracy:", round(baseline_accuracy, 4))


# --------------------------------------------------
# 4. GRID SEARCH CROSS-VALIDATION
# --------------------------------------------------

print("\n========== GRID SEARCH ==========")

# Define the hyperparameter combinations to evaluate
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 2]
}

# 3 x 3 x 2 = 18 combinations

# Use stratified 5-fold cross-validation
cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring="accuracy",
    n_jobs=-1
)

grid_start_time = time.perf_counter()

grid_search.fit(X_train, y_train)

grid_end_time = time.perf_counter()

grid_time = grid_end_time - grid_start_time

# Evaluate the best model on the test set
grid_best_model = grid_search.best_estimator_

grid_predictions = grid_best_model.predict(X_test)

grid_accuracy = accuracy_score(
    y_test,
    grid_predictions
)

print("Best Grid Search Parameters:", grid_search.best_params_)
print("Best Grid Search CV Score:", round(grid_search.best_score_, 4))
print("Grid Search Test Accuracy:", round(grid_accuracy, 4))
print("Grid Search Execution Time:", round(grid_time, 4), "seconds")


# --------------------------------------------------
# 5. RANDOMIZED SEARCH CROSS-VALIDATION
# --------------------------------------------------

print("\n========== RANDOMIZED SEARCH ==========")

# Define the hyperparameter search space
param_dist = {
    "n_estimators": [10, 25, 50, 100, 200, 300],
    "max_depth": [2, 3, 5, 7, 10, None],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10]
}

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,
    cv=cv_strategy,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)

random_start_time = time.perf_counter()

random_search.fit(X_train, y_train)

random_end_time = time.perf_counter()

random_time = random_end_time - random_start_time

# Evaluate the best model on the test set
random_best_model = random_search.best_estimator_

random_predictions = random_best_model.predict(X_test)

random_accuracy = accuracy_score(
    y_test,
    random_predictions
)

print("Best Randomized Search Parameters:", random_search.best_params_)
print("Best Randomized Search CV Score:", round(random_search.best_score_, 4))
print("Randomized Search Test Accuracy:", round(random_accuracy, 4))
print("Randomized Search Execution Time:", round(random_time, 4), "seconds")


# --------------------------------------------------
# 6. PERFORMANCE COMPARISON
# --------------------------------------------------

print("\n========== PERFORMANCE COMPARISON ==========")

comparison = pd.DataFrame({
    "Method": [
        "Baseline Random Forest",
        "Grid Search",
        "Randomized Search"
    ],
    "Test Accuracy": [
        baseline_accuracy,
        grid_accuracy,
        random_accuracy
    ],
    "Execution Time (seconds)": [
        0,  # Baseline training time is not measured here
        grid_time,
        random_time
    ]
})

print(comparison.to_string(index=False))

print("\n========== BEST MODEL ==========")

best_method = comparison.loc[
    comparison["Test Accuracy"].idxmax(),
    "Method"
]

print("Best method based on test accuracy:", best_method)

print("\n========== EXPERIMENT COMPLETED ==========")
