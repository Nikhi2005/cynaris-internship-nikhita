
"""
W4D3: Model Serialisation using Pickle and Joblib

This script:
1. Loads the Breast Cancer dataset.
2. Trains a Logistic Regression classifier.
3. Saves the trained model using Pickle and Joblib.
4. Loads both saved models.
5. Compares predictions from all three models.
6. Displays the saved model file sizes.
"""

import pickle
import joblib
import numpy as np

from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

PICKLE_PATH = ARTIFACTS_DIR / "model_pickle.pkl"
JOBLIB_PATH = ARTIFACTS_DIR / "model_joblib.joblib"


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

def load_data():
    """Load the Breast Cancer dataset and split it."""

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 3. Train the classification model
# --------------------------------------------------

def train_model(X_train, y_train):
    """Train and return a Logistic Regression model."""

    model = LogisticRegression(max_iter=5000)

    model.fit(X_train, y_train)

    return model


# --------------------------------------------------
# 4. Save the model using Pickle
# --------------------------------------------------

def save_with_pickle(model, file_path):
    """Save the trained model using Pickle."""

    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)

    except OSError as error:
        raise RuntimeError(
            f"Could not save model using Pickle: {error}"
        ) from error


# --------------------------------------------------
# 5. Save the model using Joblib
# --------------------------------------------------

def save_with_joblib(model, file_path):
    """Save the trained model using Joblib."""

    try:
        joblib.dump(model, file_path)

    except OSError as error:
        raise RuntimeError(
            f"Could not save model using Joblib: {error}"
        ) from error


# --------------------------------------------------
# 6. Load the model using Pickle
# --------------------------------------------------

def load_with_pickle(file_path):
    """Load a saved model using Pickle."""

    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)

    except (OSError, pickle.UnpicklingError) as error:
        raise RuntimeError(
            f"Could not load Pickle model: {error}"
        ) from error


# --------------------------------------------------
# 7. Load the model using Joblib
# --------------------------------------------------

def load_with_joblib(file_path):
    """Load a saved model using Joblib."""

    try:
        return joblib.load(file_path)

    except (OSError, EOFError, ValueError) as error:
        raise RuntimeError(
            f"Could not load Joblib model: {error}"
        ) from error


# --------------------------------------------------
# 8. Main execution
# --------------------------------------------------

def main():

    print("\nW4D3 - Model Serialisation")
    print("-" * 40)

    # Create artifacts folder if it does not exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print("[INFO] Loading Breast Cancer dataset...")

    X_train, X_test, y_train, y_test = load_data()

    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Testing samples: {len(X_test)}")

    # Train model
    print("\n[INFO] Training Logistic Regression model...")

    original_model = train_model(X_train, y_train)

    # Generate original predictions
    original_predictions = original_model.predict(X_test)

    # Save using Pickle
    print("\n[INFO] Saving model using Pickle...")

    save_with_pickle(original_model, PICKLE_PATH)

    # Save using Joblib
    print("[INFO] Saving model using Joblib...")

    save_with_joblib(original_model, JOBLIB_PATH)

    # Load both saved models
    print("\n[INFO] Loading saved models...")

    pickle_model = load_with_pickle(PICKLE_PATH)
    joblib_model = load_with_joblib(JOBLIB_PATH)

    # Generate predictions from loaded models
    pickle_predictions = pickle_model.predict(X_test)
    joblib_predictions = joblib_model.predict(X_test)

    # Verify prediction consistency
    pickle_match = np.array_equal(
        original_predictions, pickle_predictions
    )

    joblib_match = np.array_equal(
        original_predictions, joblib_predictions
    )

    print("\n[INFO] Prediction verification:")

    print(f"Original vs Pickle: {pickle_match}")
    print(f"Original vs Joblib: {joblib_match}")

    if not (pickle_match and joblib_match):
        raise AssertionError("Model predictions do not match!")

    print("\n[SUCCESS] All model predictions match!")

    # Display saved file sizes
    pickle_size = PICKLE_PATH.stat().st_size
    joblib_size = JOBLIB_PATH.stat().st_size

    print("\n[INFO] Saved model file sizes:")
    print(f"Pickle: {pickle_size:,} bytes")
    print(f"Joblib: {joblib_size:,} bytes")

    print("\n[INFO] Model serialisation completed successfully.")


if __name__ == "__main__":
    main()
