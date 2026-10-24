
from pathlib import Path

import pandas as pd


# Locate the processed data folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def test_all_output_files_exist():
    """Check that all four processed CSV files were created."""
    expected_files = [
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    ]

    for filename in expected_files:
        assert (OUTPUT_DIR / filename).exists()


def test_train_test_row_counts():
    """Check that the training and testing row counts are correct."""
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")
    y_train = pd.read_csv(OUTPUT_DIR / "y_train.csv")
    y_test = pd.read_csv(OUTPUT_DIR / "y_test.csv")

    assert len(X_train) == 712
    assert len(X_test) == 179
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_processed_features_have_no_missing_values():
    """Check that preprocessing removed missing feature values."""
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def test_train_and_test_have_same_features():
    """Check that both datasets contain the same feature columns."""
    X_train = pd.read_csv(OUTPUT_DIR / "X_train.csv")
    X_test = pd.read_csv(OUTPUT_DIR / "X_test.csv")

    assert list(X_train.columns) == list(X_test.columns)
