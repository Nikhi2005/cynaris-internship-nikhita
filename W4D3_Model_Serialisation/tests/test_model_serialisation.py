
"""
W4D3: Automated Tests for Model Serialisation

Tests:
1. Pickle model file creation.
2. Joblib model file creation.
3. Successful Pickle model loading.
4. Successful Joblib model loading.
5. Prediction consistency between all models.
6. Non-zero size of saved model files.
"""

import numpy as np
import pytest

from scripts.model_serialisation import (
    load_data,
    train_model,
    save_with_pickle,
    save_with_joblib,
    load_with_pickle,
    load_with_joblib
)


# --------------------------------------------------
# Fixtures: Prepare model and test data
# --------------------------------------------------

@pytest.fixture
def trained_model():
    """Train a model for testing."""

    X_train, X_test, y_train, y_test = load_data()

    model = train_model(X_train, y_train)

    return model, X_test


# --------------------------------------------------
# Test 1: Pickle file creation
# --------------------------------------------------

def test_pickle_file_exists(trained_model, tmp_path):

    model, _ = trained_model

    file_path = tmp_path / "test_model.pkl"

    save_with_pickle(model, file_path)

    assert file_path.is_file()


# --------------------------------------------------
# Test 2: Joblib file creation
# --------------------------------------------------

def test_joblib_file_exists(trained_model, tmp_path):

    model, _ = trained_model

    file_path = tmp_path / "test_model.joblib"

    save_with_joblib(model, file_path)

    assert file_path.is_file()


# --------------------------------------------------
# Test 3: Pickle model loading
# --------------------------------------------------

def test_pickle_load_success(trained_model, tmp_path):

    model, _ = trained_model

    file_path = tmp_path / "test_model.pkl"

    save_with_pickle(model, file_path)

    loaded_model = load_with_pickle(file_path)

    assert hasattr(loaded_model, "predict")


# --------------------------------------------------
# Test 4: Joblib model loading
# --------------------------------------------------

def test_joblib_load_success(trained_model, tmp_path):

    model, _ = trained_model

    file_path = tmp_path / "test_model.joblib"

    save_with_joblib(model, file_path)

    loaded_model = load_with_joblib(file_path)

    assert hasattr(loaded_model, "predict")


# --------------------------------------------------
# Test 5: Prediction consistency
# --------------------------------------------------

def test_prediction_consistency(trained_model, tmp_path):

    model, X_test = trained_model

    pickle_path = tmp_path / "test_model.pkl"
    joblib_path = tmp_path / "test_model.joblib"

    # Save the original trained model
    save_with_pickle(model, pickle_path)
    save_with_joblib(model, joblib_path)

    # Load both saved models
    pickle_model = load_with_pickle(pickle_path)
    joblib_model = load_with_joblib(joblib_path)

    # Generate predictions
    original_predictions = model.predict(X_test)
    pickle_predictions = pickle_model.predict(X_test)
    joblib_predictions = joblib_model.predict(X_test)

    # Compare all predictions
    assert np.array_equal(
        original_predictions, pickle_predictions
    )

    assert np.array_equal(
        original_predictions, joblib_predictions
    )


# --------------------------------------------------
# Test 6: Saved model files are not empty
# --------------------------------------------------

def test_file_sizes_nonzero(trained_model, tmp_path):

    model, _ = trained_model

    pickle_path = tmp_path / "test_model.pkl"
    joblib_path = tmp_path / "test_model.joblib"

    save_with_pickle(model, pickle_path)
    save_with_joblib(model, joblib_path)

    assert pickle_path.stat().st_size > 0
    assert joblib_path.stat().st_size > 0
