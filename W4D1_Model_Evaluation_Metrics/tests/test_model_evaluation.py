
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from scripts.model_evaluation import (
    load_and_prepare_data,
    create_stratified_folds,
    get_models
)


def test_dataset_shape_and_target_labels():
    X, y = load_and_prepare_data()

    assert X.shape == (569, 30)
    assert set(np.unique(y)) == {0, 1}


def test_malignant_is_positive_class():
    X, y = load_and_prepare_data()

    assert np.sum(y == 1) == 212
    assert np.sum(y == 0) == 357


def test_stratified_five_fold_configuration():
    cv = create_stratified_folds()

    assert isinstance(cv, StratifiedKFold)
    assert cv.n_splits == 5
    assert cv.shuffle is True
    assert cv.random_state == 42


def test_all_samples_are_used_in_validation_folds():
    X, y = load_and_prepare_data()
    cv = create_stratified_folds()

    validation_indices = []

    for _, test_index in cv.split(X, y):
        validation_indices.extend(test_index)

    assert len(validation_indices) == len(y)
    assert len(set(validation_indices)) == len(y)


def test_logistic_regression_uses_pipeline():
    models = get_models()

    for name, model in models[:3]:
        assert isinstance(model, Pipeline)
        assert "scaler" in model.named_steps
        assert "model" in model.named_steps


def test_tree_models_do_not_use_scaling_pipeline():
    models = get_models()

    for name, model in models[3:]:
        assert not isinstance(model, Pipeline)
