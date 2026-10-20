import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.feature_selection import SelectKBest, f_classif


# Test 1: Check all three encoders.
def test_encoders():
    grades = pd.DataFrame({"grade": ["A", "B", "C", "A"]})

    label = LabelEncoder().fit_transform(grades["grade"])
    onehot = OneHotEncoder(sparse_output=False).fit_transform(grades)
    ordinal = OrdinalEncoder(
        categories=[["A", "B", "C"]]
    ).fit_transform(grades)

    assert len(label) == 4
    assert onehot.shape == (4, 3)
    assert ordinal.shape == (4, 1)


# Test 2: Scaling must preserve the number of rows and columns.
def test_scalers():
    data = pd.DataFrame({
        "study_hrs": [2, 5, 8, 4],
        "midterm": [40, 70, 90, 60],
    })

    for scaler in [StandardScaler(), MinMaxScaler(), RobustScaler()]:
        result = scaler.fit_transform(data)

        assert result.shape == data.shape


# Test 3: SelectKBest must return exactly five features.
def test_select_k_best():
    X = pd.DataFrame({
        "f1": [1, 2, 3, 4, 5, 6],
        "f2": [6, 5, 4, 3, 2, 1],
        "f3": [1, 3, 2, 5, 4, 6],
        "f4": [2, 4, 1, 6, 3, 5],
        "f5": [5, 1, 4, 2, 6, 3],
        "f6": [3, 6, 2, 4, 1, 5],
    })

    y = [0, 0, 0, 1, 1, 1]

    selector = SelectKBest(score_func=f_classif, k=5)
    selected = selector.fit_transform(X, y)

    assert selected.shape == (6, 5)