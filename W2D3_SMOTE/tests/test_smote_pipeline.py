import pandas as pd
from collections import Counter

from imblearn.over_sampling import SMOTE


def test_smote_balances_training_data():
    X = pd.DataFrame({
        "feature": list(range(100))
    })
    y = pd.Series([0] * 80 + [1] * 20)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    counts = Counter(y_resampled)

    assert counts[0] == counts[1]
    assert len(X_resampled) > len(X)