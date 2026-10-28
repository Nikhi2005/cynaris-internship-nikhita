
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def test_random_forest_training_and_metrics():
    # Generate a small, reproducible dataset for a fast test.
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=20,
        oob_score=True,
        random_state=42,
    )
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    assert hasattr(model, "estimators_")
    assert 0.0 <= train_accuracy <= 1.0
    assert 0.0 <= test_accuracy <= 1.0
    assert 0.0 <= model.oob_score_ <= 1.0
