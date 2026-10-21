import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from W2D3_SMOTE.src.feature_engineering import (
    DATA_FILE,
    generate_dataset,
    create_features,
)


def run_smote_pipeline():
    """Train and evaluate a model using SMOTE."""

    # Load the saved dataset or generate it if missing
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, parse_dates=["order_time"])
    else:
        df = generate_dataset()

    df = create_features(df)

    # Separate input features and target
    X = df.drop(columns=["returned", "order_time"])
    y = df["returned"]

    # Split before applying SMOTE
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Identify numerical and categorical columns
    categorical_columns = ["city_tier", "age_group"]
    numerical_columns = [
        column
        for column in X.columns
        if column not in categorical_columns
    ]

    # Prepare numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numerical_columns,
            ),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_columns,
            ),
        ]
    )

    # Preprocess, apply SMOTE, and train the model
    model = ImbPipeline([
        ("preprocessing", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=5)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    print("Training class distribution:")
    print(y_train.value_counts().sort_index())

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTesting class distribution (unchanged):")
    print(y_test.value_counts().sort_index())

    print("\n=== Model Evaluation ===")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.3f}")
    print(
        f"Precision: {precision_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(
        f"Recall:    {recall_score(y_test, predictions, zero_division=0):.3f}"
    )
    print(
        f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.3f}"
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model


if __name__ == "__main__":
    run_smote_pipeline()