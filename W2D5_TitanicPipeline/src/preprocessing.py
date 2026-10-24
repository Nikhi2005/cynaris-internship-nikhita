
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Find the project folder automatically.
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_DIR / "data" / "raw" / "train.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"


def run_preprocessing():
    # 1. Load the Titanic dataset.
    df = pd.read_csv(RAW_FILE)

    print("\n--- Dataset Overview ---")
    print("Dataset shape:", df.shape)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSurvival counts:")
    print(df["Survived"].value_counts())

    # 2. Separate the target from the input features.
    y = df["Survived"]

    # Drop identifiers, names, tickets, and Cabin.
    # Cabin has many missing values and is excluded in this basic pipeline.
    X = df.drop(
        columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"]
    )

    # 3. Identify numerical and categorical columns.
    numerical_columns = ["Age", "SibSp", "Parch", "Fare"]
    categorical_columns = ["Pclass", "Sex", "Embarked"]

    # 4. Fill missing numerical values and scale the numbers.
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 5. Fill missing categorical values and encode categories.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    # 6. Combine both preprocessing pipelines.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ]
    )

    # 7. Split the data before fitting preprocessing steps.
    # Stratification preserves the approximate survival-class ratio.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 8. Fit only on training data to avoid data leakage.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 9. Use readable names for the processed features.
    feature_names = preprocessor.get_feature_names_out()

    X_train_df = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_df = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    # 10. Save the processed datasets.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_df.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    X_test_df.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

    y_train.to_frame(name="Survived").to_csv(
        OUTPUT_DIR / "y_train.csv", index=False
    )
    y_test.to_frame(name="Survived").to_csv(
        OUTPUT_DIR / "y_test.csv", index=False
    )

    # 11. Display a summary of the results.
    print("\n--- Preprocessing Results ---")
    print("Training rows:", len(X_train_df))
    print("Testing rows:", len(X_test_df))
    print("Processed feature count:", len(feature_names))
    print("Missing values in training features:", X_train_df.isnull().sum().sum())
    print("Missing values in testing features:", X_test_df.isnull().sum().sum())
    print("Saved files to:", OUTPUT_DIR)
    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()