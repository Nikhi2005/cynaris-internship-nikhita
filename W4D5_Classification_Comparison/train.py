
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Create output directories
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FIGURE_DIR = BASE_DIR / "figures"

MODEL_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Load the Breast Cancer dataset
# --------------------------------------------------

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("\nDATASET INFORMATION")
print("-------------------")
print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Class names:", dataset.target_names)
print("Class distribution:")
print(pd.Series(y).value_counts().sort_index())


# --------------------------------------------------
# 3. Split the dataset into training and testing data
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("\nDATA SPLIT")
print("----------")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 4. Define the classification models
# --------------------------------------------------

logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000)),
])

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)


# --------------------------------------------------
# 5. Train both models
# --------------------------------------------------

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model,
}

results = []

for model_name, model in models.items():

    print(f"\n{'=' * 55}")
    print(f"TRAINING: {model_name}")
    print("=" * 55)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Print classification report
    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        target_names=dataset.target_names,
    ))

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "ROC-AUC": roc_auc,
    })

    # --------------------------------------------------
    # 6. Generate and save confusion matrix
    # --------------------------------------------------

    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=dataset.target_names,
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.tight_layout()

    figure_name = model_name.lower().replace(" ", "_")
    plt.savefig(FIGURE_DIR / f"{figure_name}_confusion_matrix.png")
    plt.close()

    # --------------------------------------------------
    # 7. Generate and save ROC-AUC curve
    # --------------------------------------------------

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(7, 5))
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC = {roc_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} - ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(FIGURE_DIR / f"{figure_name}_roc_curve.png")
    plt.close()

    # --------------------------------------------------
    # 8. Save trained model using Joblib
    # --------------------------------------------------

    model_path = MODEL_DIR / f"{figure_name}.joblib"
    joblib.dump(model, model_path)

    print("Model saved:", model_path.name)


# --------------------------------------------------
# 9. Compare both models
# --------------------------------------------------

comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print("----------------")
print(comparison.to_string(index=False))


# --------------------------------------------------
# 10. Verify saved models
# --------------------------------------------------

print("\nSAVED MODEL VERIFICATION")
print("------------------------")

for model_name, model in models.items():

    figure_name = model_name.lower().replace(" ", "_")
    model_path = MODEL_DIR / f"{figure_name}.joblib"

    loaded_model = joblib.load(model_path)
    loaded_predictions = loaded_model.predict(X_test)

    original_predictions = model.predict(X_test)

    assert (loaded_predictions == original_predictions).all()

    print(f"{model_name}: Successfully loaded and verified")


print("\nW4D5 MODEL TRAINING AND EVALUATION COMPLETED!")
