# W3D2: Logistic Regression and Classification
# Cynaris Solutions Private Limited

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)

# 1. Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("BREAST CANCER CLASSIFICATION")
print("-" * 40)
print("Dataset shape:", X.shape)
print("Malignant samples:", (y == 0).sum())
print("Benign samples:", (y == 1).sum())

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train the Logistic Regression model
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Display coefficients and intercept
print("\nIntercept:", model.intercept_)

coefficients = pd.DataFrame({
    "Feature": data.feature_names,
    "Coefficient": model.coef_[0]
})
coefficients["Absolute Coefficient"] = coefficients["Coefficient"].abs()
coefficients = coefficients.sort_values(
    "Absolute Coefficient", ascending=False
)

print("\nTop 5 features by absolute coefficient:")
print(coefficients.head(5).to_string(index=False))

# 6. Generate predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 7. Evaluate the model
cm = confusion_matrix(y_test, y_pred)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nEVALUATION RESULTS")
print("-" * 40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

# 8. Plot the confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Malignant", "Benign"],
    yticklabels=["Malignant", "Benign"]
)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Breast Cancer - Confusion Matrix")
plt.tight_layout()
plt.show()

# 9. Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Breast Cancer - ROC Curve")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()



# ==================================================
# 11. MULTICLASS CLASSIFICATION USING IRIS DATASET
# ==================================================

from sklearn.datasets import load_iris
from sklearn.multiclass import OneVsRestClassifier

# Load the Iris dataset
iris = load_iris()

X_iris = iris.data
y_iris = iris.target

print("\n\nIRIS MULTICLASS CLASSIFICATION")
print("-" * 40)
print("Dataset shape:", X_iris.shape)
print("Classes:", iris.target_names)

# Split the dataset into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris
)

# Standardize the features
iris_scaler = StandardScaler()

X_train_iris_scaled = iris_scaler.fit_transform(X_train_iris)
X_test_iris_scaled = iris_scaler.transform(X_test_iris)

# Model 1: One-vs-Rest Logistic Regression
ovr_model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000)
)

ovr_model.fit(X_train_iris_scaled, y_train_iris)

# Model 2: Multinomial / Softmax Logistic Regression
softmax_model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

softmax_model.fit(X_train_iris_scaled, y_train_iris)

# Generate predictions
ovr_pred = ovr_model.predict(X_test_iris_scaled)
softmax_pred = softmax_model.predict(X_test_iris_scaled)

# Calculate evaluation metrics
comparison_results = []

for model_name, predictions in [
    ("One-vs-Rest", ovr_pred),
    ("Multinomial / Softmax", softmax_pred)
]:
    comparison_results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test_iris, predictions),
        "Precision (Macro)": precision_score(
            y_test_iris, predictions, average="macro", zero_division=0
        ),
        "Recall (Macro)": recall_score(
            y_test_iris, predictions, average="macro", zero_division=0
        ),
        "F1-Score (Macro)": f1_score(
            y_test_iris, predictions, average="macro", zero_division=0
        )
    })

# Display the comparison table
comparison_df = pd.DataFrame(comparison_results)

print("\nMODEL COMPARISON")
print("-" * 40)
print(comparison_df.round(4).to_string(index=False))