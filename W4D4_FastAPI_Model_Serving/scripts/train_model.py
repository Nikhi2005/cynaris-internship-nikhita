
"""
W4D4: FastAPI Model Serving Endpoint
Train a Logistic Regression model, track the experiment
using MLflow, and save the trained pipeline using Joblib.
"""

import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "breast_cancer.joblib"

# Create the models directory if it does not exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load the Breast Cancer dataset
# --------------------------------------------------

print("Loading Breast Cancer dataset...")

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset loaded successfully.")
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 3. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# --------------------------------------------------
# 4. Create the machine learning pipeline
# --------------------------------------------------

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

print("\nTraining Logistic Regression model...")

model.fit(X_train, y_train)

print("Model training completed.")


# --------------------------------------------------
# 6. Evaluate the trained model
# --------------------------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Evaluation")
print("-------------------------")
print(f"Accuracy: {accuracy:.4f}")


# --------------------------------------------------
# 7. Track experiment using MLflow
# --------------------------------------------------

print("\nLogging experiment to MLflow...")

mlflow.set_experiment("W4D4-FastAPI-Model-Serving")

with mlflow.start_run(run_name="Logistic-Regression-Breast-Cancer"):

    mlflow.log_param("model", "Logistic Regression")
    mlflow.log_param("max_iter", 5000)
    mlflow.log_param("test_size", 0.2)

    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, name="model")

print("MLflow experiment logged successfully.")


# --------------------------------------------------
# 8. Save the trained pipeline using Joblib
# --------------------------------------------------

joblib.dump(model, MODEL_PATH)

print("\nModel saved successfully.")
print("Saved model location:", MODEL_PATH)


# --------------------------------------------------
# 9. Verify that the saved model can be loaded
# --------------------------------------------------

loaded_model = joblib.load(MODEL_PATH)

loaded_predictions = loaded_model.predict(X_test)

predictions_match = (y_pred == loaded_predictions).all()

print("\nModel Loading Verification")
print("-------------------------")
print("Original and loaded predictions match:", predictions_match)

if not predictions_match:
    raise RuntimeError("Loaded model predictions do not match!")

print("\nW4D4 model training completed successfully!")
