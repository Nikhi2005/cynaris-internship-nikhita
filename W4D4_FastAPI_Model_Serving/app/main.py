
"""
W4D4: FastAPI Model Serving Endpoint

Loads the trained Breast Cancer classification pipeline
and exposes prediction functionality through FastAPI.
"""

import joblib
import pandas as pd

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.datasets import load_breast_cancer


# --------------------------------------------------
# 1. Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "breast_cancer.joblib"


# --------------------------------------------------
# 2. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="W4D4 Breast Cancer Prediction API",
    description="An API that serves predictions using a trained Logistic Regression model.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. Load the trained model and feature names
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}. "
        "Run scripts/train_model.py first."
    )

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = list(load_breast_cancer().feature_names)


# --------------------------------------------------
# 4. Define request data schema
# --------------------------------------------------

class PredictRequest(BaseModel):
    instances: list[dict] = Field(
        ...,
        min_length=1,
        description="List of patient measurement records."
    )


# --------------------------------------------------
# 5. Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "FastAPI model serving application is running."
    }


# --------------------------------------------------
# 6. Model prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: PredictRequest):

    try:
        input_data = pd.DataFrame(request.instances)

        missing_features = [
            feature for feature in FEATURE_NAMES
            if feature not in input_data.columns
        ]

        if missing_features:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required features: {missing_features}"
            )

        input_data = input_data[FEATURE_NAMES]

        input_data = input_data.apply(
            pd.to_numeric,
            errors="raise"
        )

        predictions = model.predict(input_data.to_numpy()).tolist()

        labels = [
            "malignant" if prediction == 0 else "benign"
            for prediction in predictions
        ]

        return {
            "predictions": predictions,
            "labels": labels
        }

    except HTTPException:
        raise

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid input data: {str(exc)}"
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )
