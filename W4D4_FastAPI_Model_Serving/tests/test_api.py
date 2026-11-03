
"""
W4D4: FastAPI Model Serving Endpoint

Automated tests for the health-check and prediction endpoints.
"""

import pytest

from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app


# --------------------------------------------------
# 1. Create FastAPI test client
# --------------------------------------------------

client = TestClient(app)


# --------------------------------------------------
# 2. Prepare a valid prediction request
# --------------------------------------------------

def get_valid_payload():
    """
    Get one real sample from the Breast Cancer dataset.
    """

    dataset = load_breast_cancer()

    feature_names = dataset.feature_names
    sample = dataset.data[0]

    instance = {
        feature: float(value)
        for feature, value in zip(feature_names, sample)
    }

    return {
        "instances": [instance]
    }


# --------------------------------------------------
# 3. Test health endpoint
# --------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --------------------------------------------------
# 4. Test successful prediction
# --------------------------------------------------

def test_predict_endpoint():
    payload = get_valid_payload()

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predictions" in result
    assert "labels" in result

    assert len(result["predictions"]) == 1
    assert result["predictions"][0] in [0, 1]

    assert result["labels"][0] in ["malignant", "benign"]


# --------------------------------------------------
# 5. Test missing required features
# --------------------------------------------------

def test_predict_missing_features():
    payload = {
        "instances": [
            {
                "mean radius": 14.2,
                "mean texture": 20.1
            }
        ]
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Missing required features" in response.json()["detail"]


# --------------------------------------------------
# 6. Test empty instances list
# --------------------------------------------------

def test_predict_empty_instances():
    payload = {
        "instances": []
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# --------------------------------------------------
# 7. Test invalid feature values
# --------------------------------------------------

def test_predict_invalid_input():
    payload = get_valid_payload()

    payload["instances"][0]["mean radius"] = "invalid"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Invalid input data" in response.json()["detail"]
