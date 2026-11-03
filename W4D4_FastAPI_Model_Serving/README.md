# W4D4: FastAPI Model Serving Endpoint

## 1. Overview

This task demonstrates how to deploy a trained Machine Learning model using FastAPI.

A Logistic Regression model is trained using the Breast Cancer Wisconsin dataset. The trained model is tracked using MLflow and saved using Joblib. FastAPI is then used to expose the model through REST API endpoints for prediction.

## 2. Objectives

- Train a Logistic Regression classification model.
- Track model parameters and evaluation metrics using MLflow.
- Save and load the trained model using Joblib.
- Develop a FastAPI application for model serving.
- Validate incoming prediction requests.
- Test API endpoints using Pytest.

## 3. Technologies Used

- Python
- Scikit-learn
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- MLflow
- Joblib
- Pytest

## 4. Project Structure

```text
W4D4_FastAPI_Model_Serving/
│
├── app/
│   └── main.py
│
├── scripts/
│   └── train_model.py
│
├── models/
│   └── breast_cancer.joblib
│
├── tests/
│   └── test_api.py
│
└── README.md
```

## 5. Dataset

The Breast Cancer Wisconsin dataset from Scikit-learn was used.

- Total samples: 569
- Total features: 30
- Training samples: 455
- Testing samples: 114
- Target classes: Malignant and Benign

## 6. Model Training

A Scikit-learn Pipeline was created using:

1. StandardScaler – scales the input features.
2. LogisticRegression – performs binary classification.

The dataset was divided into training and testing sets using an 80:20 split.

The trained model was evaluated using accuracy.

**Model Accuracy: 98.25%**

## 7. MLflow Experiment Tracking

MLflow was used to track the training experiment.

The following information was logged:

- Model name
- Maximum iterations
- Test size
- Model accuracy
- Trained model artifact

The experiment was created with the name:

`W4D4-FastAPI-Model-Serving`

Local MLflow tracking was used.

## 8. Model Serialization

The trained pipeline was saved using Joblib at:

`models/breast_cancer.joblib`

The saved model was loaded again and its predictions were compared with the original model predictions.

**Verification Result:** Original and loaded predictions match: True.

## 9. FastAPI Application

The FastAPI application loads the trained Joblib model and provides the following endpoints:

| Method | Endpoint   | Purpose                             |
| ------ | ---------- | ----------------------------------- |
| GET    | `/health`  | Checks API availability             |
| POST   | `/predict` | Returns model predictions           |
| GET    | `/docs`    | Opens interactive API documentation |

The prediction endpoint accepts input containing the required 30 features and returns the predicted class and readable label.

Prediction labels:

- 0 – Malignant
- 1 – Benign

## 10. API Testing

The API was tested using Pytest and FastAPI TestClient.

The following test cases were executed:

1. Health endpoint response.
2. Successful prediction request.
3. Missing required features.
4. Empty instances list.
5. Invalid input data.

**Test Result: 5 passed in 1.82 seconds.**

## 11. Running the Project

### Step 1: Train the model

```powershell
python scripts/train_model.py
```

### Step 2: Start the FastAPI server

```powershell
uvicorn app.main:app --reload
```

### Step 3: Open API documentation

Open the following URL in a browser:

`http://127.0.0.1:8000/docs`

### Step 4: Run API tests

```powershell
python -m pytest tests/test_api.py -v
```

## 12. Results

- Logistic Regression model trained successfully.
- Achieved 98.25% accuracy on the test dataset.
- MLflow experiment logged successfully.
- Model saved and loaded successfully using Joblib.
- FastAPI health endpoint returned HTTP 200.
- Prediction endpoint returned HTTP 200 with a valid prediction.
- All five API tests passed.

## 13. Conclusion

The W4D4 task successfully demonstrates Machine Learning model serving using FastAPI. The trained model was tracked using MLflow, serialized using Joblib, and exposed through REST API endpoints. API functionality and input validation were verified through automated testing.
