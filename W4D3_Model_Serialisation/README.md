# W4D3 – Model Serialisation using Pickle and Joblib

## 1. Overview

This practical demonstrates model serialisation using Pickle and Joblib in Python.

A Logistic Regression classification model is trained using the Breast Cancer dataset available in Scikit-learn.

The trained model is saved using both Pickle and Joblib. Both saved models are then loaded, and their predictions are compared with the original model to verify consistency.

## 2. Objectives

- Load a built-in Scikit-learn dataset.
- Train a Logistic Regression classification model.
- Save the trained model using Pickle.
- Save the same trained model using Joblib.
- Load both serialized models.
- Verify prediction consistency.
- Compare the sizes of the saved model files.
- Perform automated testing using Pytest.

## 3. Technologies Used

- Python
- Scikit-learn
- NumPy
- Pickle
- Joblib
- Pytest

## 4. Project Structure

```text
W4D3_Model_Serialisation/
│
├── scripts/
│   ├── __init__.py
│   └── model_serialisation.py
│
├── tests/
│   ├── __init__.py
│   └── test_model_serialisation.py
│
├── artifacts/
│   ├── model_pickle.pkl
│   └── model_joblib.joblib
│
└── README.md
```

## 5. Implementation

### Dataset

The Breast Cancer Wisconsin dataset from Scikit-learn is used.

- Total samples: 569
- Features: 30
- Training samples: 455
- Testing samples: 114

The dataset is divided into 80% training data and 20% testing data.

### Model Training

Logistic Regression is used as the classification algorithm.

The model is trained using the training dataset with the `fit()` method.

### Model Serialisation

The trained model is saved in two formats:

**Pickle**

- `pickle.dump()` saves the model.
- `pickle.load()` loads the saved model.

**Joblib**

- `joblib.dump()` saves the model.
- `joblib.load()` loads the saved model.

### Prediction Verification

Predictions are generated using:

1. The original trained model.
2. The model loaded using Pickle.
3. The model loaded using Joblib.

NumPy's `array_equal()` function is used to compare the predictions.

Both loaded models produced predictions identical to the original model.

## 6. Execution

Run the implementation:

```powershell
python scripts/model_serialisation.py
```

Run the automated tests:

```powershell
python -m pytest tests\test_model_serialisation.py -v
```

## 7. Results

The implementation completed successfully.

Prediction verification:

- Original vs Pickle: True
- Original vs Joblib: True

Saved model file sizes from the execution:

| Format |   File Size |
| ------ | ----------: |
| Pickle |   907 bytes |
| Joblib | 1,103 bytes |

Both serialisation methods successfully saved and restored the trained model.

## 8. Automated Testing

Six automated tests were executed using Pytest.

The tests verify:

1. Pickle model file creation.
2. Joblib model file creation.
3. Successful Pickle model loading.
4. Successful Joblib model loading.
5. Prediction consistency.
6. Non-zero saved model file sizes.

**Test Result: 6 passed**

## 9. Security Considerations

Pickle and Joblib files should only be loaded from trusted sources.

Loading untrusted serialized files can execute malicious code.

Therefore, model files should be verified and obtained from trusted sources before loading.

## 10. Conclusion

This practical demonstrates how a trained machine learning model can be saved and reused through Pickle and Joblib.

Both serialized models were successfully loaded, and their predictions matched the original model's predictions.

All six automated tests passed successfully.
