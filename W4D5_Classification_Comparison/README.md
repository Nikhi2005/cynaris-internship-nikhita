# W4D5 - Binary Classification: Logistic Regression vs Random Forest

## 1. Project Overview

This practical implements and compares two machine learning classification algorithms: Logistic Regression and Random Forest.

The Breast Cancer Wisconsin dataset from Scikit-learn is used to perform binary classification and evaluate model performance.

## 2. Objectives

- Load and understand a binary classification dataset.
- Split data into training and testing sets.
- Train Logistic Regression and Random Forest classifiers.
- Generate classification reports.
- Evaluate models using accuracy, precision, recall and ROC-AUC.
- Plot confusion matrices and ROC curves.
- Save trained models using Joblib.
- Verify saved models and perform automated testing.

## 3. Dataset

**Dataset:** Breast Cancer Wisconsin

- Total samples: 569
- Features: 30
- Classes: Malignant and Benign
- Training samples: 455
- Testing samples: 114

The dataset was divided using an 80:20 train-test split with stratification.

## 4. Algorithms Used

### Logistic Regression

A classification algorithm that estimates the probability of a sample belonging to a particular class. StandardScaler was used through a Scikit-learn Pipeline.

### Random Forest

An ensemble learning algorithm that combines multiple decision trees to generate classification predictions.

## 5. Model Performance

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------: | ------------: |
| Accuracy  |              98.25% |        95.61% |
| Precision |              98.25% |        95.61% |
| Recall    |              98.25% |        95.61% |
| ROC-AUC   |              0.9954 |        0.9937 |

Logistic Regression achieved better performance than Random Forest on the test dataset.

## 6. Confusion Matrix Results

### Logistic Regression

- True malignant predictions: 41
- True benign predictions: 71
- Incorrect predictions: 2

### Random Forest

- True malignant predictions: 39
- True benign predictions: 70
- Incorrect predictions: 5

## 7. Visualisations

The following evaluation plots are generated and saved in the figures directory:

- Logistic Regression Confusion Matrix
- Logistic Regression ROC Curve
- Random Forest Confusion Matrix
- Random Forest ROC Curve

## 8. Model Saving

Both trained models are saved using Joblib:

- models/logistic_regression.joblib
- models/random_forest.joblib

The saved models were reloaded and their predictions were verified.

## 9. Automated Testing

Three automated tests were executed successfully:

1. Dataset split validation.
2. Model training and prediction validation.
3. Saved model existence and loading validation.

**Test Result: 3 passed**

## 10. Project Structure

```text
W4D5_Classification_Comparison/
│
├── models/
│   ├── logistic_regression.joblib
│   └── random_forest.joblib
│
├── figures/
│   ├── logistic_regression_confusion_matrix.png
│   ├── logistic_regression_roc_curve.png
│   ├── random_forest_confusion_matrix.png
│   └── random_forest_roc_curve.png
│
├── tests/
│   └── test_training.py
│
├── train.py
├── README.md
└── practical-report.md
```

## 11. How to Run

Run the training script:

```bash
python train.py
```

Run automated tests:

```bash
python -m pytest tests -v
```

## 12. Conclusion

Both classification algorithms were successfully trained and evaluated using the Breast Cancer Wisconsin dataset.

Logistic Regression achieved 98.25% accuracy and a ROC-AUC score of 0.9954, while Random Forest achieved 95.61% accuracy and a ROC-AUC score of 0.9937.

The experiment demonstrates the use of classification algorithms, evaluation metrics, visualisation, model serialization and automated testing.
