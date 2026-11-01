# W4D1: Model Evaluation Metrics

**Cynaris Solutions – AI/ML Internship**

## Overview

This task implements model evaluation techniques using the Breast Cancer Wisconsin Diagnostic dataset.

The objective is to evaluate classification models using Stratified 5-Fold Cross-Validation, multiple performance metrics, and learning curves.

## Objectives

- Understand Stratified K-Fold Cross-Validation.
- Evaluate models using Accuracy, Precision, Recall, F1-score and ROC-AUC.
- Compare Logistic Regression, Decision Tree and Random Forest models.
- Understand overfitting and generalisation using learning curves.
- Prevent data leakage using Scikit-learn Pipelines.

## Dataset

The Breast Cancer Wisconsin Diagnostic dataset from Scikit-learn is used.

- Samples: 569
- Features: 30
- Malignant cases: 212
- Benign cases: 357

The target labels are transformed so that malignant cases are represented by 1 and benign cases by 0.

## Project Structure

```text
W4D1_Model_Evaluation_Metrics/
├── scripts/
│   └── model_evaluation.py
├── tests/
│   └── test_model_evaluation.py
├── outputs/
└── README.md
```

## Implementation

1. Loaded and prepared the dataset.
2. Created Stratified 5-Fold Cross-Validation.
3. Compared three Logistic Regression configurations, two Decision Tree configurations and one Random Forest model.
4. Evaluated models using five classification metrics.
5. Generated a Random Forest learning curve.
6. Added tests for dataset preparation, fold configuration and model setup.

## Best Model

Logistic Regression (C=1) achieved the following mean cross-validation scores:

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 0.9737 |
| Precision | 0.9863 |
| Recall    | 0.9436 |
| F1-score  | 0.9633 |
| ROC-AUC   | 0.9953 |

## Learning Curve Findings

The Random Forest training accuracy remained close to 1.0, while validation accuracy improved as training size increased.

The remaining training-validation gap indicates some overfitting.

## Data Leakage Prevention

StandardScaler is included in a Pipeline with Logistic Regression. This ensures that scaling is fitted independently within each cross-validation training fold.

Tree-based models are evaluated without scaling.

## How to Run

```bash
python scripts/model_evaluation.py
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Technologies Used

- Python
- NumPy
- Scikit-learn
- Pytest

## Conclusion

This task demonstrates how cross-validation and multiple evaluation metrics provide a more reliable understanding of classification model performance than accuracy alone. Learning curves help identify possible overfitting and assess generalisation.
