# W3D5: Hyperparameter Tuning — GridSearchCV

## Overview

This project explores six machine-learning classification algorithms using the Wine dataset. It compares their performance and investigates how changing hyperparameters affects model results.

## Objectives

- Compare six classification algorithms.
- Evaluate models using test accuracy, macro-F1 score, and five-fold cross-validation.
- Study the effect of different K values in K-Nearest Neighbors (KNN).
- Compare RBF and linear Support Vector Machine (SVM) configurations.
- Use GridSearchCV to identify a suitable Random Forest hyperparameter combination.
- Track experiments using MLflow.

## Dataset

The built-in scikit-learn Wine dataset is used.

- Samples: 178
- Features: 13
- Classes: 3
- Training split: 80%
- Testing split: 20%
- Random state: 42

## Algorithms

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Classifier (SVM)
5. K-Nearest Neighbors (KNN)
6. Gaussian Naive Bayes

## Experiments

- Compare the six baseline classifiers.
- Evaluate KNN with K values of 1, 3, 5, 7, 11, 15, and 21.
- Compare an RBF SVM with C=100 and a linear SVM.
- Run GridSearchCV on a Random Forest pipeline using 12 hyperparameter combinations and five-fold stratified cross-validation.
- Record test accuracy, macro-F1, and cross-validation accuracy.

## Results

The GridSearchCV experiment selected the following parameters:

- `n_estimators`: 200
- `max_depth`: None
- `min_samples_split`: 2
- Best cross-validation accuracy: 0.9791
- Test accuracy: 1.0000
- Macro-F1 score: 1.0000

These results apply to the current dataset split and do not guarantee the same performance on unseen datasets.

## Tools and Libraries

- Python
- scikit-learn
- NumPy
- Pandas
- Matplotlib
- MLflow
- pytest

## Project Structure

```text
W3D5_HyperparameterTuning/
├── wine_hyperparameter_tuning.py
├── README.md
├── tests/
│   └── test_w3d5.py
└── outputs/
    └── wine_tuning_results.csv
```

## How to Run

Run the main program from the Week 3 project root:

```powershell
python W3D5_HyperparameterTuning\wine_hyperparameter_tuning.py
```

Run the tests:

```powershell
python -m pytest W3D5_HyperparameterTuning\tests\test_w3d5.py -v
```

## Test Results

All three tests passed:

- Pipeline fit and predict
- All seven KNN values
- GridSearchCV best-parameter validation

## Experiment Tracking

MLflow experiment name: `Cynaris-W3D5-Hyperparameter-Tuning`

The experiment records selected parameters, evaluation metrics, and the CSV results artifact.
