# W2D3: Handling Imbalanced Data — SMOTE

## Objective

Understand class imbalance and use SMOTE to improve
minority-class learning during model training.

## Dataset

A reproducible synthetic e-commerce dataset containing
200 orders and a binary `returned` target.

- 0: Not returned
- 1: Returned

The expected generation probabilities are 85% and 15%.
Actual counts may vary because labels are generated randomly.

## Features

- City-tier encoding
- Date and time extraction
- Weekend and night indicators
- Value per item
- High-value order indicator
- Customer age groups

## SMOTE Workflow

1. Load the dataset.
2. Create engineered features.
3. Split into training and testing sets.
4. Preprocess features using the training data.
5. Apply SMOTE to training data only.
6. Train Logistic Regression.
7. Evaluate predictions on the original test data.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Setup

Install dependencies:

    pip install -r requirements.txt

## Generate the Dataset and Features

From the `W2D3_SMOTE` folder:

    python src/feature_engineering.py

## Run the SMOTE Pipeline

From the `Cynaris_Week2` folder:

    python -m W2D3_SMOTE.src.smote_pipeline

## Run Tests

From the `Cynaris_Week2` folder:

    python -m pytest W2D3_SMOTE\tests -v
