# W2D2: Missing Data Imputation

## Objective

Demonstrate two techniques for handling missing numerical values using Scikit-learn.

## Files

- `simple_imputer.py` — fills missing numerical values using the median strategy.
- `knn_imputation.py` — fills missing numerical values using K-Nearest Neighbors (KNN).

## Dataset

The example dataset contains five rows and three numerical columns:

- `age`
- `income`
- `score`

Each column initially contains one missing value (`NaN`).

## Method 1: SimpleImputer

- Strategy: `median`
- Missing values after imputation: 0 in all three columns.
- Filled values:
  - `age`: 27.5
  - `income`: 45000
  - `score`: 75

## Method 2: KNNImputer

- Number of neighbours: `2`
- Missing values after imputation: 0 in all three columns.
- Filled values:
  - `age`: 27.5
  - `income`: 35000
  - `score`: 80

## Result

Both methods successfully filled all missing values in the example dataset.
The estimated values differ because the methods use different imputation strategies.

**Note:** KNN imputation can be affected by differences in feature scales. Scaling may be necessary when applying it to real-world datasets.

## Verification

Both Python scripts were executed successfully, and the terminal outputs confirmed that no missing values remained after imputation.
