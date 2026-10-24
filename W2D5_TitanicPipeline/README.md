# W2D5: End-to-End Titanic Preprocessing Pipeline

## 1. Project Objective

Build an end-to-end preprocessing pipeline using the Titanic dataset. The pipeline performs exploratory data analysis (EDA), handles missing values, encodes categorical features, scales numerical features, splits the dataset, and exports ML-ready CSV files.

## 2. Dataset

The project uses the Titanic passenger dataset containing 891 rows and 12 columns.

Important missing values in the original dataset:

- Age: 177 missing values
- Cabin: 687 missing values
- Embarked: 2 missing values

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Pytest

## 4. Preprocessing Steps

1. Load the Titanic dataset and inspect its shape, missing values, and survival counts.
2. Separate the target column, `Survived`, from the input features.
3. Exclude `PassengerId`, `Name`, `Ticket`, and `Cabin` from the input features.
4. Fill missing numerical values using the median.
5. Fill missing categorical values using the most frequent value.
6. Scale numerical features using `StandardScaler`.
7. Encode categorical features using `OneHotEncoder`.
8. Split the dataset into training and testing sets using an 80:20 ratio.
9. Fit preprocessing transformations on the training set only to avoid data leakage.
10. Export the processed datasets as CSV files.

## 5. Output Files

The following files are saved in `data/processed/`:

- `X_train.csv`: Processed training features (712 rows).
- `X_test.csv`: Processed testing features (179 rows).
- `y_train.csv`: Training target labels.
- `y_test.csv`: Testing target labels.

The processed training and testing feature datasets contain 12 features each and no missing values.

## 6. Run the Project

Run the preprocessing pipeline from the Week 2 workspace:

```bash
python W2D5_TitanicPipeline/src/preprocessing.py
```

Run the automated tests:

```bash
python -m pytest W2D5_TitanicPipeline/tests -v
```

## 7. Test Results

All four automated tests passed.

The tests verify:

- All four output files exist.
- Training and testing row counts match their target labels.
- Processed feature datasets contain no missing values.
- Training and testing datasets have identical feature columns.

## 8. Scope

This project focuses on data preprocessing and exporting ML-ready features. It does not train or evaluate a classification model.
