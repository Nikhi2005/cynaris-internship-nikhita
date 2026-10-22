# W2D4: Train/Test Split, Feature Scaling and Cross-Validation

## Objective

To understand train/test splitting, compare three feature scaling methods, and evaluate a regression model using cross-validation.

## Tools Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Pytest

## Project Structure

- `data/synthetic_data.csv` — synthetic dataset.
- `src/generate_data.py` — generates the dataset.
- `src/scaling_demo.py` — performs train/test splitting and compares scalers.
- `src/cv_pipeline.py` — evaluates regression models using cross-validation.
- `tests/test_scalers.py` — tests scaling behaviour and cross-validation.

## Dataset

The dataset contains 100 records and four columns:

- `age`
- `income`
- `score`
- `target`

The income column includes extreme outliers to demonstrate the behaviour of different scaling techniques.

The target is artificially generated from the input features with random noise.

## Scaling Methods

1. **StandardScaler:** Centres features around zero and scales by standard deviation.
2. **MinMaxScaler:** Scales training feature values to the range 0–1.
3. **RobustScaler:** Uses the median and interquartile range, making it less sensitive to outliers.

## Data Leakage Prevention

The dataset is split before scaling. Each scaler is fitted only on the training data and then used to transform the test data.

During cross-validation, each scaler is included in a Scikit-learn Pipeline. This allows the scaler to be fitted separately within each training fold.

## How to Run

Generate the dataset:

```bash
python src/generate_data.py
```

Compare the scaling methods:

```bash
python src/scaling_demo.py
```

Run cross-validation:

```bash
python src/cv_pipeline.py
```

Run the tests:

```bash
python -m pytest tests/test_scalers.py -v
```

## Results

- Training dataset: 80 records.
- Testing dataset: 20 records.
- Cross-validation: 5 folds.
- All four tests passed.
- The three scalers achieved an average R² score of approximately 0.99 on the synthetic regression task.

## Conclusion

This project demonstrates train/test splitting, feature scaling, cross-validation, and basic automated testing while following practices that help prevent data leakage.
