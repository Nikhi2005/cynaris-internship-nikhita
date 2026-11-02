# W4D2: Bias-Variance Tradeoff & Regularisation

## 1. Objective

The objective of this practical is to understand the Bias-Variance Tradeoff and perform hyperparameter tuning using GridSearchCV and RandomizedSearchCV with a Random Forest Classifier.

The experiment compares a baseline model with two hyperparameter search strategies using the Wine Recognition dataset.

## 2. Dataset Description

**Dataset:** Wine Recognition Dataset

**Source:** Scikit-learn built-in dataset (`load_wine()`)

Dataset details:

- Total samples: 178
- Number of features: 13
- Target classes: 3
- Training samples: 142
- Testing samples: 36

The dataset contains chemical measurements of wine samples. The target variable represents three different wine classes.

## 3. Concepts Used

### Bias-Variance Tradeoff

- **Bias:** Error caused by a model being too simple to capture patterns in the data.
- **Variance:** Error caused by a model being too sensitive to variations in training data.
- **Overfitting:** A model performs well on training data but poorly on unseen data.
- **Underfitting:** A model fails to learn the important patterns in the data.

### Hyperparameter Tuning

Hyperparameters are settings specified before model training. Tuning them helps identify a suitable model configuration.

The following parameters were explored:

| Hyperparameter      | Description                                                  |
| ------------------- | ------------------------------------------------------------ |
| `n_estimators`      | Number of decision trees in the Random Forest                |
| `max_depth`         | Maximum depth of each decision tree                          |
| `min_samples_leaf`  | Minimum number of samples required at a leaf node            |
| `max_features`      | Number or proportion of features considered for a split      |
| `min_samples_split` | Minimum number of samples required to split an internal node |

### GridSearchCV

GridSearchCV evaluates every specified combination of hyperparameters using cross-validation and selects the combination with the best average validation score.

### RandomizedSearchCV

RandomizedSearchCV evaluates a specified number of randomly selected hyperparameter combinations instead of trying every possible combination.

## 4. Implementation Details

### Step 1: Dataset Loading

Loaded the Wine Recognition dataset using `load_wine()` from Scikit-learn and converted the feature data into a Pandas DataFrame.

### Step 2: Train-Test Split

Split the dataset into:

- 80% training data
- 20% testing data

Used `stratify=y` to preserve class proportions and `random_state=42` for reproducibility.

### Step 3: Baseline Model

Created and trained a default `RandomForestClassifier` and evaluated its test accuracy.

### Step 4: Grid Search

Configured the following parameter grid:

```python
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, None],
    "min_samples_leaf": [1, 2]
}
```

Total combinations: 18

Cross-validation: 5 folds

Total model fits: 18 × 5 = 90

### Step 5: Randomized Search

Configured the following search space:

```python
param_dist = {
    "n_estimators": [10, 25, 50, 100, 200, 300],
    "max_depth": [2, 3, 5, 7, 10, None],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", "log2", None],
    "min_samples_split": [2, 5, 10]
}
```

Number of sampled combinations: 20

Cross-validation: 5 folds

Total model fits: 20 × 5 = 100

### Step 6: Performance Evaluation

Compared the baseline, Grid Search and Randomized Search using:

- Test accuracy
- Best hyperparameters
- Best cross-validation accuracy
- Execution time

Used `time.perf_counter()` to measure the execution time of both search procedures.

## 5. Experimental Results

### Baseline Random Forest

- Test Accuracy: **1.0000 (100%)**

### Grid Search Results

Best parameters:

```python
{
    "max_depth": 3,
    "min_samples_leaf": 1,
    "n_estimators": 200
}
```

- Best CV Score: **0.9791**
- Test Accuracy: **1.0000 (100%)**
- Execution Time: **7.4152 seconds**

### Randomized Search Results

Best parameters:

```python
{
    "n_estimators": 200,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "max_features": "log2",
    "max_depth": 3
}
```

- Best CV Score: **0.9791**
- Test Accuracy: **0.9722 (97.22%)**
- Execution Time: **3.7822 seconds**

### Performance Comparison

| Method                 | Test Accuracy | Execution Time |
| ---------------------- | ------------: | -------------: |
| Baseline Random Forest |          100% |   Not measured |
| Grid Search            |          100% | 7.4152 seconds |
| Randomized Search      |        97.22% | 3.7822 seconds |

## 6. Testing

Automated tests were implemented using `pytest`.

The following checks were performed:

1. Verified successful program execution.
2. Verified Wine dataset dimensions.
3. Verified training and testing sample counts.
4. Verified execution of baseline, Grid Search and Randomized Search.
5. Verified generation of the final performance comparison.

**Test command:**

```powershell
python -m pytest W4D2_BiasVariance_Regularisation\tests\test_hyperparameter_tuning.py -v
```

**Test result:**

```text
5 passed in 13.44s
```

All five automated tests passed successfully.

## 7. Observations and Conclusion

The baseline Random Forest achieved 100% test accuracy. Grid Search also achieved 100%, while Randomized Search achieved 97.22%.

Both search methods obtained the same best cross-validation score of 0.9791, but selected different hyperparameter configurations.

Randomized Search completed faster in this experiment, taking approximately 3.78 seconds compared with 7.42 seconds for Grid Search.

The experiment demonstrates that hyperparameter tuning helps explore different model configurations, but it does not guarantee better test accuracy than a baseline model.

The practical successfully demonstrated the use of GridSearchCV, RandomizedSearchCV, stratified train-test splitting and model performance evaluation.

## 8. Technologies and Libraries

- Python
- Pandas
- Scikit-learn
- Pytest

## 9. Project Structure

```text
W4D2_BiasVariance_Regularisation/
│
├── scripts/
│   └── hyperparameter_tuning.py
│
├── tests/
│   └── test_hyperparameter_tuning.py
│
└── README.md
```

---
