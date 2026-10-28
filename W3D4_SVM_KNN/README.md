# W3D4: Decision Tree and Random Forest Experiments

## Overview

This practical compares a single Decision Tree with Random Forest classifiers using a synthetic binary-classification dataset.

**Lesson title:** SVM & KNN — When to Use What

**Scope note:** The supplied lesson title refers to SVM and KNN, but the provided playground code and resources focus on Decision Trees and Random Forests. This implementation follows the supplied practical material.

## Dataset and Setup

- Dataset: Generated using `sklearn.datasets.make_classification`
- Samples: 1,000
- Features: 20
- Informative features: 10
- Redundant features: 5
- Train/test split: 80% / 20%
- Random state: 42

## Experiments

1. Baseline Decision Tree.
2. Baseline Random Forest with 100 trees and out-of-bag scoring.
3. Random Forest with 500 trees.
4. Random Forest with `max_depth=5`.

Each experiment reports training accuracy, test accuracy, and 5-fold cross-validation accuracy. Random Forest experiments also report the OOB score.

## Results

| Experiment                | Train Accuracy | Test Accuracy | 5-Fold CV |
| ------------------------- | -------------: | ------------: | --------: |
| Decision Tree baseline    |         1.0000 |        0.8050 |    0.8260 |
| Random Forest baseline    |         1.0000 |        0.9100 |    0.9330 |
| Random Forest — 500 trees |         1.0000 |        0.9050 |    0.9360 |
| Random Forest — depth 5   |         0.9575 |        0.8800 |    0.9120 |

Baseline Random Forest OOB score: **0.9200**

## Feature Importance

The baseline Random Forest feature-importance plot is saved at:

`outputs/rf_feature_importance.png`

## MLflow

Experiment name: `Cynaris-W3D4-Tree-Ensemble`

The script logs model parameters and evaluation metrics. The feature-importance plot is logged as an artifact.

## Testing

Run the automated test from the Week 3 workspace:

```powershell
python -m pytest W3D4_SVM_KNN\tests\test_w3d4.py -v
```

Observed result: **1 test passed**.

## Files

- `decision_tree_random_forest.py` — model training, experiments, evaluation, and MLflow tracking.
- `tests/test_w3d4.py` — automated Random Forest training and metric checks.
- `outputs/rf_feature_importance.png` — feature-importance visualization.
- `evidence/` — execution evidence.

## Conclusion

In this run, the baseline Random Forest achieved higher test accuracy than the single Decision Tree. Increasing the number of trees slightly improved cross-validation accuracy but did not improve test accuracy. Limiting tree depth reduced the gap between training and test performance, although its test accuracy was lower in this experiment.

These findings describe the current dataset and random split; results may differ on other datasets.
