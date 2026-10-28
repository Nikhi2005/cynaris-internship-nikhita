# W3D3: Decision Trees and Random Forests

## Overview

This task explores Decision Tree and Random Forest classifiers using the Iris dataset. It compares tree depths, Gini impurity and entropy, and parameters that help control overfitting.

## Objectives

- Train Decision Tree classifiers with different maximum depths.
- Compare Gini impurity and entropy.
- Evaluate `min_samples_leaf` and `min_samples_split`.
- Visualize the selected Decision Tree and feature importance.
- Compare Decision Tree and Random Forest performance.
- Track experiment parameters, metrics, and artifacts using MLflow.
- Use a LangGraph workflow to summarize the final model comparison.

## Dataset

The scikit-learn Iris dataset contains 150 samples, four numerical features, and three flower classes: setosa, versicolor, and virginica.

The data is split into 80% training and 20% testing data using `random_state=42` and stratification. Five-fold cross-validation is applied to the training data.

## Implementation

All model experiments are implemented in `decision_trees.py`.

The program evaluates:

- Tree depths: 1, 2, 3, 5, and unrestricted.
- Splitting criteria: Gini and entropy.
- Overfitting controls: `min_samples_leaf` and `min_samples_split`.
- A depth-3 Decision Tree and a 100-estimator Random Forest.

## Results

| Metric                  | Decision Tree (depth 3) | Random Forest |
| ----------------------- | ----------------------: | ------------: |
| Training accuracy       |                  98.33% |       100.00% |
| Testing accuracy        |                  96.67% |        90.00% |
| Mean 5-fold CV accuracy |                  93.33% |        95.00% |

These results are specific to the selected data split. The Random Forest has higher mean cross-validation accuracy, while the selected Decision Tree has higher test accuracy.

## Output Files

- `decision_tree.png` — visual representation of the selected tree.
- `feature_importance.png` — feature-importance chart.
- `decision_tree_rules.txt` — learned decision rules.
- `random_forest_classification_report.txt` — Random Forest evaluation report.

## MLflow Tracking

Experiment name: `Cynaris-W3D3-Decision-Trees`

The program logs model parameters and evaluation metrics. It also logs the generated plots, decision rules, and Random Forest classification report as artifacts.

Local MLflow tracking files are kept outside the task deliverables.

## Requirements

Install the packages listed in `requirements.txt` in an appropriate Python environment.

## Run

From the Week 3 workspace, activate the virtual environment and run:

```powershell
python W3D3_DecisionTrees\decision_trees.py
```

The program prints the model evaluations, saves visualizations and reports, records MLflow experiments, and executes the LangGraph comparison workflow.
