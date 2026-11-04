@'

# W4D6: Responsible AI — Bias, Fairness & Transparency

## Overview

This task explores gender fairness in an Adult Income classification model using IBM AI Fairness 360 (AIF360), Logistic Regression, SHAP explanations, and Reweighing bias mitigation.

## Dataset

- Dataset: UCI Adult Income Dataset
- Records loaded: 45,222
- Training records: 31,655
- Testing records: 13,567
- Protected attribute: Gender (sex)
- Gender was excluded from predictive features but retained for fairness evaluation.

## Methodology

1. Loaded and prepared the Adult Income dataset using AIF360.
2. Split the dataset into training and testing sets.
3. Trained a baseline Logistic Regression classifier.
4. Evaluated fairness using Disparate Impact, Statistical Parity Difference, and Equal Opportunity Difference.
5. Applied AIF360 Reweighing to the training data and used instance weights during model training.
6. Compared fairness metrics before and after mitigation.
7. Used SHAP LinearExplainer to generate a model explanation summary plot.

## Fairness Results

| Metric                        | Before Reweighing | After Reweighing |
| ----------------------------- | ----------------: | ---------------: |
| Disparate Impact              |          0.343029 |         0.612674 |
| Statistical Parity Difference |         -0.170336 |        -0.083341 |
| Equal Opportunity Difference  |         -0.065959 |         0.159404 |

## Findings

- Disparate Impact improved from 0.343029 to 0.612674.
- Statistical Parity Difference moved closer to zero.
- Equal Opportunity Difference increased in absolute magnitude, indicating that this metric did not improve.
- Disparate Impact remained below the 0.80 reference threshold after mitigation.
- Reweighing improved some fairness measures but did not eliminate the observed disparity.

## SHAP Explainability

SHAP LinearExplainer was used to explain the Logistic Regression model. The generated summary plot shows the contribution of input features to model predictions.

## Indian AI Bias Case Study

The analysis examines gender-related errors in commercial facial-processing tools evaluated on Indian faces.

Reference:
Jain, G., & Parsheera, S. (2021). _Cinderella's Shoe Won't Fit Soundarya: An Audit of Facial Processing Tools on Indian Faces._ arXiv:2112.09326.

https://arxiv.org/abs/2112.09326

## Deliverables

- `W4D6_ResponsibleAI.py` — Complete implementation.
- `SHAP_Summary_Plot.png` — SHAP feature-importance summary.
- `Fairness_Comparison.csv` — Before-and-after fairness metrics.
- `Adult_Income_Model_Card.pdf` — Model card.
- `Indian_AI_Bias_Case_Analysis.txt` — Indian AI bias case analysis.

## Limitations

This is an experimental fairness analysis using the Adult Income dataset. Reweighing did not bring Disparate Impact above 0.80, and fairness metrics showed different outcomes. The model should not be treated as validated for real-world credit or employment decisions.
