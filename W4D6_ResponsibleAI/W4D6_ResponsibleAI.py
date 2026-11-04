
# ==============================================================
# W4D6: RESPONSIBLE AI — BIAS, FAIRNESS & TRANSPARENCY
# Company: Cynaris Solutions Private Limited
#
# Purpose:
# 1. Load the UCI Adult Income dataset using AIF360.
# 2. Train a Logistic Regression classification model.
# 3. Measure gender fairness using AIF360 metrics.
# 4. Apply Reweighing to mitigate bias.
# 5. Generate a SHAP summary plot.
# 6. Generate a one-page Model Card PDF.
# 7. Save an Indian AI bias case analysis.
# ==============================================================

import os
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. BASIC CONFIGURATION
# --------------------------------------------------------------

np.random.seed(42)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 65)
print("W4D6: RESPONSIBLE AI ANALYSIS")
print("=" * 65)


# --------------------------------------------------------------
# 2. LOAD THE ADULT INCOME DATASET
# --------------------------------------------------------------

print("\n[1] Loading Adult Income Dataset...")

# Compatibility fix:
# AIF360's older dataset loader expects traditional NumPy-compatible
# object/string columns. Pandas 3 uses StringDtype by default.

original_read_csv = pd.read_csv

def compatible_read_csv(*args, **kwargs):
    df = original_read_csv(*args, **kwargs)

    for column in df.columns:
        if isinstance(df[column].dtype, pd.StringDtype):
            df[column] = df[column].astype(object)

    return df

pd.read_csv = compatible_read_csv

# Load the dataset
adult_dataset = AdultDataset()

print("Dataset loaded successfully!")
print("Dataset shape:", adult_dataset.features.shape)
print("Protected attributes:", adult_dataset.protected_attribute_names)
print("Available labels:", adult_dataset.label_names)

# Gender is the protected attribute.
# Male = 1 and Female = 0.

protected_attribute = "sex"

privileged_groups = [{protected_attribute: 1.0}]
unprivileged_groups = [{protected_attribute: 0.0}]

# --------------------------------------------------------------
# 3. SPLIT THE DATASET INTO TRAINING AND TESTING DATA
# --------------------------------------------------------------

print("\n[2] Splitting dataset...")

train_dataset, test_dataset = adult_dataset.split(
    [0.70],
    shuffle=True
)

print("Training records:", train_dataset.features.shape[0])
print("Testing records:", test_dataset.features.shape[0])

# Remove gender from predictive features.
# Gender remains available as a protected attribute for fairness
# evaluation. Removing gender alone does not eliminate proxy bias.

feature_indices = [
    i for i, name in enumerate(adult_dataset.feature_names)
    if name != protected_attribute
]

feature_names = [
    adult_dataset.feature_names[i] for i in feature_indices
]

X_train = train_dataset.features[:, feature_indices]
X_test = test_dataset.features[:, feature_indices]

y_train = train_dataset.labels.ravel()
y_test = test_dataset.labels.ravel()

X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

print("Predictive features:", len(feature_names))


# --------------------------------------------------------------
# 4. TRAIN BASELINE LOGISTIC REGRESSION MODEL
# --------------------------------------------------------------

print("\n[3] Training baseline Logistic Regression...")

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(X_train, y_train)

baseline_predictions = baseline_model.predict(X_test)

print("Baseline model trained successfully!")


# --------------------------------------------------------------
# 5. CALCULATE BASELINE FAIRNESS METRICS
# --------------------------------------------------------------

print("\n[4] Evaluating baseline fairness...")

baseline_predicted_dataset = test_dataset.copy(deepcopy=True)

baseline_predicted_dataset.labels = baseline_predictions.reshape(-1, 1)

baseline_metric = ClassificationMetric(
    test_dataset,
    baseline_predicted_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

baseline_di = baseline_metric.disparate_impact()
baseline_spd = baseline_metric.statistical_parity_difference()
baseline_eod = baseline_metric.equal_opportunity_difference()

print("\nBASELINE FAIRNESS RESULTS")
print("-" * 40)
print(f"Disparate Impact: {baseline_di:.4f}")
print(f"Statistical Parity Difference: {baseline_spd:.4f}")
print(f"Equal Opportunity Difference: {baseline_eod:.4f}")

print("\n80% Rule:")
if baseline_di < 0.80:
    print("Potential gender bias detected.")
else:
    print("Disparate impact is not below the 0.80 threshold.")

# A ratio near 1 indicates parity.
# A ratio below 0.80 is a potential adverse-impact warning.


# --------------------------------------------------------------
# 6. APPLY REWEIGHING BIAS MITIGATION
# --------------------------------------------------------------

print("\n[5] Applying AIF360 Reweighing...")

reweighing = Reweighing(
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

reweighted_train_dataset = reweighing.fit_transform(
    train_dataset
)

# Use the generated instance weights during model training.

instance_weights = reweighted_train_dataset.instance_weights

mitigated_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

mitigated_model.fit(
    X_train,
    y_train,
    sample_weight=instance_weights
)

mitigated_predictions = mitigated_model.predict(X_test)

print("Reweighing applied successfully!")


# --------------------------------------------------------------
# 7. EVALUATE FAIRNESS AFTER REWEIGHING
# --------------------------------------------------------------

print("\n[6] Evaluating fairness after Reweighing...")

mitigated_predicted_dataset = test_dataset.copy(deepcopy=True)

mitigated_predicted_dataset.labels = mitigated_predictions.reshape(-1, 1)

mitigated_metric = ClassificationMetric(
    test_dataset,
    mitigated_predicted_dataset,
    unprivileged_groups=unprivileged_groups,
    privileged_groups=privileged_groups
)

mitigated_di = mitigated_metric.disparate_impact()
mitigated_spd = mitigated_metric.statistical_parity_difference()
mitigated_eod = mitigated_metric.equal_opportunity_difference()

print("\nCOMPARISON OF FAIRNESS METRICS")
print("-" * 65)

comparison = pd.DataFrame({
    "Metric": [
        "Disparate Impact",
        "Statistical Parity Difference",
        "Equal Opportunity Difference"
    ],
    "Before Reweighing": [
        baseline_di,
        baseline_spd,
        baseline_eod
    ],
    "After Reweighing": [
        mitigated_di,
        mitigated_spd,
        mitigated_eod
    ]
})

print(comparison.to_string(index=False))

print("\nInterpretation:")
print("Disparate Impact closer to 1 indicates better parity.")
print("Parity differences closer to 0 indicate better parity.")
print("Mitigation results should be evaluated alongside model performance.")


# --------------------------------------------------------------
# 8. GENERATE SHAP EXPLANATIONS
# --------------------------------------------------------------

print("\n[7] Generating SHAP summary plot...")

try:
    # Use a small background sample to explain the model.

    background = shap.sample(
        X_train_df,
        min(100, len(X_train_df)),
        random_state=42
    )

    explanation_data = X_test_df.sample(
        n=min(500, len(X_test_df)),
        random_state=42
    )

    explainer = shap.LinearExplainer(
        baseline_model,
        background
    )

    shap_values = explainer.shap_values(explanation_data)

    # Handle different SHAP output formats.

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    plt.figure(figsize=(10, 7))

    shap.summary_plot(
        shap_values,
        explanation_data,
        show=False
    )

    plt.title("SHAP Summary Plot - Adult Income Model")

    plt.tight_layout()

    shap_path = os.path.join(
        OUTPUT_DIR,
        "SHAP_Summary_Plot.png"
    )

    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("SHAP plot saved:", shap_path)

except Exception as error:
    print("SHAP explanation could not be generated.")
    print("Error:", error)


# --------------------------------------------------------------
# 9. SAVE FAIRNESS COMPARISON
# --------------------------------------------------------------

print("\n[8] Saving fairness comparison...")

comparison_path = os.path.join(
    OUTPUT_DIR,
    "Fairness_Comparison.csv"
)

comparison.to_csv(comparison_path, index=False)

print("Fairness comparison saved:", comparison_path)


# --------------------------------------------------------------
# 10. CREATE ONE-PAGE MODEL CARD PDF
# --------------------------------------------------------------

print("\n[9] Creating Model Card PDF...")

model_card_text = f"""
MODEL CARD: ADULT INCOME CLASSIFICATION

1. MODEL DETAILS
Model: Logistic Regression
Dataset: UCI Adult Income Dataset
Framework: Scikit-learn and IBM AI Fairness 360
Protected Attribute: Gender

2. INTENDED USE
Experimental binary classification of income categories
using demographic and employment-related features.
This model is intended for educational fairness analysis only.

3. MODEL LIMITATIONS
The dataset represents a particular historical population.
Its predictions should not be treated as objective measures
of individual ability, worth, or financial reliability.

4. EVALUATION DATA
The dataset was divided into 70% training and 30% testing data.
The same test dataset was used for both models.

5. FAIRNESS EVALUATION

Baseline Model:
Disparate Impact: {baseline_di:.4f}
Statistical Parity Difference: {baseline_spd:.4f}
Equal Opportunity Difference: {baseline_eod:.4f}

After Reweighing:
Disparate Impact: {mitigated_di:.4f}
Statistical Parity Difference: {mitigated_spd:.4f}
Equal Opportunity Difference: {mitigated_eod:.4f}

6. MITIGATION
AIF360 Reweighing was applied to training instances.
The generated instance weights were used during
Logistic Regression model training.

7. EXPLAINABILITY
SHAP LinearExplainer was used to identify the contribution
of input features to model predictions.

8. ETHICAL CONSIDERATIONS
Gender-based disparities may remain even when gender is
excluded from predictive features because other variables
can act as proxies. Fairness metrics do not prove that a
model is completely unbiased.

9. CAVEATS
This is an experimental educational model.
It has not been validated for real-world employment,
credit, lending, or other high-impact decisions.
"""

pdf_path = os.path.join(
    OUTPUT_DIR,
    "Adult_Income_Model_Card.pdf"
)

with PdfPages(pdf_path) as pdf:

    fig = plt.figure(figsize=(8.27, 11.69))

    fig.text(
        0.08,
        0.96,
        "CYNARIS SOLUTIONS — MODEL CARD",
        fontsize=15,
        fontweight="bold",
        va="top"
    )

    wrapped_text = "\n".join(
        textwrap.fill(line, width=88) if line else ""
        for line in model_card_text.strip().split("\n")
    )

    fig.text(
        0.08,
        0.91,
        wrapped_text,
        fontsize=8.5,
        va="top",
        linespacing=1.5,
        family="monospace"
    )

    plt.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("Model Card PDF saved:", pdf_path)


# --------------------------------------------------------------
# 11. INDIAN AI BIAS CASE ANALYSIS
# --------------------------------------------------------------

print("\n[10] Saving Indian AI Bias Case Analysis...")

india_case = """
INDIAN AI BIAS CASE STUDY

Title:
Cinderella's Shoe Won't Fit Soundarya: An Audit of Facial
Processing Tools on Indian Faces.

Researchers:
Gaurav Jain and Smriti Parsheera.

Case Summary:

The study examined four commercial facial-processing tools
using 32,184 images of Indian electoral candidates obtained
from publicly available Election Commission of India data.

The audit identified differences in gender-classification
errors between male and female faces. Female faces had
higher classification error rates than male faces, with the
highest reported female error rate reaching 14.68%.

This case demonstrates how facial-processing systems
developed or evaluated using insufficiently representative
data can perform unevenly across demographic groups.
Such differences can create unfair outcomes when facial
AI is used in public services, identity systems, or other
important applications.

Root Cause and Risk:

The audit highlights unequal performance across Indian
gender groups. Limited representation and insufficient
subgroup testing are possible contributing risks; the
reported error difference alone does not establish one
specific technical cause.

Proposed Improvements:

1. Use diverse and representative Indian facial datasets.
2. Evaluate accuracy separately across gender groups.
3. Conduct independent bias audits before deployment.
4. Improve transparency about model limitations.
5. Monitor subgroup performance after deployment.

Conclusion:

AI systems should be evaluated across the populations
they are intended to serve. Accuracy alone is insufficient;
fairness, transparency, and accountability are also essential.

Reference:
Jain, G. and Parsheera, S.
"Cinderella's Shoe Won't Fit Soundarya: An Audit of Facial
Processing Tools on Indian Faces."
"""

case_path = os.path.join(
    OUTPUT_DIR,
    "Indian_AI_Bias_Case_Analysis.txt"
)

with open(case_path, "w", encoding="utf-8") as file:
    file.write(india_case.strip())

print("Indian AI case analysis saved:", case_path)


# --------------------------------------------------------------
# 12. FINAL SUMMARY
# --------------------------------------------------------------

print("\n" + "=" * 65)
print("W4D6 ANALYSIS COMPLETED")
print("=" * 65)

print("\nGenerated deliverables:")

print("1. Fairness_Comparison.csv")
print("2. Adult_Income_Model_Card.pdf")
print("3. Indian_AI_Bias_Case_Analysis.txt")

if os.path.exists(os.path.join(OUTPUT_DIR, "SHAP_Summary_Plot.png")):
    print("4. SHAP_Summary_Plot.png")

print("\nAll available outputs are saved in:")
print(OUTPUT_DIR)

print("\nResponsible AI analysis completed!")
