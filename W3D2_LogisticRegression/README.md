# Objective

To implement Logistic Regression using Scikit-Learn for binary classification, evaluate its performance using classification metrics, and visualize the results using a confusion matrix and ROC curve.

# Dataset

The Breast Cancer Wisconsin dataset is loaded from Scikit-Learn.

- Total samples: 569
- Total features: 30
- Malignant samples: 212
- Benign samples: 357
- Classification type: Binary classification

# Class labels:

- 0 — Malignant
- 1 — Benign

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

# Implementation

1. Loaded the Breast Cancer Wisconsin dataset.
2. Divided the data into training and testing sets using an 80:20 split.
3. Used stratified sampling to preserve the class distribution.
4. Standardized features using StandardScaler.
5. Trained a Logistic Regression model.
6. Displayed the model intercept and the five features with the largest absolute coefficients.
7. Evaluated the model using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Visualized the confusion matrix and ROC curve.

# Results

Metric Score
Accuracy 0.9825
Precision 0.9861
Recall 0.9861
F1-score 0.9861
ROC-AUC 0.9954

Confusion Matrix:
[[41, 1],
        [ 1, 71]]

The model correctly classified 41 malignant samples and 71 benign samples in the test set. It misclassified one sample from each class.

# Important Features

The five features with the largest absolute coefficients were:

1. Worst texture
2. Radius error
3. Worst concave points
4. Worst area
5. Worst radius

These coefficients are based on standardized features.

# How to Run:

    Install the required libraries:
    python -m pip install numpy pandas scikit-learn matplotlib seaborn
    Run the program from the Week 3 workspace:
    python W3D2_LogisticRegression/logistic_regression.py

# Conclusion

    The Logistic Regression model achieved 98.25% accuracy and a ROC-AUC of 0.9954 on the test set. The confusion matrix shows only two misclassifications. These results indicate strong performance on this particular test split, but they do not guarantee the same performance on new datasets.

## Multiclass Classification Using Iris

# Objective

Compare One-vs-Rest (OvR) and Multinomial / Softmax Logistic Regression on a three-class classification problem.

# Dataset

The Iris dataset was loaded using Scikit-Learn.

Total samples: 150

Total features: 4

Classes: Setosa, Versicolor, Virginica

Training/testing split: 80% / 20%

Test samples: 30

# Model Comparison

Metric One-vs-Rest Multinomial / Softmax
Accuracy 0.9000 0.9333
Macro Precision 0.9024 0.9333
Macro Recall 0.9000 0.9333
Macro F1-score 0.8997 0.9333

# Interpretation

Multinomial / Softmax Logistic Regression achieved the higher score for all four metrics on this test split. It correctly classified 28 out of 30 test samples, compared with 27 out of 30 for One-vs-Rest.

These results apply to the selected train/test split and do not establish that one approach will always be superior.

# Conclusion

The implementation demonstrates both binary and multiclass classification using Scikit-Learn. The binary model was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, a confusion matrix, and an ROC curve. The Iris experiment compared One-vs-Rest and Multinomial / Softmax Logistic Regression using accuracy and macro-averaged precision, recall, and F1-score.
