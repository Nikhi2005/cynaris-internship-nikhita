# 1. Objective

The objective of this task is to understand and implement Linear Regression using Scikit-Learn. The task includes training regression models, evaluating their performance using different metrics, and visualizing prediction results.

The following models are implemented:

Linear Regression

Ridge Regression

Lasso Regression

The models are evaluated and compared using MSE, RMSE, MAE, and R².

# 2. Dataset Description

A synthetic housing dataset containing 200 house records is generated using NumPy.

The dataset contains three input features and one target variable.

Input Features

Feature

Description

size_sqft

Size of the house in square feet

bedrooms

Number of bedrooms

age_years

Age of the house in years

Target Variable

Variable

Description

price

Generated house price

The dataset is divided into training and testing sets using an 80:20 split.

Total records: 200

Training records: 160

Testing records: 40

Random state: 42

Note: This is a synthetic dataset created for learning and demonstrating regression techniques. The generated relationships should not be interpreted as verified real-world housing-market relationships.

# 3. Technologies and Libraries

The following Python libraries are used:

NumPy: Numerical operations and data generation.

pandas: Dataset organization and results comparison.

Matplotlib: Data visualization.

Scikit-Learn: Model training, preprocessing, train-test splitting, and evaluation.

# 4. Project Structure

W3D1_LinearRegression/
│
├── linear_regression.py
└── README.md

# 5. How to Run the Program

Activate the existing project virtual environment before running the script.

From the project directory, execute:

python linear_regression.py

The program generates the housing dataset, trains the models, displays evaluation results, prints a comparison table, and generates the required plots.

# 6. Implementation Steps

The implementation follows these steps:

Import the required Python libraries.

Generate a synthetic housing dataset containing 200 records.

Display the first five rows and dataset dimensions.

Separate the input features (X) and target variable (y).

Split the dataset into training and testing sets.

Train the Linear Regression model.

Print the learned coefficients and intercept.

Calculate MSE, RMSE, MAE, and R² for Linear Regression.

Train Ridge and Lasso Regression using standardized input features.

Generate predictions using all three models.

Compare the models using an evaluation table.

Plot actual versus predicted prices.

Plot residuals against predicted prices.

# 7. Models Implemented

Linear Regression

Linear Regression is used as the baseline model. It learns the relationship between the input features and house prices by minimizing the sum of squared prediction errors.

Ridge Regression

Ridge Regression applies L2 regularization, which penalizes large coefficients to help control model complexity.

Lasso Regression

Lasso Regression applies L1 regularization. It can reduce some feature coefficients to zero.

For Ridge and Lasso, StandardScaler is used within a Scikit-Learn pipeline. The scaler is fitted on the training data to avoid data leakage.

All three models use the same training and testing split for comparison.

# 8. Evaluation Metrics

The models are evaluated using four metrics.

Metric

Description

Preferred Result

MSE

Mean Squared Error

Lower

RMSE

Root Mean Squared Error

Lower

MAE

Mean Absolute Error

Lower

R²

Coefficient of Determination

Generally closer to 1

MSE: Measures the average squared difference between actual and predicted prices.

RMSE: Measures prediction error in the target variable's units.

MAE: Measures the average absolute prediction error.

R²: Indicates how much of the variation in the target variable is explained by the model on the test set.

# 9. Linear Regression Results

The Linear Regression model produced the following test-set results:

Metric

Result

MSE

482645007.4603

RMSE

21969.1831

MAE

17755.6648

R² Score

0.952253

The R² score of approximately 0.9523 indicates that the model explains about 95.23% of the variation in house prices in this test set. It does not mean that 95.23% of individual predictions are correct.

Learned Coefficients

Feature

Coefficient

size_sqft

151.6919

bedrooms

19098.3023

age_years

-1041.4745

Intercept

1409.5570

The model learned positive coefficients for house size and number of bedrooms and a negative coefficient for house age. These relationships reflect the synthetic data-generation formula.

# 10. Comparison of Regression Models

The following results were obtained using the same test dataset.

Model

MSE

RMSE

MAE

R² Score

Linear Regression

4.826450e+08

21969.1831

17755.6648

0.9523

Ridge Regression

4.786349e+08

21877.7255

17756.9694

0.9527

Lasso Regression

4.826288e+08

21968.8133

17755.7150

0.9523

Observation

Ridge Regression performed slightly better on this particular test split, with the lowest MSE and RMSE and the highest R² score.

The differences between the models are small. These results apply to the generated dataset and the selected train-test split.

# 11. Data Visualizations

Actual vs. Predicted Prices

This scatter plot compares actual house prices with prices predicted by the Linear Regression model.

The dashed diagonal line represents perfect predictions. Points closer to the line indicate predictions closer to the actual values.

Residuals Plot

Residuals are calculated as:

Residual = Actual Price - Predicted Price

The residuals plot displays prediction errors against predicted house prices. The horizontal zero line represents predictions with no error.

Positive residuals indicate underprediction, while negative residuals indicate overprediction.

The plots help inspect model predictions and identify possible patterns in the errors.

# 12. Conclusion##

The W3D1 task successfully demonstrates a regression workflow using Scikit-Learn.

Linear Regression, Ridge Regression, and Lasso Regression were trained and evaluated using a synthetic housing dataset. The implementation calculates MSE, RMSE, MAE, and R², prints model coefficients, compares model performance, and generates actual-versus-predicted and residuals plots.

Ridge Regression achieved a slightly better result on the selected test split. The task provided practical experience with regression models, regularization, model evaluation, and data visualization.
