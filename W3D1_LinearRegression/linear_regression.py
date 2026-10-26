import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make the generated data reproducible
np.random.seed(42)

# Create data for 200 houses
number_of_houses = 200

# Generate house features
size_sqft = np.random.randint(500, 3000, number_of_houses)
bedrooms = np.random.randint(1, 6, number_of_houses)
age_years = np.random.randint(0, 51, number_of_houses)

# Generate house prices with some random variation
noise = np.random.normal(0, 20000, number_of_houses)

price = (
    150 * size_sqft
    + 20000 * bedrooms
    - 1000 * age_years
    + noise
)

# Organize the data into a table
data = pd.DataFrame({
    "size_sqft": size_sqft,
    "bedrooms": bedrooms,
    "age_years": age_years,
    "price": price
})

# Display the first five rows
print("First five rows of the housing dataset:")
print(data.head())

# Display the dataset dimensions
print("\nDataset shape:", data.shape)

# Separate input features (X) and target (y)
X = data[["size_sqft", "bedrooms", "age_years"]]
y = data["price"]

# Display their shapes
print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Display the sizes of the datasets
print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# Create the Linear Regression model
linear_model = LinearRegression()

# Train the model using the training data
linear_model.fit(X_train, y_train)

# Predict prices for the testing data
y_pred = linear_model.predict(X_test)

# Display the learned coefficients and intercept
print("\n--- Linear Regression Model ---")
print("Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Display a few predicted and actual prices
print("\nActual prices:", y_test.iloc[:5].to_numpy())
print("Predicted prices:", y_pred[:5])

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation results
print("\n--- Linear Regression Evaluation ---")
print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R² Score:", r2)

# Create the Ridge Regression model
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

# Train the Ridge model
ridge_model.fit(X_train, y_train)


# Create the Lasso Regression model
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(alpha=1.0, max_iter=10000))
])

# Train the Lasso model
lasso_model.fit(X_train, y_train)


print("\n--- Ridge and Lasso Models Trained Successfully ---")

# Generate predictions using all three models
linear_pred = linear_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)

print("\nPredictions generated for all three models!")


# Store the predictions from all three models
predictions = {
    "Linear Regression": linear_pred,
    "Ridge Regression": ridge_pred,
    "Lasso Regression": lasso_pred
}

# Create an empty list to store evaluation results
results = []

# Evaluate each model
for model_name, predictions_array in predictions.items():

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, predictions_array)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions_array)
    r2 = r2_score(y_test, predictions_array)

    # Store the results
    results.append({
        "Model": model_name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2
    })

# Display the comparison table
results_df = pd.DataFrame(results)

print("\n--- Model Comparison Table ---")
print(results_df.round(4).to_string(index=False))

# Plot actual vs predicted prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, linear_pred, alpha=0.7, label="Predictions")

# Draw a reference line for perfect predictions
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Linear Regression: Actual vs Predicted Prices")
plt.legend()
plt.tight_layout()
plt.show()

# Calculate residuals
residuals = y_test - linear_pred

# Plot residuals
plt.figure(figsize=(8, 6))

plt.scatter(linear_pred, residuals, alpha=0.7)

# Draw a horizontal line at zero
plt.axhline(y=0, color="red", linestyle="--")

plt.xlabel("Predicted House Prices")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residuals Plot")

plt.tight_layout()
plt.show()