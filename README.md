# EXP-03-Predictive Analytics

## A. Regression Analysis: Build and Evaluate Linear Regression Models and Interpret Regression Coefficients

## Tools:
Python (Scikit-learn), R

## Python Code for Linear Regression

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Example dataset (predicting 'y' based on 'x')
data = {
    'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'y': [2, 4, 5, 4, 5, 7, 8, 9, 10, 12]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the predictor and target variables
X = df[['x']]  # Predictor variable (independent)
y = df['y']    # Target variable (dependent)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable using the test set
y_pred = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output regression coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

# Model evaluation results
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plotting the regression line
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()
```

## OUTPUT:

### 1. Regression Coefficients:
- **Intercept (b₀):** The value of `y` when `x` is 0. It represents the baseline value.
- **Coefficient (b₁):** The change in `y` for each unit increase in `x`.

**Example output:**

  **Intercept:** 0.6237623762376234  
  **Coefficient:** 1.0693069306930694
  
### 2. Model Evaluation:
- **Mean Squared Error (MSE):** A measure of the average squared difference between actual and predicted values. Lower values indicate a better fit.
- **R-squared (R²):** A measure of how well the independent variable(s) explain the variability in the dependent variable. An R² closer to 1 means a better model fit.

**Example output:**

  **Mean Squared Error:** 0.5315165179884329  
  **R-squared:** 0.9114139136685945

### 3. Regression Line Visualization:
The plot will show the data points and the fitted regression line.

![image](https://github.com/user-attachments/assets/ad3b5d6d-285e-4e81-ab05-7489c396ccfe)

## Explanation

### Linear Regression Model
The model assumes a linear relationship between the predictor variable (x) and the target variable (y). 

### Intercept and Coefficient
- **Intercept**: The intercept (b₀) is where the line crosses the y-axis.
- **Coefficient**: The coefficient (b₁) indicates how much y increases for each unit increase in x.

### Model Evaluation
- **Mean Squared Error (MSE)** helps to evaluate the error between the actual and predicted values.
- **R-squared** helps to determine the proportion of variance explained by the model.
- 
## B. Classification Models

## Logistic Regression, Decision Trees and Model Accuracy Assessment
In this section, we will implement Logistic Regression and Decision Trees and assess model accuracy with confusion matrices.

## Tools
- Python (Scikit-learn)

## Python Code for Logistic Regression and Decision Trees

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Example dataset (Iris dataset for classification)
from sklearn.datasets import load_iris
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)

# Decision Tree Model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)

# Confusion Matrix for Logistic Regression
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)

# Confusion Matrix for Decision Tree
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

# Output confusion matrices and accuracies
print("Logistic Regression Accuracy:", accuracy_log_reg)
print("Decision Tree Accuracy:", accuracy_dt)

print("\nConfusion Matrix for Logistic Regression:")
print(conf_matrix_log_reg)

print("\nConfusion Matrix for Decision Tree:")
print(conf_matrix_dt)

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[0])
ax[0].set_title('Logistic Regression Confusion Matrix')
ax[0].set_xlabel('Predicted')
ax[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax[1])
ax[1].set_title('Decision Tree Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')

plt.tight_layout()
```
## Output

### Model Accuracy
- **Logistic Regression Accuracy**: A percentage of correctly classified instances.
- **Decision Tree Accuracy**: Similarly, the percentage of correct predictions.

**Example output:**
    **Logistic Regression Accuracy:** 0.9777777777777777  
    **Decision Tree Accuracy:** 0.9777777777777777

### Confusion Matrices
The confusion matrix displays the true positive (TP)**, false positive (FP), true negative (TN), and false negative (FN) predictions for each class.

### Example Output (for Logistic Regression and Decision Tree)

[[16  0  0]
 [ 0 14  1]
 [ 0  0 14]]    

### Confusion Matrix for Logistic Regression:

[[16  0  0]
 [ 0 14  1]
 [ 0  0 14]]
 
### Definitions:
- **True Positives (TP)**: Diagonal elements (e.g., `16` for class `0` in Logistic Regression).
- **False Positives (FP)**: Off-diagonal elements (e.g., `1` for class `1` predicting class `2`).
- **False Negatives (FN)**: Off-diagonal elements (e.g., `1` for class `2` predicting class `1`).
- 
### Confusion Matrix Visualization
The confusion matrices will be displayed as **heatmaps** for easy interpretation. The heatmaps highlight the counts for each prediction outcome.

![image](https://github.com/user-attachments/assets/cea6698f-3812-4d58-ac33-6b1ee964b018)


## Explanation:

### Logistic Regression:
- Logistic Regression is a statistical method used for **binary** or **multiclass classification problems**.
- The model predicts the **probability** of a particular class.

### Decision Trees:
- Decision Trees split the data into subsets based on **feature values**.
- Each **node** in the tree represents a feature decision, and each **leaf** represents a class label.

### Confusion Matrix:
- A Confusion Matrix is a useful tool for understanding the performance of a classification model.
- It shows the **true vs. predicted labels**, helping to calculate key metrics like:
  - Accuracy
  - Precision
  - Recall
  - F1-score

### Accuracy:
- Accuracy is calculated as:
  \[
  \text{Accuracy} = \frac{\text{Correct Predictions (Sum of Diagonal Elements)}}{\text{Total Predictions (Sum of All Elements in the Matrix)}}
  \]

### Heatmap Visualization:
- The heatmaps allow us to visually identify:
  - Where predictions are **correct**.
  - Where the model is making **mistakes**.

