

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ----------------------------------------------------------------------
# 1. Load the whole dataset
# ----------------------------------------------------------------------
df = pd.read_csv("Salary_Data.csv")

X = df.iloc[:, :-1].values   # YearsExperience
y = df.iloc[:, -1].values    # Salary

# ----------------------------------------------------------------------
# 2. Split the whole dataset into training (80%) and testing (20%)
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# ----------------------------------------------------------------------
# 3. Fit the Simple Linear Regression model
# ----------------------------------------------------------------------
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# ----------------------------------------------------------------------
# 4. Scores: Training score, Testing score, RMSE
# ----------------------------------------------------------------------
train_score = regressor.score(X_train, y_train)
test_score = regressor.score(X_test, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

# ----------------------------------------------------------------------
# 5. Plot: Predicted vs Actual (all data) with best fit line
# ----------------------------------------------------------------------
y_all = np.concatenate([y_train, y_test])
y_pred_all = np.concatenate([y_pred_train, y_pred_test])

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color="blue", label="Test data", s=60)
plt.scatter(y_train, y_pred_train, color="green", alpha=0.5, label="Train data", s=40)

# Best fit line through predicted vs actual points
m, b = np.polyfit(y_all, y_pred_all, 1)
x_line = np.linspace(y_all.min(), y_all.max(), 100)
plt.plot(x_line, m * x_line + b, color="red", linestyle="--", label="Best Fit Line")

plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Simple Linear Regression: Predicted vs Actual Salary")
plt.legend(loc="upper left")

# Display scores and RMSE directly on the plot
score_text = (
    f"Training Score (R$^2$): {train_score:.4f}\n"
    f"Testing Score (R$^2$) : {test_score:.4f}\n"
    f"RMSE              : {rmse:.2f}"
)
plt.gca().text(
    0.97, 0.03, score_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    fontfamily="monospace",
    verticalalignment="bottom",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9),
)

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("salary_predicted_vs_actual.png", dpi=150)
plt.show()
