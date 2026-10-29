
# ============================================================
# AIM: IMPLEMENT LOGISTIC REGRESSION
# DATASET: SOCIAL NETWORK ADS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# 1. DATA PREPROCESSING
# ============================================================

data = pd.read_csv("Social_Network_Ads.csv")

original_rows = len(data)
duplicate_rows = data.duplicated().sum()

data = data.drop_duplicates()

remaining_rows = len(data)


print("\n" + "=" * 60)
print("                 DATA PREPROCESSING")
print("=" * 60)

print("Original Number of Rows              :", original_rows)
print("Number of Duplicate Rows             :", duplicate_rows)
print("Number of Rows After Removing Duplicates:", remaining_rows)

# ============================================================
# 2. SELECT INPUT AND OUTPUT VARIABLES
# ============================================================

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=0
)

print("\nTraining Dataset Size : 80%  (", len(X_train), "records )")
print("Testing Dataset Size  : 20%  (", len(X_test), "records )")

# ============================================================
# 4. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================================================
# 5. LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(random_state=0)

model.fit(X_train, y_train)

# ============================================================
# 6. PREDICT TEST RESULT
# ============================================================

y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("                 PREDICTED TEST RESULTS")
print("=" * 60)

print("Predicted Values:")
print(y_pred)

# ============================================================
# 7. PROBABILISTIC VALUES
# ============================================================

probability = model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("                  PROBABILISTIC VALUES")
print("=" * 60)

print("Probability of Purchase:")
print(np.round(probability, 4))

# ============================================================
# 8. ACTUAL VS PREDICTED VS PROBABILITY
# ============================================================

print("\n" + "=" * 60)
print("          ACTUAL vs PREDICTED vs PROBABILITY")
print("=" * 60)

actual = y_test.to_numpy()

print("Actual     :", actual)
print("Predicted  :", y_pred)
print("Probability:", np.round(probability, 4))

# ============================================================
# 9. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

TN, FP, FN, TP = cm.ravel()

print("\n" + "=" * 60)
print("                   CONFUSION MATRIX")
print("=" * 60)

print("""
                    Predicted
                   0         1
              +---------+---------+
Actual     0  |   {:3d}   |   {:3d}   |
              +---------+---------+
           1  |   {:3d}   |   {:3d}   |
              +---------+---------+
""".format(TN, FP, FN, TP))

print("True Negative  (TN) :", TN)
print("False Positive (FP) :", FP)
print("False Negative (FN) :", FN)
print("True Positive  (TP) :", TP)


# ============================================================
# 10. VISUALIZE CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    linewidths=1,
    linecolor='black',
    xticklabels=['Not Purchased', 'Purchased'],
    yticklabels=['Not Purchased', 'Purchased']
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

# ============================================================
# 11. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
specificity = TN / (TN + FP)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ============================================================
# 12. FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("                    FINAL RESULT")
print("=" * 60)

print("Accuracy    :", round(accuracy * 100, 2), "%")
print("Precision   :", round(precision * 100, 2), "%")
print("Specificity :", round(specificity * 100, 2), "%")
print("Recall      :", round(recall * 100, 2), "%")
print("F1 Score    :", round(f1 * 100, 2), "%")

# ============================================================
# 12. LOGISTIC REGRESSION VISUALIZATION
#     AGE ON X-AXIS
#     SALARY ON Y-AXIS
#     DATA POINTS + DECISION BOUNDARY
# ============================================================

# Create a grid covering Age and Estimated Salary
age_min = data['Age'].min() - 2
age_max = data['Age'].max() + 2

salary_min = data['EstimatedSalary'].min() - 5000
salary_max = data['EstimatedSalary'].max() + 5000

age_grid, salary_grid = np.meshgrid(
    np.linspace(age_min, age_max, 300),
    np.linspace(salary_min, salary_max, 300)
)

# Convert grid into dataframe
grid_data = pd.DataFrame({
    'Age': age_grid.ravel(),
    'EstimatedSalary': salary_grid.ravel()
})

# Scale grid using the trained scaler
grid_scaled = scaler.transform(grid_data)

# Calculate probability of Purchased = 1
prob_grid = model.predict_proba(grid_scaled)[:, 1]

prob_grid = prob_grid.reshape(age_grid.shape)


# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(9, 6))

# Probability regions
plt.contourf(
    age_grid,
    salary_grid,
    prob_grid,
    levels=20,
    cmap='coolwarm',
    alpha=0.25
)

# Logistic regression decision boundary
contour = plt.contour(
    age_grid,
    salary_grid,
    prob_grid,
    levels=[0.5],
    colors='black',
    linewidths=2
)

plt.clabel(
    contour,
    inline=True,
    fontsize=10,
    fmt={0.5: 'Probability = 0.5'}
)


# Plot actual data points
plt.scatter(
    data.loc[data['Purchased'] == 0, 'Age'],
    data.loc[data['Purchased'] == 0, 'EstimatedSalary'],
    marker='o',
    label='Not Purchased',
    edgecolors='black'
)

plt.scatter(
    data.loc[data['Purchased'] == 1, 'Age'],
    data.loc[data['Purchased'] == 1, 'EstimatedSalary'],
    marker='o',
    label='Purchased',
    edgecolors='black'
)


plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Logistic Regression - Age vs Estimated Salary")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
