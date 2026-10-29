
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# -------------------- 1. GATHER DATA --------------------

data = pd.read_csv("Social_Network_Ads.csv")


# -------------------- 2. PREPROCESS DATA --------------------

print("\n" + "=" * 60)
print("                 DATA PREPROCESSING")
print("=" * 60)

print("Original Number of Rows        :", len(data))

duplicate_count = data.duplicated().sum()
print("Duplicate Rows Found           :", duplicate_count)

data = data.drop_duplicates()

print("Rows After Removing Duplicates :", len(data))


# -------------------- 3. SEPARATE VARIABLES --------------------

X = data[['Age', 'EstimatedSalary']]
y = data['Purchased']

print("\n" + "=" * 60)
print("              VARIABLE SEPARATION")
print("=" * 60)

print("Independent Variables : Age, EstimatedSalary")
print("Dependent Variable    : Purchased")


# -------------------- 4. SPLIT DATASET --------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    train_size=0.80,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("                 DATASET SPLITTING")
print("=" * 60)

print("Training Dataset Size : 80%")
print("Testing Dataset Size  : 20%")
print("Training Records      :", len(X_train))
print("Testing Records       :", len(X_test))


# -------------------- 5. FEATURE SCALING --------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -------------------- 6. CREATE SVM MODEL --------------------

svm_model = SVC(
    kernel='rbf',
    random_state=42
)


# -------------------- 7. TRAIN SVM MODEL --------------------

svm_model.fit(X_train_scaled, y_train)


# -------------------- 8. CLASSIFY TRAINING DATA --------------------

y_train_pred = svm_model.predict(X_train_scaled)


# -------------------- 9. CLASSIFY TESTING DATA --------------------

y_test_pred = svm_model.predict(X_test_scaled)


# ================================================================
# 10. TRAINING DATA RESULTS
# ================================================================

train_accuracy = accuracy_score(y_train, y_train_pred)

train_cm = confusion_matrix(y_train, y_train_pred)

train_tn, train_fp, train_fn, train_tp = train_cm.ravel()

train_correct = np.sum(y_train == y_train_pred)
train_incorrect = np.sum(y_train != y_train_pred)

print("\n" + "=" * 60)
print("              TRAINING DATA RESULTS")
print("=" * 60)

print("SVM Kernel             : RBF")
print("Training Records       :", len(X_train))
print("Correctly Classified   :", train_correct)
print("Incorrectly Classified :", train_incorrect)
print("Training Accuracy      : {:.2f}%".format(train_accuracy * 100))

print("\nConfusion Matrix Values")
print("-----------------------")
print("TN : {}".format(train_tn))
print("FP : {}".format(train_fp))
print("FN : {}".format(train_fn))
print("TP : {}".format(train_tp))


# ================================================================
# 11. TESTING DATA RESULTS
# ================================================================

test_accuracy = accuracy_score(y_test, y_test_pred)

test_cm = confusion_matrix(y_test, y_test_pred)

test_tn, test_fp, test_fn, test_tp = test_cm.ravel()

test_correct = np.sum(y_test == y_test_pred)
test_incorrect = np.sum(y_test != y_test_pred)

print("\n" + "=" * 60)
print("               TESTING DATA RESULTS")
print("=" * 60)

print("SVM Kernel             : RBF")
print("Testing Records        :", len(X_test))
print("Correctly Classified   :", test_correct)
print("Incorrectly Classified :", test_incorrect)
print("Testing Accuracy       : {:.2f}%".format(test_accuracy * 100))

print("\nConfusion Matrix Values")
print("-----------------------")
print("TN : {}".format(test_tn))
print("FP : {}".format(test_fp))
print("FN : {}".format(test_fn))
print("TP : {}".format(test_tp))


# ================================================================
# 12. CREATE MESH GRID FOR RBF DECISION BOUNDARY
# ================================================================

x_min = X['Age'].min() - 2
x_max = X['Age'].max() + 2

y_min = X['EstimatedSalary'].min() - 5000
y_max = X['EstimatedSalary'].max() + 5000

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

grid = pd.DataFrame({
    'Age': xx.ravel(),
    'EstimatedSalary': yy.ravel()
})

grid_scaled = scaler.transform(grid)

Z = svm_model.predict(grid_scaled)
Z = Z.reshape(xx.shape)


# ================================================================
# 13. TRAINING SET GRAPH
# ================================================================

plt.figure(figsize=(9, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.18
)

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    linewidths=2
)

plt.scatter(
    X_train.loc[y_train == 0, 'Age'],
    X_train.loc[y_train == 0, 'EstimatedSalary'],
    label='Not Purchased (0)',
    edgecolor='black',
    s=45
)

plt.scatter(
    X_train.loc[y_train == 1, 'Age'],
    X_train.loc[y_train == 1, 'EstimatedSalary'],
    label='Purchased (1)',
    edgecolor='black',
    s=45
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("SVM Classification - Training Set (RBF Kernel)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# ================================================================
# 14. TESTING SET GRAPH
# ================================================================

plt.figure(figsize=(9, 6))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.18
)

plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    linewidths=2
)

plt.scatter(
    X_test.loc[y_test == 0, 'Age'],
    X_test.loc[y_test == 0, 'EstimatedSalary'],
    label='Not Purchased (0)',
    edgecolor='black',
    s=55
)

plt.scatter(
    X_test.loc[y_test == 1, 'Age'],
    X_test.loc[y_test == 1, 'EstimatedSalary'],
    label='Purchased (1)',
    edgecolor='black',
    s=55
)

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("SVM Classification - Testing Set (RBF Kernel)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


