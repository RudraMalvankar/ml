# ML Practicals: All 6 Experiments

# Experiment 1A: Simple Linear Regression

```python
# Experiment 1A: Simple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_excel("Salary_Data.xlsx")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Training Score:", model.score(X_train, y_train))
print("Testing Score:", model.score(X_test, y_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(X_test, y_test)
order = np.argsort(X_test[:, 0])
plt.plot(X_test[order], y_pred[order], color="red")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
```

---

# Experiment 1B: Multiple Linear Regression

```python
# Experiment 1B: Multiple Linear Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_excel("50_Startups.xlsx")

X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].values

X["State"] = LabelEncoder().fit_transform(X["State"])
X = X.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Training Score:", model.score(X_train, y_train))
print("Testing Score:", model.score(X_test, y_test))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.show()
```

---

# Experiment 2: Logistic Regression

```python
# Experiment 2: Logistic Regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Social_Network_Ads.csv")
df = df.drop_duplicates()

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Probabilities:", model.predict_proba(X_test)[:, 1])

plt.scatter(df["Age"], df["EstimatedSalary"], c=y)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.show()
```

---

# Experiment 3: Support Vector Machine (SVM)

```python
# Experiment 3: Support Vector Machine (SVM)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Social_Network_Ads.csv")
df = df.drop_duplicates()

X = df[["Age", "EstimatedSalary"]]
y = df["Purchased"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC(kernel="rbf", random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))

plt.scatter(df["Age"], df["EstimatedSalary"], c=y)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.show()
```

---

# Experiment 4: DBSCAN

```python
# Experiment 4: DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

points = {
    "X": [2, 3, 3, 4, 7, 7, 8, 8, 9, 1, 6, 10],
    "Y": [2, 2, 3, 3, 7, 8, 7, 8, 7, 8, 2, 3]
}

df = pd.DataFrame(points)
X = df[["X", "Y"]].values

epsilon = 1.9
min_pts = 4

dist_matrix = cdist(X, X, metric="euclidean")

types = []

for row in dist_matrix:
    neighbours = np.where(row <= epsilon)[0]

    if len(neighbours) >= min_pts:
        types.append("Core")
    else:
        types.append("Noise")

df["Point_Type"] = types
print(df)

for kind, group in df.groupby("Point_Type"):
    plt.scatter(group["X"], group["Y"], label=kind)

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()
```

---

# Experiment 5: Principal Component Analysis (PCA)

```python
# Experiment 5: Principal Component Analysis (PCA)
import numpy as np
import pandas as pd

X = np.array([2.5, 0.5, 2.2, 1.9, 3.1,
              2.3, 2.0, 1.0, 1.5, 1.1])

Y = np.array([2.4, 0.7, 2.9, 2.2, 3.0,
              2.7, 1.6, 1.1, 1.6, 0.9])

n = len(X)

x_mean = np.mean(X)
y_mean = np.mean(Y)

x_centered = X - x_mean
y_centered = Y - y_mean

var_x = np.sum(x_centered ** 2) / (n - 1)
var_y = np.sum(y_centered ** 2) / (n - 1)
cov_xy = np.sum(x_centered * y_centered) / (n - 1)

C = np.array([
    [var_x, cov_xy],
    [cov_xy, var_y]
])

eigenvalues, eigenvectors = np.linalg.eig(C)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

u = eigenvectors[:, 0]

PC1 = x_centered * u[0] + y_centered * u[1]

print("Covariance Matrix:\\n", C)
print("Eigenvalues:", eigenvalues)
print("Eigenvector:", u)
print("Principal Component:", PC1)
```

---

# Experiment 6: XGBoost Classification

```python
# Experiment 6: XGBoost Classification
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score
)

df = pd.read_csv("Xgboost.csv.csv")
df = df.drop_duplicates()

X = df.drop(columns=["Sample code number", "Class"])
y = df["Class"].map({2: 0, 4: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\\n", confusion_matrix(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
```

---

