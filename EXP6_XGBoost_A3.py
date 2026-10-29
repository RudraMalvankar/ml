
# AIM: Implementation of XGBoost (Extreme Gradient Boosting)
# Dataset: Breast Cancer Wisconsin (Original)

from pathlib import Path
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score,
    recall_score, f1_score
)

LINE = "=" * 60
PURPLE = "#7C3AED"
PINK = "#DB2777"
BLUE = "#2563EB"
CYAN = "#0891B2"
GREY = "#475569"
LIGHT = "#F1F5F9"
DARK = "#1E293B"


def section(title):
    print("\n" + LINE)
    print(title.center(60))
    print(LINE)


def show_matrix(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print("\nConfusion Matrix Values")
    print("-" * 23)
    print(f"TN : {tn}")
    print(f"FP : {fp}")
    print(f"FN : {fn}")
    print(f"TP : {tp}")
    return tn, fp, fn, tp


def locate_csv():
    # Looks beside this Python file, inside its subfolders, and in ML66 on Desktop.
    script_dir = Path(__file__).resolve().parent
    search_dirs = [script_dir, script_dir / "ML66", Path.home() / "Desktop" / "ML66"]
    candidates = []
    for folder in search_dirs:
        if folder.exists():
            candidates.extend(folder.glob("*.csv"))
    # Prefer the expected dataset name, ignoring case and accidental double extension.
    for path in candidates:
        if path.name.lower() in {"xgboost.csv", "xgboost.csv.csv"}:
            return path
    if candidates:
        print("CSV files found:")
        for path in candidates:
            print(" -", path)
    raise FileNotFoundError(
        "Xgboost.csv not found. Put the CSV in the same folder as this .py file, "
        "or in Desktop\\ML66. Check the exact filename."
    )


# STEP 1: Load and preprocess dataset
csv_path = locate_csv()
print("Reading dataset from:", csv_path)
df = pd.read_csv(csv_path)
original_rows = len(df)
duplicates = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

section("DATA PREPROCESSING")
print(f"{'CSV file':<35}: {csv_path.name}")
print(f"{'Original Number of Rows':<35}: {original_rows}")
print(f"{'Duplicate Rows Found':<35}: {duplicates}")
print(f"{'Rows After Removing Duplicates':<35}: {len(df)}")

# STEP 2: Separate input features and target
X = df.drop(columns=["Sample code number", "Class"])
y = df["Class"].map({2: 0, 4: 1})
if y.isna().any():
    raise ValueError("Class column must contain only 2 (Benign) and 4 (Malignant).")

section("VARIABLE SEPARATION")
indep = textwrap.fill(", ".join(X.columns), width=78,
                      initial_indent=" " * 25,
                      subsequent_indent=" " * 25).lstrip()
print(f"{'Independent Variables':<23}: {indep}")
print(f"{'Dependent Variable':<23}: Class (0 = Benign, 1 = Malignant)")

# STEP 3: Split data into training and testing sets
TEST_SIZE = 0.2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42, stratify=y
)

section("DATASET SPLITTING")
print(f"{'Training Dataset Size':<25}: 80%")
print(f"{'Testing Dataset Size':<25}: 20%")
print(f"{'Training Records':<25}: {len(X_train)}")
print(f"{'Testing Records':<25}: {len(X_test)}")

# STEP 4: Create and train XGBoost model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric="error"
)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=False)

# STEP 5: Predict and evaluate
train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

section("TRAINING DATA RESULTS")
train_correct = int((train_pred == y_train).sum())
print(f"{'Model':<25}: XGBoost Classifier")
print(f"{'Training Records':<25}: {len(y_train)}")
print(f"{'Correctly Classified':<25}: {train_correct}")
print(f"{'Incorrectly Classified':<25}: {len(y_train) - train_correct}")
print(f"{'Training Accuracy':<25}: {accuracy_score(y_train, train_pred) * 100:.2f}%")
show_matrix(y_train, train_pred)

section("TESTING DATA RESULTS")
test_correct = int((y_pred == y_test).sum())
print(f"{'Model':<25}: XGBoost Classifier")
print(f"{'Testing Records':<25}: {len(y_test)}")
print(f"{'Correctly Classified':<25}: {test_correct}")
print(f"{'Incorrectly Classified':<25}: {len(y_test) - test_correct}")
print(f"{'Testing Accuracy':<25}: {accuracy_score(y_test, y_pred) * 100:.2f}%")
tn, fp, fn, tp = show_matrix(y_test, y_pred)

section("FINAL RESULT")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
spec = tn / (tn + fp) if (tn + fp) else 0
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
print(f"{'Accuracy':<12}: {acc * 100:.2f} %")
print(f"{'Precision':<12}: {prec * 100:.2f} %")
print(f"{'Specificity':<12}: {spec * 100:.2f} %")
print(f"{'Recall':<12}: {rec * 100:.2f} %")
print(f"{'F1 Score':<12}: {f1 * 100:.2f} %")

# STEP 6: Visualizations. Each plt.show() opens a graph window; close it to continue.
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#CBD5E1"})

# Figure 1: First boosted tree (shown as a tree diagram)
booster_df = model.get_booster().trees_to_dataframe()

def draw_tree(tree_id, title):
    tree = booster_df[booster_df["Tree"] == tree_id].set_index("ID")
    positions = {}
    leaf_counter = [0]

    def layout(node, depth=0):
        row = tree.loc[node]
        if row["Feature"] == "Leaf":
            positions[node] = (leaf_counter[0], -depth)
            leaf_counter[0] += 1
        else:
            left_x = layout(row["Yes"], depth + 1)
            right_x = layout(row["No"], depth + 1)
            positions[node] = ((left_x + right_x) / 2, -depth)
        return positions[node][0]

    layout(f"{tree_id}-0")
    fig, ax = plt.subplots(figsize=(15, 7))
    leaf_gains = tree.loc[tree["Feature"] == "Leaf", "Gain"].abs()
    max_gain = max(float(leaf_gains.max()), 1e-9)
    for node, row in tree.iterrows():
        x, y_pos = positions[node]
        if row["Feature"] != "Leaf":
            for child, branch in ((row["Yes"], "Yes"), (row["No"], "No")):
                cx, cy = positions[child]
                ax.plot([x, cx], [y_pos, cy], color=GREY, linewidth=1.5)
                ax.text((x + cx) / 2, (y_pos + cy) / 2, branch,
                        ha="center", fontsize=8, color=GREY, backgroundcolor="white")
    for node, row in tree.iterrows():
        x, y_pos = positions[node]
        if row["Feature"] == "Leaf":
            val = float(row["Gain"])
            label = f"Leaf\n{val:+.2f}"
            color = "#FCE7F3" if val > 0 else "#DBEAFE"
        else:
            label = f"{row['Feature']}\n< {row['Split']:.2f}?"
            color = LIGHT
        ax.text(x, y_pos, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec=DARK))
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.axis("off")
    fig.tight_layout()
    plt.show()

n_trees = booster_df["Tree"].nunique()
draw_tree(0, f"First Boosted Decision Tree (Tree 1 of {n_trees})")

# Figure 2: Feature importance
importance = model.get_booster().get_score(importance_type="gain")
imp = pd.Series({feature: importance.get(f"f{i}", importance.get(feature, 0.0))
                  for i, feature in enumerate(X.columns)})
# XGBoost often names columns by feature name; use the model's actual importance mapping.
if sum(imp.values) == 0:
    imp = pd.Series({X.columns[int(k[1:])]: v for k, v in importance.items() if k.startswith("f")})
imp = imp.sort_values()
fig, ax = plt.subplots(figsize=(10, 6))
bar_colors = [CYAN if i >= max(0, len(imp) - 3) else "#C4B5FD" for i in range(len(imp))]
bars = ax.barh(imp.index, imp.values, color=bar_colors)
ax.set_title("Feature Importance (Gain)", fontsize=14, fontweight="bold", loc="left")
ax.set_xlabel("Importance")
ax.grid(axis="x", alpha=0.2)
for bar, value in zip(bars, imp.values):
    ax.text(value, bar.get_y() + bar.get_height()/2, f" {value:.1f}", va="center", fontsize=8)
fig.tight_layout()
plt.show()

# Figure 3: Training vs testing error
history = model.evals_result()
train_err = np.array(history["validation_0"]["error"]) * 100
test_err = np.array(history["validation_1"]["error"]) * 100
rounds = np.arange(1, len(train_err) + 1)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(rounds, train_err, color=BLUE, linewidth=2, label="Training error")
ax.plot(rounds, test_err, color=PINK, linewidth=2, label="Testing error")
ax.set_xlabel("Boosting round (tree number)")
ax.set_ylabel("Error (%)")
ax.set_title("Training vs Testing Error", fontsize=14, fontweight="bold", loc="left")
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
plt.show()

# Figure 4: Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
fig, ax = plt.subplots(figsize=(7, 6))
image = ax.imshow(cm, cmap="PuBuGn")
ax.set_xticks([0, 1], labels=["Benign", "Malignant"])
ax.set_yticks([0, 1], labels=["Benign", "Malignant"])
ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title("XGBoost Confusion Matrix", fontsize=14, fontweight="bold")
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() else 0
        ax.text(j, i, f"{cm[i, j]}\n{pct:.1f}%", ha="center", va="center",
                fontsize=16, fontweight="bold", color="white" if cm[i, j] > cm.max()/2 else DARK)
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
plt.show()

# Figure 5: Final boosted tree
# The final tree is the last individual tree; final prediction combines all trees.
draw_tree(n_trees - 1, f"Final Boosted Decision Tree (Tree {n_trees} of {n_trees})")

print("\nAll results and graphs displayed successfully.")
