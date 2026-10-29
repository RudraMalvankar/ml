import numpy as np
import pandas as pd

# Raw Data
X = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2.0, 1.0, 1.5, 1.1])
Y = np.array([2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9])
n = len(X)

print("=" * 65)
print("                PCA(Principal Component Analysis) ")
print("=" * 65)

# Step 1: Calculate Mean
x̄ = np.sum(X) / n
ȳ = np.sum(Y) / n
print("\n--- STEP 1: Mean Values ---")
print(f"x̄ = {x̄:.4f}")
print(f"ȳ = {ȳ:.4f}")

# Step 2: Center the Data
x_centered = X - x̄
y_centered = Y - ȳ
centered_df = pd.DataFrame({"xᵢ - x̄": x_centered, "yᵢ - ȳ": y_centered})
print("\n--- STEP 2: Centered Data ---")
print(centered_df.to_string(index=False))

# Step 3: Find Covariance Matrix using Exact Formulas from Image
var_x = np.sum(x_centered**2) / (n - 1)
var_y = np.sum(y_centered**2) / (n - 1)
cov_xy = (1 / (n - 1)) * np.sum((x_centered**2) * (y_centered**2))

C = np.array([[var_x, cov_xy], [cov_xy, var_y]])

print("\n--- STEP 3: Covariance Matrix (C) ---")
print(f"var(x)   = {var_x:.4f}")
print(f"var(y)   = {var_y:.4f}")
print(f"cov(x,y) = {cov_xy:.4f}\n")
print(f"C = [[{C[0,0]:.4f}, {C[0,1]:.4f}],")
print(f"     [{C[1,0]:.4f}, {C[1,1]:.4f}]]")

# Step 4: Eigenvalues of Covariance Matrix det([S - λI]) = 0
eigenvalues, eigenvectors = np.linalg.eig(C)

# Sort eigenvalues in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\n--- STEP 4: Eigen values ---")
print(f"λ₁= {eigenvalues[0]:.4f}")
print(f"λ₂= {eigenvalues[1]:.4f}")

# Step 5: Compute Eigenvector for Highest Eigenvalue
u_highest = eigenvectors[:, 0]
print("\n--- STEP 5: Compute Eigen vector ---")
print(f"Selected λ = {eigenvalues[0]:.4f} (max value)")
print(f"Eigenvector U = [{u_highest[0]:.4f}, {u_highest[1]:.4f}]ᵀ")

# Step 6: Find Principal Component using [xᵢ - x̄, yᵢ - ȳ]ᵀ
PC1 = x_centered * u_highest[0] + y_centered * u_highest[1]

pc_df = pd.DataFrame(
    {
        "xᵢ - x̄": x_centered,
        "yᵢ - ȳ": y_centered,
        "Principal Component (PC1)": PC1,
    }
)

print(
    "\n--- STEP 6: Principal Component Values using [xᵢ - x̄, yᵢ - ȳ]ᵀ ---"
)
print(pc_df.to_string(index=False))
print("=" * 65)
