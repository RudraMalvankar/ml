
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# 1. Dataset setup
points = {
    'Point': [f'P{i}' for i in range(1, 13)],
    'X': [3, 4, 5, 6, 7, 6, 7, 8, 3, 2, 3, 2],
    'Y': [7, 6, 5, 4, 3, 2, 2, 4, 3, 6, 5, 4]
}

df = pd.DataFrame(points)
X = df[['X', 'Y']].values

epsilon = 1.9
min_pts = 4

# 2. Euclidean Distance Matrix
dist_matrix = cdist(X, X, metric='euclidean')

# DataFrame for Distance Table
point_labels = [f'P{i}' for i in range(1, 13)]
dist_df = pd.DataFrame(np.round(dist_matrix, 2), index=point_labels, columns=point_labels)

# 3. Track Neighbors & Identify Core Points
neighbors_idx_list = []
neighbors_names_list = []
point_types = []

for i, row in enumerate(dist_matrix):
    n_idx = [j for j, d in enumerate(row) if d <= epsilon]
    neighbors_idx_list.append(n_idx)
    neighbors_names_list.append(", ".join([f'P{j+1}' for j in n_idx]))
    
    if len(n_idx) >= min_pts:
        point_types.append('Core')
    else:
        point_types.append('Pending')

# Classify Border and Noise points
for i, p_type in enumerate(point_types):
    if p_type == 'Pending':
        has_core_neighbor = any(point_types[j] == 'Core' for j in neighbors_idx_list[i])
        point_types[i] = 'Border' if has_core_neighbor else 'Noise'

# Primary cluster labeling for display in the dataframe table
cluster_labels = [-1] * len(X)
core_indices = [i for i, t in enumerate(point_types) if t == 'Core'] # P2 (idx 1), P5 (idx 4), P11 (idx 10)

for c_id, core_idx in enumerate(core_indices, start=1):
    cluster_labels[core_idx] = c_id
    for n_idx in neighbors_idx_list[core_idx]:
        if point_types[n_idx] != 'Noise' and cluster_labels[n_idx] == -1:
            cluster_labels[n_idx] = c_id

# Update DataFrame attributes
df['Neighbors'] = neighbors_names_list
df['Neighbors_Count'] = [len(n) for n in neighbors_idx_list]
df['Point_Type'] = point_types
df['Cluster'] = [f'Cluster {c}' if c != -1 else 'Noise' for c in cluster_labels]

# --- PRINT TABLES ---
print("======================== EUCLIDEAN DISTANCE MATRIX ========================")
print(dist_df.to_string())

print("\n================== DBSCAN CLASSIFICATION TABLE =====================")
print(df[['Point', 'X', 'Y', 'Neighbors', 'Neighbors_Count', 'Point_Type', 'Cluster']].to_string(index=False))

# --- PRINT EXPLICIT CLUSTER MEMBERSHIP ---
print("\n========================= CLUSTER POINTS ===========================")
# Cluster 1 (Centered at Core P2 -> P1, P2, P3, P11)
c1_members = [f'P{j+1}' for j in neighbors_idx_list[1]]
print(f"• Cluster 1 (Core Point: P2)  : {', '.join(c1_members)}")

# Cluster 2 (Centered at Core P5 -> P4, P5, P6, P7, P8)
c2_members = [f'P{j+1}' for j in neighbors_idx_list[4]]
print(f"• Cluster 2 (Core Point: P5)  : {', '.join(c2_members)}")

# Cluster 3 (Centered at Core P11 -> P2, P10, P11, P12)
c3_members = [f'P{j+1}' for j in neighbors_idx_list[10]]
# Re-order nicely: P2, P10, P11, P12
c3_members.sort(key=lambda x: int(x[1:]))
print(f"• Cluster 3 (Core Point: P11) : {', '.join(c3_members)}")

noise_members = df[df['Cluster'] == 'Noise']['Point'].tolist()
print(f"• Noise Points                : {', '.join(noise_members)}")

# 5. Clean Visualization Plot
plt.figure(figsize=(7, 6))

colors = {'Core': 'green', 'Border': 'orange', 'Noise': 'red'}
markers = {'Core': 'o', 'Border': '^', 'Noise': 'x'}

# Draw clean dashed epsilon circles around core points
core_points = df[df['Point_Type'] == 'Core']
for idx, row in core_points.iterrows():
    circle = plt.Circle(
        (row['X'], row['Y']), epsilon, 
        color='green', fill=False, linestyle='--', linewidth=1, alpha=0.5
    )
    plt.gca().add_patch(circle)

# Plot Core, Border, and Noise points
for p_type in ['Core', 'Border', 'Noise']:
    sub = df[df['Point_Type'] == p_type]
    plt.scatter(
        sub['X'], sub['Y'], 
        c=colors[p_type], 
        marker=markers[p_type], 
        s=120 if p_type != 'Noise' else 90, 
        label=f'{p_type} Point',
        zorder=3
    )

# Label all points
for idx, row in df.iterrows():
    plt.text(row['X'] + 0.15, row['Y'] + 0.1, row['Point'], fontsize=9, fontweight='bold')

plt.title(f'DBSCAN Point Classification (eps={epsilon}, minPts={min_pts})', fontsize=11)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(-0.5, 10.5)
plt.ylim(-0.5, 8.5)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()
