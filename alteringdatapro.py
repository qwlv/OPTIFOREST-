import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Load original dataset
input_path = r"C:\Users\direc\Documents\GitHub\OptIForest\data\wine.csv"
df = pd.read_csv(input_path, header=None)

# Separate features and label
X = df.iloc[:, :-1].copy()
y = df.iloc[:, -1].copy()

# Step 1: Add Gaussian noise to 30% of the rows
np.random.seed(42)
noise_fraction = 0.3
num_noisy = int(noise_fraction * len(X))
noise_indices = np.random.choice(X.index, num_noisy, replace=False)
X.loc[noise_indices] += np.random.normal(loc=0, scale=0.5, size=X.loc[noise_indices].shape)

# Step 2: Flip 10% of the labels (simulate mislabeling)
flip_fraction = 0.1
num_flips = int(flip_fraction * len(y))
flip_indices = np.random.choice(y.index, num_flips, replace=False)
y.loc[flip_indices] = 1 - y.loc[flip_indices]  # Assuming binary labels 0 and 1

# Step 3: Shuffle two feature columns to break correlation
X.iloc[:, 2] = np.random.permutation(X.iloc[:, 2].values)
X.iloc[:, 5] = np.random.permutation(X.iloc[:, 5].values)

# Step 4: Add derived features (sum, mean, std)
X['feature_sum'] = X.sum(axis=1)
X['feature_mean'] = X.mean(axis=1)
X['feature_std'] = X.std(axis=1)

# 🔧 Fix column name types
X.columns = X.columns.astype(str)

# Step 5: Cluster-based anomaly labeling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Cluster-based anomaly labeling (optional override)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Label smallest cluster as anomaly
counts = np.bincount(cluster_labels)
anomaly_cluster = np.argmin(counts)
y_clustered = (cluster_labels == anomaly_cluster).astype(int)

# Step 6: Save the enhanced dataset
X_final = pd.concat([pd.DataFrame(X), pd.Series(y_clustered)], axis=1)
output_path = r"C:\Users\direc\Documents\GitHub\OptIForest\data\wine_enhanced.csv"
X_final.to_csv(output_path, index=False, header=False)

print(f"✅ Enhanced dataset saved to:\n{output_path}")
print(f"📊 Shape: {X_final.shape} (Rows: {X_final.shape[0]}, Features: {X_final.shape[1]-1}, Label column: 1)")