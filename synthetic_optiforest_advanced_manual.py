# Filename: synthetic_optiforest_advanced_manual.py
"""
Synthetic Advanced Dataset Generator (No make_classification)
Author: Lavansh Kumar Singh, 2025
Generates a realistic high-dimensional anomaly detection dataset
with correlated features, structured outliers, and engineered variables.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

np.random.seed(42)

# --------------------------------------
# 1. Create clustered normal data
# --------------------------------------
n_clusters = 5
samples_per_cluster = 500
n_features = 25

cluster_means = np.random.uniform(-3, 3, size=(n_clusters, n_features))
cluster_covs = [np.diag(np.random.uniform(0.5, 2.0, n_features)) for _ in range(n_clusters)]

data_parts = []
for i in range(n_clusters):
    X_cluster = np.random.multivariate_normal(cluster_means[i], cluster_covs[i], samples_per_cluster)
    data_parts.append(X_cluster)

X_normal = np.vstack(data_parts)
y_normal = np.zeros(X_normal.shape[0])

# --------------------------------------
# 2. Add correlated and redundant features
# --------------------------------------
# Create 5 redundant features as linear combinations of others
redundant = X_normal[:, :5] * 0.8 + X_normal[:, 5:10] * 0.2 + np.random.normal(0, 0.1, (X_normal.shape[0], 5))
X_normal = np.hstack([X_normal, redundant])

# --------------------------------------
# 3. Inject anomalies (structured outliers)
# --------------------------------------
n_outliers = 250
outliers = np.random.laplace(loc=0, scale=5, size=(n_outliers, X_normal.shape[1]))
# add extreme shifts in first few features
outliers[:, :10] += np.random.uniform(8, 12, size=(n_outliers, 10))
y_outliers = np.ones(n_outliers)

X = np.vstack([X_normal, outliers])
y = np.concatenate([y_normal, y_outliers])

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df["label"] = y

# --------------------------------------
# 4. Add nonlinear and statistical transformations
# --------------------------------------
df["f_sum"] = df.iloc[:, :-1].sum(axis=1)
df["f_mean"] = df.iloc[:, :-1].mean(axis=1)
df["f_var"] = df.iloc[:, :-1].var(axis=1)
df["f_log"] = np.log(np.abs(df["f0"]) + 1)
df["f_sqrt"] = np.sqrt(np.abs(df["f1"]))

# --------------------------------------
# 5. Introduce correlated drift (sensor bias)
# --------------------------------------
for i in range(5):
    df[f"f{i}"] += np.random.normal(0, 0.7, len(df)) * (df["f_sum"] / df["f_sum"].std())

# --------------------------------------
# 6. Cluster-based label refinement
# --------------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=["label"]))

kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df["cluster"] = clusters

cluster_counts = df["cluster"].value_counts()
smallest_clusters = cluster_counts[cluster_counts < cluster_counts.mean()].index
df["label"] = df["cluster"].apply(lambda x: 1 if x in smallest_clusters else 0)

# --------------------------------------
# 7. Shuffle, scale, and save
# --------------------------------------
df.drop(columns=["cluster"], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

scaled = scaler.fit_transform(df.drop(columns=["label"]))
df_scaled = pd.DataFrame(scaled, columns=[c for c in df.columns if c != "label"])
df_scaled["label"] = df["label"]

output_path = "data/synthetic_optiforest_advanced222.csv"
df_scaled.to_csv(output_path, index=False, header=False)

print(f"✅ Dataset created successfully at {output_path}")
print(f"Rows: {df_scaled.shape[0]}, Columns: {df_scaled.shape[1]}")
print(f"Anomaly ratio: {df_scaled['label'].mean():.3f}")
