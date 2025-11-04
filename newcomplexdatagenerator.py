# Filename: synthetic_optiforest_advanced.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ------------------------------
# 1. Generate base data
# ------------------------------
# Multi-cluster structure with correlation between features
X, y = make_classification(
    n_samples=2500,
    n_features=25,
    n_informative=15,
    n_redundant=5,
    n_repeated=0,
    n_classes=2,
    weights=[0.9, 0.1],  # imbalance
    class_sep=1.5,
    flip_y=0.02,
    n_clusters_per_class=3,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
df['label'] = y

# ------------------------------
# 2. Inject realistic anomalies
# ------------------------------
# Add structured outliers: feature drift, extreme values, and correlated spikes
n_outliers = 200
outliers = np.random.normal(0, 6, size=(n_outliers, X.shape[1]))
outlier_df = pd.DataFrame(outliers, columns=[f"f{i}" for i in range(X.shape[1])])
outlier_df['label'] = 1  # anomalous class

# Combine and shuffle
df = pd.concat([df, outlier_df], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

# ------------------------------
# 3. Add nonlinear transformations
# ------------------------------
df['f_sum'] = df.iloc[:, :-1].sum(axis=1)
df['f_mean'] = df.iloc[:, :-1].mean(axis=1)
df['f_var'] = df.iloc[:, :-1].var(axis=1)
df['f_log'] = np.log(np.abs(df['f0']) + 1)
df['f_sqrt'] = np.sqrt(np.abs(df['f1']))

# ------------------------------
# 4. Introduce correlated noise (simulating sensor or cluster drift)
# ------------------------------
for i in range(5):
    df[f"f{i}"] += np.random.normal(0, 0.8, size=len(df)) * (df['f_sum'] / df['f_sum'].std())

# ------------------------------
# 5. Cluster-based anomaly refinement
# ------------------------------
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop(columns=['label']))

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['cluster'] = clusters

# smallest cluster = anomaly
cluster_counts = df['cluster'].value_counts()
smallest_clusters = cluster_counts[cluster_counts < cluster_counts.mean()].index
df['label'] = df['cluster'].apply(lambda x: 1 if x in smallest_clusters else 0)

# ------------------------------
# 6. Final clean-up and export
# ------------------------------
df.drop(columns=['cluster'], inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Scale all features
scaled = scaler.fit_transform(df.drop(columns=['label']))
df_scaled = pd.DataFrame(scaled, columns=[c for c in df.columns if c != 'label'])
df_scaled['label'] = df['label']

# Save to CSV (no headers, as expected by OptIForest)
output_path = "data/synthetic_optiforest_advanced.csv"
df_scaled.to_csv(output_path, index=False, header=False)

print(f"✅ Advanced synthetic dataset created and saved at: {output_path}")
print(f"Rows: {df_scaled.shape[0]}, Columns: {df_scaled.shape[1]}")
print(f"Anomaly ratio: {df_scaled['label'].mean():.3f}")
