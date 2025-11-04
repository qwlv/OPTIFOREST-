import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# === Step 1: Load base dataset ===
input_path = r"C:\Users\direc\Documents\GitHub\OPTIFOREST-\data\annthyroid_21feat_normalised.csv"

if not os.path.exists(input_path):
    raise FileNotFoundError(f"❌ File not found at: {input_path}")

df = pd.read_csv(input_path)
print(f"✅ Original dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# === Step 2: Ensure numeric data only ===
df = df.select_dtypes(include=[np.number])
X = df.values

# === Step 3: Add Gaussian noise (measurement errors) ===
np.random.seed(42)
noise = np.random.normal(0, 0.25, X.shape)
X_noisy = X + noise

# === Step 4: Simulate feature scaling drift (medical device calibration) ===
scaling_factors = np.random.uniform(0.8, 1.25, X.shape[1])
X_scaled = X_noisy * scaling_factors

# === Step 5: Create derived & nonlinear features ===
df_new = pd.DataFrame(X_scaled)
df_new["feature_sum"] = df_new.sum(axis=1)
df_new["feature_mean"] = df_new.mean(axis=1)
df_new["feature_std"] = df_new.std(axis=1)
df_new["sqrt_f0"] = np.sqrt(np.abs(df_new.iloc[:, 0]))
df_new["log_f1"] = np.log1p(np.abs(df_new.iloc[:, 1]))
df_new["interaction_0_1"] = df_new.iloc[:, 0] * df_new.iloc[:, 1]

# === Step 6: Inject extreme outliers (rare, faulty readings) ===
num_outliers = int(0.05 * len(df_new))
outlier_idx = np.random.choice(df_new.index, num_outliers, replace=False)
df_new.loc[outlier_idx] += np.random.normal(3, 1.5, size=df_new.loc[outlier_idx].shape)

# === Step 7: Random feature permutation (destroy linear correlations) ===
shuffle_cols = np.random.choice(df_new.columns[:-3], size=3, replace=False)
for col in shuffle_cols:
    df_new[col] = np.random.permutation(df_new[col].values)

# === Step 8: Feature dropout (simulate missing sensor values) ===
dropout_rows = np.random.choice(df_new.index, size=int(0.05 * len(df_new)), replace=False)
dropout_cols = np.random.choice(df_new.columns, size=2, replace=False)
df_new.loc[dropout_rows, dropout_cols] = 0

# === Step 9: Add synthetic anomalies (high variance, low density clusters) ===
n_anomalies = int(0.02 * len(df_new))
synthetic_anomalies = np.random.uniform(low=-10, high=10, size=(n_anomalies, df_new.shape[1]))
df_synthetic = pd.DataFrame(synthetic_anomalies, columns=df_new.columns)

# Merge both normal + anomaly samples
df_combined = pd.concat([df_new, df_synthetic], ignore_index=True)


df_combined.columns = df_combined.columns.astype(str)


# === Step 10: Recluster to assign new labels ===
scaler = StandardScaler()
X_scaled_final = scaler.fit_transform(df_combined)

pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled_final)

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca)
counts = np.bincount(cluster_labels)
anomaly_cluster = np.argmin(counts)

df_combined["label"] = (cluster_labels == anomaly_cluster).astype(int)

# === Step 11: Save final dataset ===
output_path = r"C:\Users\direc\Documents\GitHub\OPTIFOREST-\data\annthyroid_complex.csv"
df_combined.to_csv(output_path, index=False, header=False)

print(f"✅ annthyroid_complex.csv created successfully at:\n{output_path}")
print(f"📊 Final shape: {df_combined.shape[0]} rows × {df_combined.shape[1]-1} features + 1 label column")
print(f"⚡ Added {n_anomalies} synthetic extreme points + {num_outliers} injected outliers.")
