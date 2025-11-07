'''
Purpose:
    This script prepares a normalized, anomaly-labeled version of the 
    Annthyroid dataset for use in OptIForest experiments.

Rationale:
    - The original ADBench version of Annthyroid includes 21 numerical features.
    - Since OptIForest expects numeric inputs and a final binary label column, 
      this script creates a simple but meaningful anomaly labeling scheme 
      using unsupervised clustering (K-Means).
    - The smaller of the two clusters is treated as the "anomaly" class,
      reflecting typical imbalance in anomaly detection datasets.
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
input_path = r"C:\Users\direc\Documents\GitHub\ADBenchmarks-anomaly-detection-datasets-main\numerical data\DevNet datasets\annthyroid_21feat_normalised.csv"
df = pd.read_csv(input_path)

# Step 2: Drop non-numeric columns (if any)
# If the dataset has a time or ID column, drop it here
df = df.select_dtypes(include=[np.number])

# Step 3: Add anomaly labels using clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Label the smaller cluster as anomaly
counts = np.bincount(cluster_labels)
anomaly_cluster = np.argmin(counts)
df['label'] = (cluster_labels == anomaly_cluster).astype(int)

# Step 4: Save to OptIForest data folder (no headers)
output_path = r"C:\Users\direc\Documents\GitHub\OptIForest\data\annthyroid_clean.csv"
df.to_csv(output_path, index=False, header=False)

print(f"✅ Dataset saved for OptIForest at:\n{output_path}")
print(f"📊 Shape: {df.shape} (Rows: {df.shape[0]}, Features: {df.shape[1]-1}, Label column: 1)")