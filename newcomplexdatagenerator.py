'''
Purpose:
    This script creates a reproducible, parametrically controlled synthetic dataset
    designed to test the robustness of anomaly detection models such as OptIForest.

Rationale for Data Generation Method:
    While scikit-learn’s `make_classification` function is sometimes viewed as a basic
    synthetic data generator, it was intentionally selected in this project to establish
    a reproducible and parameterized foundation for constructing the synthetic complex dataset.
    This choice aligns with the principle of controlled replication, enabling direct comparison
    with the original OptIForest study under consistent conditions.

    The controlled generation process offered by `make_classification` allows explicit tuning
    of feature informativity, redundancy, class imbalance, and injected noise—all crucial
    for evaluating the algorithm’s sensitivity to structured and unstructured variance.
    Moreover, it supports precise adjustment of complexity parameters
    (e.g., number of informative features, correlation strength, and class separation),
    allowing systematic testing of OptIForest’s robustness.

Scalability:
    The modular design of this code allows researchers to modify parameters such as
    sample size, number of features, proportion of outliers, or data distribution range
    to generate multiple variants (e.g., synthetic_simple, synthetic_advanced).
    Each run can simulate different complexity scenarios for model benchmarking.
'''



import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import os

# Parameters
n_samples = 1200
n_features = 25
n_outliers = 80
random_state = 42

# Step 1: Generate normal data
X_normal, y_normal = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=18,
    n_redundant=5,
    flip_y=0.01,
    weights=[0.98],  # 98% normal
    random_state=random_state
)

# Step 2: Generate synthetic outliers
X_outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, n_features))
y_outliers = np.ones(n_outliers)  # Label anomalies as 1

# Step 3: Combine normal + outlier data
X_combined = np.vstack([X_normal, X_outliers])
y_combined = np.hstack([y_normal, y_outliers])

# Step 4: Feature engineering
X_df = pd.DataFrame(X_combined)
X_df['feature_sum'] = X_df.sum(axis=1)
X_df['feature_mean'] = X_df.mean(axis=1)
X_df['feature_std'] = X_df.std(axis=1)
X_df['log_feature_0'] = np.log1p(np.abs(X_df[0]))
X_df['sqrt_feature_1'] = np.sqrt(np.abs(X_df[1]))

# Step 5: Inject noise into 10% of rows
np.random.seed(42)
noise_rows = np.random.choice(X_df.index, size=int(0.1 * len(X_df)), replace=False)
X_df.loc[noise_rows] += np.random.normal(loc=0, scale=1.0, size=X_df.loc[noise_rows].shape)

# Step 6: Simulate feature dropout
dropout_rows = np.random.choice(X_df.index, size=int(0.05 * len(X_df)), replace=False)
X_df.loc[dropout_rows, 3] = 0  # Zero out column 3

# Step 7: Apply scaling drift to first few columns
X_df.iloc[:, :5] *= np.random.uniform(0.7, 1.3)

# Step 8: Final assembly
X_final = pd.concat([X_df, pd.Series(y_combined, name='label')], axis=1)

# Step 9: Save to CSV
output_path = r"C:\Users\direc\Documents\GitHub\OptIForest\data\synthetic_complex.csv"
X_final.to_csv(output_path, index=False, header=False)

print(f"✅ Complex synthetic dataset saved to:\n{output_path}")
print(f"📊 Shape: {X_final.shape} (Rows: {X_final.shape[0]}, Features: {X_final.shape[1]-1}, Label column: 1)")