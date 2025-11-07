# beer_dataset_generator.py
"""
Beer Quality and Process Anomalies Dataset Generator
Author: Lavansh Kumar Singh (2025)
Purpose:
    This script constructs a realistic, domain-inspired synthetic dataset 
    that simulates the brewing process in beer production. It generates 
    correlated process features, environmental variables, and production 
    anomalies that mimic quality control issues in industrial settings.

Rationale for Dataset Design:
    The brewing process provides a structured and interpretable analogy 
    for anomaly detection, where most batches follow a stable process 
    distribution and occasional faulty batches deviate significantly.
    
    The dataset intentionally models *real-world complexity*:
    - Continuous variables with realistic industrial ranges.
    - Feature correlations between process parameters.
    - Synthetic noise and nonlinear interactions.
    - Injected anomalies with physically interpretable deviations.

    This design allows OptIForest and other anomaly detection algorithms 
    to be evaluated under realistic multi-feature, continuous-variable 
    conditions rather than purely statistical simulations.

Scalability:
    The modular architecture allows parameter tuning for:
      • Different production domains (e.g., wine, dairy, chemical mixing).
      • Varying anomaly ratios or drift intensity.
      • Controlled experiments across multiple datasets (e.g., beers_simple, beers_drifted, beers_extreme).
    Thus, this framework can generate a *family of correlated anomaly datasets* 
    for model benchmarking.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Generate base brewing data
# -----------------------------
np.random.seed(42)
n_samples = 4000

# Core brewing parameters (realistic ranges)
alcohol_pct = np.random.normal(5.5, 1.0, n_samples)       # Alcohol %
bitterness_ibu = np.random.normal(35, 10, n_samples)      # Bitterness
color_srm = np.random.normal(15, 4, n_samples)            # Color
gravity = np.random.normal(1.050, 0.010, n_samples)       # Original gravity
fermentation_temp = np.random.normal(20, 2.5, n_samples)  # Temperature (°C)
ph_level = np.random.normal(4.3, 0.15, n_samples)         # Acidity
co2_volume = np.random.normal(2.4, 0.3, n_samples)        # Carbonation
yeast_percent = np.random.normal(0.6, 0.1, n_samples)     # Yeast ratio
malt_ratio = np.random.normal(70, 5, n_samples)           # % malt content
sugar_conc = np.random.normal(12, 2, n_samples)           # Sugar content (g/L)

# Combine into DataFrame
df = pd.DataFrame({
    "alcohol_pct": alcohol_pct,
    "bitterness_ibu": bitterness_ibu,
    "color_srm": color_srm,
    "gravity": gravity,
    "fermentation_temp": fermentation_temp,
    "ph_level": ph_level,
    "co2_volume": co2_volume,
    "yeast_percent": yeast_percent,
    "malt_ratio": malt_ratio,
    "sugar_conc": sugar_conc
})

# -----------------------------
# 2. Add feature interactions
# -----------------------------
df["ratio_abv_ibu"] = df["alcohol_pct"] / (df["bitterness_ibu"] + 1)
df["temp_gravity_product"] = df["fermentation_temp"] * df["gravity"]
df["mean_process"] = df[["gravity","fermentation_temp","ph_level"]].mean(axis=1)
df["std_process"] = df[["gravity","fermentation_temp","ph_level"]].std(axis=1)

# -----------------------------
# 3. Introduce correlated noise
# -----------------------------
noise = np.random.normal(0, 0.2, df.shape)
mask = np.random.choice([0, 1], size=df.shape, p=[0.85, 0.15])
df += noise * mask

# -----------------------------
# 4. Simulate anomalies (faulty batches)
# -----------------------------
n_anomalies = int(0.05 * n_samples)
anomaly_indices = np.random.choice(df.index, n_anomalies, replace=False)

# Create strong deviations in certain features
df.loc[anomaly_indices, "ph_level"] += np.random.uniform(0.8, 1.2, n_anomalies)
df.loc[anomaly_indices, "alcohol_pct"] -= np.random.uniform(1.5, 2.5, n_anomalies)
df.loc[anomaly_indices, "fermentation_temp"] += np.random.uniform(8, 12, n_anomalies)
df.loc[anomaly_indices, "co2_volume"] += np.random.uniform(1.5, 3.0, n_anomalies)
df.loc[anomaly_indices, "color_srm"] += np.random.uniform(6, 10, n_anomalies)

# Label column (1 = anomaly, 0 = normal)
labels = np.zeros(n_samples)
labels[anomaly_indices] = 1
df["label"] = labels

# -----------------------------
# 5. Standardize & Export
# -----------------------------
scaler = StandardScaler()
scaled = pd.DataFrame(scaler.fit_transform(df.drop(columns=["label"])))
scaled["label"] = labels

# Save for OptIForest
output_path = "data/beers_complex.csv"
scaled.to_csv(output_path, header=False, index=False)
print(f"✅ Dataset saved as {output_path}")
print(f"Rows: {scaled.shape[0]}, Columns: {scaled.shape[1]}")
print("Anomalies:", int(sum(labels)), "out of", n_samples)
