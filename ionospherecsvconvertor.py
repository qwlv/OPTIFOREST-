"""
ionospherecsvconvertor.py
----------------------------------
Purpose:
    This script converts the Ionosphere dataset stored in NumPy binary format 
    (X.npy for features, y.npy for labels) into a single CSV file compatible 
    with the OptIForest project pipeline.

Functionality:
    • Loads feature and label arrays (X.npy and y.npy) from the specified directory.
    • Combines them horizontally so that the label column appears as the last column.
    • Saves the merged dataset as 'ionosphere.csv' (or a custom filename) 
      without headers or index, matching the input format expected by demo.py.

Usage:
    Run directly from command line:
        python ionospherecsvconvertor.py

Output:
    A CSV file containing 351 rows (samples) and 35 columns 
    (34 numeric features + 1 binary label column), 
    saved in the designated output directory.

Author:
    Lavansh Kumar Singh
    Master of Data Science, Macquarie University
    (for OptIForest replication and dataset preprocessing tasks)
"""

import numpy as np
import pandas as pd
import os

# Paths
x_path = r"C:\Users\direc\Downloads\18_Ionosphere\X.npy"
y_path = r"C:\Users\direc\Downloads\18_Ionosphere\y.npy"
output_path = r"C:\Users\direc\Downloads\18_Ionosphere\ionsphere.csv"

# Step 1: Load arrays
X = np.load(x_path)
y = np.load(y_path)

print("✅ Loaded X and y successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)

# Step 2: Combine them (X as features, y as last column)
combined = np.hstack((X, y.reshape(-1, 1)))

# Step 3: Convert to DataFrame and save as CSV (no header, no index)
df = pd.DataFrame(combined)
df.to_csv(output_path, index=False, header=False)

print(f"📁 Saved combined CSV to: {output_path}")
print(f"📊 Shape: {df.shape} (Rows: {df.shape[0]}, Features: {df.shape[1]-1}, Label column: 1)")
