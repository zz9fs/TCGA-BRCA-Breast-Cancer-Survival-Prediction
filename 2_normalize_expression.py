import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("Step 2: Normalizing Gene Expression Data")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the merged data
print("Loading merged data...")
try:
    merged_data = pd.read_csv('processed_data/merged_data.csv')
    print(f"Loaded merged data: {merged_data.shape[0]} samples, {merged_data.shape[1]} features")
except FileNotFoundError:
    print("Merged data file not found. Please run the merge script first.")
    exit(1)

# 2. Separate clinical (metadata) and expression data
print("Separating clinical and expression data...")
# First 3 columns are patient_id, survival_time, and event
clinical_cols = ['patient_id', 'survival_time', 'event']
clinical_data = merged_data[clinical_cols]
expression_data = merged_data.drop(columns=clinical_cols)

print(f"Clinical data: {clinical_data.shape[1]} features")
print(f"Expression data: {expression_data.shape[1]} genes")

# 3. Check for zero or negative values before log transformation
print("Checking expression data for non-positive values...")
min_value = expression_data.min().min()
print(f"Minimum expression value: {min_value}")

# 4. Apply log2 transformation to expression data
print("Applying log2 transformation...")
# Add a small offset to avoid log(0) if needed
offset = 1.0
if min_value <= 0:
    print(f"Adding offset of {offset} to avoid log of zero or negative values")
    expression_data_log2 = np.log2(expression_data + offset)
else:
    expression_data_log2 = np.log2(expression_data)

# 5. Z-score normalization (standardization)
print("Applying z-score normalization...")
scaler = StandardScaler()
expression_data_scaled = pd.DataFrame(
    scaler.fit_transform(expression_data_log2),
    columns=expression_data_log2.columns,
    index=expression_data_log2.index
)

# 6. Create visualization to show the effect of normalization
print("Creating visualization of normalization effect...")
# Sample 100 random genes
np.random.seed(42)  # for reproducibility
sample_genes = np.random.choice(expression_data.columns, size=min(100, len(expression_data.columns)), replace=False)

# Plot distributions before and after normalization for sampled genes
plt.figure(figsize=(15, 5))

# Before normalization (log2 only)
plt.subplot(1, 2, 1)
for gene in sample_genes[:10]:  # Just plot 10 genes for clarity
    plt.hist(expression_data_log2[gene], alpha=0.3, bins=20)
plt.title('Gene Expression Distribution After Log2 Transformation')
plt.xlabel('Log2 Expression')
plt.ylabel('Frequency')

# After z-score normalization
plt.subplot(1, 2, 2)
for gene in sample_genes[:10]:  # Just plot 10 genes for clarity
    plt.hist(expression_data_scaled[gene], alpha=0.3, bins=20)
plt.title('Gene Expression Distribution After Z-score Normalization')
plt.xlabel('Standardized Expression')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('plots/normalization_effect.png')
print("Visualization saved to plots/normalization_effect.png")

# 7. Merge normalized expression data back with clinical data
print("Merging normalized expression data with clinical data...")
normalized_data = pd.concat([clinical_data, expression_data_scaled], axis=1)
print(f"Final normalized dataset: {normalized_data.shape[0]} samples, {normalized_data.shape[1]} features")

# 8. Save the normalized data
print("Saving normalized data...")
normalized_data.to_csv('processed_data/normalized_data.csv', index=False)
print("Normalized data saved to processed_data/normalized_data.csv")

# 9. Also save a version with just log2 transformation (without z-score) for reference
log2_data = pd.concat([clinical_data, expression_data_log2], axis=1)
log2_data.to_csv('processed_data/log2_data.csv', index=False)
print("Log2 transformed data (without z-score) saved to processed_data/log2_data.csv") 