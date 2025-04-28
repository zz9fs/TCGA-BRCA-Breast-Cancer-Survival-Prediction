import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

print("Step 7: Detecting and Handling Outliers")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the clean training data
print("Loading clean training data...")
try:
    train_data = pd.read_csv('processed_data/train_data_clean.csv')
    test_data = pd.read_csv('processed_data/test_data_clean.csv')
    print(f"Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"Loaded testing data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
except FileNotFoundError:
    print("Clean data files not found. Please run the missing data handling script first.")
    exit(1)

# 2. Separate metadata and expression data
print("Separating metadata and gene expression data...")
meta_cols = ['patient_id', 'survival_time', 'event']
X_train = train_data.drop(columns=meta_cols)
y_train = train_data[meta_cols]
X_test = test_data.drop(columns=meta_cols)
y_test = test_data[meta_cols]

# 3. Identify outliers in gene expression data
print("Identifying outliers using Z-score method...")
# Calculate z-scores for each feature (gene)
z_scores = stats.zscore(X_train)
z_score_df = pd.DataFrame(z_scores, columns=X_train.columns)

# Define outlier threshold (common threshold is z-score > 3 or < -3)
outlier_threshold = 3.0
is_outlier = np.abs(z_scores) > outlier_threshold

# Count outliers per sample (row) and per gene (column)
outliers_per_sample = is_outlier.sum(axis=1)
outliers_per_gene = is_outlier.sum(axis=0)

print(f"Found samples with outliers: {(outliers_per_sample > 0).sum()} out of {len(outliers_per_sample)}")
print(f"Found genes with outliers: {(outliers_per_gene > 0).sum()} out of {len(outliers_per_gene)}")

# 4. Visualize outlier distribution
print("Creating outlier visualization...")
plt.figure(figsize=(15, 10))

# Plot number of outliers per sample
plt.subplot(2, 2, 1)
plt.hist(outliers_per_sample, bins=20)
plt.title('Number of Outlier Genes per Sample')
plt.xlabel('Count of Outlier Genes')
plt.ylabel('Number of Samples')

# Plot number of outliers per gene
plt.subplot(2, 2, 2)
plt.hist(outliers_per_gene, bins=20)
plt.title('Number of Outlier Samples per Gene')
plt.xlabel('Count of Outlier Samples')
plt.ylabel('Number of Genes')

# Plot distribution of top 5 genes with most outliers
plt.subplot(2, 2, 3)
top_outlier_genes = outliers_per_gene.sort_values(ascending=False).head(5).index
for gene in top_outlier_genes:
    sns.kdeplot(X_train[gene], label=gene)
plt.title('Distribution of Top 5 Genes with Most Outliers')
plt.xlabel('Normalized Expression Value')
plt.legend()

# Plot heatmap of samples with most outliers
plt.subplot(2, 2, 4)
top_outlier_samples = outliers_per_sample.sort_values(ascending=False).head(10).index
top_gene_outliers = outliers_per_gene.sort_values(ascending=False).head(10).index
outlier_heatmap = X_train.loc[top_outlier_samples, top_gene_outliers]
sns.heatmap(outlier_heatmap, cmap='coolwarm', center=0)
plt.title('Heatmap of Top Outlier Samples and Genes')

plt.tight_layout()
plt.savefig('plots/outlier_analysis.png')
print("Outlier visualization saved to plots/outlier_analysis.png")

# 5. Handle outliers using capping (winsorization)
print("Handling outliers using winsorization (capping)...")
X_train_capped = X_train.copy()
X_test_capped = X_test.copy()

# For each gene (column), apply capping at specified percentiles
lower_percentile = 0.01
upper_percentile = 0.99

for column in X_train.columns:
    # Calculate percentile thresholds from training data
    lower_bound = X_train[column].quantile(lower_percentile)
    upper_bound = X_train[column].quantile(upper_percentile)
    
    # Apply capping to training data
    X_train_capped[column] = X_train[column].clip(lower=lower_bound, upper=upper_bound)
    
    # Apply same capping to test data (using training data thresholds)
    X_test_capped[column] = X_test[column].clip(lower=lower_bound, upper=upper_bound)

# 6. Compare original and capped distributions for some example genes
print("Creating comparison of original vs. capped distributions...")
plt.figure(figsize=(15, 10))

# Select a few genes that had outliers
example_genes = top_outlier_genes[:3]  # Take top 3 genes with most outliers

for i, gene in enumerate(example_genes):
    plt.subplot(3, 2, i*2+1)
    plt.hist(X_train[gene], bins=30, alpha=0.7)
    plt.axvline(X_train[gene].quantile(lower_percentile), color='r', linestyle='--')
    plt.axvline(X_train[gene].quantile(upper_percentile), color='r', linestyle='--')
    plt.title(f'Original Distribution: {gene}')
    
    plt.subplot(3, 2, i*2+2)
    plt.hist(X_train_capped[gene], bins=30, alpha=0.7)
    plt.title(f'Capped Distribution: {gene}')

plt.tight_layout()
plt.savefig('plots/outlier_capping_comparison.png')
print("Outlier capping comparison saved to plots/outlier_capping_comparison.png")

# 7. Create final datasets with capped values
print("Creating final datasets with capped values...")
train_data_capped = pd.concat([y_train, X_train_capped], axis=1)
test_data_capped = pd.concat([y_test, X_test_capped], axis=1)

# 8. Save capped datasets
print("Saving datasets with capped outliers...")
train_data_capped.to_csv('processed_data/train_data_capped.csv', index=False)
test_data_capped.to_csv('processed_data/test_data_capped.csv', index=False)

print(f"Capped training data saved ({train_data_capped.shape[0]} samples, {train_data_capped.shape[1]} features)")
print(f"Capped testing data saved ({test_data_capped.shape[0]} samples, {test_data_capped.shape[1]} features)")

# 9. Recalculate z-scores after capping to verify outlier reduction
z_scores_after = stats.zscore(X_train_capped)
is_outlier_after = np.abs(z_scores_after) > outlier_threshold
outliers_per_sample_after = is_outlier_after.sum(axis=1)
outliers_per_gene_after = is_outlier_after.sum(axis=0)

print("Outlier reduction results:")
print(f"Samples with outliers before: {(outliers_per_sample > 0).sum()}")
print(f"Samples with outliers after: {(outliers_per_sample_after > 0).sum()}")
print(f"Total outlier values before: {is_outlier.sum().sum()}")
print(f"Total outlier values after: {is_outlier_after.sum().sum()}") 