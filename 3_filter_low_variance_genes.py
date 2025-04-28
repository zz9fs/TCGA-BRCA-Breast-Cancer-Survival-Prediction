import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold

print("Step 3: Filtering Low-Variance Genes")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the normalized data
print("Loading normalized data...")
try:
    normalized_data = pd.read_csv('processed_data/normalized_data.csv')
    print(f"Loaded normalized data: {normalized_data.shape[0]} samples, {normalized_data.shape[1]} features")
except FileNotFoundError:
    print("Normalized data file not found. Please run the normalization script first.")
    exit(1)

# 2. Separate clinical and expression data
print("Separating clinical and expression data...")
clinical_cols = ['patient_id', 'survival_time', 'event']
clinical_data = normalized_data[clinical_cols]
expression_data = normalized_data.drop(columns=clinical_cols)

print(f"Clinical data: {clinical_data.shape[1]} features")
print(f"Expression data before filtering: {expression_data.shape[1]} genes")

# 3. Calculate variance for each gene
print("Calculating gene expression variance...")
gene_variance = expression_data.var().sort_values(ascending=False)

# 4. Visualize gene variance distribution
print("Creating gene variance distribution plot...")
plt.figure(figsize=(10, 6))
plt.hist(gene_variance, bins=100)
plt.axvline(x=0.1, color='r', linestyle='--')  # Example threshold line
plt.title('Gene Expression Variance Distribution')
plt.xlabel('Variance')
plt.ylabel('Number of Genes')
plt.yscale('log')  # Log scale for better visualization
plt.savefig('plots/gene_variance_distribution.png')
print("Variance distribution plot saved to plots/gene_variance_distribution.png")

# 5. Plot cumulative variance to help with threshold selection
print("Creating cumulative variance plot...")
plt.figure(figsize=(10, 6))
sorted_var = gene_variance.sort_values(ascending=False)
cumulative_var = np.cumsum(sorted_var) / sum(sorted_var)
plt.plot(range(len(cumulative_var)), cumulative_var)
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Top Genes')
plt.ylabel('Cumulative Variance Proportion')
plt.grid(True)
plt.savefig('plots/cumulative_variance.png')
print("Cumulative variance plot saved to plots/cumulative_variance.png")

# 6. Apply variance threshold to filter genes
# Using VarianceThreshold from sklearn
print("Filtering low-variance genes...")
variance_threshold = 0.1  # This is an example threshold - adjust based on your data
print(f"Using variance threshold: {variance_threshold}")

var_filter = VarianceThreshold(threshold=variance_threshold)
filtered_expression = pd.DataFrame(
    var_filter.fit_transform(expression_data),
    index=expression_data.index
)

# Get the feature names that were kept
expression_data_array = expression_data.values
support = var_filter.get_support()
filtered_columns = expression_data.columns[support]
filtered_expression.columns = filtered_columns

print(f"Expression data after filtering: {filtered_expression.shape[1]} genes")
print(f"Removed {expression_data.shape[1] - filtered_expression.shape[1]} low-variance genes")

# 7. Calculate what percentage of genes were kept
kept_percentage = (filtered_expression.shape[1] / expression_data.shape[1]) * 100
print(f"Kept {kept_percentage:.2f}% of the original genes")

# 8. Merge filtered expression data with clinical data
print("Merging filtered expression data with clinical data...")
filtered_data = pd.concat([clinical_data, filtered_expression], axis=1)
print(f"Final filtered dataset: {filtered_data.shape[0]} samples, {filtered_data.shape[1]} features")

# 9. Save the filtered data
print("Saving filtered data...")
filtered_data.to_csv('processed_data/filtered_data.csv', index=False)
print("Filtered data saved to processed_data/filtered_data.csv")

# 10. Save the list of high-variance genes for reference
high_variance_genes = pd.DataFrame({
    'gene': filtered_columns,
    'variance': gene_variance[filtered_columns].values
})
high_variance_genes.to_csv('processed_data/high_variance_genes.csv', index=False)
print(f"List of {len(filtered_columns)} high-variance genes saved to processed_data/high_variance_genes.csv") 