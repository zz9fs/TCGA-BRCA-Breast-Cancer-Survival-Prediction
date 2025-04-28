import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("Step 8: Multicollinearity Analysis and Feature Selection")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the capped training data
print("Loading capped training data...")
try:
    train_data = pd.read_csv('processed_data/train_data_capped.csv')
    test_data = pd.read_csv('processed_data/test_data_capped.csv')
    print(f"Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"Loaded testing data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
except FileNotFoundError:
    print("Capped data files not found. Please run the outlier handling script first.")
    try:
        train_data = pd.read_csv('processed_data/train_data_clean.csv')
        test_data = pd.read_csv('processed_data/test_data_clean.csv')
        print("Using clean data instead.")
    except FileNotFoundError:
        print("Clean data files not found either. Please run the preprocessing pipeline first.")
        exit(1)

# 2. Separate metadata and expression data
print("Separating metadata and gene expression data...")
meta_cols = ['patient_id', 'survival_time', 'event']
X_train = train_data.drop(columns=meta_cols)
y_train = train_data[meta_cols]
X_test = test_data.drop(columns=meta_cols)
y_test = test_data[meta_cols]

# 3. Analyze correlation between genes (multicollinearity)
print("Analyzing gene correlation...")
# To avoid memory issues with large correlation matrices, sample a subset of genes
np.random.seed(42)
if X_train.shape[1] > 500:
    print(f"Sampling 500 genes from {X_train.shape[1]} for correlation analysis")
    gene_sample = np.random.choice(X_train.columns, size=500, replace=False)
    X_train_sample = X_train[gene_sample]
else:
    X_train_sample = X_train
    
# Calculate correlation matrix
corr_matrix = X_train_sample.corr()

# 4. Visualize correlation matrix
print("Creating correlation heatmap...")
plt.figure(figsize=(16, 12))
plt.subplot(2, 2, 1)
mask = np.triu(np.ones_like(corr_matrix), k=1)
sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0, linewidths=0.5, square=True)
plt.title('Gene Expression Correlation Matrix (Sample)')

# 5. Identify highly correlated gene pairs
high_corr_threshold = 0.8
high_corr_pairs = []

# Only check upper triangle of correlation matrix to avoid duplicates
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > high_corr_threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print(f"Found {len(high_corr_pairs)} highly correlated gene pairs (|r| > {high_corr_threshold})")
if len(high_corr_pairs) > 0:
    print("Top 5 highly correlated pairs:")
    for pair in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
        print(f"{pair[0]} and {pair[1]}: r = {pair[2]:.3f}")

# 6. Perform feature selection using univariate methods
print("Performing univariate feature selection...")
# Using regression since our target is continuous (survival time)
# We'll only use data points where event=1 (deceased) for this analysis
survival_time = y_train.loc[y_train['event'] == 1, 'survival_time'].values
X_train_deceased = X_train.loc[y_train['event'] == 1]

# F-regression for univariate feature selection
selector_f = SelectKBest(f_regression, k=50)
X_train_f_selected = selector_f.fit_transform(X_train_deceased, survival_time)
f_scores = selector_f.scores_
f_pvalues = selector_f.pvalues_

# Mutual information for nonlinear relationships
selector_mi = SelectKBest(mutual_info_regression, k=50)
X_train_mi_selected = selector_mi.fit_transform(X_train_deceased, survival_time)
mi_scores = selector_mi.scores_

# 7. Visualize feature importance scores
print("Creating feature importance visualization...")
plt.subplot(2, 2, 2)
plt.hist(-np.log10(f_pvalues), bins=30)
plt.axvline(-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
plt.title('F-Test p-value Distribution (-log10 scale)')
plt.xlabel('-log10(p-value)')
plt.ylabel('Count')
plt.legend()

# Get the top genes by F-score
gene_importance = pd.DataFrame({
    'gene': X_train.columns,
    'f_score': f_scores,
    'p_value': f_pvalues,
    'mi_score': mi_scores
})
top_genes_f = gene_importance.sort_values('f_score', ascending=False).head(20)
top_genes_mi = gene_importance.sort_values('mi_score', ascending=False).head(20)

plt.subplot(2, 2, 3)
sns.barplot(x='f_score', y='gene', data=top_genes_f.head(10))
plt.title('Top 10 Genes by F-Score')

plt.subplot(2, 2, 4)
sns.barplot(x='mi_score', y='gene', data=top_genes_mi.head(10))
plt.title('Top 10 Genes by Mutual Information')

plt.tight_layout()
plt.savefig('plots/feature_selection.png')
print("Feature selection visualization saved to plots/feature_selection.png")

# 8. Save feature importance scores
print("Saving feature importance scores...")
gene_importance.to_csv('processed_data/gene_importance.csv', index=False)
print(f"Gene importance scores saved to processed_data/gene_importance.csv")

# 9. Create reduced datasets using top features from univariate selection
print("Creating reduced datasets with top features...")
# Get the union of top genes from both methods
top_genes_f_set = set(top_genes_f['gene'])
top_genes_mi_set = set(top_genes_mi['gene'])
top_genes_combined = list(top_genes_f_set.union(top_genes_mi_set))
print(f"Selected {len(top_genes_combined)} genes using univariate methods")

# Create reduced datasets
X_train_reduced = X_train[top_genes_combined]
X_test_reduced = X_test[top_genes_combined]

train_data_reduced = pd.concat([y_train, X_train_reduced], axis=1)
test_data_reduced = pd.concat([y_test, X_test_reduced], axis=1)

print(f"Reduced training data: {train_data_reduced.shape[0]} samples, {train_data_reduced.shape[1]} features")
print(f"Reduced testing data: {test_data_reduced.shape[0]} samples, {test_data_reduced.shape[1]} features")

# Save reduced datasets
train_data_reduced.to_csv('processed_data/train_data_reduced.csv', index=False)
test_data_reduced.to_csv('processed_data/test_data_reduced.csv', index=False)
print("Reduced datasets saved to processed_data directory")

# 10. PCA as an alternative dimensionality reduction approach
print("Performing PCA analysis...")
# Standardize data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA
n_components = 50  # Choose number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Create DataFrame with PCA components
pca_columns = [f'PC{i+1}' for i in range(n_components)]
X_train_pca_df = pd.DataFrame(X_train_pca, columns=pca_columns, index=X_train.index)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_columns, index=X_test.index)

# 11. Visualize PCA results
print("Creating PCA visualization...")
plt.figure(figsize=(12, 8))

# Plot explained variance ratio
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.subplot(2, 2, 1)
plt.bar(range(1, len(explained_variance)+1), explained_variance)
plt.plot(range(1, len(explained_variance)+1), cumulative_variance, 'r-')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')
plt.grid(True)

# Plot some patient samples in 2D PCA space, colored by survival status
plt.subplot(2, 2, 2)
event_status = y_train['event'].values
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=event_status, alpha=0.6, cmap='coolwarm')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Patients in PCA Space (colored by event)')
plt.colorbar(label='Event Status')

# Plot survival time in PCA space (only for deceased patients)
plt.subplot(2, 2, 3)
deceased_mask = y_train['event'] == 1
if deceased_mask.sum() > 0:
    survival_time_normalized = y_train.loc[deceased_mask, 'survival_time'].values
    survival_time_normalized = (survival_time_normalized - survival_time_normalized.min()) / (survival_time_normalized.max() - survival_time_normalized.min())
    plt.scatter(X_train_pca[deceased_mask, 0], X_train_pca[deceased_mask, 1], c=survival_time_normalized, alpha=0.6, cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Deceased Patients in PCA Space (colored by survival time)')
    plt.colorbar(label='Normalized Survival Time')
else:
    plt.text(0.5, 0.5, 'No deceased patients in training set', ha='center', va='center')

plt.tight_layout()
plt.savefig('plots/pca_analysis.png')
print("PCA visualization saved to plots/pca_analysis.png")

# 12. Create datasets with PCA features
print("Creating PCA-transformed datasets...")
train_data_pca = pd.concat([y_train.reset_index(drop=True), X_train_pca_df.reset_index(drop=True)], axis=1)
test_data_pca = pd.concat([y_test.reset_index(drop=True), X_test_pca_df.reset_index(drop=True)], axis=1)

print(f"PCA training data: {train_data_pca.shape[0]} samples, {train_data_pca.shape[1]} features")
print(f"PCA testing data: {test_data_pca.shape[0]} samples, {test_data_pca.shape[1]} features")

# Save PCA datasets
train_data_pca.to_csv('processed_data/train_data_pca.csv', index=False)
test_data_pca.to_csv('processed_data/test_data_pca.csv', index=False)
print("PCA datasets saved to processed_data directory")

print("Multicollinearity analysis and feature selection completed") 