import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

print("EDA Step 4: Comprehensive Biomarker Analysis")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('results'):
    os.makedirs('results')

# Load processed data
try:
    # Try to load the full feature dataset
    train_data = pd.read_csv('processed_data/train_data_capped.csv')
    test_data = pd.read_csv('processed_data/test_data_capped.csv')
    combined_data = pd.concat([train_data, test_data])
    print(f"Loaded full feature data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")
except FileNotFoundError:
    # Fall back to the reduced dataset
    train_data = pd.read_csv('processed_data/train_data_survival_fixed.csv')
    test_data = pd.read_csv('processed_data/test_data_survival_fixed.csv')
    combined_data = pd.concat([train_data, test_data])
    print(f"Loaded reduced feature data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")

# Separate metadata and gene expression data
meta_cols = ['patient_id', 'survival_time', 'event']
gene_cols = [col for col in combined_data.columns if col not in meta_cols]

print(f"Analyzing {len(gene_cols)} genes as potential biomarkers")

# 1. Calculate variance of each gene
gene_variance = combined_data[gene_cols].var().sort_values(ascending=False)
top_variable_genes = gene_variance.head(100).index.tolist()

print("Identified top 100 variable genes as potential biomarkers")

# 2. Create variance distribution plot with thresholds
plt.figure(figsize=(12, 8))
plt.hist(gene_variance.values, bins=50)
plt.axvline(x=gene_variance.iloc[99], color='r', linestyle='--', 
            label=f'Top 100 genes threshold ({gene_variance.iloc[99]:.4f})')
plt.axvline(x=gene_variance.iloc[49], color='g', linestyle='--', 
            label=f'Top 50 genes threshold ({gene_variance.iloc[49]:.4f})')
plt.axvline(x=gene_variance.iloc[19], color='b', linestyle='--', 
            label=f'Top 20 genes threshold ({gene_variance.iloc[19]:.4f})')
plt.title('Gene Expression Variance Distribution')
plt.xlabel('Variance')
plt.ylabel('Number of Genes')
plt.legend()
plt.yscale('log')
plt.savefig('plots/biomarker_variance_distribution.png')

# 3. Create a correlation heatmap of top variable genes
plt.figure(figsize=(16, 14))
top_genes_data = combined_data[top_variable_genes[:50]]
corr_matrix = top_genes_data.corr()
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
plt.title('Correlation Matrix of Top 50 Variable Genes')
plt.tight_layout()
plt.savefig('plots/biomarker_correlation_heatmap.png')

# 4. Run PCA specifically on top variable genes
X = combined_data[top_variable_genes].values
X_scaled = StandardScaler().fit_transform(X)
pca = PCA(n_components=10)
pca_result = pca.fit_transform(X_scaled)

# Create dataframe with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(10)])
pca_df['survival_time'] = combined_data['survival_time'].values
pca_df['event'] = combined_data['event'].values
pca_df['patient_id'] = combined_data['patient_id'].values

# Plot first two principal components colored by survival status
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', hue='event', 
                palette={0: 'blue', 1: 'red'},
                data=pca_df, alpha=0.7)
plt.title('PCA of Top Variable Genes')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(title='Event', labels=['Censored', 'Deceased'])
plt.savefig('plots/biomarker_pca.png')

# 5. Try to identify molecular subtypes using clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pca_df[['PC1', 'PC2']])
pca_df['cluster'] = clusters

# Plot clusters
plt.figure(figsize=(12, 10))
sns.scatterplot(x='PC1', y='PC2', hue='cluster', 
                palette='viridis',
                data=pca_df, alpha=0.7)
plt.title('Potential Molecular Subtypes Based on Top Variable Genes')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.legend(title='Cluster')
plt.savefig('plots/biomarker_clusters.png')

# 6. Survival analysis by cluster
plt.figure(figsize=(12, 10))
for cluster in sorted(pca_df['cluster'].unique()):
    mask = pca_df['cluster'] == cluster
    kmf = KaplanMeierFitter()
    kmf.fit(pca_df.loc[mask, 'survival_time'], 
            pca_df.loc[mask, 'event'],
            label=f'Cluster {cluster} (n={mask.sum()})')
    kmf.plot_survival_function()

plt.title('Survival by Molecular Subtype Cluster')
plt.xlabel('Time (Days)')
plt.ylabel('Survival Probability')

# Perform multivariate log-rank test
multi_lr_results = multivariate_logrank_test(
    pca_df['survival_time'], 
    pca_df['cluster'],
    pca_df['event']
)
plt.text(0.1, 0.1, f'Log-rank p-value: {multi_lr_results.p_value:.4f}', 
         transform=plt.gca().transAxes)

plt.tight_layout()
plt.savefig('plots/biomarker_cluster_survival.png')

# 7. Save top biomarkers information
top_biomarkers = pd.DataFrame({
    'gene': gene_variance.head(100).index,
    'variance': gene_variance.head(100).values
})
top_biomarkers.to_csv('results/top_biomarkers.csv', index=False)

# 8. Analyze variance within/between clusters for top genes
biomarker_by_cluster = []
for gene in top_variable_genes[:20]:
    # Get overall variance
    overall_var = combined_data[gene].var()
    
    # Get cluster-specific variance
    cluster_vars = []
    for cluster in sorted(pca_df['cluster'].unique()):
        mask = pca_df['cluster'] == cluster
        patient_ids = pca_df.loc[mask, 'patient_id']
        gene_values = combined_data.loc[combined_data['patient_id'].isin(patient_ids), gene]
        cluster_vars.append(gene_values.var())
    
    # Calculate between-cluster variance
    means_by_cluster = []
    for cluster in sorted(pca_df['cluster'].unique()):
        mask = pca_df['cluster'] == cluster
        patient_ids = pca_df.loc[mask, 'patient_id']
        gene_values = combined_data.loc[combined_data['patient_id'].isin(patient_ids), gene]
        means_by_cluster.append(gene_values.mean())
    
    between_var = np.var(means_by_cluster) * len(means_by_cluster)
    
    biomarker_by_cluster.append({
        'gene': gene,
        'overall_variance': overall_var,
        'within_cluster_variance': np.mean(cluster_vars),
        'between_cluster_variance': between_var,
        'variance_ratio': between_var / np.mean(cluster_vars) if np.mean(cluster_vars) > 0 else 0
    })

biomarker_cluster_df = pd.DataFrame(biomarker_by_cluster)
biomarker_cluster_df = biomarker_cluster_df.sort_values('variance_ratio', ascending=False)
biomarker_cluster_df.to_csv('results/biomarker_cluster_variance.csv', index=False)

# Plot top genes by variance ratio (between/within clusters)
plt.figure(figsize=(14, 8))
biomarker_subset = biomarker_cluster_df.head(15)
sns.barplot(x='variance_ratio', y='gene', data=biomarker_subset)
plt.title('Top Biomarkers by Between/Within Cluster Variance Ratio')
plt.xlabel('Variance Ratio (Between/Within)')
plt.tight_layout()
plt.savefig('plots/biomarker_variance_ratio.png')

print("Biomarker analysis completed. Results saved to results directory and plots saved to plots directory.")