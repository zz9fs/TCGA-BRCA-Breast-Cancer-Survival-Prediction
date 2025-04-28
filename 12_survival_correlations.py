import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler

print("EDA Step 3: Univariate Survival Correlations")

# Create plots and results directories if they don't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('results'):
    os.makedirs('results')

# Load processed data
try:
    # First try to load the full feature data
    train_data = pd.read_csv('processed_data/train_data_capped.csv')
    print(f"Loaded full feature training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
except FileNotFoundError:
    # Fall back to the reduced dataset
    train_data = pd.read_csv('processed_data/train_data_survival_fixed.csv')
    print(f"Loaded reduced feature training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")

# Identify metadata columns and gene expression columns
meta_cols = ['patient_id', 'survival_time', 'event']
gene_cols = [col for col in train_data.columns if col not in meta_cols]

print(f"Analyzing {len(gene_cols)} genes for survival correlation")

# 1. Find top variable genes
gene_variance = train_data[gene_cols].var().sort_values(ascending=False)
top_variable_genes = gene_variance.head(20).index.tolist()

print(f"Selected top 20 variable genes for analysis")

# 2. Prepare data for Cox analysis
cox_data = train_data[meta_cols + top_variable_genes].copy()
# Standardize gene expression values for Cox model
scaler = StandardScaler()
cox_data[top_variable_genes] = scaler.fit_transform(cox_data[top_variable_genes])

# 3. Run univariate Cox regression for each gene
cox_results = []
for gene in top_variable_genes:
    # Prepare data for this gene
    gene_data = cox_data[['survival_time', 'event', gene]].copy()
    gene_data = gene_data.dropna()
    
    # Fit Cox model
    cph = CoxPHFitter()
    try:
        cph.fit(gene_data, duration_col='survival_time', event_col='event')
        
        # Extract results - Fix the deprecated Series access
        hr = np.exp(cph.params_.iloc[0])
        hr_lower = np.exp(cph.confidence_intervals_.iloc[0, 0])
        hr_upper = np.exp(cph.confidence_intervals_.iloc[0, 1])
        p_value = cph.summary.p.iloc[0]
        
        cox_results.append({
            'gene': gene,
            'hazard_ratio': hr,
            'hr_lower_ci': hr_lower,
            'hr_upper_ci': hr_upper,
            'p_value': p_value,
            'log_rank_p': None  # Will fill this in next step
        })
    except Exception as e:
        print(f"Warning: Cox model failed for gene {gene}")
        # Add a placeholder with null values to maintain the gene in results
        cox_results.append({
            'gene': gene,
            'hazard_ratio': np.nan,
            'hr_lower_ci': np.nan,
            'hr_upper_ci': np.nan,
            'p_value': np.nan,
            'log_rank_p': None
        })

# 4. Also compute log-rank test for each gene (high vs low expression)
for result in cox_results:
    gene = result['gene']
    median_val = train_data[gene].median()
    high_exp = train_data[gene] > median_val
    
    # Perform log-rank test
    try:
        lr_results = logrank_test(
            train_data.loc[high_exp, 'survival_time'], 
            train_data.loc[~high_exp, 'survival_time'],
            train_data.loc[high_exp, 'event'], 
            train_data.loc[~high_exp, 'event']
        )
        result['log_rank_p'] = lr_results.p_value
    except Exception as e:
        print(f"Warning: Log-rank test failed for gene {gene}")
        result['log_rank_p'] = np.nan

# 5. Sort results by p-value and save to CSV
cox_results_df = pd.DataFrame(cox_results)
# Handle case where all Cox models failed
if cox_results_df['p_value'].notna().any():
    # Sort only if there are non-null p-values
    cox_results_df = cox_results_df.sort_values('p_value', na_position='last')
else:
    # Sort by log-rank p-value if no Cox p-values available
    if cox_results_df['log_rank_p'].notna().any():
        cox_results_df = cox_results_df.sort_values('log_rank_p', na_position='last')
        print("No valid Cox models. Sorting by log-rank p-values instead.")
    else:
        print("Warning: No valid p-values for sorting. Showing unsorted results.")

cox_results_df.to_csv('results/gene_survival_correlations.csv', index=False)

# 6. Check if we have any significant genes before attempting to plot
significant_genes = cox_results_df[cox_results_df['p_value'].notna() & (cox_results_df['p_value'] < 0.05)]

if len(significant_genes) > 0:
    print(f"Found {len(significant_genes)} genes significantly associated with survival (p<0.05)")
    
    # Plot forest plot for top significant genes
    plt.figure(figsize=(12, max(6, len(significant_genes[:10]) * 0.5)))
    
    genes_to_plot = significant_genes.head(10) if len(significant_genes) >= 10 else significant_genes
    genes_to_plot = genes_to_plot.sort_values('hazard_ratio')
    
    y_pos = np.arange(len(genes_to_plot))
    
    plt.errorbar(
        x=genes_to_plot['hazard_ratio'],
        y=y_pos,
        xerr=[genes_to_plot['hazard_ratio'] - genes_to_plot['hr_lower_ci'], 
             genes_to_plot['hr_upper_ci'] - genes_to_plot['hazard_ratio']],
        fmt='o',
        capsize=5
    )
    
    # Add reference line at HR=1
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    
    plt.yticks(y_pos, genes_to_plot['gene'])
    plt.xlabel('Hazard Ratio (95% CI)')
    plt.title('Univariate Cox Regression: Genes Associated with Survival')
    
    # Add p-value annotations
    for i, row in enumerate(genes_to_plot.itertuples()):
        plt.text(
            max(row.hr_upper_ci, row.hazard_ratio) + 0.1, 
            i, 
            f'p={row.p_value:.4f}',
            va='center'
        )
    
    plt.tight_layout()
    plt.savefig('plots/cox_forest_plot.png')
else:
    print("No genes significantly associated with survival (p<0.05) using Cox regression")
    
    # Try using log-rank test results instead
    log_rank_significant = cox_results_df[cox_results_df['log_rank_p'].notna() & (cox_results_df['log_rank_p'] < 0.05)]
    
    if len(log_rank_significant) > 0:
        print(f"Found {len(log_rank_significant)} genes with significant log-rank test (p<0.05)")
        
        # Create alternative visualization using log-rank p-values
        plt.figure(figsize=(12, 8))
        genes_to_plot = log_rank_significant.head(10) if len(log_rank_significant) >= 10 else log_rank_significant
        genes_to_plot = genes_to_plot.sort_values('log_rank_p')
        
        plt.barh(genes_to_plot['gene'], -np.log10(genes_to_plot['log_rank_p']))
        plt.axvline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        plt.xlabel('-log10(p-value)')
        plt.title('Log-Rank Test: Genes Associated with Survival')
        plt.tight_layout()
        plt.savefig('plots/logrank_pvalues.png')
        print("Created alternative visualization using log-rank test p-values")
    else:
        print("No genes with significant association to survival found with either method")

# 7. Create heatmap of top genes vs survival time using genes with best log-rank p-values
if cox_results_df['log_rank_p'].notna().any():
    best_logrank_genes = cox_results_df.sort_values('log_rank_p').head(10)['gene'].tolist()
else:
    # Fall back to top variable genes if no valid log-rank tests
    best_logrank_genes = top_variable_genes[:10]

plt.figure(figsize=(15, 10))

# Prepare data
heatmap_data = train_data[['survival_time', 'event'] + best_logrank_genes].copy()
heatmap_data = heatmap_data.sort_values('survival_time')

# Normalize gene expression for visualization
heatmap_genes = heatmap_data[best_logrank_genes]
heatmap_genes_norm = (heatmap_genes - heatmap_genes.mean()) / heatmap_genes.std()

# Create the heatmap
ax = sns.heatmap(heatmap_genes_norm.T, cmap='viridis', center=0)

# Add survival time points on top of the heatmap
ax2 = ax.twiny()
ax2.plot(np.arange(len(heatmap_data)), heatmap_data['survival_time'], 'r-')
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel('Survival Time (Days)')

# Add event indicators
events = heatmap_data['event'] == 1
ax3 = ax.twiny()
ax3.scatter(np.where(events)[0], [len(best_logrank_genes) + 0.5] * events.sum(), marker='x', color='black', alpha=0.5)
ax3.set_xlim(ax.get_xlim())
ax3.set_ylim(ax.get_ylim())
ax3.axis('off')

ax.set_ylabel('Genes')
ax.set_yticks(np.arange(len(best_logrank_genes)) + 0.5)
ax.set_yticklabels(best_logrank_genes)

plt.title('Gene Expression Heatmap vs. Survival Time')
plt.tight_layout()
plt.savefig('plots/gene_survival_heatmap.png')

print("Survival correlation analysis completed. Results saved to results directory and plots saved to plots directory.")