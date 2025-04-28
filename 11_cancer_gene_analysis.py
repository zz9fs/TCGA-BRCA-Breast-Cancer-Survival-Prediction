import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("EDA Step 2: Known Cancer Gene Analysis")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
    
# Mapping of gene symbols to Ensembl IDs - we'll need to adapt this based on available genes
# These are examples and may not match your actual data's gene IDs
GENE_MAPPING = {
    'BRCA1': 'ENSG00000012048',  # Breast cancer 1 gene
    'BRCA2': 'ENSG00000139618',  # Breast cancer 2 gene
    'ERBB2': 'ENSG00000141736',  # HER2/neu
    'ESR1': 'ENSG00000091831',   # Estrogen receptor
    'PGR': 'ENSG00000082175',    # Progesterone receptor
    'MKI67': 'ENSG00000148773',  # Ki-67 (proliferation marker)
    'TP53': 'ENSG00000141510',   # p53
    'MYC': 'ENSG00000136997'     # c-Myc
}

# Load processed data with all features
try:
    # Try to load data with full feature set
    full_train_data = pd.read_csv('processed_data/train_data_capped.csv')
    full_test_data = pd.read_csv('processed_data/test_data_capped.csv')
    combined_data = pd.concat([full_train_data, full_test_data])
    print(f"Loaded full feature data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")
except FileNotFoundError:
    # Fall back to the reduced feature set
    full_train_data = pd.read_csv('processed_data/train_data_survival_fixed.csv')
    full_test_data = pd.read_csv('processed_data/test_data_survival_fixed.csv')
    combined_data = pd.concat([full_train_data, full_test_data])
    print(f"Loaded reduced feature data: {combined_data.shape[0]} samples, {combined_data.shape[1]} features")

# Identify available cancer genes in our dataset
available_genes = []
for gene_symbol, ensembl_id in GENE_MAPPING.items():
    if ensembl_id in combined_data.columns:
        available_genes.append((gene_symbol, ensembl_id))
    elif gene_symbol in combined_data.columns:
        available_genes.append((gene_symbol, gene_symbol))
        
if not available_genes:
    # If we can't find the genes by their expected IDs, try to find them by substring matching
    all_gene_columns = [col for col in combined_data.columns 
                       if col not in ['patient_id', 'survival_time', 'event']]
    
    for gene_symbol, ensembl_id in GENE_MAPPING.items():
        # Try to find columns containing the gene symbol or Ensembl ID
        matching_cols = [col for col in all_gene_columns 
                         if gene_symbol in col or ensembl_id in col]
        if matching_cols:
            available_genes.append((gene_symbol, matching_cols[0]))

if not available_genes:
    print("Warning: None of the known cancer genes found in dataset.")
    print("Using top variance genes from the dataset instead.")
    # Get metadata columns
    meta_cols = ['patient_id', 'survival_time', 'event']
    # Calculate variance for each gene
    gene_vars = combined_data.drop(columns=meta_cols).var().sort_values(ascending=False)
    # Get top 8 variable genes
    top_genes = gene_vars.head(8)
    available_genes = [(f"Gene {i+1}", gene) for i, gene in enumerate(top_genes.index)]

print(f"Analyzing {len(available_genes)} genes: {[g[0] for g in available_genes]}")

# 1. Distribution of cancer gene expression
plt.figure(figsize=(15, 10))
for i, (gene_symbol, gene_id) in enumerate(available_genes[:min(8, len(available_genes))]):
    plt.subplot(2, 4, i+1)
    
    # Plot histogram with KDE
    sns.histplot(combined_data[gene_id], kde=True)
    plt.title(f'{gene_symbol} Expression')
    plt.xlabel('Expression Value')
    
    # Add log transformed version as a second line
    if (combined_data[gene_id] > 0).all():
        log_data = np.log1p(combined_data[gene_id])
        twin_ax = plt.twinx()
        sns.kdeplot(log_data, color='r', ax=twin_ax)
        twin_ax.set_ylabel('Log Density', color='r')
        twin_ax.tick_params(axis='y', colors='r')

plt.tight_layout()
plt.savefig('plots/cancer_gene_distributions.png')

# 2. Boxplots by survival status
plt.figure(figsize=(15, 8))
for i, (gene_symbol, gene_id) in enumerate(available_genes[:min(8, len(available_genes))]):
    plt.subplot(2, 4, i+1)
    
    # Create boxplot comparing expression between survived and deceased
    sns.boxplot(x='event', y=gene_id, data=combined_data)
    plt.title(f'{gene_symbol}')
    plt.xlabel('Event (0=Censored, 1=Deceased)')
    plt.ylabel('Expression')

plt.tight_layout()
plt.savefig('plots/cancer_gene_boxplots.png')

# 3. Kaplan-Meier curves stratified by gene expression (high vs low)
plt.figure(figsize=(20, 15))
for i, (gene_symbol, gene_id) in enumerate(available_genes[:min(8, len(available_genes))]):
    plt.subplot(2, 4, i+1)
    
    # Create high/low expression groups based on median
    median_val = combined_data[gene_id].median()
    high_exp = combined_data[gene_id] > median_val
    
    # Plot KM curves for high vs low expression
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(combined_data.loc[high_exp, 'survival_time'], 
                combined_data.loc[high_exp, 'event'], 
                label=f"High {gene_symbol}")
    
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(combined_data.loc[~high_exp, 'survival_time'], 
               combined_data.loc[~high_exp, 'event'], 
               label=f"Low {gene_symbol}")
    
    ax = plt.gca()
    kmf_high.plot_survival_function(ax=ax)
    kmf_low.plot_survival_function(ax=ax)
    
    # Perform log-rank test
    results = logrank_test(combined_data.loc[high_exp, 'survival_time'], 
                          combined_data.loc[~high_exp, 'survival_time'],
                          combined_data.loc[high_exp, 'event'], 
                          combined_data.loc[~high_exp, 'event'])
    
    plt.title(f'{gene_symbol} (p={results.p_value:.4f})')
    plt.xlabel('Time (Days)')
    plt.ylabel('Survival Probability')

plt.tight_layout()
plt.savefig('plots/cancer_gene_survival.png')

print("Cancer gene analysis completed. Plots saved to plots directory.")