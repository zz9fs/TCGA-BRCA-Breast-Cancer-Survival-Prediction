import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import requests
import json
import time
from collections import Counter
import warnings
from scipy.stats import sem
import re
from matplotlib.ticker import MaxNLocator

# Suppress warnings
warnings.filterwarnings("ignore")

print("Feature Importance and Biological Interpretation Analysis")

# Create directories for outputs
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/features'):
    os.makedirs('plots/features')
if not os.path.exists('results/features'):
    os.makedirs('results/features')

# -------------------- 1. Load Gene Signatures from All Models --------------------

gene_signatures = {}
gene_coefficients = {}
model_names = ['lasso', 'ridge', 'elasticnet', 'decision_tree', 'random_forest']

# Load gene lists and their importance measures
for model_name in model_names:
    # Different files have different formats
    try:
        if model_name in ['lasso', 'elasticnet']:
            # Cox models with gene selection
            df = pd.read_csv(f'results/{model_name}_selected_genes.csv')
            gene_signatures[model_name] = df['gene'].tolist()
            gene_coefficients[model_name] = dict(zip(df['gene'], df['coefficient']))
            print(f"Loaded {len(gene_signatures[model_name])} genes from {model_name}")
            
        elif model_name == 'ridge':
            # Ridge typically keeps all genes but we want top ones
            try:
                df = pd.read_csv('results/ridge_gene_coefficients.csv')
                # Get absolute coefficient for ranking
                df['abs_coefficient'] = df['coefficient'].abs()
                df = df.sort_values('abs_coefficient', ascending=False)
                # Take top 50 genes for comparison
                top_genes = df.head(50)
                gene_signatures[model_name] = top_genes['gene'].tolist()
                gene_coefficients[model_name] = dict(zip(top_genes['gene'], top_genes['coefficient']))
                print(f"Loaded top {len(gene_signatures[model_name])} genes from {model_name}")
            except FileNotFoundError:
                try:
                    # Try alternative file format
                    df = pd.read_csv('results/ridge_selected_genes.csv')
                    gene_signatures[model_name] = df['gene'].tolist()
                    gene_coefficients[model_name] = dict(zip(df['gene'], df['coefficient']))
                    print(f"Loaded {len(gene_signatures[model_name])} genes from {model_name}")
                except FileNotFoundError:
                    print(f"Warning: Could not find gene list for {model_name}")
        
        elif model_name in ['decision_tree', 'random_forest']:
            # Tree models with feature importance
            df = pd.read_csv(f'results/{model_name}_feature_importance.csv')
            df = df.sort_values('importance', ascending=False)
            gene_signatures[model_name] = df['gene'].tolist()
            gene_coefficients[model_name] = dict(zip(df['gene'], df['importance']))
            print(f"Loaded {len(gene_signatures[model_name])} genes from {model_name}")
            
    except FileNotFoundError:
        print(f"Warning: No gene signature found for {model_name}")

if not gene_signatures:
    print("No gene signatures found. Please run model training first.")
    exit(1)

# -------------------- 2. Identify Top Genes from Each Model --------------------

# Function to get top N genes from a model
def get_top_genes(model_name, n=20):
    if model_name not in gene_signatures:
        return []
    
    genes = gene_signatures[model_name]
    if len(genes) <= n:
        return genes
    
    # If we have coefficients, use them to rank
    if model_name in gene_coefficients:
        # Get absolute values for ranking
        coeffs = {gene: abs(val) for gene, val in gene_coefficients[model_name].items()}
        sorted_genes = sorted(coeffs.keys(), key=lambda x: coeffs[x], reverse=True)
        return sorted_genes[:n]
    
    # Otherwise just take first N from the list (assumes already sorted)
    return genes[:n]

# Get top 20 genes from each model
top_genes = {}
for model in gene_signatures:
    top_genes[model] = get_top_genes(model, 20)
    print(f"Top 20 genes from {model}: {', '.join(top_genes[model][:5])}, ...")

# -------------------- 3. Compare Gene Lists Across Models --------------------

# Find genes that appear in multiple models
all_genes = []
for model in top_genes:
    all_genes.extend(top_genes[model])

gene_counts = Counter(all_genes)
consensus_genes = [gene for gene, count in gene_counts.items() if count > 1]

print(f"\nFound {len(consensus_genes)} genes that appear in multiple models")

# Create a model-gene matrix (which models selected which genes)
unique_genes = set(all_genes)
model_gene_matrix = pd.DataFrame(0, index=list(unique_genes), columns=list(gene_signatures.keys()))

for model, genes in top_genes.items():
    model_gene_matrix.loc[genes, model] = 1

# Add a count column
model_gene_matrix['models_count'] = model_gene_matrix.sum(axis=1)
model_gene_matrix = model_gene_matrix.sort_values('models_count', ascending=False)

# Save the cross-model gene comparison
model_gene_matrix.to_csv('results/features/cross_model_gene_comparison.csv')

# -------------------- 4. Identify Linear vs. Nonlinear Gene Sets --------------------

# Group models by type
linear_models = ['lasso', 'ridge', 'elasticnet']
nonlinear_models = ['decision_tree', 'random_forest']

# Get genes appearing in linear models
linear_genes = []
for model in linear_models:
    if model in top_genes:
        linear_genes.extend(top_genes[model])
linear_gene_set = set(linear_genes)

# Get genes appearing in nonlinear models
nonlinear_genes = []
for model in nonlinear_models:
    if model in top_genes:
        nonlinear_genes.extend(top_genes[model])
nonlinear_gene_set = set(nonlinear_genes)

# Find unique and common genes
common_gene_set = linear_gene_set.intersection(nonlinear_gene_set)
linear_only_genes = linear_gene_set - common_gene_set
nonlinear_only_genes = nonlinear_gene_set - common_gene_set

print(f"\nLinear vs. Nonlinear Gene Comparison:")
print(f"Common to both: {len(common_gene_set)} genes")
print(f"Linear models only: {len(linear_only_genes)} genes")
print(f"Nonlinear models only: {len(nonlinear_only_genes)} genes")

# Create Venn diagram (if matplotlib-venn is available)
try:
    from matplotlib_venn import venn2
    
    plt.figure(figsize=(10, 8))
    venn2([linear_gene_set, nonlinear_gene_set], 
          set_labels=('Linear Models', 'Nonlinear Models'))
    plt.title('Gene Overlap: Linear vs. Nonlinear Models')
    plt.savefig('plots/features/linear_nonlinear_venn.png')
    print("Created Venn diagram of linear vs. nonlinear gene overlap")
except ImportError:
    print("matplotlib-venn not installed. Skipping Venn diagram.")

# -------------------- 5. Visualize Key Gene Importance Across Models --------------------

# Bar chart of top consensus genes (appearing in most models)
top_consensus = model_gene_matrix.head(min(15, len(model_gene_matrix)))

plt.figure(figsize=(12, 8))
ax = sns.heatmap(top_consensus.iloc[:, :-1], cmap='Blues', cbar=False)
plt.title('Top Genes Selected Across Models')
plt.tight_layout()
plt.savefig('plots/features/top_genes_heatmap.png')

# Bar chart of top genes with their importance/coefficients
# Choose a primary model with coefficients (prefer Elastic Net or Random Forest)
primary_model = None
for model in ['elasticnet', 'lasso', 'random_forest', 'ridge', 'decision_tree']:
    if model in gene_coefficients and len(gene_coefficients[model]) > 0:
        primary_model = model
        break

if primary_model:
    # Create coefficient bar chart for top genes
    top_n = min(20, len(gene_signatures[primary_model]))
    top_primary_genes = get_top_genes(primary_model, top_n)
    
    coef_values = [gene_coefficients[primary_model][gene] for gene in top_primary_genes]
    
    # For linear models, sort by coefficient value (positive to negative)
    if primary_model in linear_models:
        # Sort by coefficient value
        sorted_idx = np.argsort(coef_values)
        top_primary_genes = [top_primary_genes[i] for i in sorted_idx]
        coef_values = [coef_values[i] for i in sorted_idx]
        
        # Determine color based on positive/negative coefficient
        colors = ['red' if val > 0 else 'blue' for val in coef_values]
        
        plt.figure(figsize=(12, 8))
        plt.barh(top_primary_genes, coef_values, color=colors)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Top Gene Coefficients from {primary_model.upper()} Model')
        plt.xlabel('Coefficient (Log Hazard Ratio)')
        plt.tight_layout()
        plt.savefig(f'plots/features/{primary_model}_coefficients.png')
    else:
        # For tree models, sort by importance (descending)
        sorted_idx = np.argsort(coef_values)[::-1]
        top_primary_genes = [top_primary_genes[i] for i in sorted_idx]
        coef_values = [coef_values[i] for i in sorted_idx]
        
        plt.figure(figsize=(12, 8))
        plt.barh(top_primary_genes, coef_values)
        plt.title(f'Top Gene Importance from {primary_model.upper()} Model')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.savefig(f'plots/features/{primary_model}_importance.png')
    
    print(f"Created visualization of top genes from {primary_model}")

# -------------------- 6. Perform Simple Pathway Analysis --------------------

# Function to clean gene IDs (extract gene symbol from Ensembl if needed)
def clean_gene_ids(gene_list):
    cleaned_genes = []
    for gene in gene_list:
        # If it's an Ensembl ID (e.g., ENSG00000123456.7), extract the base ID
        if gene.startswith('ENSG'):
            # Remove version number if present
            base_id = gene.split('.')[0]
            cleaned_genes.append(base_id)
        else:
            cleaned_genes.append(gene)
    return cleaned_genes

# Get top genes to analyze
if common_gene_set:
    # Use genes common to multiple models
    genes_for_pathway = list(common_gene_set)
elif primary_model:
    # Use top genes from primary model
    genes_for_pathway = get_top_genes(primary_model, 30)
else:
    # Use top consensus genes
    genes_for_pathway = top_consensus.index.tolist()[:30]

# Clean gene IDs
clean_genes = clean_gene_ids(genes_for_pathway)

# Check if we're dealing with Ensembl IDs
is_ensembl = any(gene.startswith('ENSG') for gene in clean_genes)

print(f"\nPreparing pathway analysis for {len(clean_genes)} genes")
print(f"Using Ensembl IDs: {is_ensembl}")

# Print genes for manual pathway analysis
with open('results/features/genes_for_pathway_analysis.txt', 'w') as f:
    f.write("# Genes for Pathway Analysis\n")
    f.write("# Copy these IDs for use with tools like Enrichr, DAVID, or g:Profiler\n\n")
    for gene in clean_genes:
        f.write(f"{gene}\n")

print("Saved gene list for pathway analysis to results/features/genes_for_pathway_analysis.txt")
print("Please use this list with online tools like Enrichr, DAVID, or g:Profiler for pathway enrichment")

# -------------------- 7. Generate Final Gene Signature and Literature Evidence Template --------------------

# Select final gene signature based on best model
final_model = None
try:
    # Check for comprehensive comparison file
    comparison_df = pd.read_csv('results/comprehensive_model_comparison.csv')
    final_model = comparison_df.iloc[0]['Model']
    print(f"\nSelected {final_model} as final model based on performance metrics")
except (FileNotFoundError, IndexError, KeyError):
    # Fall back to consensus approach
    if primary_model:
        final_model = primary_model
        print(f"\nSelected {final_model} as final model (default primary model)")
    else:
        # No clear choice, use model with most genes available
        models_with_genes = {model: len(genes) for model, genes in gene_signatures.items() if len(genes) > 0}
        if models_with_genes:
            final_model = max(models_with_genes.items(), key=lambda x: x[1])[0]
            print(f"\nSelected {final_model} as final model (most genes available)")

# Get final gene signature
if final_model:
    final_signature = get_top_genes(final_model, 20)
    
    # Create CSV template for literature evidence
    literature_template = pd.DataFrame({
        'gene': final_signature,
        'direction': [np.sign(gene_coefficients[final_model].get(gene, 0)) if final_model in linear_models else 'N/A' 
                      for gene in final_signature],
        'importance_score': [abs(gene_coefficients[final_model].get(gene, 0)) for gene in final_signature],
        'known_in_literature': '',
        'evidence_summary': '',
        'reference': ''
    })
    
    literature_template.to_csv('results/features/final_gene_signature_evidence.csv', index=False)
    
    print(f"Created template for literature evidence of final gene signature from {final_model}")
    print("Please edit results/features/final_gene_signature_evidence.csv to add literature evidence")

# -------------------- 8. Summarize Results --------------------

# Generate a summary report
summary = {
    'total_genes_analyzed': len(unique_genes),
    'genes_in_multiple_models': len(consensus_genes),
    'linear_only_genes': len(linear_only_genes),
    'nonlinear_only_genes': len(nonlinear_only_genes),
    'common_genes': len(common_gene_set),
    'final_model': final_model,
    'final_signature_size': len(final_signature) if 'final_signature' in locals() else 0
}

# Save summary as JSON
with open('results/features/feature_analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=4)

print("\nFeature importance analysis complete.")
print(f"Final gene signature from {final_model}: {', '.join(final_signature[:5])}, ...")
print("Results saved to results/features/ and plots/features/ directories") 