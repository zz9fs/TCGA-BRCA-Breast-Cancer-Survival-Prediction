import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Comparing Cox Models with Different Regularization Approaches")

# Create directories for outputs if they don't exist
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('results'):
    os.makedirs('results')

# ---- PART 1: Compare Gene Signatures ----
print("\nComparing gene signatures between models...")

# Load gene signatures from result files
gene_signatures = {}
model_names = ['lasso', 'ridge', 'elasticnet']

for model_name in model_names:
    try:
        # Try to load selected genes from CSV
        signature_df = pd.read_csv(f'results/{model_name}_selected_genes.csv')
        gene_signatures[model_name] = signature_df['gene'].tolist()
        print(f"Loaded {len(gene_signatures[model_name])} genes for {model_name}")
    except FileNotFoundError:
        try:
            # For Ridge, try alternative filename
            if model_name == 'ridge':
                coef_df = pd.read_csv(f'results/ridge_gene_coefficients.csv')
                top_genes = coef_df.sort_values('abs_coefficient', ascending=False).head(50)['gene'].tolist()
                gene_signatures[model_name] = top_genes
                print(f"Loaded top 50 genes for {model_name}")
        except FileNotFoundError:
            print(f"Warning: Could not find gene signature for {model_name}")

# Print gene signature size comparison
if gene_signatures:
    print("\nGene signature sizes:")
    for model, genes in gene_signatures.items():
        print(f"{model}: {len(genes)} genes")

    # Find common genes between models
    if len(gene_signatures) > 1:
        common_genes = set.intersection(*[set(genes) for genes in gene_signatures.values()])
        print(f"\nFound {len(common_genes)} genes common to all models:")
        if len(common_genes) < 20:  # Show them all if there are fewer than 20
            print(", ".join(common_genes))
        else:
            print(", ".join(list(common_genes)[:10]) + "... and others")
        
        # Save common genes to CSV
        pd.DataFrame({'common_genes': list(common_genes)}).to_csv('results/common_genes.csv', index=False)
    
    # Try to create Venn diagram
    try:
        from matplotlib_venn import venn2, venn3
        
        if len(gene_signatures) == 2:
            plt.figure(figsize=(10, 8))
            models_to_plot = list(gene_signatures.keys())
            venn2([set(gene_signatures[models_to_plot[0]]), 
                set(gene_signatures[models_to_plot[1]])],
                set_labels=models_to_plot)
            plt.title('Overlap of Genes Selected by Different Models')
            plt.savefig('plots/gene_signature_venn2.png')
            print("\nCreated Venn diagram of overlapping genes (saved to plots/gene_signature_venn2.png)")
            
        elif len(gene_signatures) == 3:
            plt.figure(figsize=(10, 8))
            venn3([set(gene_signatures['lasso']), 
                set(gene_signatures['ridge']), 
                set(gene_signatures['elasticnet'])],
                set_labels=['LASSO', 'Ridge', 'Elastic Net'])
            plt.title('Overlap of Genes Selected by Different Models')
            plt.savefig('plots/gene_signature_venn3.png')
            print("\nCreated Venn diagram of overlapping genes (saved to plots/gene_signature_venn3.png)")
    except (ImportError, NameError):
        print("\nCould not create Venn diagram (matplotlib-venn not available)")

# ---- PART 2: Compare Model Performance ----
print("\nComparing model performance metrics...")

# Extract performance from individual model results
performance = {}

try:
    # From LASSO model
    with open('results/lasso_cv_results.csv', 'r') as f:
        lasso_cv = pd.read_csv(f)
        best_lasso = lasso_cv['mean_test_score'].max()
        performance['lasso'] = {
            'c_index': best_lasso,
            'n_features': len(gene_signatures.get('lasso', [])),
        }
        print(f"LASSO C-index: {best_lasso:.4f}")
except FileNotFoundError:
    print("Could not find LASSO results")

try:
    # From Ridge model
    with open('results/ridge_cv_results.csv', 'r') as f:
        ridge_cv = pd.read_csv(f)
        best_ridge = ridge_cv['mean_test_score'].max()
        performance['ridge'] = {
            'c_index': best_ridge,
            'n_features': len(gene_signatures.get('ridge', [])),
        }
        print(f"Ridge C-index: {best_ridge:.4f}")
except FileNotFoundError:
    print("Could not find Ridge results")

try:
    # From Elastic Net model
    with open('results/elasticnet_cv_results.csv', 'r') as f:
        enet_cv = pd.read_csv(f)
        best_enet = enet_cv['mean_test_score'].max()
        performance['elasticnet'] = {
            'c_index': best_enet,
            'n_features': len(gene_signatures.get('elasticnet', [])),
        }
        print(f"Elastic Net C-index: {best_enet:.4f}")
except FileNotFoundError:
    print("Could not find Elastic Net results")

# Use test results if available
for model_name in model_names:
    try:
        # Look for the best model test result
        with open(f'results/{model_name}_test_results.txt', 'r') as f:
            for line in f:
                if 'Test set C-index:' in line:
                    c_index = float(line.split(':')[1].strip())
                    if model_name in performance:
                        performance[model_name]['test_c_index'] = c_index
                    else:
                        performance[model_name] = {
                            'c_index': c_index,
                            'test_c_index': c_index,
                            'n_features': len(gene_signatures.get(model_name, [])),
                        }
                    print(f"{model_name} Test C-index: {c_index:.4f}")
                    break
    except FileNotFoundError:
        pass

# Create performance comparison table
if performance:
    performance_df = pd.DataFrame(performance).T
    performance_df.index.name = 'Model'
    performance_df = performance_df.reset_index()
    performance_df = performance_df.sort_values('c_index', ascending=False)
    performance_df.to_csv('results/model_performance_comparison.csv', index=False)

    print("\nModel Performance Comparison:")
    print(performance_df)

    # Create bar chart of C-index comparison
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='c_index', data=performance_df)
    plt.title('Model Performance Comparison (C-index)')
    plt.ylim(0.5, max(0.85, performance_df['c_index'].max() + 0.05))  # Start at 0.5 (random)
    for i, row in enumerate(performance_df.itertuples()):
        plt.text(i, row.c_index + 0.01, f'{row.c_index:.4f}', ha='center')
    plt.tight_layout()
    plt.savefig('plots/model_cindex_comparison.png')
    print("Created C-index comparison plot (saved to plots/model_cindex_comparison.png)")

    # Create scatter plot of model complexity vs. performance
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='n_features', y='c_index', data=performance_df, s=100)
    plt.title('Model Complexity vs. Performance')
    plt.xlabel('Number of Features')
    plt.ylabel('C-index')
    for i, row in enumerate(performance_df.itertuples()):
        plt.annotate(row.Model, (row.n_features, row.c_index), 
                    xytext=(5, 5), textcoords='offset points')
    plt.tight_layout()
    plt.savefig('plots/model_complexity_vs_performance.png')
    print("Created complexity vs. performance plot (saved to plots/model_complexity_vs_performance.png)")

    # Identify the best model
    best_model_name = performance_df.iloc[0]['Model']
    best_score = performance_df.iloc[0]['c_index']
    print(f"\nBest performing model: {best_model_name} (C-index: {best_score:.4f})")

print("\nComparison analysis complete. Results saved to results/ and plots/ directories.")