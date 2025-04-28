import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index
import pickle
import warnings
from sklearn.feature_selection import VarianceThreshold
from lifelines.statistics import logrank_test
from tqdm import tqdm  # For progress bar

# Suppress convergence warnings during grid search
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Training Cox Proportional Hazards Model with Elastic Net (L1+L2) Regularization")

# Create directories for outputs
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load training and testing data
try:
    # Try full feature dataset first
    train_data = pd.read_csv('processed_data/train_data_capped.csv')
    test_data = pd.read_csv('processed_data/test_data_capped.csv')
    dataset_type = "full"
    print(f"Using full feature dataset: {train_data.shape[1]} features")
except FileNotFoundError:
    # Fall back to reduced feature dataset
    train_data = pd.read_csv('processed_data/train_data_survival_fixed.csv')
    test_data = pd.read_csv('processed_data/test_data_survival_fixed.csv')
    dataset_type = "reduced"
    print(f"Using reduced feature dataset: {train_data.shape[1]} features")

# Separate features and target
meta_cols = ['patient_id', 'survival_time', 'event']
gene_cols = [col for col in train_data.columns if col not in meta_cols]

# Convert data to sklearn-survival format
y_train = np.array([(bool(e), t) for e, t in zip(train_data['event'], train_data['survival_time'])], 
                  dtype=[('event', bool), ('time', float)])
X_train = train_data[gene_cols].values

y_test = np.array([(bool(e), t) for e, t in zip(test_data['event'], test_data['survival_time'])], 
                 dtype=[('event', bool), ('time', float)])
X_test = test_data[gene_cols].values

# Check for class imbalance and potentially use balanced data
print(f"Training data: {len(train_data)} samples, {sum(train_data['event'])} events ({sum(train_data['event'])/len(train_data):.1%})")

# Optional: Reduce dimensionality before model fitting (as we did with LASSO and Ridge)
if len(gene_cols) > 10000 and dataset_type == "full":
    print("Performing feature preselection before Elastic Net...")
    # Use variance filtering to select top 5000 genes
    selector = VarianceThreshold()
    X_var = selector.fit_transform(X_train)
    var_scores = selector.variances_
    top_indices = np.argsort(-var_scores)[:5000]  # Keep top 5000 variable genes
    
    X_train = X_train[:, top_indices]
    X_test = X_test[:, top_indices]
    gene_cols = [gene_cols[i] for i in top_indices]
    print(f"Reduced feature set to {len(gene_cols)} genes based on variance")

# Add univariate filtering
print("Performing univariate filtering for survival-associated genes...")
p_values = []
for i in tqdm(range(X_train.shape[1])):
    # Split by median expression
    median = np.median(X_train[:, i])
    high_exp = X_train[:, i] > median
    
    # Compute log-rank test
    result = logrank_test(
        y_train['time'][high_exp], y_train['time'][~high_exp],
        y_train['event'][high_exp], y_train['event'][~high_exp]
    )
    p_values.append(result.p_value)

# Keep top 500 genes with lowest p-values
top_survival_indices = np.argsort(p_values)[:500]
X_train = X_train[:, top_survival_indices]
X_test = X_test[:, top_survival_indices]
gene_cols = [gene_cols[i] for i in top_survival_indices]
print(f"Further reduced to {len(gene_cols)} survival-associated genes")

# Define preprocessing and model pipeline with more limited parameters
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('cox', CoxnetSurvivalAnalysis(max_iter=50000)) # Add max_iter to prevent excessive runtime
])

# Define simplified parameter grid - use much smaller alpha values to prevent all zeros
param_grid = {
    'cox__alphas': [[1.0, 0.1, 0.01]],  # Using smaller alpha values to avoid all-zero coefficients
    'cox__l1_ratio': [0.3, 0.5, 0.7]  # Focus on middle l1_ratio values
}

# Implement k-fold cross-validation with k=5
n_folds = 5
cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Use concordance index as our scoring metric
def c_index_scorer(estimator, X, y):
    prediction = estimator.predict(X)
    return concordance_index_censored(y['event'], y['time'], prediction)[0]

# Create GridSearchCV object
grid_search = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring=c_index_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

print("Starting hyperparameter grid search with cross-validation...")
grid_search.fit(X_train, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best hyperparameters: {best_params}")
print(f"Best cross-validated C-index: {best_score:.4f}")

# Evaluate on test set
test_prediction = best_model.predict(X_test)
test_cindex = concordance_index_censored(y_test['event'], y_test['time'], test_prediction)[0]
print(f"Test set C-index: {test_cindex:.4f}")

# Extract coefficients from Elastic Net model
cox_model = best_model.named_steps['cox']
alphas = cox_model.alphas_
print(f"Available alphas: {alphas}")

# Find which alpha in our original grid matches the best one
# The default behavior of CoxnetSurvivalAnalysis is to use the last alpha (smallest)
# So we'll get the last index which should have most non-zero coefficients
best_alpha_idx = len(alphas) - 1  # Use last alpha (least regularization)
print(f"Using alpha index: {best_alpha_idx} (alpha={alphas[best_alpha_idx]:.4f})")

# Get coefficients
coefficients = cox_model.coef_
if hasattr(coefficients, 'shape') and len(coefficients.shape) > 1:
    # If coefficients is multi-dimensional, get appropriate column
    coefficients = coefficients[:, best_alpha_idx]

# Find nonzero coefficients with a small threshold to account for numerical precision
nonzero_idx = np.where(np.abs(coefficients) > 1e-5)[0]  # Use small threshold instead of zero
nonzero_coeffs = coefficients[nonzero_idx]
selected_genes = [gene_cols[i] for i in nonzero_idx]

print(f"Elastic Net selected {len(selected_genes)} genes out of {len(gene_cols)}")

# Create dataframe of selected genes and their coefficients
gene_coef_df = pd.DataFrame({
    'gene': selected_genes,
    'coefficient': nonzero_coeffs
})
gene_coef_df = gene_coef_df.sort_values('coefficient', ascending=False)

# Save results
gene_coef_df.to_csv('results/elasticnet_selected_genes.csv', index=False)
with open('models/elasticnet_cox_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Create visualization of selected genes
plt.figure(figsize=(12, max(8, len(selected_genes) * 0.25)))
if len(selected_genes) > 30:
    # If many genes, just show top and bottom 15
    top_genes = gene_coef_df.head(15)
    bottom_genes = gene_coef_df.tail(15)
    plot_df = pd.concat([top_genes, bottom_genes])
    plt.barh(y=plot_df['gene'], width=plot_df['coefficient'])
    plt.title('Top and Bottom 15 Genes Selected by Elastic Net Cox Model')
else:
    # If fewer genes, show all
    plt.barh(y=gene_coef_df['gene'], width=gene_coef_df['coefficient'])
    plt.title('Genes Selected by Elastic Net Cox Model')

plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.xlabel('Coefficient (Log Hazard Ratio)')
plt.tight_layout()
plt.savefig('plots/elasticnet_selected_genes.png')

# Save cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('results/elasticnet_cv_results.csv', index=False)

# Create heatmap of C-index scores across l1_ratio and alpha values
scores = grid_search.cv_results_['mean_test_score']
l1_ratios = param_grid['cox__l1_ratio']
alphas = param_grid['cox__alphas'][0]

# Reshape scores for heatmap
score_grid = np.zeros((len(l1_ratios), len(alphas)))
for i, l1_ratio in enumerate(l1_ratios):
    for j, alpha in enumerate(alphas):
        # Find the index in the cv_results that matches this parameter combination
        idx = 0
        for k, params in enumerate(grid_search.cv_results_['params']):
            if params['cox__l1_ratio'] == l1_ratio and alpha in params['cox__alphas']:
                idx = k
                break
        score_grid[i, j] = scores[idx]

# Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(score_grid, annot=True, fmt=".3f", cmap="YlGnBu",
            xticklabels=[f"{a:.2f}" for a in alphas],
            yticklabels=[f"{l:.1f}" for l in l1_ratios])
plt.title('Cross-Validated C-index by Elastic Net Parameters')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('L1 Ratio (0=Ridge, 1=LASSO)')
plt.tight_layout()
plt.savefig('plots/elasticnet_parameter_heatmap.png')

# Final summary
print("\nModel training complete.")
print(f"Elastic Net Cox model selected {len(selected_genes)} genes as predictors")
print(f"Test set C-index: {test_cindex:.4f}")
print(f"Results saved to:\n- results/elasticnet_selected_genes.csv\n- models/elasticnet_cox_model.pkl")
print(f"Visualizations saved to:\n- plots/elasticnet_selected_genes.png\n- plots/elasticnet_parameter_heatmap.png") 