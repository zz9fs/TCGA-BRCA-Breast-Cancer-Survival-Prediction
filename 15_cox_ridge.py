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

print("Training Cox Proportional Hazards Model with Ridge (L2) Regularization")

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

# Optional: Reduce dimensionality before model fitting if dataset is very large
if len(gene_cols) > 10000 and dataset_type == "full":
    print("Performing feature preselection before Ridge...")
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

# Define preprocessing and model pipeline
# Using scikit-survival's implementation which handles high-dimensional data better
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('cox', CoxnetSurvivalAnalysis(l1_ratio=0.001))  # Almost pure Ridge (0.001 instead of 0.0)
])

# Define parameter grid for hyperparameter search
# Use a wide range of alphas to ensure we find optimal regularization strength
param_grid = {
    'cox__alphas': [[10.0, 5.0, 1.0, 0.5, 0.1, 0.05, 0.01]]  # Remove small values to avoid numerical issues
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

# Extract coefficients from Ridge model - get the best alpha's coefficients
coefficients = best_model.named_steps['cox'].coef_
print(f"Coefficient shape: {coefficients.shape}")

# Need to get the coefficients for the best alpha value
# Find the best alpha index - we don't have direct access to best_alpha, so get it from alphas parameter
alphas = best_model.named_steps['cox'].alphas_
# Assuming the best alpha is the one with highest score from cross-validation
# Since we're only using one set of alphas, we'll take the coefficients from the first alpha in our list
best_alpha_idx = 0  # Since we only have one alpha set, use the first one
print(f"Available alphas: {alphas}")
print(f"Using alpha index: {best_alpha_idx}")

# Extract coefficients for best alpha
coefficients = coefficients[:, best_alpha_idx]

# Create dataframe of gene coefficients, sorted by absolute value
gene_coef_df = pd.DataFrame({
    'gene': gene_cols,
    'coefficient': coefficients,
    'abs_coefficient': np.abs(coefficients)
})
gene_coef_df = gene_coef_df.sort_values('abs_coefficient', ascending=False)

# Save results
gene_coef_df.to_csv('results/ridge_gene_coefficients.csv', index=False)
with open('models/ridge_cox_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Create visualization of top genes by coefficient magnitude
plt.figure(figsize=(12, 10))
top_genes = gene_coef_df.head(30)
plt.barh(y=top_genes['gene'], width=top_genes['coefficient'])
plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
plt.title('Top 30 Genes with Highest Ridge Coefficients')
plt.xlabel('Coefficient (Log Hazard Ratio)')
plt.tight_layout()
plt.savefig('plots/ridge_top_genes.png')

# Save cross-validation results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.to_csv('results/ridge_cv_results.csv', index=False)

# Final summary
print("\nModel training complete.")
print(f"Ridge Cox model coefficients saved for all {len(gene_cols)} genes")
print(f"Test set C-index: {test_cindex:.4f}")
print(f"Results saved to:\n- results/ridge_gene_coefficients.csv\n- models/ridge_cox_model.pkl")
print(f"Visualizations saved to:\n- plots/ridge_top_genes.png") 