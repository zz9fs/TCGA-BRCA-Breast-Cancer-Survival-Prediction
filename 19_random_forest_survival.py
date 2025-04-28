import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from lifelines.statistics import logrank_test
import pickle
import warnings
from tqdm import tqdm
import shap
from sklearn.inspection import permutation_importance

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Training Random Survival Forest Model")

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

# Convert data to scikit-survival format
y_train = Surv.from_arrays(train_data['event'].astype(bool), train_data['survival_time'])
X_train = train_data[gene_cols].values

y_test = Surv.from_arrays(test_data['event'].astype(bool), test_data['survival_time'])
X_test = test_data[gene_cols].values

# Check for class imbalance
print(f"Training data: {len(train_data)} samples, {sum(train_data['event'])} events ({sum(train_data['event'])/len(train_data):.1%})")

# Feature selection: First use variance filtering
if len(gene_cols) > 1000 and dataset_type == "full":
    print("Performing variance-based feature selection...")
    selector = VarianceThreshold()
    X_var = selector.fit_transform(X_train)
    var_scores = selector.variances_
    top_indices = np.argsort(-var_scores)[:1000]  # Keep top 1000 variable genes
    
    X_train = X_train[:, top_indices]
    X_test = X_test[:, top_indices]
    selected_genes = [gene_cols[i] for i in top_indices]
    print(f"Selected {len(selected_genes)} genes based on variance")
else:
    selected_genes = gene_cols

# Feature selection: Then use univariate filtering with log-rank test
if len(selected_genes) > 100:
    print("Performing univariate filtering for survival-associated genes...")
    p_values = []
    for i in tqdm(range(X_train.shape[1])):
        # Split by median expression
        median = np.median(X_train[:, i])
        high_exp = X_train[:, i] > median
        
        # Get survival times and events for each group
        high_times = train_data.loc[high_exp, 'survival_time'].values
        high_events = train_data.loc[high_exp, 'event'].values
        low_times = train_data.loc[~high_exp, 'survival_time'].values
        low_events = train_data.loc[~high_exp, 'event'].values
        
        # Compute log-rank test
        result = logrank_test(high_times, low_times, high_events, low_events)
        p_values.append(result.p_value)
    
    # Keep top 100 genes with lowest p-values
    top_survival_indices = np.argsort(p_values)[:100]
    X_train = X_train[:, top_survival_indices]
    X_test = X_test[:, top_survival_indices]
    selected_genes = [selected_genes[i] for i in top_survival_indices]
    print(f"Further reduced to {len(selected_genes)} survival-associated genes")

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt', 'log2']
}

# Define Random Survival Forest model
rsf = RandomSurvivalForest(random_state=42, n_jobs=-1)

# Use custom scorer for grid search with concordance index
def rsf_score(estimator, X, y):
    prediction = estimator.predict(X)
    result = concordance_index_censored(y['event'], y['time'], prediction)
    return result[0]  # Return c-index

# Setup cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform grid search
print("Starting grid search with cross-validation...")
grid_search = GridSearchCV(
    estimator=rsf,
    param_grid=param_grid,
    cv=cv,
    scoring=rsf_score,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Get best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validated C-index: {best_score:.4f}")

# Evaluate on test set
test_pred = best_model.predict(X_test)
test_cindex = concordance_index_censored(y_test['event'], y_test['time'], test_pred)[0]
print(f"Test set C-index: {test_cindex:.4f}")

# Use permutation importance instead of the built-in feature_importances_
print("Calculating permutation feature importance...")
perm_importance = permutation_importance(
    best_model, 
    X_test, 
    y_test, 
    n_repeats=10, 
    random_state=42,
    scoring=rsf_score  # Use the same scoring function
)

# Get sorted indices of feature importance
importances = perm_importance.importances_mean
indices = np.argsort(importances)[::-1]
top_genes = [selected_genes[i] for i in indices[:20]]  # Top 20 genes

# Create dataframe of feature importances
importance_df = pd.DataFrame({
    'gene': [selected_genes[i] for i in indices],
    'importance': importances[indices],
    'std': perm_importance.importances_std[indices]
})
importance_df.to_csv('results/random_forest_feature_importance.csv', index=False)

# Visualize feature importances
plt.figure(figsize=(12, 8))
plt.bar(range(20), importances[indices[:20]], yerr=perm_importance.importances_std[indices[:20]])
plt.xticks(range(20), top_genes, rotation=90)
plt.xlabel('Gene')
plt.ylabel('Feature Importance')
plt.title('Top 20 Feature Importances - Random Survival Forest')
plt.tight_layout()
plt.savefig('plots/random_forest_feature_importance.png')

# Save the model
with open('models/random_forest_survival_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Update comparison with previous models
print("\nComparison with previous models:")
try:
    comparison_df = pd.read_csv('results/model_performance_comparison.csv')
    
    # Add random forest results
    new_row = pd.DataFrame([{
        'Model': 'random_forest',
        'c_index': test_cindex,
        'n_features': len(selected_genes)
    }])
    
    comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
    comparison_df = comparison_df.sort_values('c_index', ascending=False)
    
    print(comparison_df)
    comparison_df.to_csv('results/model_performance_comparison.csv', index=False)
except FileNotFoundError:
    print("No comparison file found. Creating new comparison.")
    pd.DataFrame([{
        'Model': 'random_forest',
        'c_index': test_cindex,
        'n_features': len(selected_genes)
    }]).to_csv('results/model_performance_comparison.csv', index=False)

# Check for overlap with Cox model selected genes
try:
    cox_genes = pd.read_csv('results/common_genes.csv')['common_genes'].tolist()
    rf_top_genes = importance_df.head(20)['gene'].tolist()
    
    overlap = set(cox_genes).intersection(set(rf_top_genes))
    
    print(f"\nFound {len(overlap)} genes in both Cox models and top 20 RF genes:")
    if overlap:
        print(", ".join(overlap))
    
    # Save these consensus genes
    pd.DataFrame({'consensus_genes': list(overlap)}).to_csv('results/consensus_genes.csv', index=False)
except FileNotFoundError:
    print("\nCould not find common Cox genes file for comparison")

print("\nRandom Survival Forest model training complete:")
print(f"- C-index on test set: {test_cindex:.4f}")
print(f"- Model uses {len(selected_genes)} features")
print("- Feature importances saved to results/random_forest_feature_importance.csv")
print("- Visualizations saved to plots/random_forest_*.png") 