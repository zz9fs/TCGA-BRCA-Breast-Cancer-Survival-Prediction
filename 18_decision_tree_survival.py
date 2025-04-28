import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from lifelines.statistics import logrank_test
import pickle
import warnings
from tqdm import tqdm
import seaborn as sns

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Training Decision Tree for Survival Prediction (5-Year Classification Approach)")

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

# Define 5-year threshold for classification
FIVE_YEAR_THRESHOLD = 365 * 5  # 5 years in days

# Convert survival data to 5-year binary classification
def convert_to_classification(df):
    # 1: Died within 5 years
    # 0: Survived beyond 5 years OR censored beyond 5 years
    y_class = []
    for _, row in df.iterrows():
        if row['event'] == 1 and row['survival_time'] <= FIVE_YEAR_THRESHOLD:
            y_class.append(1)  # Died within 5 years
        else:
            # Either survived beyond 5 years or was censored beyond 5 years
            if row['survival_time'] > FIVE_YEAR_THRESHOLD:
                y_class.append(0)  # Definitely survived beyond 5 years
            else:
                # Censored before 5 years - can't know for sure
                # For now, we'll classify as survived (but this is a simplification)
                y_class.append(0)
    return np.array(y_class)

# Prepare features and target
meta_cols = ['patient_id', 'survival_time', 'event']
gene_cols = [col for col in train_data.columns if col not in meta_cols]

X_train = train_data[gene_cols].values
y_train = convert_to_classification(train_data)

X_test = test_data[gene_cols].values
y_test = convert_to_classification(test_data)

# Print class distribution
print(f"Training data: {len(y_train)} samples")
print(f"Class distribution: {np.sum(y_train)} died within 5 years ({np.sum(y_train)/len(y_train):.1%}), " 
      f"{len(y_train) - np.sum(y_train)} survived beyond 5 years ({1-np.sum(y_train)/len(y_train):.1%})")

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

# Feature selection: Then use univariate filtering
if len(selected_genes) > 100:
    print("Performing univariate filtering for survival-associated genes...")
    # Use Chi-squared or similar for classification problems
    from sklearn.feature_selection import SelectKBest, chi2
    
    # Feature values must be non-negative for chi2
    # Using a scaler that makes all values non-negative
    min_values = np.min(X_train, axis=0)
    X_train_shifted = X_train - min_values + 0.1  # Add 0.1 to avoid zeros
    X_test_shifted = X_test - min_values + 0.1
    
    # Select top 50 features
    selector = SelectKBest(chi2, k=50)
    X_train = selector.fit_transform(X_train_shifted, y_train)
    X_test = selector.transform(X_test_shifted)
    
    # Update selected genes
    mask = selector.get_support()
    selected_genes = [selected_genes[i] for i in range(len(selected_genes)) if mask[i]]
    print(f"Further reduced to {len(selected_genes)} survival-associated genes")

# Standardize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid for Decision Tree
param_grid = {
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [5, 10, 20],
    'min_samples_leaf': [5, 10, 15],
    'class_weight': [None, 'balanced']
}

# Cross-validation
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='roc_auc',  # Use ROC AUC as the score metric
    cv=kf,
    n_jobs=-1,
    verbose=1
)

print("Starting grid search with cross-validation...")
grid_search.fit(X_train, y_train)

# Best parameters and model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
cv_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best cross-validation ROC AUC: {cv_score:.4f}")

# Evaluate on test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_pred)
test_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Test set accuracy: {test_accuracy:.4f}")
print(f"Test set ROC AUC: {test_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Survived >5yrs', 'Died ≤5yrs'],
            yticklabels=['Survived >5yrs', 'Died ≤5yrs'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/decision_tree_confusion_matrix.png')

# Feature importances
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Save feature importances
feature_importance_df = pd.DataFrame({
    'gene': [selected_genes[i] for i in indices],
    'importance': importances[indices]
})
feature_importance_df.to_csv('results/decision_tree_feature_importance.csv', index=False)

# Plot top 20 feature importances
n_top_features = min(20, len(selected_genes))
plt.figure(figsize=(12, 8))
plt.bar(range(n_top_features), importances[indices[:n_top_features]], align='center')
plt.xticks(range(n_top_features), [selected_genes[i] for i in indices[:n_top_features]], rotation=90)
plt.title('Top Feature Importances in Decision Tree')
plt.tight_layout()
plt.savefig('plots/decision_tree_feature_importance.png')

# Plot decision tree (limit to max_depth=3 for visualization)
plt.figure(figsize=(20, 10))
visualization_model = DecisionTreeClassifier(
    max_depth=3, 
    random_state=42,
    **{k: v for k, v in best_params.items() if k != 'max_depth'}
)
visualization_model.fit(X_train, y_train)
plot_tree(
    visualization_model, 
    feature_names=selected_genes,
    class_names=['Survived >5yrs', 'Died ≤5yrs'],
    filled=True, 
    rounded=True
)
plt.title('Decision Tree for 5-Year Survival Prediction (Simplified for Visualization)')
plt.savefig('plots/decision_tree_visualization.png', dpi=300, bbox_inches='tight')

# Save model
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save results
results = {
    'best_params': best_params,
    'cv_score': cv_score,
    'test_accuracy': test_accuracy,
    'test_auc': test_auc,
    'n_features': len(selected_genes),
    'selected_genes': selected_genes
}

# Compare with linear models
print("\nComparison with previous models:")
try:
    comparison_df = pd.read_csv('results/model_performance_comparison.csv')
    
    # Add decision tree results
    new_row = pd.DataFrame([{
        'Model': 'decision_tree',
        'c_index': test_auc,  # Note: Using AUC as approximation of C-index
        'n_features': len(selected_genes)
    }])
    
    comparison_df = pd.concat([comparison_df, new_row], ignore_index=True)
    comparison_df = comparison_df.sort_values('c_index', ascending=False)
    
    print(comparison_df)
    comparison_df.to_csv('results/model_performance_comparison.csv', index=False)
except FileNotFoundError:
    print("No comparison file found. Creating new comparison.")
    pd.DataFrame([{
        'Model': 'decision_tree',
        'c_index': test_auc,
        'n_features': len(selected_genes)
    }]).to_csv('results/model_performance_comparison.csv', index=False)

print("\nDecision Tree model training complete:")
print(f"- ROC AUC on test set: {test_auc:.4f}")
print(f"- Model uses {len(selected_genes)} features")
print(f"- Visualization saved to plots/decision_tree_visualization.png")
print(f"- Feature importances saved to results/decision_tree_feature_importance.csv") 