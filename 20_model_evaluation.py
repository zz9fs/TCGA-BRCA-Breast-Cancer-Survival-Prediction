import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import mean_squared_error, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
import re

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

print("Comprehensive Model Evaluation and Comparison")

# Create directories for outputs
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('plots'):
    os.makedirs('plots')
if not os.path.exists('plots/evaluation'):
    os.makedirs('plots/evaluation')

# Load model comparison data instead of raw models (simpler approach)
try:
    model_comparison = pd.read_csv('results/model_performance_comparison.csv')
    print(f"Loaded model comparison data with {len(model_comparison)} models")
except FileNotFoundError:
    print("No model comparison file found. Please run comparison scripts first.")
    model_comparison = pd.DataFrame(columns=['Model', 'c_index', 'n_features'])

# Load test data 
try:
    # Try full feature dataset first
    test_data = pd.read_csv('processed_data/test_data_capped.csv')
    dataset_type = "full"
    print(f"Using full test dataset: {test_data.shape[1]} features")
except FileNotFoundError:
    # Fall back to reduced feature dataset
    test_data = pd.read_csv('processed_data/test_data_survival_fixed.csv')
    dataset_type = "reduced"
    print(f"Using reduced test dataset: {test_data.shape[1]} features")

# Prepare test data
meta_cols = ['patient_id', 'survival_time', 'event']
gene_cols = [col for col in test_data.columns if col not in meta_cols]

# For KM plots, convert to binary classification
FIVE_YEAR_THRESHOLD = 365 * 5  # 5 years in days
test_data['five_year_outcome'] = np.where(
    (test_data['event'] == 1) & (test_data['survival_time'] <= FIVE_YEAR_THRESHOLD), 
    1, 0
)

# Normalize column names (replace spaces/special chars with underscores)
model_comparison.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in model_comparison.columns]

# Determine model types based on name (simplified approach)
model_types = {}
for model_name in model_comparison['Model']:
    if model_name in ['lasso', 'ridge', 'elasticnet']:
        model_types[model_name] = 'cox'
    elif model_name == 'decision_tree':
        model_types[model_name] = 'classification'
    elif model_name == 'random_forest':
        model_types[model_name] = 'survival'
    else:
        model_types[model_name] = 'unknown'

# Add model type to comparison dataframe
model_comparison['Model_Type'] = model_comparison['Model'].map(model_types)

# Rename columns to standard names if needed
if 'c_index' in model_comparison.columns:
    model_comparison.rename(columns={'c_index': 'C_index_AUC'}, inplace=True)
if 'n_features' in model_comparison.columns:
    model_comparison.rename(columns={'n_features': 'Features'}, inplace=True)

# Ensure dataframe is sorted by performance
model_comparison = model_comparison.sort_values('C_index_AUC', ascending=False)

# Print comprehensive comparison
print("\nComprehensive Model Comparison:")
print(model_comparison)

# ------------------ Generate Visualizations ------------------

# 1. Bar chart of C-index/AUC comparison
plt.figure(figsize=(10, 6))
colors = ['steelblue' if model_types.get(model, 'unknown') == 'cox' else 
          'forestgreen' if model_types.get(model, 'unknown') == 'survival' else 
          'darkorange' if model_types.get(model, 'unknown') == 'classification' else 'gray' 
          for model in model_comparison['Model']]

ax = sns.barplot(x='Model', y='C_index_AUC', data=model_comparison)
plt.title('Model Performance Comparison (C-index/AUC)')
plt.ylim(0.5, max(0.85, model_comparison['C_index_AUC'].max() + 0.05))  # Start at 0.5 (random)

# Add text labels for each bar
for i, bar in enumerate(ax.patches):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{model_comparison['C_index_AUC'].iloc[i]:.4f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig('plots/evaluation/performance_comparison.png')
print("Created performance comparison plot")

# 2. Scatter plot of model complexity vs. performance
if 'Features' in model_comparison.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Features', y='C_index_AUC', data=model_comparison, 
                   s=100, hue='Model_Type')
    plt.title('Model Complexity vs. Performance')
    
    # Add model name labels
    for i, row in model_comparison.iterrows():
        plt.annotate(row['Model'], (row['Features'], row['C_index_AUC']), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/complexity_vs_performance.png')
    print("Created complexity vs. performance plot")

# 3. Simple comparison table with interpretability notes
comparison_table = model_comparison.copy()
comparison_table['Interpretability'] = ''

for i, row in comparison_table.iterrows():
    model = row['Model']
    if model == 'lasso':
        comparison_table.at[i, 'Interpretability'] = 'High - Sparse coefficients'
    elif model == 'ridge':
        comparison_table.at[i, 'Interpretability'] = 'Medium - All features used'
    elif model == 'elasticnet':
        comparison_table.at[i, 'Interpretability'] = 'High - Balance of L1/L2'
    elif model == 'decision_tree':
        comparison_table.at[i, 'Interpretability'] = 'Very High - Visual decision rules'
    elif model == 'random_forest':
        comparison_table.at[i, 'Interpretability'] = 'Medium - Feature importance ranks'

# Save enhanced comparison table
comparison_table.to_csv('results/enhanced_model_comparison.csv', index=False)
print("Created enhanced comparison table with interpretability notes")

# 4. Kaplan-Meier curve for 5-year outcome (demonstration)
# This is a simplified version that doesn't require model predictions
plt.figure(figsize=(10, 6))
kmf_died = KaplanMeierFitter()
kmf_survived = KaplanMeierFitter()

# Split by 5-year outcome
died_mask = test_data['five_year_outcome'] == 1
survived_mask = ~died_mask

died_events = test_data.loc[died_mask, 'event'].values
died_times = test_data.loc[died_mask, 'survival_time'].values
survived_events = test_data.loc[survived_mask, 'event'].values
survived_times = test_data.loc[survived_mask, 'survival_time'].values

# Fit KM curves
kmf_died.fit(died_times, died_events, label=f"Died within 5 years (n={sum(died_mask)})")
kmf_survived.fit(survived_times, survived_events, label=f"Survived beyond 5 years (n={sum(survived_mask)})")

# Plot
ax = kmf_died.plot(ci_show=False)
kmf_survived.plot(ax=ax, ci_show=False)

# Add log-rank p-value
results = logrank_test(died_times, survived_times, died_events, survived_events)
p_value = results.p_value

plt.title('Kaplan-Meier Curves by 5-Year Outcome')
plt.text(0.1, 0.1, f'Log-rank p-value: {p_value:.6f}', transform=plt.gca().transAxes)
plt.xlabel('Time (Days)')
plt.ylabel('Survival Probability')
plt.tight_layout()
plt.savefig('plots/evaluation/observed_outcome_kaplan_meier.png')
print("Created Kaplan-Meier plot for observed outcomes")

# 5. Create a radar chart comparing models across multiple dimensions
if len(model_comparison) >= 3:
    # Normalize metrics to 0-1 scale for radar chart
    radar_df = model_comparison.copy()
    
    # Add additional dimensions
    radar_df['Simplicity'] = 1 - (radar_df['Features'] / radar_df['Features'].max())  # Inverse of feature count
    
    # Interpretability score based on model type
    model_interpret = {
        'lasso': 0.9,
        'ridge': 0.6,
        'elasticnet': 0.8,
        'decision_tree': 1.0,
        'random_forest': 0.5
    }
    radar_df['Interpretability'] = radar_df['Model'].map(lambda x: model_interpret.get(x, 0.5))
    
    # Predictive power is the C-index/AUC
    radar_df['Predictive_Power'] = radar_df['C_index_AUC']
    
    # Setup radar chart
    categories = ['Predictive_Power', 'Simplicity', 'Interpretability']
    N = len(categories)
    
    # Create angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Draw one line per model and fill area
    for i, model in enumerate(radar_df['Model'].head(3)):  # Top 3 models only
        values = radar_df.loc[radar_df['Model'] == model, categories].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right')
    plt.title('Model Comparison - Multiple Dimensions')
    
    plt.tight_layout()
    plt.savefig('plots/evaluation/model_radar_comparison.png')
    print("Created radar chart comparing top models across multiple dimensions")

# Final summary
print("\nEvaluation complete. Results and visualizations saved.")
print("Best model according to performance metrics:", model_comparison['Model'].iloc[0])
print(f"C-index/AUC: {model_comparison['C_index_AUC'].iloc[0]:.4f}")
print(f"Features: {model_comparison['Features'].iloc[0]}") 