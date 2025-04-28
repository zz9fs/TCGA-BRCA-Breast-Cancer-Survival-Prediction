import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("EDA Step 1: Kaplan-Meier Survival Analysis")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load processed data
train_data = pd.read_csv('processed_data/train_data_survival_fixed.csv')
test_data = pd.read_csv('processed_data/test_data_survival_fixed.csv')
combined_data = pd.concat([train_data, test_data])

print(f"Loaded data: {combined_data.shape[0]} samples")

# Overall survival Kaplan-Meier curve
plt.figure(figsize=(12, 8))

# Plot overall survival curve
kmf = KaplanMeierFitter()
kmf.fit(combined_data['survival_time'], combined_data['event'], label="Overall Survival")
ax = kmf.plot_survival_function(ci_show=True)
median_survival = kmf.median_survival_time_
plt.axhline(y=0.5, color='r', linestyle='--')
plt.axvline(x=median_survival, color='r', linestyle='--')
plt.title(f'Kaplan-Meier Overall Survival Curve (N={combined_data.shape[0]})')
plt.xlabel('Time (Days)')
plt.ylabel('Survival Probability')
plt.text(median_survival + 100, 0.52, f'Median Survival: {median_survival:.1f} days')
plt.savefig('plots/kaplan_meier_overall.png')

# Plot train vs test KM curves
plt.figure(figsize=(12, 8))
kmf_train = KaplanMeierFitter()
kmf_train.fit(train_data['survival_time'], train_data['event'], label=f"Training (n={train_data.shape[0]})")
ax = kmf_train.plot_survival_function(ci_show=True)

kmf_test = KaplanMeierFitter()
kmf_test.fit(test_data['survival_time'], test_data['event'], label=f"Testing (n={test_data.shape[0]})")
kmf_test.plot_survival_function(ax=ax, ci_show=True)

# Perform log-rank test to compare curves
results = logrank_test(train_data['survival_time'], test_data['survival_time'], 
                       train_data['event'], test_data['event'])
p_value = results.p_value

plt.title(f'Kaplan-Meier Survival: Training vs Testing Sets (log-rank p={p_value:.4f})')
plt.xlabel('Time (Days)')
plt.ylabel('Survival Probability')
plt.savefig('plots/kaplan_meier_train_test.png')

# Print summary statistics
print("\nSurvival Summary Statistics:")
print(f"Number of patients: {combined_data.shape[0]}")
print(f"Number of events (deaths): {combined_data['event'].sum()}")
print(f"Number of censored cases: {(combined_data['event'] == 0).sum()}")
print(f"Censoring rate: {(combined_data['event'] == 0).sum() / combined_data.shape[0]:.2%}")
print(f"Median follow-up time: {combined_data['survival_time'].median():.1f} days")
print(f"Median survival time: {median_survival:.1f} days")

print("Kaplan-Meier analysis completed. Plots saved to plots directory.")