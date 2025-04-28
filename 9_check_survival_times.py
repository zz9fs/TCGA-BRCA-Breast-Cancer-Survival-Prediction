import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

print("Step 9: Checking and Fixing Survival Times")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the reduced datasets (could also use other versions)
print("Loading datasets...")
try:
    train_data = pd.read_csv('processed_data/train_data_reduced.csv')
    test_data = pd.read_csv('processed_data/test_data_reduced.csv')
    print(f"Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"Loaded testing data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
except FileNotFoundError:
    print("Reduced data files not found. Trying other datasets...")
    try:
        train_data = pd.read_csv('processed_data/train_data_capped.csv')
        test_data = pd.read_csv('processed_data/test_data_capped.csv')
        print("Using capped data instead.")
    except FileNotFoundError:
        try:
            train_data = pd.read_csv('processed_data/train_data_clean.csv')
            test_data = pd.read_csv('processed_data/test_data_clean.csv')
            print("Using clean data instead.")
        except FileNotFoundError:
            print("No processed data files found. Please run the preprocessing pipeline first.")
            exit(1)

# 2. Check survival times in train and test sets
print("Checking survival times...")
survival_cols = ['survival_time', 'event']
train_survival = train_data[survival_cols]
test_survival = test_data[survival_cols]

# Look for negative or zero survival times, which would be invalid
train_invalid_times = train_survival[train_survival['survival_time'] <= 0]
test_invalid_times = test_survival[test_survival['survival_time'] <= 0]

print(f"Invalid survival times in training set: {len(train_invalid_times)} (survival time <= 0)")
print(f"Invalid survival times in testing set: {len(test_invalid_times)} (survival time <= 0)")

# 3. Visualize survival time distribution
print("Creating survival time distribution visualization...")
plt.figure(figsize=(15, 10))

# Histogram of all survival times
plt.subplot(2, 2, 1)
plt.hist(train_survival['survival_time'], bins=30, alpha=0.7, label='Train')
plt.hist(test_survival['survival_time'], bins=30, alpha=0.7, label='Test')
plt.xlabel('Survival Time (days)')
plt.ylabel('Count')
plt.title('Overall Survival Time Distribution')
plt.legend()

# Boxplot of survival times by event status (train)
plt.subplot(2, 2, 2)
sns.boxplot(x='event', y='survival_time', data=train_survival)
plt.title('Survival Time by Event Status (Training Set)')
plt.xlabel('Event (1=Deceased, 0=Censored)')
plt.ylabel('Survival Time (days)')

# Survival time by event status (density plot)
plt.subplot(2, 2, 3)
for event in [0, 1]:
    train_subset = train_survival[train_survival['event'] == event]
    if len(train_subset) > 0:
        sns.kdeplot(train_subset['survival_time'], label=f'Event={event}')
plt.title('Survival Time Density by Event Status')
plt.xlabel('Survival Time (days)')
plt.legend()

# Log-transformed survival time
plt.subplot(2, 2, 4)
plt.hist(np.log1p(train_survival['survival_time']), bins=30, alpha=0.7, label='Train')
plt.hist(np.log1p(test_survival['survival_time']), bins=30, alpha=0.7, label='Test')
plt.xlabel('Log Survival Time')
plt.ylabel('Count')
plt.title('Log-Transformed Survival Time Distribution')
plt.legend()

plt.tight_layout()
plt.savefig('plots/survival_time_analysis.png')
print("Survival time analysis saved to plots/survival_time_analysis.png")

# 4. Fix invalid survival times if needed
if len(train_invalid_times) > 0 or len(test_invalid_times) > 0:
    print("Fixing invalid survival times...")
    
    # Get the median survival time for each event type
    train_medians = train_survival.groupby('event')['survival_time'].median()
    
    # For training set
    if len(train_invalid_times) > 0:
        for idx in train_invalid_times.index:
            event = train_data.loc[idx, 'event']
            # Replace with median survival time for that event type
            train_data.loc[idx, 'survival_time'] = train_medians[event]
    
    # For test set
    if len(test_invalid_times) > 0:
        for idx in test_invalid_times.index:
            event = test_data.loc[idx, 'event']
            # Replace with median survival time for that event type from training set
            test_data.loc[idx, 'survival_time'] = train_medians[event]
    
    print("Invalid survival times have been fixed")
    
    # Check if all survival times are now valid
    invalid_after_fix_train = train_data[train_data['survival_time'] <= 0].shape[0]
    invalid_after_fix_test = test_data[test_data['survival_time'] <= 0].shape[0]
    
    print(f"Invalid survival times after fixing - Train: {invalid_after_fix_train}, Test: {invalid_after_fix_test}")
    
    # Save the fixed datasets with a different suffix
    train_data.to_csv('processed_data/train_data_survival_fixed.csv', index=False)
    test_data.to_csv('processed_data/test_data_survival_fixed.csv', index=False)
    print("Fixed datasets saved to processed_data directory")
else:
    print("All survival times are valid (positive). No fixes needed.")

    # Still save the datasets to maintain consistency in the pipeline
    train_data.to_csv('processed_data/train_data_survival_fixed.csv', index=False)
    test_data.to_csv('processed_data/test_data_survival_fixed.csv', index=False)
    print("Datasets saved to processed_data directory with _survival_fixed suffix")

# 5. Summarize survival time statistics
print("\nSurvival Time Summary Statistics:")
# Overall statistics
print("Overall:")
train_stats = train_data['survival_time'].describe()
print(f"  Training: min={train_stats['min']:.1f}, max={train_stats['max']:.1f}, mean={train_stats['mean']:.1f}, median={train_stats['50%']:.1f}")
test_stats = test_data['survival_time'].describe()
print(f"  Testing: min={test_stats['min']:.1f}, max={test_stats['max']:.1f}, mean={test_stats['mean']:.1f}, median={test_stats['50%']:.1f}")

# By event status
print("\nBy Event Status (Training):")
for event in [0, 1]:
    subset = train_data[train_data['event'] == event]
    if len(subset) > 0:
        stats = subset['survival_time'].describe()
        print(f"  Event={event}: count={stats['count']}, min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}, median={stats['50%']:.1f}")

print("\nBy Event Status (Testing):")
for event in [0, 1]:
    subset = test_data[test_data['event'] == event]
    if len(subset) > 0:
        stats = subset['survival_time'].describe()
        print(f"  Event={event}: count={stats['count']}, min={stats['min']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}, median={stats['50%']:.1f}")

print("\nSurvival time validation and correction completed") 