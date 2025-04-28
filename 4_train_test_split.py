import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("Step 4: Train-Test Split")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the filtered data
print("Loading filtered data...")
try:
    filtered_data = pd.read_csv('processed_data/filtered_data.csv')
    print(f"Loaded filtered data: {filtered_data.shape[0]} samples, {filtered_data.shape[1]} features")
except FileNotFoundError:
    print("Filtered data file not found. Please run the filtering script first.")
    exit(1)

# 2. Separate features and target variables
print("Separating features and target variables...")
# Target variables: patient_id, survival_time, event
target_cols = ['patient_id', 'survival_time', 'event']
X = filtered_data.drop(columns=target_cols)
y = filtered_data[target_cols]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# 3. Split data into training and testing sets (70% train, 30% test)
print("Splitting data into training and testing sets...")
# Stratify by event to ensure similar censoring distribution in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y['event']
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")

# 4. Check if stratification preserved the event distribution
train_event_dist = y_train['event'].value_counts(normalize=True)
test_event_dist = y_test['event'].value_counts(normalize=True)

print("Event distribution in training set:")
print(train_event_dist)
print("Event distribution in testing set:")
print(test_event_dist)

# 5. Visualize survival time distribution in train and test sets
print("Creating survival time distribution plot...")
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(y_train['survival_time'], bins=30, alpha=0.7, label='Train')
plt.hist(y_test['survival_time'], bins=30, alpha=0.7, label='Test')
plt.xlabel('Survival Time (days)')
plt.ylabel('Count')
plt.title('Survival Time Distribution')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_train.loc[y_train['event'] == 1, 'survival_time'], bins=30, alpha=0.7, label='Train (event=1)')
plt.hist(y_test.loc[y_test['event'] == 1, 'survival_time'], bins=30, alpha=0.7, label='Test (event=1)')
plt.xlabel('Survival Time (days) for Deceased Patients')
plt.ylabel('Count')
plt.title('Survival Time Distribution for Events')
plt.legend()

plt.tight_layout()
plt.savefig('plots/train_test_survival_distribution.png')
print("Survival distribution plot saved to plots/train_test_survival_distribution.png")

# 6. Combine features and targets for train and test sets
print("Combining features and targets for train and test sets...")
train_data = pd.concat([y_train, X_train], axis=1)
test_data = pd.concat([y_test, X_test], axis=1)

print(f"Final training dataset: {train_data.shape[0]} samples, {train_data.shape[1]} features")
print(f"Final testing dataset: {test_data.shape[0]} samples, {test_data.shape[1]} features")

# 7. Save the training and testing datasets
print("Saving training and testing datasets...")
train_data.to_csv('processed_data/train_data.csv', index=False)
test_data.to_csv('processed_data/test_data.csv', index=False)

print("Training data saved to processed_data/train_data.csv")
print("Testing data saved to processed_data/test_data.csv")

# 8. Also save X and y separately for easy modeling
print("Saving features and targets separately...")
# For X, remove patient_id from features since it's not used for modeling
X_train_final = X_train
X_test_final = X_test

# Save X train/test (features only)
X_train_final.to_csv('processed_data/X_train.csv', index=False)
X_test_final.to_csv('processed_data/X_test.csv', index=False)

# For y, we need survival_time and event for survival analysis
y_train_final = y_train[['survival_time', 'event']]
y_test_final = y_test[['survival_time', 'event']]

# Save y train/test (targets only)
y_train_final.to_csv('processed_data/y_train.csv', index=False)
y_test_final.to_csv('processed_data/y_test.csv', index=False)

print("Features (X) and targets (y) saved separately in processed_data/ directory")

# 9. Save patient IDs separately for reference
train_ids = y_train[['patient_id']]
test_ids = y_test[['patient_id']]
train_ids.to_csv('processed_data/train_patient_ids.csv', index=False)
test_ids.to_csv('processed_data/test_patient_ids.csv', index=False)

print("Patient IDs saved separately in processed_data/ directory") 