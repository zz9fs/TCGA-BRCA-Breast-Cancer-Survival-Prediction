import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

print("Step 6: Analyzing and Addressing Class Imbalance")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the clean training data
print("Loading clean training data...")
try:
    train_data = pd.read_csv('processed_data/train_data_clean.csv')
    print(f"Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
except FileNotFoundError:
    print("Clean training data not found. Please run the missing data handling script first.")
    exit(1)

# 2. Analyze class distribution
print("Analyzing class distribution...")
event_distribution = train_data['event'].value_counts()
event_percentages = 100 * event_distribution / len(train_data)

print("Event distribution in training data:")
print(f"Event=0 (censored): {event_distribution[0]} samples ({event_percentages[0]:.2f}%)")
print(f"Event=1 (deceased): {event_distribution[1]} samples ({event_percentages[1]:.2f}%)")

# 3. Visualize class distribution
print("Creating class distribution visualization...")
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.bar(['Censored (0)', 'Deceased (1)'], event_distribution.values, color=['skyblue', 'salmon'])
plt.title('Event Distribution')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(event_distribution.values):
    plt.text(i, v + 5, str(v), ha='center')

plt.subplot(1, 2, 2)
plt.pie(event_distribution.values, labels=['Censored (0)', 'Deceased (1)'], 
        autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
plt.title('Event Distribution (%)')

plt.tight_layout()
plt.savefig('plots/class_imbalance.png')
print("Class distribution visualization saved to plots/class_imbalance.png")

# 4. Decide whether to address class imbalance
# If the imbalance is severe (less than 30% of minority class)
imbalance_ratio = min(event_distribution) / max(event_distribution)
print(f"Imbalance ratio (minority/majority): {imbalance_ratio:.2f}")

# 5. Prepare data for potential resampling
meta_cols = ['patient_id', 'survival_time', 'event']
X = train_data.drop(columns=meta_cols)
y = train_data['event']  # Target is the event status

# 6. If severely imbalanced, create resampled versions
if imbalance_ratio < 0.7:  # This threshold can be adjusted
    print("Class imbalance detected. Creating resampled datasets...")
    
    # Create oversampled version using SMOTE
    print("Creating SMOTE-oversampled dataset...")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    
    # Create new dataframe with resampled data
    X_smote_df = pd.DataFrame(X_smote, columns=X.columns)
    # Since SMOTE creates synthetic samples, we need to handle patient_id and survival_time differently
    # We'll set patient_id to synthetic IDs and use median survival time for the class
    smote_df = pd.DataFrame()
    smote_df['event'] = y_smote
    # Create synthetic patient IDs
    smote_df['patient_id'] = [f"SYNTH-{i:04d}" if idx >= len(train_data) else train_data.iloc[idx]['patient_id'] 
                             for i, idx in enumerate(range(len(y_smote)))]
    
    # For survival time, use original if available, or median of the class
    class_medians = train_data.groupby('event')['survival_time'].median()
    smote_df['survival_time'] = [train_data.iloc[idx]['survival_time'] if idx < len(train_data) 
                               else class_medians[y_smote[idx]] 
                               for idx in range(len(y_smote))]
    
    # Combine with the feature data
    smote_data = pd.concat([smote_df[meta_cols], X_smote_df], axis=1)
    
    # Save oversampled data
    smote_data.to_csv('processed_data/train_data_smote.csv', index=False)
    print(f"SMOTE-oversampled data saved ({smote_data.shape[0]} samples, {smote_data.shape[1]} features)")
    print(f"New class distribution: {Counter(y_smote)}")
    
    # Also create undersampled version
    print("Creating undersampled dataset...")
    # Find the count of the minority class
    min_class_count = min(event_distribution)
    undersampler = RandomUnderSampler(sampling_strategy={0: min_class_count, 1: min_class_count}, 
                                     random_state=42)
    X_under, y_under = undersampler.fit_resample(X, y)
    
    # Get the indices of the selected samples
    indices = undersampler.sample_indices_
    
    # Create undersampled dataframe with all columns
    under_data = train_data.iloc[indices].copy()
    
    # Save undersampled data
    under_data.to_csv('processed_data/train_data_undersampled.csv', index=False)
    print(f"Undersampled data saved ({under_data.shape[0]} samples, {under_data.shape[1]} features)")
    print(f"New class distribution: {Counter(y_under)}")
    
    # Plot the resampled distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(['Censored (0)', 'Deceased (1)'], event_distribution.values, color=['skyblue', 'salmon'])
    plt.title('Original Distribution')
    plt.ylabel('Count')
    
    plt.subplot(1, 3, 2)
    smote_counts = Counter(y_smote)
    plt.bar(['Censored (0)', 'Deceased (1)'], [smote_counts[0], smote_counts[1]], color=['skyblue', 'salmon'])
    plt.title('SMOTE-Oversampled')
    
    plt.subplot(1, 3, 3)
    under_counts = Counter(y_under)
    plt.bar(['Censored (0)', 'Deceased (1)'], [under_counts[0], under_counts[1]], color=['skyblue', 'salmon'])
    plt.title('Undersampled')
    
    plt.tight_layout()
    plt.savefig('plots/resampled_distributions.png')
    print("Resampled distributions visualization saved to plots/resampled_distributions.png")
else:
    print("Class distribution is acceptable. No resampling necessary.") 