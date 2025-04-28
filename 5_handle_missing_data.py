import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

print("Step 5: Handling Missing Data")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
if not os.path.exists('plots'):
    os.makedirs('plots')

# 1. Load the training and testing data
print("Loading training and testing data...")
try:
    train_data = pd.read_csv('processed_data/train_data.csv')
    test_data = pd.read_csv('processed_data/test_data.csv')
    print(f"Loaded training data: {train_data.shape[0]} samples, {train_data.shape[1]} features")
    print(f"Loaded testing data: {test_data.shape[0]} samples, {test_data.shape[1]} features")
except FileNotFoundError:
    print("Train/test data files not found. Please run the train_test_split script first.")
    exit(1)

# 2. Separate metadata and expression data
print("Separating metadata and expression data...")
meta_cols = ['patient_id', 'survival_time', 'event']
X_train = train_data.drop(columns=meta_cols)
y_train = train_data[meta_cols]
X_test = test_data.drop(columns=meta_cols)
y_test = test_data[meta_cols]

# 3. Check for missing values in training set
print("Checking for missing values in training set...")
train_missing = X_train.isnull().sum()
train_missing_percent = (train_missing / len(X_train)) * 100
train_missing_summary = pd.DataFrame({
    'Missing Values': train_missing,
    'Percentage': train_missing_percent
})
train_missing_summary = train_missing_summary[train_missing_summary['Missing Values'] > 0].sort_values('Percentage', ascending=False)

print(f"Total features with missing values in training set: {len(train_missing_summary)}")
if len(train_missing_summary) > 0:
    print("Top features with missing values:")
    print(train_missing_summary.head(10))

# 4. Check for missing values in test set
print("Checking for missing values in test set...")
test_missing = X_test.isnull().sum()
test_missing_percent = (test_missing / len(X_test)) * 100
test_missing_summary = pd.DataFrame({
    'Missing Values': test_missing,
    'Percentage': test_missing_percent
})
test_missing_summary = test_missing_summary[test_missing_summary['Missing Values'] > 0].sort_values('Percentage', ascending=False)

print(f"Total features with missing values in test set: {len(test_missing_summary)}")
if len(test_missing_summary) > 0:
    print("Top features with missing values:")
    print(test_missing_summary.head(10))

# 5. Visualize missing data patterns
if len(train_missing_summary) > 0 or len(test_missing_summary) > 0:
    print("Creating missing data visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot percentage of missing values for each feature
    plt.subplot(2, 1, 1)
    if len(train_missing_summary) > 0:
        missing_data = train_missing_summary.head(20)  # Show top 20 features with missing values
        plt.barh(y=missing_data.index, width=missing_data['Percentage'])
        plt.xlabel('Percentage of Missing Values')
        plt.title('Top Features with Missing Values (Training Set)')
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'No missing values in training set', ha='center', va='center', fontsize=12)
    
    plt.savefig('plots/missing_data_patterns.png')
    print("Missing data visualization saved to plots/missing_data_patterns.png")

# 6. Impute missing values using mean imputation
if len(train_missing_summary) > 0 or len(test_missing_summary) > 0:
    print("Imputing missing values...")
    
    # Initialize imputer
    imputer = SimpleImputer(strategy='mean')
    
    # Fit and transform training data
    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform test data (using mean from training data)
    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    print("Missing values have been imputed using mean imputation")
    
    # Check for any remaining missing values
    remaining_train_missing = X_train_imputed.isnull().sum().sum()
    remaining_test_missing = X_test_imputed.isnull().sum().sum()
    
    print(f"Remaining missing values in training set: {remaining_train_missing}")
    print(f"Remaining missing values in test set: {remaining_test_missing}")
    
    # Save imputed data
    X_train = X_train_imputed
    X_test = X_test_imputed
else:
    print("No missing values found in the data. Imputation not needed.")

# 7. Combine metadata and imputed expression data
print("Combining metadata and imputed expression data...")
train_data_final = pd.concat([y_train, X_train], axis=1)
test_data_final = pd.concat([y_test, X_test], axis=1)

print(f"Final training dataset: {train_data_final.shape[0]} samples, {train_data_final.shape[1]} features")
print(f"Final testing dataset: {test_data_final.shape[0]} samples, {test_data_final.shape[1]} features")

# 8. Save the final clean datasets
print("Saving final clean datasets...")
train_data_final.to_csv('processed_data/train_data_clean.csv', index=False)
test_data_final.to_csv('processed_data/test_data_clean.csv', index=False)

# Also save X and y separately
X_train.to_csv('processed_data/X_train_clean.csv', index=False)
X_test.to_csv('processed_data/X_test_clean.csv', index=False)
y_train.to_csv('processed_data/y_train_clean.csv', index=False)
y_test.to_csv('processed_data/y_test_clean.csv', index=False)

print("Final datasets saved to processed_data/ directory")
print("Data preprocessing pipeline completed successfully!") 