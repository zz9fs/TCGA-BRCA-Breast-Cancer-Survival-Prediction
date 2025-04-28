import os
import subprocess
import time

print("=== TCGA-BRCA Breast Cancer Survival Prediction ===")
print("=== Advanced Data Preprocessing Pipeline ===")

# Create directories
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
    print("Created directory: processed_data/")
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created directory: plots/")

# List of preprocessing steps
preprocessing_steps = [
    {"script": "6_analyze_class_imbalance.py", "name": "Analyzing Class Imbalance"},
    {"script": "7_detect_outliers.py", "name": "Detecting and Handling Outliers"},
    {"script": "8_multicollinearity_analysis.py", "name": "Multicollinearity Analysis and Feature Selection"},
    {"script": "9_check_survival_times.py", "name": "Checking and Fixing Survival Times"}
]

# Run each preprocessing step
for i, step in enumerate(preprocessing_steps):
    script = step["script"]
    name = step["name"]
    
    print(f"\n{'='*50}")
    print(f"Step {i+1}/{len(preprocessing_steps)}: {name}")
    print(f"Running script: {script}")
    print(f"{'='*50}\n")
    
    start_time = time.time()
    
    # Run the script as a subprocess
    try:
        result = subprocess.run(['python', script], check=True)
        if result.returncode == 0:
            end_time = time.time()
            duration = end_time - start_time
            print(f"\nStep {i+1} completed successfully in {duration:.2f} seconds.")
        else:
            print(f"\nError running step {i+1}. Process returned non-zero code: {result.returncode}")
            break
    except subprocess.CalledProcessError as e:
        print(f"\nError running step {i+1}: {e}")
        break
    except Exception as e:
        print(f"\nUnexpected error running step {i+1}: {e}")
        break

print("\n" + "="*50)
print("Advanced data preprocessing pipeline execution completed!")
print("Check processed_data/ directory for the processed datasets")
print("Check plots/ directory for visualizations")
print("="*50)

print("\nFinal processed datasets available:")
print(" - train_data_survival_fixed.csv & test_data_survival_fixed.csv: Final datasets with all preprocessing steps")
print(" - train_data_reduced.csv & test_data_reduced.csv: Datasets with selected features")
print(" - train_data_pca.csv & test_data_pca.csv: Datasets with PCA transformation")
print(" - train_data_smote.csv: SMOTE-oversampled dataset to address class imbalance (if applicable)")
print(" - train_data_undersampled.csv: Undersampled dataset to address class imbalance (if applicable)")
print("\nRecommendation: Use train_data_survival_fixed.csv & test_data_survival_fixed.csv for model training") 