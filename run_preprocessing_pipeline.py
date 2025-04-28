import os
import subprocess
import time

print("=== TCGA-BRCA Breast Cancer Survival Prediction ===")
print("=== Data Preprocessing Pipeline ===")

# Create directories
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')
    print("Created directory: processed_data/")
if not os.path.exists('plots'):
    os.makedirs('plots')
    print("Created directory: plots/")

# List of preprocessing steps
preprocessing_steps = [
    {"script": "1_merge_clinical_genomic.py", "name": "Merging Clinical and Genomic Data"},
    {"script": "2_normalize_expression.py", "name": "Normalizing Gene Expression"},
    {"script": "3_filter_low_variance_genes.py", "name": "Filtering Low-Variance Genes"},
    {"script": "4_train_test_split.py", "name": "Train-Test Split"},
    {"script": "5_handle_missing_data.py", "name": "Handling Missing Data"}
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
print("Data preprocessing pipeline execution completed!")
print("Check processed_data/ directory for the processed datasets")
print("Check plots/ directory for visualizations")
print("="*50) 