import gzip
import pandas as pd
import numpy as np
import os

print("Step 1: Merging Clinical and Genomic Data")

# Create output directory if it doesn't exist
if not os.path.exists('processed_data'):
    os.makedirs('processed_data')

# 1. Load the clinical data
print("Loading clinical data...")
with gzip.open('TCGA-BRCA.GDC_phenotype.tsv.gz', 'rt') as f:
    clinical_data = pd.read_csv(f, sep='\t')

print(f"Clinical data loaded: {clinical_data.shape[0]} samples, {clinical_data.shape[1]} features")

# 2. Extract survival information
print("Extracting survival information...")

# Key columns for survival analysis
survival_time_col = 'days_to_death.demographic'
last_followup_col = 'days_to_last_follow_up.diagnoses'
vital_status_col = 'vital_status.demographic'

# Create a survival DataFrame with patient IDs, survival time, and event status
survival_data = pd.DataFrame()

# Use patient ID from the samples column - this will be used to match with expression data
survival_data['patient_id'] = clinical_data['submitter_id.samples'].apply(lambda x: x[:12])

# Extract survival time (days_to_death if deceased, days_to_last_follow_up if alive)
survival_data['survival_time'] = clinical_data.apply(
    lambda row: row[survival_time_col] if pd.notna(row[survival_time_col]) 
    else row[last_followup_col], axis=1
)

# Extract event status (1 if deceased, 0 if alive/censored)
survival_data['event'] = clinical_data[vital_status_col].apply(
    lambda x: 1 if x == 'Dead' else 0
)

# Drop rows with missing survival time
survival_data = survival_data.dropna(subset=['survival_time'])

print(f"Survival data extracted: {survival_data.shape[0]} samples with valid survival information")
print(f"Event distribution: {survival_data['event'].value_counts().to_dict()}")

# 3. Load expression data (headers only first to get sample IDs)
print("Loading expression data headers...")
with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
    header = f.readline().strip().split('\t')
    
# Extract sample IDs from expression data
expression_samples = [col[:12] for col in header[1:]]  # Skip first column (gene IDs)
print(f"Expression data has {len(expression_samples)} samples")

# 4. Find the intersection of samples between clinical and expression data
common_samples = set(survival_data['patient_id']).intersection(set(expression_samples))
print(f"Found {len(common_samples)} samples common to both clinical and expression data")

# 5. Filter survival data to include only samples that have expression data
survival_data_filtered = survival_data[survival_data['patient_id'].isin(common_samples)]
print(f"Filtered survival data to {survival_data_filtered.shape[0]} samples")

# 6. Load the full expression data
print("Loading full expression data (this may take a while)...")
with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
    expression_data = pd.read_csv(f, sep='\t')
    
print(f"Expression data loaded: {expression_data.shape[0]} genes, {expression_data.shape[1]} columns")

# 7. Prepare expression data for merging
# Transpose expression data to have samples as rows and genes as columns
print("Transposing expression data...")
genes = expression_data.iloc[:, 0]  # Save gene IDs
exp_data_T = expression_data.iloc[:, 1:].T  # Transpose all but the first column
exp_data_T.columns = genes  # Set gene IDs as column names
exp_data_T.index = [idx[:12] for idx in exp_data_T.index]  # Set sample IDs as index, truncate to patient ID

# 8. Filter expression data to include only samples with survival information
exp_data_filtered = exp_data_T.loc[exp_data_T.index.isin(survival_data_filtered['patient_id'])]
print(f"Filtered expression data to {exp_data_filtered.shape[0]} samples")

# 9. Merge survival and expression data
print("Merging survival and expression data...")
# Reset index to make patient_id a column for the merge
exp_data_filtered = exp_data_filtered.reset_index().rename(columns={'index': 'patient_id'})

# Merge on patient_id
merged_data = pd.merge(survival_data_filtered, exp_data_filtered, on='patient_id')
print(f"Final merged dataset: {merged_data.shape[0]} samples, {merged_data.shape[1]} features")

# 10. Save the merged data
print("Saving merged data...")
merged_data.to_csv('processed_data/merged_data.csv', index=False)
print("Merged data saved to processed_data/merged_data.csv")

# Also save a smaller version with just survival data for reference
survival_data_filtered.to_csv('processed_data/survival_data.csv', index=False)
print("Survival data saved to processed_data/survival_data.csv") 