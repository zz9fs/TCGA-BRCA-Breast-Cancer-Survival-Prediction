import gzip
import pandas as pd
import numpy as np

print("Analyzing TCGA-BRCA dataset for survival prediction...")

# Read the full phenotype data
print("Reading phenotype data...")
with gzip.open('TCGA-BRCA.GDC_phenotype.tsv.gz', 'rt') as f:
    phenotype_data = pd.read_csv(f, sep='\t')

# Print basic dataset information
print(f"Total samples: {phenotype_data.shape[0]}")
print(f"Total features: {phenotype_data.shape[1]}")

# Check for survival-related columns
survival_cols = [col for col in phenotype_data.columns if any(term in col.lower() for term in 
                ['survival', 'death', 'deceased', 'alive', 'vital', 'follow', 'recurrence', 'progression'])]

print("\nSurvival-related columns found:")
for col in survival_cols:
    print(f"- {col}")
    # Show value counts if categorical
    if phenotype_data[col].dtype == 'object':
        value_counts = phenotype_data[col].value_counts().head(5)
        print(f"  Top values: {dict(value_counts)}")
    # Show basic stats if numerical
    else:
        non_null = phenotype_data[col].dropna()
        if len(non_null) > 0:
            print(f"  Stats: min={non_null.min()}, max={non_null.max()}, mean={non_null.mean():.2f}, non-null count={len(non_null)}")

# Check sample columns to understand if we can link to expression data
sample_cols = [col for col in phenotype_data.columns if 'sample' in col.lower() or 'patient' in col.lower() or 'id' in col.lower()]
print("\nSample identifier columns:")
for col in sample_cols[:10]:  # Limit to first 10 to avoid too much output
    unique_count = phenotype_data[col].nunique()
    print(f"- {col}: {unique_count} unique values")

# Check expression data structure (just the header to understand sample IDs)
print("\nChecking expression data structure...")
try:
    with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
        # Just read the header row
        header = f.readline().strip().split('\t')
        print(f"Expression data has {len(header)} columns")
        print(f"First column (likely gene IDs): {header[0]}")
        print(f"Other columns (likely sample IDs): {header[1:6]}... (and {len(header)-6} more)")
        
        # Count samples
        sample_count = len(header) - 1  # Subtract 1 for the gene ID column
        print(f"Expression data contains {sample_count} samples")
except Exception as e:
    print("Error reading expression data:", e) 