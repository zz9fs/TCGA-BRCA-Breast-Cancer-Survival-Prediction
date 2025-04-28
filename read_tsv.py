import gzip
import pandas as pd

# Read phenotype data
print("Reading phenotype data...")
with gzip.open('TCGA-BRCA.GDC_phenotype.tsv.gz', 'rt') as f:
    phenotype_data = pd.read_csv(f, sep='\t', nrows=10)
    print("Phenotype data shape:", phenotype_data.shape)
    print("Phenotype data columns:", phenotype_data.columns.tolist())
    print("Phenotype data sample:")
    print(phenotype_data.head(5))

# Read expression data (just peek at it since it's large)
print("\nReading expression data (sample)...")
try:
    with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
        # Read just the first few rows to see structure
        exp_data = pd.read_csv(f, sep='\t', nrows=10)
        print("Expression data shape (sample):", exp_data.shape)
        print("Expression data columns (sample):", exp_data.columns.tolist()[:5], "...")
        print("Expression data sample:")
        print(exp_data.head(3))
except Exception as e:
    print("Error reading expression data:", e) 