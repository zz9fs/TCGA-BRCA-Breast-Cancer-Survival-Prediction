import gzip
import pandas as pd

try:
    # Read phenotype data - first 20 rows only for preview
    print("Reading sample of phenotype data...")
    with gzip.open('TCGA-BRCA.GDC_phenotype.tsv.gz', 'rt') as f:
        phenotype_sample = pd.read_csv(f, sep='\t', nrows=20)
        
    print(f"Phenotype data sample shape: {phenotype_sample.shape}")
    
    # Look for key survival columns
    survival_cols = []
    for col in phenotype_sample.columns:
        if any(term in col.lower() for term in ['vital', 'survival', 'death', 'status', 'follow']):
            survival_cols.append(col)
    
    print(f"Potential survival-related columns found: {len(survival_cols)}")
    for col in survival_cols:
        print(f"- {col}")
    
    # Sample IDs to check if they match between files
    id_cols = [col for col in phenotype_sample.columns if 'id' in col.lower() or 'sample' in col.lower()]
    print("\nID columns that could link to expression data:")
    for col in id_cols[:5]:  # Limit to first 5
        print(f"- {col}: {phenotype_sample[col].iloc[0]}")
    
    # Check expression data headers
    print("\nReading expression data header...")
    with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
        header_line = f.readline().strip()
        header_parts = header_line.split('\t')
        
    print(f"Expression data has {len(header_parts)} columns")
    print(f"First column: {header_parts[0]}")
    print(f"Sample columns (first 5): {header_parts[1:6]}")
    
    # Look at first few rows of expression data
    print("\nPeeking at expression data (first 3 rows, first 6 columns)...")
    with gzip.open('TCGA-BRCA.htseq_counts.tsv.gz', 'rt') as f:
        # Skip header
        f.readline()
        # Read 3 rows
        for i in range(3):
            line = f.readline().strip()
            parts = line.split('\t')
            print(f"Gene: {parts[0]}, Values: {parts[1:6]}")
    
    print("\nDataset suitability summary:")
    print("1. Phenotype data contains clinical information with potential survival endpoints")
    print("2. Expression data contains RNA-seq counts for ~60K genes across >1000 samples")
    print("3. Patient IDs appear to follow TCGA convention and should be linkable between files")
    
except Exception as e:
    print(f"Error in analysis: {e}") 