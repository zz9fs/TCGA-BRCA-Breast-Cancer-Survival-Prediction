# Data Access Instructions

## TCGA Breast Cancer (BRCA) Dataset

This project uses gene expression and survival data from The Cancer Genome Atlas (TCGA) Breast Cancer dataset. Follow these steps to obtain the data:

### Option 1: Using TCGA Data Portal

1. Visit the [GDC Data Portal](https://portal.gdc.cancer.gov/)
2. Search for "TCGA-BRCA"
3. Filter for RNA-Seq gene expression data
4. Download the gene expression data (HTSeq-FPKM)
5. Download the clinical data with survival information

### Option 2: Using R Bioconductor

You can use the `TCGAbiolinks` package in R to download the data:

```R
library(TCGAbiolinks)

# Download gene expression data
query_expr <- GDCquery(
    project = "TCGA-BRCA",
    data.category = "Transcriptome Profiling",
    data.type = "Gene Expression Quantification",
    workflow.type = "HTSeq - FPKM"
)
GDCdownload(query_expr)
expr_data <- GDCprepare(query_expr)

# Download clinical data
query_clin <- GDCquery(
    project = "TCGA-BRCA",
    data.category = "Clinical",
    data.type = "Clinical Supplement"
)
GDCdownload(query_clin)
clin_data <- GDCprepare(query_clin)
```

### Option 3: Using Preprocessed Files

For convenience, preprocessed files are available at the following location:
[Link to your preprocessed files if you want to share them]

## Data Preprocessing

After obtaining the raw data, preprocessing steps include:

1. Merging gene expression and clinical data
2. Log-transforming FPKM values
3. Handling missing values
4. Creating survival time and event variables
5. Train/test splitting

All preprocessing steps are documented in the scripts:

- `01_data_preparation.py`
- `02_data_normalization.py`

## Final Data Format

Place the processed data files in the `processed_data/` directory:

- `train_data_capped.csv`: Training data with gene expression and survival info
- `test_data_capped.csv`: Test data with gene expression and survival info

Each CSV file should have the following columns:

- `patient_id`: Unique identifier for each patient
- `survival_time`: Time to event in days
- `event`: Binary indicator (1=event occurred, 0=censored)
- Gene expression columns (e.g., ENSG00000000003.14, ENSG00000000005.5, etc.)
