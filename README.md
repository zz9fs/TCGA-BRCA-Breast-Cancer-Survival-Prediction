# Breast Cancer Survival Analysis: A Comprehensive Machine Learning Approach

## Project Overview

This project implements a comprehensive machine learning pipeline for predicting breast cancer patient survival from gene expression data. It compares linear techniques (Cox regression with LASSO, Ridge, and Elastic Net regularization) and nonlinear approaches (Decision Trees and Random Survival Forests) to identify prognostic gene signatures.

## Key Features

- Comprehensive survival analysis using various ML approaches
- Feature selection and gene signature identification
- Model comparison and performance evaluation
- Biological interpretation of machine learning results

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Data Preparation

1. Download the processed TCGA-BRCA dataset (Gene expression + survival data)
2. Place files in the `processed_data/` directory:
   - `train_data_capped.csv`
   - `test_data_capped.csv`

## Pipeline Execution

Run scripts in the following order:

1. **Cox Regression with LASSO Regularization**

   ```bash
   python 14_cox_lasso.py
   ```

2. **Cox Regression with Ridge Regularization**

   ```bash
   python 15_cox_ridge.py
   ```

3. **Cox Regression with Elastic Net Regularization**

   ```bash
   python 16_cox_elastic_net.py
   ```

4. **Decision Tree Survival Analysis**

   ```bash
   python 18_decision_tree_survival.py
   ```

5. **Random Survival Forest Analysis**

   ```bash
   python 19_random_forest_survival.py
   ```

6. **Model Evaluation and Comparison**

   ```bash
   python 20_model_evaluation_fixed.py
   ```

7. **Feature Importance Analysis**
   ```bash
   python 21_feature_interpretation.py
   ```

## Results

- Random Forest achieved the best performance (C-index: 0.759)
- Elastic Net was the best linear model (C-index: 0.749)
- Different modeling approaches identified largely distinct gene signatures
- Key genes were found that have prognostic value across multiple models
