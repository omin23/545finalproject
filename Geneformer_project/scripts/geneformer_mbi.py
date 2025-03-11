from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import scanpy as sc
import datetime
import pandas as pd
import numpy as np
import os
import scvi

current_date = datetime.datetime.now()
datestamp = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}{current_date.hour:02d}{current_date.minute:02d}{current_date.second:02d}"
datestamp_min = f"{str(current_date.year)[-2:]}{current_date.month:02d}{current_date.day:02d}"

# set the output directory and prefix
output_prefix = "mbi_classifier_test"
output_dir = f"Geneformer_project/runs/{datestamp}"

# get the data from the scanpy package
adata = scvi.data.pbmc_dataset()
    

genes = np.array(adata.var_names)
expr_matrix = adata.X

def tokenize_cell_expression(expression_row):
    """Sort genes by expression level and create a gene sequence"""
    sorted_gene_indices = np.argsort(-expression_row) 
    sorted_genes = genes[sorted_gene_indices]
    return " ".join(sorted_genes)

# Convert each cell's expression into a tokenized sequence
adata.obs["gene_sequence"] = [tokenize_cell_expression(expr_matrix[i, :]) for i in range(expr_matrix.shape[0])]

# Assume we have class labels for classification (modify as needed)
df = adata.obs[["gene_sequence"]]

df["description"] = "This dataset contains gene sequences and binary labels."

print(df.head())  # Preview

