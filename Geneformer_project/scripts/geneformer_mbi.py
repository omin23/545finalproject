from datasets import Dataset, DatasetDict
from Geneformer.geneformer import tokenizer 
import scanpy as sc
import harmonypy as hm
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

# Tokenize sequences


# Convert each cell's expression into a tokenized sequence
adata.obs["gene_sequence"] = [tokenizer(expr_matrix[i, :]) for i in range(expr_matrix.shape[0])]

# Assume we have class labels for classification (modify as needed)
df = adata.obs[["gene_sequence"]]

df["description"] = "This dataset contains gene sequences and binary labels."

# normalize
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Select highly variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var["highly_variable"]]


sc.pp.pca(adata, n_comps=50)
adata.obsm["X_pca"] = hm.run_harmony(adata.obsm["X_pca"], adata.obs, "batch").Z_corr
adata.write("integrated_dataset.h5ad")

genes = np.array(adata.var_names)
expr_matrix = adata.X 

def tokenize_cell_expression(expression_row):
    """Sort genes by expression level and create a gene sequence"""
    sorted_gene_indices = np.argsort(-expression_row)  # Descending order
    sorted_genes = genes[sorted_gene_indices]
    return " ".join(sorted_genes)

# Convert each cell into a tokenized sequence
adata.obs["gene_sequence"] = [tokenize_cell_expression(expr_matrix[i, :]) for i in range(expr_matrix.shape[0])]

# Save tokenized dataset
adata.write("tokenized_integrated_dataset.h5ad")


# TOKENIZER 
tokenizer = AutoTokenizer.from_pretrained("theislab/geneformer")

# Tokenize sequences
adata.obs["tokenized"] = adata.obs["gene_sequence"].apply(lambda x: tokenizer(x, truncation=True, padding="max_length"))

# Save processed dataset
adata.write("final_geneformer_ready_dataset.h5ad")
