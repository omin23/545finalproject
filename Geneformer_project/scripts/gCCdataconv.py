import sys
sys.path.append('/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/scFoundationModels')
import scFoundationModels.notebooks.anndata2embedding.embed as embed
import scanpy as sc

# Load the dataset
adata = sc.read("Datasets/filtered_ms_adata.h5ad")

# Input the anndata in the embed script 
adata_update = embed.embed(adata, model="geneformer", model_directory="/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Geneformer/gf-12L-95M-i4096", output_file="filtered_ms_geneformer.h5ad")

