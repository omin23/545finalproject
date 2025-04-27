import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project'))
import scFoundationModels.notebooks.anndata2embedding.embed as embed
import rich.console as Console
import scanpy as sc
con = Console.Console()

# df = pd.read_csv("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/mouse_brain_processed_RNA.csv")
# adata = sc.read_h5ad("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/mouse_brain1.h5ad")
adata = sc.read_h5ad("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/filtered_ms_adata.h5ad")



# # Input the anndata in the embed script 
# adata_update = embed.embed(adata, model="geneformer", model_directory="/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Geneformer/gf-12L-95M-i4096", output_file="mouse_brain_gene.h5ad", verbose=True)
adata_update = embed.embed(adata, model="geneformer", model_directory="/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Geneformer/gf-12L-95M-i4096", output_file="tester.h5ad", verbose=True)
# # Save the updated AnnData object
# adata_update.write("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/runs/mouse_geneformer.h5ad")
adata_update.write("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/runs/test.h5ad")
# # print("Embedding process completed.")

