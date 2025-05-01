import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project'))
import scFoundationModels.notebooks.anndata2embedding.embed as embed
import rich.console as Console
import scanpy as sc
con = Console.Console()


adata = sc.read_h5ad("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/mouse_brain1.h5ad")


adata_update = embed.embed(adata, model="geneformer", model_directory="/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Geneformer/gf-12L-95M-i4096", output_file="mouse_brain_gene.h5ad", verbose=True)

adata_update.write("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/runs/mouse_geneformer.h5ad")
