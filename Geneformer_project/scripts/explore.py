import scanpy as sc
import anndata as ad
import pandas as pd
from rich.console import Console
import numpy as np
import os


con = Console()

df = pd.read_csv("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/mouse_brain_processed_RNA.csv")
df = df.T
con.print(df.head(), style="bold red")
con.print("DataFrame loaded successfully.", style="bold green")
# con.print(df["cell_type"], style="bold blue")
con.print(df.index, style="bold blue")

adata = ad.AnnData(X=df.values)
adata.obs_names = df.index
adata.var_names = df.columns
con.print("AnnData object created successfully.", style="bold green")
adata.write("/Users/macbook/Desktop/EECS545/545finalproject/Geneformer_project/Datasets/mouse_brain_processed.h5ad")


