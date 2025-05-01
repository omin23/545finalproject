import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from performer_pytorch import PerformerLM
import os

# ---------------------
# Argument parsing
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=865)
parser.add_argument("--n_clusters", type=int, default=7)
parser.add_argument("--output_prefix", type=str, default="prediction_mouse")
args = parser.parse_args()

# ---------------------
# Constants
# ---------------------
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Load data
# ---------------------
adata = sc.read_h5ad(args.data_path)
X_raw = adata.X
cell_types = adata.obs["cell_type"].values
n_cells = adata.n_obs

# ---------------------
# Split data
# ---------------------
indices = np.arange(n_cells)
train_idx, temp_idx = train_test_split(indices, test_size=0.3, stratify=cell_types, random_state=42)
valid_idx, test_idx = train_test_split(temp_idx, test_size=0.7, stratify=cell_types[temp_idx], random_state=42)

adata_valid = adata[valid_idx].copy()
adata_test = adata[test_idx].copy()

os.makedirs("data", exist_ok=True)
adata_valid.write("data/mouse_valid_0421.h5ad")
adata_test.write("data/mouse_test_0421.h5ad")
print("Saved data/mouse_valid_0421.h5ad and data/mouse_test_0421.h5ad")

# ---------------------
# Load pretrained model
# ---------------------
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=True
)
ckpt = torch.load(args.model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# ---------------------
# Prediction function
# ---------------------
def run_predict(adata_split, name_suffix):
    X_split = adata_split.X
    cell_types = adata_split.obs["cell_type"].values
    cls_vectors = []

    with torch.no_grad():
        for i in range(X_split.shape[0]):
            x = X_split[i].toarray()[0] if hasattr(X_split[i], 'toarray') else X_split[i]
            x = np.clip(x, 0, CLASS - 2)
            x_tensor = torch.from_numpy(x).long()
            x_tensor = torch.cat([x_tensor, torch.tensor([0])])
            x_tensor = x_tensor.unsqueeze(0).to(device)

            emb = model(x_tensor)
            cls_token = emb[:, 0, :]
            cls_vectors.append(cls_token.squeeze(0).cpu().numpy())

    cls_matrix = np.vstack(cls_vectors)

    # KMeans clustering and alignment
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(cls_matrix)

    true_labels = pd.Categorical(cell_types).codes
    conf_mat = confusion_matrix(true_labels, cluster_labels)
    row_ind, col_ind = linear_sum_assignment(-conf_mat)
    mapping = {cluster: target for cluster, target in zip(col_ind, row_ind)}
    aligned_preds = [mapping[c] for c in cluster_labels]
    aligned_labels = pd.Categorical.from_codes(aligned_preds, categories=pd.Categorical(cell_types).categories)

    df_out = pd.DataFrame({
        "cell_type": cell_types,
        "cluster_raw": cluster_labels,
        "cluster_aligned": aligned_labels
    })
    output_path = f"{args.output_prefix}_{name_suffix}.csv"
    df_out.to_csv(output_path, index=False)
    print(f"Saved prediction to: {output_path}")

# ---------------------
# Run on both sets
# ---------------------
run_predict(adata_test, "test")
run_predict(adata_valid, "valid")
