import pandas as pd
import numpy as np
import torch
from torch import nn
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from performer_pytorch import PerformerLM
import os

# ---------------------
# Fixed parameters (local version, no argparse)
# ---------------------
data_path = "./data/mouse_all.h5ad"
model_path = "pretrain_mouse_nosplit_best.pth"
bin_num = 5
gene_num = 865
n_clusters = 7
output_prefix = "prediction_mouse_all"

# ---------------------
# Constants
# ---------------------
SEQ_LEN = gene_num + 1
CLASS = bin_num + 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------
# Load data
# ---------------------
adata = sc.read_h5ad(data_path)
X_raw = adata.X
cell_types = adata.obs["cell_type"].values
n_cells = adata.n_obs

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
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# ---------------------
# Run prediction
# ---------------------
cls_vectors = []

with torch.no_grad():
    for i in range(X_raw.shape[0]):
        x = X_raw[i].toarray()[0] if hasattr(X_raw[i], 'toarray') else X_raw[i]
        x = np.clip(x, 0, CLASS - 2)
        x_tensor = torch.from_numpy(x).long()
        x_tensor = torch.cat([x_tensor, torch.tensor([0])])  # Add CLS token at end
        x_tensor = x_tensor.unsqueeze(0).to(device)

        emb = model(x_tensor)
        cls_token = emb[:, 0, :]  # Extract CLS token
        cls_vectors.append(cls_token.squeeze(0).cpu().numpy())

cls_matrix = np.vstack(cls_vectors)

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(cls_matrix)

# Hungarian alignment
true_labels = pd.Categorical(cell_types).codes
conf_mat = confusion_matrix(true_labels, cluster_labels)
row_ind, col_ind = linear_sum_assignment(-conf_mat)
mapping = {cluster: target for cluster, target in zip(col_ind, row_ind)}
aligned_preds = [mapping[c] for c in cluster_labels]
aligned_labels = pd.Categorical.from_codes(aligned_preds, categories=pd.Categorical(cell_types).categories)

# Save result
df_out = pd.DataFrame({
    "cell_type": cell_types,
    "cluster_raw": cluster_labels,
    "cluster_aligned": aligned_labels
})
output_path = f"{output_prefix}.csv"
df_out.to_csv(output_path, index=False)
print(f"Saved prediction to: {output_path}")

