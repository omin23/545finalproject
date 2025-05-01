import pandas as pd
import numpy as np
import torch
from torch import nn
import scanpy as sc
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from performer_pytorch import PerformerLM
from tqdm import tqdm
import os

# ---------------------
# Fixed parameters
# ---------------------
data_path = "./data/filtered_ms_adata.h5ad"
model_path = "pretrain_MS_nosplit_best.pth"
bin_num = 5
gene_num = 3000
n_clusters = 18
output_prefix = "prediction_MS_all"
batch_size = 64

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
X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X  # 全部提前转dense
cell_types = adata.obs["celltype"].values
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
# Run batch prediction
# ---------------------
cls_vectors = []

with torch.no_grad():
    for i in tqdm(range(0, n_cells, batch_size), desc="Predicting"):
        x_batch = X_raw[i:i + batch_size]
        x_batch = np.clip(x_batch, 0, CLASS - 2).astype(int)

        # 构建 CLS token 输入：每行加一位
        cls_token = np.zeros((x_batch.shape[0], SEQ_LEN), dtype=int)
        cls_token[:, :-1] = x_batch
        x_tensor = torch.from_numpy(cls_token).long().to(device)

        emb = model(x_tensor)
        cls_batch = emb[:, 0, :].cpu().numpy()
        cls_vectors.extend(cls_batch)

cls_matrix = np.vstack(cls_vectors)

# ---------------------
# KMeans clustering
# ---------------------
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(cls_matrix)

# ---------------------
# Hungarian alignment
# ---------------------
true_labels = pd.Categorical(cell_types).codes
conf_mat = confusion_matrix(true_labels, cluster_labels)
row_ind, col_ind = linear_sum_assignment(-conf_mat)
mapping = {cluster: target for cluster, target in zip(col_ind, row_ind)}
aligned_preds = [mapping[c] for c in cluster_labels]
aligned_labels = pd.Categorical.from_codes(aligned_preds, categories=pd.Categorical(cell_types).categories)

# ---------------------
# Save result
# ---------------------
df_out = pd.DataFrame({
    "cell_type": cell_types,
    "cluster_raw": cluster_labels,
    "cluster_aligned": aligned_labels
})
output_path = f"{output_prefix}.csv"
df_out.to_csv(output_path, index=False)
print(f"\n✅ Saved prediction to: {output_path}")
