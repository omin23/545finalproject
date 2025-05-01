import numpy as np
import torch
import scanpy as sc
from performer_pytorch import PerformerLM

# ====== 手动参数设置 ======
data_path = "data/mouse_all.h5ad"
model_path = "pretrain_mouse_nosplit_best.pth"
bin_num = 5
gene_num = 865
output_prefix = "mouse"

# ====== 常量定义 ======
SEQ_LEN = gene_num + 1
CLASS = bin_num + 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 加载数据 ======
adata = sc.read_h5ad(data_path)
X_raw = adata.X
cell_types = adata.obs["cell_type"].values
n_cells = adata.n_obs
assert n_cells == X_raw.shape[0] == len(cell_types)
print(f"Loaded {n_cells} cells with shape {X_raw.shape}, gene_num = {gene_num}")

# ====== Binning ======
X_dense = X_raw.toarray() if not isinstance(X_raw, np.ndarray) else X_raw
if X_dense.max() > bin_num:
    print(f"[INFO] Max count {X_dense.max():.2f} > bin_num={bin_num}, applying binning...")
    bins = np.linspace(0, np.max(X_dense), bin_num + 1)
    X_binned = np.digitize(X_dense, bins) - 1
    X_binned = np.clip(X_binned, 0, bin_num - 1)
else:
    print("[INFO] Input already looks like token IDs, skipping binning.")
    X_binned = X_dense.astype(int)

# ====== 加载预训练模型 ======
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

# ====== 提取 CLS embedding ======
print("Extracting CLS embeddings...")
cls_vectors = []
with torch.no_grad():
    for i in range(n_cells):
        x = X_binned[i]
        x_tensor = torch.from_numpy(x).long()
        x_tensor = torch.cat([x_tensor, torch.tensor([0])])  # CLS token
        x_tensor = x_tensor.unsqueeze(0).to(device)

        embedding = model(x_tensor, return_encodings=True)  # shape: (1, seq_len, dim)
        cls_token = embedding[:, 0, :]                       # (1, dim)
        cls_vectors.append(cls_token.squeeze(0).cpu().numpy())

cls_matrix = np.vstack(cls_vectors)
print(f"CLS embedding shape: {cls_matrix.shape}")

# ====== 保存输出 ======
np.save(f"{output_prefix}_X_cls.npy", cls_matrix)
np.save(f"{output_prefix}_y_cls.npy", cell_types)
print(f"Saved: {output_prefix}_X_cls.npy and {output_prefix}_y_cls.npy")