# get_embedding_float.py

import argparse
import scanpy as sc
import numpy as np
import torch
from performer_lm_float import PerformerLM_Float

def load_data(h5ad_path):
    adata = sc.read_h5ad(h5ad_path)
    if not isinstance(adata.X, np.ndarray):
        X = adata.X.toarray()
    else:
        X = adata.X
    return X, adata

def load_model(model_path, input_dim, device):
    model = PerformerLM_Float(input_dim=input_dim, dim=200, depth=6, heads=10)
    ckpt = torch.load(model_path, map_location=device)

    state_dict = ckpt.get("model_state_dict", ckpt)
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"✅ Loaded {len(pretrained_dict)} parameters from checkpoint.")
    return model

def get_embeddings(model, X, device, batch_size=64):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=device)
            emb = model(batch)
            all_embeddings.append(emb.cpu().numpy())
            if (i // batch_size + 1) % 10 == 0:
                print(f"Progress: {i + batch_size} / {X.shape[0]} cells")
    return np.vstack(all_embeddings)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("📥 Loading data...")
    X, _ = load_data(args.input)

    print("📦 Loading model...")
    model = load_model(args.model, input_dim=X.shape[1], device=device)

    print("🧠 Generating embeddings...")
    embeddings = get_embeddings(model, X, device)

    print("💾 Saving to", args.output)
    np.save(args.output, embeddings)
    print("✅ Done!")

if __name__ == '__main__':
    main()
