# ========= Imports ==========
import pandas as pd
import scanpy as sc
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder

# ========= Create output directory ==========
os.makedirs("figures_0421", exist_ok=True)

# ========= Evaluation Function ==========
def evaluate(pred_csv, true_h5ad, tag):
    print(f"===== Evaluating on {tag} set =====")
    
    # Step 1: Load predictions and ground truth
    df = pd.read_csv(pred_csv)
    adata = sc.read_h5ad(true_h5ad)
    assert len(df) == adata.n_obs, "Mismatch in cell count"

    # Step 2: Encode labels
    true_labels = adata.obs["cell_type"].values
    pred_labels = df["cluster_aligned"].values

    le_true = LabelEncoder()
    le_pred = LabelEncoder()

    y_true = le_true.fit_transform(true_labels)
    y_pred = le_pred.fit_transform(pred_labels)

    # Step 3: Evaluation metrics
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Accuracy:         {acc:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"Precision (Macro): {prec_macro:.4f}")
    print(f"Recall (Macro):    {recall_macro:.4f}\n")

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=le_true.classes_))

    # Step 4: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=le_pred.classes_,
                yticklabels=le_true.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    plt.savefig(f"figures_0421/confusion_matrix_{tag}.png", dpi=300)
    plt.close()
    print(f"Confusion matrix saved to figures_0421/confusion_matrix_{tag}.png\n")

    # Step 5: UMAP Visualization
    adata.obs["cluster_aligned"] = df["cluster_aligned"].values
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.settings.figdir = "figures_0421"
    sc.pl.umap(adata, color="cell_type", title=f"UMAP: True ({tag})", show=False,
               save=f"_true_{tag}_0421.svg")
    sc.pl.umap(adata, color="cluster_aligned", title=f"UMAP: Predicted ({tag})", show=False,
               save=f"_predicted_{tag}_0421.svg")
    print(f"UMAP saved to figures_0421/umap_true_{tag}_0421.svg and umap_predicted_{tag}_0421.svg\n")

# ========= Main Logic ==========
if __name__ == "__main__":
    evaluate("prediction_mouse_test.csv", "data/mouse_test_0421.h5ad", tag="mouse_test")
    evaluate("prediction_mouse_valid.csv", "data/mouse_valid_0421.h5ad", tag="mouse_valid")
