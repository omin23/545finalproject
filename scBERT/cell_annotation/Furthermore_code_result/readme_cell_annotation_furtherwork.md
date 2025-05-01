# scBERT Pipeline: Pretraining, Inference, and Evaluation

After training a downstream classifier and observing poor performance, I conducted a zero-shot evaluation using KMeans clustering followed by Hungarian alignment. This approach allows me to assess whether the pre-trained encoder (scBERT) has actually captured meaningful biological structure in the data, independent of any classifier. If the embeddings themselves are not informative, it suggests that the problem lies in the encoder rather than the classifier.

This repository implements a full pipeline for:
- Pretraining a Performer-based language model on single-cell RNA-seq data using masked language modeling
- Performing **zero-shot cell type clustering** using CLS token embeddings
- Evaluating predictions via classification metrics and UMAP visualizations

Supported datasets:
- Mouse dataset
- Multiple Sclerosis (MS) dataset

---

## Pipeline Overview

### 1. Pretraining (Masked Language Modeling)

Scripts:
- `pretrain_mouse_0420.py`
- `pretrain_MS_0420.py`

These scripts train a PerformerLM model with masked language modeling using distributed training and validation.

**Example usage:**

```bash
# Mouse dataset
python pretrain_mouse_0420.py \
  --train_path data/mouse_train.h5ad \
  --valid_path data/mouse_valid.h5ad \
  --gene_num 865 \
  --bin_num 5 \
  --ckpt_dir ckpts_mouse/ \
  --model_name pretrain_mouse

# MS dataset
python pretrain_MS_0420.py \
  --train_path data/MS_train.h5ad \
  --valid_path data/MS_valid.h5ad \
  --gene_num 3000 \
  --bin_num 5 \
  --ckpt_dir ckpts_MS/ \
  --model_name pretrain_MS
```

Checkpoints are saved under the specified directory, with the best model (lowest validation loss) saved as `*_best.pth`.

---

### 2. Zero-shot Prediction and Clustering

Scripts:
- `predict_mouse_0421.py`, `predict_mouse_0421_appended.py`
- `predict_MS_0421.py`, `predict_MS_0421_appended.py`

You can either:
- Use the `*_0421.py` scripts to split the dataset into **validation/test subsets**, then predict
- Or use the `*_appended.py` scripts to **predict directly on full datasets** without splitting

**Example usage (split sets):**

```bash
# Mouse
python predict_mouse_0421.py \
  --data_path data/mouse_all.h5ad \
  --model_path ckpts_mouse/pretrain_mouse_best.pth \
  --gene_num 865 \
  --bin_num 5 \
  --n_clusters 7 \
  --output_prefix prediction_mouse

# MS
python predict_MS_0421.py \
  --data_path data/filtered_ms_data.h5ad \
  --model_path ckpts_MS/pretrain_MS_best.pth \
  --gene_num 3000 \
  --bin_num 5 \
  --n_clusters 18 \
  --output_prefix prediction_MS
```

**Example usage (full data):**

```bash
# Mouse
python predict_mouse_0421_appended.py

# MS
python predict_MS_0421_appended.py
```

**Outputs:**
- Predictions: `prediction_mouse_test.csv`, `prediction_MS_all.csv`, etc.
- Optional `.h5ad` split subsets saved under `data/`

---

### 3. Evaluation and Visualization

Scripts:
- `evaluate_mouse_0421.py`, `evaluate_mouse_0421_updated.py`
- `evaluate_MS_0421.py`, `evaluate_MS_0421_updated.py`

Each script:
- Compares predicted clusters to ground-truth cell types
- Computes classification metrics: Accuracy, F1 (macro/micro), Precision, Recall
- Generates confusion matrix
- Draws UMAP plots of true and predicted annotations

The `*_updated.py` versions also support **UMAP visualization on full datasets** (`prediction_*_all.csv`).

**Example usage:**

```bash
# Mouse (split sets + full-data UMAP)
python evaluate_mouse_0421_updated.py

# MS (split sets + full-data UMAP)
python evaluate_MS_0421_updated.py
```

**Outputs saved to `figures_0421/`:**
- `confusion_matrix_mouse_test.png`
- `umap_true_mouse_test_0421.svg`, `umap_predicted_mouse_all_0421.svg`
- `confusion_matrix_MS_test.png`, `umap_predicted_MS_all_0421.svg`, etc.

---

## Requirements

```bash
pip install torch scanpy pandas numpy scikit-learn matplotlib seaborn performer-pytorch tqdm
```

---

## Folder Structure

```
.
├── pretrain_mouse_0420.py
├── pretrain_MS_0420.py
├── predict_mouse_0421.py
├── predict_mouse_0421_appended.py
├── predict_MS_0421.py
├── predict_MS_0421_appended.py
├── evaluate_mouse_0421.py
├── evaluate_mouse_0421_updated.py
├── evaluate_MS_0421.py
├── evaluate_MS_0421_updated.py
├── ckpts_mouse/               # Mouse model checkpoints
├── ckpts_MS/                  # MS model checkpoints
├── data/                      # .h5ad inputs + .csv predictions
└── figures_0421/              # Visual outputs (UMAPs, confusion matrices)
```

---

## Notes

- No fine-tuning is performed: all clustering is **zero-shot** using pretrained CLS embeddings.
- Binarized gene expression matrices are used (`bin_num` controls the number of bins).
- Cluster alignment uses the **Hungarian algorithm** based on confusion matrix.

