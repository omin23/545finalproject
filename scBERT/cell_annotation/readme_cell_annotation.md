# scBERT Pipeline: Pretraining, Embedding Extraction, and Classification

This repository contains scripts to pretrain a scBERT model on single-cell RNA-seq data,  
extract CLS embeddings, and classify cell types using a transformer-based classifier.  
Two datasets are used in this pipeline:
- **Mouse dataset**
- **Multiple Sclerosis (MS) dataset**

---

## 1. Pretraining (`pretrain_0420.py`)

- **Input**: `.h5ad` file (entire dataset, no splits)  
- **Model**: `PerformerLM` (trained with masked language modeling)  
- **Output**: Checkpoints (`.pth`), best model separately saved  

This script is used twice, once per dataset:
- **Mouse**: `data/mouse_all.h5ad` → `pretrain_mouse_nosplit_best.pth`
- **MS**: `data/filtered_ms_adata.h5ad` → `pretrain_MS_nosplit_best.pth`

**Example (Mouse)**:
```bash
python pretrain_0420.py \
    --data_path data/mouse_all.h5ad \
    --gene_num 865 \
    --ckpt_dir ckpts/ \
    --model_name pretrain_mouse \
    --bin_num 5 \
    --epoch 50 \
    --batch_size 8
```

**Example (MS)**:
```bash
python pretrain_0420.py \
    --data_path data/filtered_ms_adata.h5ad \
    --gene_num 3000 \
    --ckpt_dir ckpts/ \
    --model_name pretrain_MS \
    --bin_num 5 \
    --epoch 50 \
    --batch_size 8
```

---

## 2. Embedding Extraction

**Scripts**:
- `extract_tokens_local_mouse.py`
- `extract_tokens_local_ms.py`

- **Input**: `.h5ad` + corresponding pretrained `.pth`  
- **Output**: `X_cls.npy` (CLS token embeddings), `y_cls.npy` (cell type labels)

**Example**:
```bash
# For Mouse
python extract_tokens_local_mouse.py

# For MS
python extract_tokens_local_ms.py
```

---

## 3. Classification

**Scripts**:
- `scBERT_mouse_classifier_train_test_validation.py`
- `scBERT_MS_classifier_train_test_validation.py`

- **Input**: CLS embedding + cell type labels  
- **Function**: Train a transformer classifier with train/test/valid splits  
- **Output**: Accuracy, Macro/Micro F1, Precision/Recall, UMAP visualizations

**Example**:
```bash
python scBERT_mouse_classifier_train_test_validation.py
python scBERT_MS_classifier_train_test_validation.py
```

---

## Output Files

- `mouse_X_cls.npy` / `ms_X_cls.npy` : CLS token embeddings  
- `mouse_y_cls.npy` / `ms_y_cls.npy` : Cell type labels  
- `figures_0420/*.svg` : UMAP visualizations of embedding space  
- `ckpts/` : Model checkpoints saved during pretraining

---

## Dependencies

- Python >= 3.8  
- PyTorch  
- scanpy  
- numpy  
- scikit-learn  
- umap-learn  
- matplotlib  
- seaborn
