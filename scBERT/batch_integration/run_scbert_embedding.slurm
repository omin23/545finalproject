#!/bin/bash
#SBATCH --job-name=scbert_embed
#SBATCH --output=scbert_embed.out
#SBATCH --error=scbert_embed.err
#SBATCH --time=02:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ⚙️ Load environment
module purge
module load python/3.10.4 cuda/11.8.0

# ⚠️ 记得根据你的实际情况激活 virtualenv 或 conda
source ~/scbert_env/bin/activate

# 🧠 Run embedding generation
python scbert_get_embedding.py \
  --input ./data/pbmc_raw_normalized.h5ad \
  --model ./data/pbmc_batchaware_120.pth \
  --output ./data/pbmc_scBERT_batchaware_embedding_120.npy

