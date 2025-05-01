#!/bin/bash
#SBATCH --job-name=scbert_pbmc_pretrain
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=2

module load python/3.10.4
module load cuda/11.8.0

# Activate your venv
source ~/scbert_env/bin/activate

cd ~/scBERT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  pretrain.py \
  --data_path ./data/pbmc_raw_normalized.h5ad \
  --model_name pbmc_pretrain \
  --ckpt_dir ./pbmc_ckpt \
  --epoch 20 \
  --batch_size 2
