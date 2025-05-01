# -*- coding: utf-8 -*-

import os
import argparse
import math
from functools import reduce

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import scanpy as sc
from performer_pytorch import PerformerLM
from utils import (
    seed_all, CosineAnnealingWarmupRestarts, SequentialDistributedSampler,
    get_reduced, distributed_concat, save_ckpt, save_best_ckpt
)

# ------------------------ Argument parser ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, default=3000)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=60)
parser.add_argument("--valid_every", type=int, default=1)
parser.add_argument("--mask_prob", type=float, default=0.15)
parser.add_argument("--replace_prob", type=float, default=0.9)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--train_path", type=str, default='data/MS_train.h5ad')
parser.add_argument("--valid_path", type=str, default='data/MS_valid.h5ad')
parser.add_argument("--ckpt_dir", type=str, default='ckpts_MS/')
parser.add_argument("--model_name", type=str, default='pretrain_MS_0420')
args = parser.parse_args()

# ------------------------ Distributed setup ------------------------
local_rank = args.local_rank
rank = int(os.environ.get("RANK", 0))
is_master = (rank == 0)

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = dist.get_world_size()
seed_all(args.seed + rank)

# ------------------------ Constants ------------------------
SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]

# ------------------------ Masking utilities ------------------------
def prob_mask_like(t, prob):
    return torch.rand_like(t.float()) < prob

def mask_with_tokens(t, token_ids):
    mask = torch.full_like(t, False, dtype=torch.bool)
    for token in token_ids:
        mask |= (t == token)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len = mask.shape
    max_masked = math.ceil(prob * seq_len)
    rand = torch.rand((batch, seq_len), device=mask.device).masked_fill(~mask, -1e9)
    _, sampled_idx = rand.topk(max_masked, dim=-1)
    final_mask = torch.zeros((batch, seq_len), device=mask.device)
    final_mask.scatter_(1, sampled_idx, 1)
    return final_mask.bool()

def data_mask(data, mask_prob=args.mask_prob, replace_prob=args.replace_prob):
    no_mask = mask_with_tokens(data, MASK_IGNORE_TOKEN_IDS + [PAD_TOKEN_ID])
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_input = data.clone()
    replace = prob_mask_like(data, replace_prob)
    masked_input[mask & replace] = MASK_TOKEN_ID
    labels = data.masked_fill(~mask, PAD_TOKEN_ID)
    return masked_input, labels

# ------------------------ Dataset ------------------------
class SCDataset(Dataset):
    def __init__(self, array):
        self.data = torch.from_numpy(array).long()

    def __getitem__(self, index):
        x = torch.clamp(self.data[index], max=(CLASS - 2))
        return torch.cat((x, torch.tensor([0]))).to(device)

    def __len__(self):
        return self.data.shape[0]

# ------------------------ Load data ------------------------
adata_train = sc.read_h5ad(args.train_path)
adata_valid = sc.read_h5ad(args.valid_path)

X_train = adata_train.X.toarray() if not isinstance(adata_train.X, np.ndarray) else adata_train.X
X_valid = adata_valid.X.toarray() if not isinstance(adata_valid.X, np.ndarray) else adata_valid.X

train_dataset = SCDataset(X_train)
valid_dataset = SCDataset(X_valid)

train_sampler = DistributedSampler(train_dataset)
valid_sampler = SequentialDistributedSampler(valid_dataset, batch_size=args.batch_size, world_size=world_size)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)

# ------------------------ Model setup ------------------------
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    local_attn_heads=0,
    g2v_position_emb=args.pos_embed
).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=args.learning_rate,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID).to(device)

# ------------------------ Training loop ------------------------
best_val_loss = float("inf")

for epoch in range(1, args.epoch + 1):
    train_loader.sampler.set_epoch(epoch)
    model.train()
    dist.barrier()
    total_loss = 0.0

    for step, batch in enumerate(train_loader, 1):
        batch = batch.to(device)
        batch, labels = data_mask(batch)
        logits = model(batch)
        loss = loss_fn(logits.transpose(1, 2), labels) / args.grad_acc

        if step % args.grad_acc != 0:
            with model.no_sync():
                loss.backward()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / step
    avg_train_loss = get_reduced(avg_train_loss, local_rank, 0, world_size)
    scheduler.step()
    dist.barrier()

    if is_master:
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

    # ---------- validation ----------
    if epoch % args.valid_every == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in valid_loader:
                val_batch = val_batch.to(device)
                val_batch, labels = data_mask(val_batch)
                val_logits = model(val_batch)
                loss = loss_fn(val_logits.transpose(1, 2), labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(valid_loader)
        avg_val_loss = get_reduced(avg_val_loss, local_rank, 0, world_size)

        if is_master:
            print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.6f}")
            save_ckpt(epoch, model, optimizer, scheduler, avg_train_loss, args.model_name, args.ckpt_dir)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_best_ckpt(epoch, model, optimizer, scheduler, avg_val_loss, args.model_name, args.ckpt_dir)
