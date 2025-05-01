# -*- coding: utf-8 -*-
# Pretrain any dataset with full data (no split)

import argparse
import os
import math
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
from performer_pytorch import PerformerLM
from utils import (
    seed_all, CosineAnnealingWarmupRestarts,
    get_reduced, save_ckpt, save_best_ckpt
)

# ------------------------ Argument parser ------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--bin_num", type=int, default=5)
parser.add_argument("--gene_num", type=int, required=True)
parser.add_argument("--epoch", type=int, default=50)
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_acc", type=int, default=16)
parser.add_argument("--mask_prob", type=float, default=0.15)
parser.add_argument("--replace_prob", type=float, default=0.9)
parser.add_argument("--pos_embed", type=bool, default=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--ckpt_dir", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
args = parser.parse_args()

# ------------------------ Setup ------------------------
local_rank = args.local_rank
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
torch.distributed.init_process_group(backend="nccl")
rank = int(os.environ.get("RANK", 0))
is_master = (rank == 0)
world_size = torch.distributed.get_world_size()
seed_all(args.seed + rank)

SEQ_LEN = args.gene_num + 1
CLASS = args.bin_num + 2
MASK_TOKEN_ID = CLASS - 1
PAD_TOKEN_ID = CLASS - 1
MASK_IGNORE_TOKEN_IDS = [0]

# ------------------------ Data ------------------------
adata = sc.read_h5ad(args.data_path)
X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X

class SCDataset(torch.utils.data.Dataset):
    def __init__(self, array):
        self.data = torch.from_numpy(array).long()

    def __getitem__(self, idx):
        x = torch.clamp(self.data[idx], max=(CLASS - 2))
        return torch.cat([x, torch.tensor([0])]).to(device)

    def __len__(self):
        return self.data.shape[0]

dataset = SCDataset(X)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

# ------------------------ Masking ------------------------
def prob_mask_like(t, prob): return torch.rand_like(t.float()) < prob
def mask_with_tokens(t, token_ids): return torch.stack([t == token for token in token_ids]).any(dim=0)
def get_mask_subset_with_prob(mask, prob):
    batch, seq_len = mask.shape
    max_masked = math.ceil(prob * seq_len)
    rand = torch.rand((batch, seq_len), device=mask.device).masked_fill(~mask, -1e9)
    _, idx = rand.topk(max_masked, dim=-1)
    final_mask = torch.zeros((batch, seq_len), device=mask.device)
    final_mask.scatter_(1, idx, 1)
    return final_mask.bool()

def data_mask(data, mask_prob=args.mask_prob, replace_prob=args.replace_prob):
    no_mask = mask_with_tokens(data, MASK_IGNORE_TOKEN_IDS + [PAD_TOKEN_ID])
    mask = get_mask_subset_with_prob(~no_mask, mask_prob)
    masked_input = data.clone()
    replace = prob_mask_like(data, replace_prob)
    masked_input[mask & replace] = MASK_TOKEN_ID
    labels = data.masked_fill(~mask, PAD_TOKEN_ID)
    return masked_input, labels

# ------------------------ Model ------------------------
model = PerformerLM(
    num_tokens=CLASS,
    dim=200,
    depth=6,
    max_seq_len=SEQ_LEN,
    heads=10,
    g2v_position_emb=args.pos_embed
).to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
scheduler = CosineAnnealingWarmupRestarts(optimizer, 15, 2, args.learning_rate, 1e-6, 5, 0.9)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID).to(device)

# ------------------------ Training ------------------------
best_loss = float("inf")
for epoch in range(1, args.epoch + 1):
    sampler.set_epoch(epoch)
    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader, 1):
        batch = batch.to(device)
        batch, labels = data_mask(batch)
        logits = model(batch)
        loss = loss_fn(logits.transpose(1, 2), labels) / args.grad_acc

        if step % args.grad_acc != 0:
            with model.no_sync(): loss.backward()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e2)
            optimizer.step(); optimizer.zero_grad()

        total_loss += loss.item()

    avg_loss = total_loss / step
    avg_loss = get_reduced(avg_loss, local_rank, 0, world_size)
    scheduler.step()

    if is_master:
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")
        save_ckpt(epoch, model, optimizer, scheduler, avg_loss, args.model_name, args.ckpt_dir)
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_best_ckpt(epoch, model, optimizer, scheduler, avg_loss, args.model_name, args.ckpt_dir)
