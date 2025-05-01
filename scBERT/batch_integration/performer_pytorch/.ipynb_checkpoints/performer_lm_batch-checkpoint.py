import torch
from torch import nn
from performer_pytorch import PerformerLM

class PerformerLM_Batch(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, max_seq_len, num_batches):
        super().__init__()
        
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.batch_emb = nn.Embedding(num_batches, dim)
        self.dropout = nn.Dropout(0.1)

        # Do not pass num_tokens to PerformerLM to avoid second embedding layer
        self.performer = PerformerLM(
            dim=dim,
            depth=depth,
            heads=heads,
            max_seq_len=max_seq_len,
            g2v_position_emb=False,
            num_tokens=num_tokens  # <== Prevent PerformerLM from creating token_emb
        )

    def forward(self, x, batch):
        bsz, seq_len = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        pos = pos.unsqueeze(0).expand(bsz, -1, -1)
        batch_tok = self.batch_emb(batch).unsqueeze(1)
        x = tok + pos + batch_tok
        x = self.dropout(x)
        x = self.performer.performer(x)     # 只走 transformer block
        x = self.performer.norm(x)          # 加 norm
        return x

