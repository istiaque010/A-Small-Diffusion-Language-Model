"""Bidirectional Transformer denoiser.

Same block design as Project 1's GPT (pre-LayerNorm, learned positions, GELU MLP x4, biases, dropout, tied
embedding/output), with one change: attention has NO causal mask, so every position sees the whole window,
left and right. A key-padding mask is supported (positions marked as padding are never attended to).
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from mdlm.tokens import VOCAB


@dataclass
class MDLMConfig:
    vocab_size: int = VOCAB
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True
    time_cond: bool = False  # optional masking-level embedding (not used in the main runs)


class BidirectionalSelfAttention(nn.Module):
    def __init__(self, cfg: MDLMConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding: torch.Tensor | None = None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for z in (q, k, v))
        attn_mask = None
        if key_padding is not None:  # (B, T) True = padding -> not attended
            attn_mask = (~key_padding)[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False,
                                           dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: MDLMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = BidirectionalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias), nn.Dropout(cfg.dropout))

    def forward(self, x, key_padding=None):
        x = x + self.attn(self.ln1(x), key_padding)
        return x + self.mlp(self.ln2(x))


class MaskedDiffusionLM(nn.Module):
    def __init__(self, cfg: MDLMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.time_emb = nn.Sequential(nn.Linear(1, cfg.n_embd), nn.GELU(), nn.Linear(cfg.n_embd, cfg.n_embd)) \
            if cfg.time_cond else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying, as in Project 1
        self.apply(self._init)
        for n, p in self.named_parameters():  # GPT-2 style scaled init of residual projections
            if n.endswith("attn.proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, t: torch.Tensor | None = None, key_padding: torch.Tensor | None = None):
        """idx: (B, T) token ids with [MASK] at corrupted positions. Returns logits (B, T, vocab)."""
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        if self.time_emb is not None:
            assert t is not None, "time-conditioned model needs t"
            x = x + self.time_emb(t.view(B, 1, 1).float())
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x, key_padding)
        return self.head(self.ln_f(x))


def masked_ce(logits: torch.Tensor, x0: torch.Tensor, masked: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy over masked (non-padding) positions only: the guide's L_mask."""
    return F.cross_entropy(logits[masked], x0[masked])


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    return {"total": total, "non_embedding": total - emb}


def save_checkpoint(path, model, optimizer, step, best_val, cfg, run_cfg):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
                "step": step, "best_val": best_val, "model_config": asdict(model.cfg), "config": cfg,
                "run": run_cfg}, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MaskedDiffusionLM(MDLMConfig(**ck["model_config"]))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck
