"""Training examples: 256-token windows that start at a story's <BOS>.

Project 1 stores every split as one long stream: <BOS> story <EOS> <BOS> story <EOS> ... . The autoregressive
models read arbitrary non-overlapping windows of that stream. A diffusion model generates a whole window at once
starting from position 0, so its windows are *story-aligned*: each starts at a <BOS> and continues with the
following text (the story, its <EOS>, and the beginning of the next story if the story is short). No padding is
needed, so every position is a real token. Generation then starts from <BOS> and the story ends at the first <EOS>.
"""
import os

import numpy as np
import torch

import mdlm
from mdlm.tokens import BOS_ID
from src.dataset import load_tokens


def story_starts(split: str, block: int) -> np.ndarray:
    """Positions of every <BOS> whose window of `block` tokens fits in the stream (cached in data/)."""
    os.makedirs(mdlm.CACHE_DIR, exist_ok=True)
    path = os.path.join(mdlm.CACHE_DIR, f"{split}_starts.npy")
    ids = load_tokens(mdlm.DATA_DIR, split)
    if os.path.exists(path):
        starts = np.load(path)
    else:
        starts = np.flatnonzero(np.asarray(ids) == BOS_ID).astype(np.int64)
        np.save(path, starts)
    return starts[starts + block <= len(ids)]


class StoryWindows(torch.utils.data.Dataset):
    def __init__(self, split: str, block: int = 256, limit: int | None = None):
        self.ids = load_tokens(mdlm.DATA_DIR, split)
        self.block = block
        self.starts = story_starts(split, block)
        if limit is not None:
            self.starts = self.starts[:limit]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return torch.from_numpy(np.asarray(self.ids[s:s + self.block], dtype=np.int64))


def fixed_subset(split: str, n: int, block: int = 256, seed: int = 0) -> torch.Tensor:
    """n story-aligned windows chosen with a fixed seed (for validation / test protocols). Shape (n, block)."""
    ds = StoryWindows(split, block)
    idx = np.random.default_rng(seed).choice(len(ds), size=min(n, len(ds)), replace=False)
    idx.sort()
    return torch.stack([ds[int(i)] for i in idx])
