"""Iterative parallel unmasking (Phases 9-11).

Start from a window of [MASK] tokens with <BOS> at position 0 (and an optional prompt right after it). At every
step the model predicts all masked positions at once; a *schedule* decides HOW MANY positions to reveal, a
*selection rule* decides WHICH ones, and a *sampling rule* decides WHAT token goes there. Positions that are not
revealed stay masked and are predicted again at the next step. Revealed tokens are never changed again.

Schedules (fraction of the initially masked positions still masked after step s of S):
  linear      r(s) = 1 - s/S
  cosine      r(s) = cos(pi/2 * s/S)            (reveals few tokens early, many late)
  confidence  linear quota, but also reveals every position whose confidence >= tau (adaptive: can finish early)
Selection: "confidence" (highest probability of the chosen token first) or "random".
Sampling: argmax, temperature, top-k, top-p (Project 1's select_next), applied per masked position.

The story is the text between <BOS> and the first <EOS>; a window without <EOS> counts as "hit the length limit".
"""
import math
import time

import torch
import torch.nn.functional as F

from mdlm.tokens import BOS_ID, EOS_ID, MASK_ID, PAD_ID
from src.generate import select_next

SCHEDULES = ("linear", "cosine", "confidence")


def remaining_fraction(schedule: str, s: int, steps: int) -> float:
    if schedule in ("linear", "confidence"):
        return 1.0 - s / steps
    if schedule == "cosine":
        return math.cos(0.5 * math.pi * s / steps)
    raise ValueError(schedule)


@torch.no_grad()
def generate(model, prompts: list[list[int]], length: int = 256, steps: int = 16, schedule: str = "linear",
             select: str = "confidence", method: str = "top_p", temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = 0.95, tau: float = 0.9, generator: torch.Generator | None = None, trace: bool = False):
    """prompts: token ids WITHOUT <BOS> (may be empty). Returns (x (B, length), info dict)."""
    model.eval()
    B = len(prompts)
    x = torch.full((B, length), MASK_ID, dtype=torch.long)
    x[:, 0] = BOS_ID
    for b, p in enumerate(prompts):
        x[b, 1:1 + len(p)] = torch.tensor(p, dtype=torch.long)
    n0 = (x == MASK_ID).sum(1)  # masked positions per row at the start
    nfe = 0
    revealed_per_step = []
    states = [x.clone()] if trace else None
    for s in range(1, steps + 1):
        is_mask = x == MASK_ID
        if not is_mask.any():
            break
        logits = model(x)
        nfe += 1
        logits[..., MASK_ID] = float("-inf")
        logits[..., PAD_ID] = float("-inf")
        flat = logits[is_mask]  # (M, V) for masked positions only
        tok = select_next(flat, method=method, temperature=temperature, top_k=top_k, top_p=top_p,
                          generator=generator).squeeze(1)
        conf = F.softmax(flat, dim=-1).gather(1, tok[:, None]).squeeze(1)  # model probability of the chosen token
        cand_tok = torch.full_like(x, -1)
        cand_conf = torch.full(x.shape, -1.0)
        cand_tok[is_mask] = tok
        cand_conf[is_mask] = conf
        if select == "random":
            score = torch.where(is_mask, torch.rand(x.shape, generator=generator), torch.full(x.shape, -1.0))
        elif select == "confidence":
            score = cand_conf
        else:
            raise ValueError(select)
        left = is_mask.sum(1)
        target_left = torch.floor(n0.float() * remaining_fraction(schedule, s, steps)).long()
        k = (left - target_left).clamp(min=1)
        if schedule == "confidence":
            k = torch.maximum(k, ((cand_conf >= tau) & is_mask).sum(1))
        if s == steps:
            k = left
        k = torch.minimum(k, left)
        order = score.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(length).expand(B, length))
        reveal = is_mask & (rank < k[:, None])
        x = torch.where(reveal, cand_tok, x)
        revealed_per_step.append(reveal.sum(1).tolist())
        if trace:
            states.append(x.clone())
    return x, {"nfe": nfe, "revealed_per_step": revealed_per_step, "states": states}


def split_story(ids: list[int], prompt_len: int) -> dict:
    """Window ids (starting with <BOS>) -> story ids between <BOS> and the first <EOS>."""
    body = ids[1:]
    ended = EOS_ID in body
    story = body[:body.index(EOS_ID)] if ended else body
    return {"story_ids": story, "new_ids": story[prompt_len:], "ended_with_eos": ended}


def generate_stories(model, tok, prompts: list[str], seed: int, **kw) -> list[dict]:
    """Generate one story per prompt (one batch). Returns dicts like Project 1's samples."""
    g = torch.Generator().manual_seed(seed)
    p_ids = [tok.encode(p) for p in prompts]
    t0 = time.perf_counter()
    x, info = generate(model, p_ids, generator=g, **kw)
    dt = time.perf_counter() - t0
    out = []
    for b, p in enumerate(prompts):
        sp = split_story(x[b].tolist(), len(p_ids[b]))
        completion = tok.decode(sp["new_ids"])
        out.append({"prompt": p, "completion": completion, "text": tok.decode(sp["story_ids"]),
                    "new_tokens": len(sp["new_ids"]), "ended_with_eos": sp["ended_with_eos"],
                    "nfe": info["nfe"], "batch_latency_s": dt, "window_ids": x[b].tolist()})
    return out
