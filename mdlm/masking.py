"""Forward (corruption) process of absorbing-state / masked diffusion.

For a clean window x0, sample a masking level t in (0, 1] per sequence and replace each eligible token with
[MASK] independently with probability m(t). Default m(t) = t (linear), as in the guide.

Eligible positions: every position except position 0 (the anchoring <BOS>) and padding. Unlike the guide's
suggestion, <EOS> and later <BOS> tokens ARE maskable: they mark where a story ends, and a model that never has
to predict them cannot decide where a generated story stops. (Our windows contain no padding.)
"""
import math

import torch

from mdlm.tokens import MASK_ID, PAD_ID

T_EPS = 1e-3  # t ~ U(T_EPS, 1): avoids t = 0 (nothing masked) and the 1/t blow-up in the bound


def mask_prob(t: torch.Tensor, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        return t
    if schedule == "cosine":
        return 1 - torch.cos(0.5 * math.pi * t)
    raise ValueError(schedule)


def eligible_positions(x0: torch.Tensor) -> torch.Tensor:
    elig = x0 != PAD_ID
    elig[:, 0] = False
    return elig


def sample_t(batch: int, generator: torch.Generator | None = None) -> torch.Tensor:
    return T_EPS + (1 - T_EPS) * torch.rand(batch, generator=generator)


def corrupt(x0: torch.Tensor, t: torch.Tensor, generator: torch.Generator | None = None, schedule: str = "linear"):
    """Returns (x_t, masked) where masked is a bool tensor of the positions replaced by [MASK].

    Every row gets at least one masked eligible position (if it has any), so every example contributes to the loss.
    """
    elig = eligible_positions(x0)
    p = mask_prob(t, schedule).view(-1, 1)
    u = torch.rand(x0.shape, generator=generator)
    masked = (u < p) & elig
    empty = ~masked.any(1) & elig.any(1)
    if empty.any():
        # mask the eligible position with the smallest u (a uniformly random eligible position)
        u_e = torch.where(elig, u, torch.full_like(u, 2.0))
        j = u_e.argmin(1)
        rows = empty.nonzero().squeeze(1)
        masked[rows, j[rows]] = True
    xt = torch.where(masked, torch.full_like(x0, MASK_ID), x0)
    return xt, masked
