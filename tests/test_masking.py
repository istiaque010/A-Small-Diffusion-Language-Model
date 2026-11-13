"""Phase 4 tests: the forward masking process."""
import torch

import mdlm  # noqa: F401
from mdlm.masking import corrupt, eligible_positions, sample_t
from mdlm.tokens import BOS_ID, EOS_ID, MASK_ID, PAD_ID


def windows(B=64, T=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(4, 8192, (B, T), generator=g)
    x[:, 0] = BOS_ID
    x[:, 100] = EOS_ID
    return x


def test_position_zero_bos_is_never_masked():
    x = windows()
    xt, m = corrupt(x, torch.ones(len(x)), torch.Generator().manual_seed(1))
    assert not m[:, 0].any() and (xt[:, 0] == BOS_ID).all()


def test_t_one_masks_every_eligible_position_and_eos_is_maskable():
    x = windows()
    xt, m = corrupt(x, torch.ones(len(x)), torch.Generator().manual_seed(1))
    assert m[:, 1:].all() and (xt[:, 1:] == MASK_ID).all()


def test_mask_rate_matches_t():
    x = windows(B=2000)
    for t in (0.2, 0.5, 0.8):
        _, m = corrupt(x, torch.full((len(x),), t), torch.Generator().manual_seed(2))
        rate = m[:, 1:].float().mean().item()
        assert abs(rate - t) < 0.01, (t, rate)


def test_at_least_one_masked_and_only_masked_positions_change():
    x = windows()
    xt, m = corrupt(x, torch.full((len(x),), 1e-6), torch.Generator().manual_seed(3))
    assert m.sum(1).min().item() >= 1
    assert torch.equal(xt[~m], x[~m]) and (xt[m] == MASK_ID).all()


def test_padding_is_never_masked():
    x = windows()
    x[:, 200:] = PAD_ID
    assert not eligible_positions(x)[:, 200:].any()
    _, m = corrupt(x, torch.ones(len(x)), torch.Generator().manual_seed(4))
    assert not m[:, 200:].any()


def test_reproducible_with_same_seed():
    x = windows()
    t = sample_t(len(x), torch.Generator().manual_seed(5))
    a = corrupt(x, t, torch.Generator().manual_seed(6))[0]
    b = corrupt(x, t, torch.Generator().manual_seed(6))[0]
    assert torch.equal(a, b)
