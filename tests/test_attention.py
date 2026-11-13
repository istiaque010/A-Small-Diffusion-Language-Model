"""Phase 5-6 tests: the denoiser sees BOTH sides, shapes are right, padding is ignored, loss uses masked positions."""
import torch

import mdlm  # noqa: F401
from mdlm.masking import corrupt
from mdlm.model import MaskedDiffusionLM, MDLMConfig, masked_ce
from mdlm.tokens import MASK_ID, VOCAB


def tiny():
    torch.manual_seed(0)
    return MaskedDiffusionLM(MDLMConfig(n_layer=2, n_head=2, n_embd=32, block_size=16, dropout=0.0)).eval()


def test_logits_shape():
    m = tiny()
    assert m(torch.randint(0, VOCAB, (3, 16))).shape == (3, 16, VOCAB)


def test_uses_right_context():
    """Changing a token to the RIGHT of position 3 changes the prediction at position 3 (a causal model could not)."""
    m = tiny()
    x = torch.randint(4, 8000, (1, 16))
    x[0, 3] = MASK_ID
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 8000 + 4
    assert not torch.allclose(m(x)[0, 3], m(y)[0, 3])


def test_padding_mask_blocks_attention_to_pad():
    m = tiny()
    x = torch.randint(4, 8000, (1, 16))
    pad = torch.zeros(1, 16, dtype=torch.bool)
    pad[0, 12:] = True
    y = x.clone()
    y[0, 12:] = 5  # change only padded positions
    assert torch.allclose(m(x, key_padding=pad)[0, :12], m(y, key_padding=pad)[0, :12], atol=1e-6)


def test_loss_only_counts_masked_positions_and_gradients_flow():
    m = tiny().train()
    x0 = torch.randint(4, 8000, (4, 16))
    x0[:, 0] = 2
    xt, masked = corrupt(x0, torch.full((4,), 0.5), torch.Generator().manual_seed(0))
    logits = m(xt)
    loss = masked_ce(logits, x0, masked)
    manual = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB)[masked.reshape(-1)], x0.reshape(-1)[masked.reshape(-1)])
    assert torch.isfinite(loss) and torch.allclose(loss, manual)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
