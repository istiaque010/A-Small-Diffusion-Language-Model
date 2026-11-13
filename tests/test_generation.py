"""Phase 9-10 tests: iterative unmasking finishes, keeps the prompt, respects the step budget, reproducible."""
import pytest
import torch

import mdlm  # noqa: F401
from mdlm.generate import SCHEDULES, generate, remaining_fraction, split_story
from mdlm.model import MaskedDiffusionLM, MDLMConfig
from mdlm.tokens import BOS_ID, EOS_ID, MASK_ID


def tiny():
    torch.manual_seed(0)
    return MaskedDiffusionLM(MDLMConfig(n_layer=1, n_head=2, n_embd=32, block_size=32, dropout=0.0)).eval()


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("steps", [1, 4, 8, 32, 64])
def test_no_masks_left_and_step_budget_respected(schedule, steps):
    x, info = generate(tiny(), [[10, 11], []], length=32, steps=steps, schedule=schedule,
                       generator=torch.Generator().manual_seed(0))
    assert not (x == MASK_ID).any()
    assert info["nfe"] <= steps
    assert (x[:, 0] == BOS_ID).all() and x[0, 1].item() == 10 and x[0, 2].item() == 11


def test_reveals_follow_the_schedule():
    _, info = generate(tiny(), [[]], length=32, steps=4, schedule="linear", method="greedy",
                       generator=torch.Generator().manual_seed(0))
    # 31 masked; positions left after each step: floor(31*0.75)=23, floor(31*0.5)=15, floor(31*0.25)=7, 0
    assert [r[0] for r in info["revealed_per_step"]] == [8, 8, 8, 7]


def test_remaining_fraction_endpoints():
    for s in ("linear", "cosine"):
        assert remaining_fraction(s, 0, 8) == pytest.approx(1.0) and remaining_fraction(s, 8, 8) == pytest.approx(0.0)


def test_reproducible_and_random_selection_works():
    a = generate(tiny(), [[]], length=32, steps=8, select="random", generator=torch.Generator().manual_seed(3))[0]
    b = generate(tiny(), [[]], length=32, steps=8, select="random", generator=torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b)


def test_split_story_cuts_at_first_eos():
    out = split_story([BOS_ID, 10, 11, 12, EOS_ID, BOS_ID, 13], prompt_len=1)
    assert out["story_ids"] == [10, 11, 12] and out["new_ids"] == [11, 12] and out["ended_with_eos"]
    assert not split_story([BOS_ID, 10, 11], 0)["ended_with_eos"]
