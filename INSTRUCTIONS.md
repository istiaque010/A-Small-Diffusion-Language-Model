# Project 3 — Build a Small Diffusion Language Model: phase-by-phase instructions

Source plan: `docs/Project_3_Small_Diffusion_Language_Model_Guide.docx` (kept locally, not in the repository).

**Goal:** train a small masked (absorbing-state) diffusion language model from scratch on TinyStories, generate
stories by iterative parallel unmasking, and measure quality and speed against the autoregressive models of
Projects 1 and 2.

**Main question:** can a small bidirectional Transformer learn to generate coherent stories by progressively
recovering masked tokens, and how do the number of refinement steps and the unmasking schedule affect quality and
speed?

```
TinyStories (Project 1) → BPE tokenizer + [MASK] → story-aligned windows → mask with probability t
   → bidirectional Transformer → CE on masked tokens → iterative parallel unmasking → evaluation vs AR
```

This is a discrete masked-diffusion prototype. It does not reproduce any published recipe (e.g. LLaDA or MDLM) in
full.

## How to use this file
1. Work through the phases in order. Each has **Learn**, **Files**, **Run**, **Check** and **Result**.
2. Run everything from the `project3` folder with `.venv` active (Windows: `.venv\Scripts\activate`).
3. The status in the table below changes from ⬜ to ✅ when the phase is done and checked.

## Progress

| # | Phase | Status |
|---|---|---|
| 1 | Environment setup | ✅ Done (2026-11-11): same versions as Projects 1–2, CPU only |
| 2 | Reuse and inspect TinyStories | ✅ Done (2026-11-11): Project 1 splits; story-aligned windows |
| 3 | Tokenizer and special tokens | ✅ Done (2026-11-11): Project 1 BPE + [MASK] = 8192 |
| 4 | Forward masking process | ✅ Done (2026-11-11): 6 tests pass |
| 5 | Bidirectional Transformer denoiser | ✅ Done (2026-11-11): 4 tests pass, mutation-checked |
| 6 | Masked-token training loss | ✅ Done (2026-11-11): smoke run, loss falls |
| 7 | Training loop | ✅ Done: small 1.32 h (val CE 3.86), base 9.54 h (val CE 3.05) |
| 8 | Verify denoising (reconstruction) | ✅ Done: base 69% at 20% masked, 9% at 100% |
| 9 | Iterative non-autoregressive generation | ✅ Code + 19 tests; results with Phase 10 |
| 10 | Generation schedules | ✅ Done: cosine ≈ linear; random order needed |
| 11 | Sampling experiments | ✅ Done: top-k 20 / temp 0.7 best balance |
| 12 | Compare with the autoregressive model | ✅ Done: AR better; diffusion faster only at few steps |
| 13 | Error analysis | ✅ Done: local incoherence 97% vs 7% (AI labels) |
| 14 | Research experiments and ablations | ✅ Done (2 retrain ablations not run) |
| 15 | Repository structure | ✅ see Reference |
| 16 | Final report | ✅ Done: README.md |

## Decisions for this PC and this project (senior-researcher choices, with reasons)

- **CPU, not GPU.** The guide assumes Colab/CUDA. Everything here runs on the CPU (Intel Core Ultra 7, 24 threads),
  fp32, as in Projects 1–2. "GPU memory" is reported as peak RAM.
- **Two model sizes, each matched to an existing autoregressive (AR) model**, so the comparison changes only the
  modelling approach, not size or data:
  - `mdlm_small`: 4 layers, 4 heads, width 256 (5.32M parameters), 3,000 steps × 64 × 256 tokens = 49M tokens —
    same shape and budget as Project 2's *student from scratch* (AR, test perplexity 9.63).
  - `mdlm_base`: 6 layers, 6 heads, width 384 (13.89M parameters), 10,000 steps = 164M tokens, same optimizer
    settings — same shape and budget as the Project 1 *teacher* (AR, test perplexity 5.74).
- **Story-aligned windows.** Training examples are 256-token windows that start at a story's `<BOS>` (followed by
  the story, its `<EOS>`, and the start of the next story if there is room). Generation starts from `<BOS>` and the
  story ends at the first generated `<EOS>`. No padding is needed.
- **`<EOS>` is maskable.** The guide suggests never masking BOS/EOS/PAD. We keep the leading `<BOS>` fixed but let
  `<EOS>` (and later `<BOS>`) be masked: a model that never predicts `<EOS>` cannot decide where a story ends.
- **Loss** is the guide's: mean cross-entropy over masked positions, t ~ U(0.001, 1), m(t) = t. No masking-level
  (time) input in the main runs; the option exists (`time_cond`) but its ablation is not run (cost: a full retrain).
- **A common quality measure for AR and diffusion stories:** the perplexity that the Project 1 teacher assigns to
  the generated continuation (*teacher perplexity*). Low values can also come from repetitive text, so it is always
  reported together with Distinct-2 and the repetition flag.
- **Settings are chosen on development prompts** (beginnings of validation stories). The 8 test prompts of Project 1
  are used only for the final numbers. Selection rule, fixed in advance: lowest teacher perplexity among settings
  with Distinct-2 ≥ 0.70 and ≤ 15% repetitive stories.
- **Change to the selection rule (made 2026-11-11, after the small model's grid, before any base-model result).**
  With most-confident-first ordering, *no* setting of `mdlm_small` passed the rule: the model keeps filling slots
  with safe, frequent phrases ("the park", "with her toys"), so Distinct-2 stayed ≤ 0.67 and longer runs became
  repetitive. Two changes: (1) the grid now also searches **random ordering** (which positions to reveal), which the
  guide lists as an option; (2) if still nothing passes, a **fallback** keeps only the repetition limit (≤ 15%) and
  takes the lowest teacher perplexity. Which rule was used is saved in `grid_dev.json` (`selected.rule`).
  The error-analysis stories use the selected step count and ordering (before: always 32 steps, confidence order).
- **Code package `mdlm/`** (not `src/`), so it never collides with Project 1's `src` package, which is reused
  read-only (tokenizer wrapper, data loader, sampling and evaluation helpers, AR models).
- **Another program was using the CPU** during this project's training (a separate experiment started at 11:24 on
  2026-11-11). Training wall-clock times are therefore not comparable with Projects 1–2. Speed benchmarks are run
  only when the CPU-load check passes.

---

## Phase 1 — Environment setup

**Learn:** a separate `.venv` with exactly Project 1/2's package versions, so the tokenizer and numbers match.

**Files:** `requirements.txt`, `requirements.lock.txt`

**Run:**
```
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Check:** `python -c "import torch; print(torch.__version__)"` prints `2.14.1+cpu`.

**Result:** ✅ Python 3.12.10, PyTorch 2.14.1+cpu, tokenizers 0.23.3, NumPy 2.5.4. No GPU (CPU, 24 threads).
Seeds: 1337 for training (Python, NumPy, PyTorch via `set_seed`), fixed seeds for every evaluation protocol.

---

## Phase 2 — Reuse and inspect TinyStories

**Learn:** reuse Project 1's cleaned, leakage-checked splits (no story appears in two splits — verified for the
Project 2 paper). Train → optimization; validation → monitoring and setting selection; test → final numbers only.

**Files:** `mdlm/data.py` (reads `../project1/data/full/*.bin`)

**Result:** ✅

| Split | Stories | Tokens | Mean / median / 95th pct. story length | Stories longer than 256 tokens |
|---|---|---|---|---|
| train | 1,772,187 | 372.6M | 210 / 182 / 428 | 16.0% |
| val | 10,995 | 2.27M | 206 / 181 / 407 | 14.8% |
| test | 10,995 | 2.26M | 206 / 180 / 409 | 14.6% |

About 15% of stories do not fit in a 256-token window; in those windows there is no `<EOS>`, so the model also
learns that a window can end mid-story. Generated stories are compared with the AR models under the same
256-token limit (Phase 12).

---

## Phase 3 — Tokenizer and special tokens

**Learn:** keep Project 1's 8,192-token BPE so every existing id keeps its meaning; add one special token.
```
Original:  <BOS> The boy found a dog . <EOS>
Corrupt:   <BOS> The [MASK] found [MASK] dog . <EOS>
```

**Files:** `mdlm/tokens.py` → writes `tokenizer/tokenizer.json`

**Result:** ✅ `[MASK]` = id 8192; ids 0–8191 unchanged (PAD 0, UNK 1, BOS 2, EOS 3). Round trip:
`"The boy found a dog."` → `[2, 1607, 527, 595, 262, 645, 17, 3]` → `"The boy found a dog."`. The diffusion
model's vocabulary is 8,193 (one extra embedding row). Our windows contain no PAD, so PAD never enters the loss.

---

## Phase 4 — Forward masking process

**Learn:** for each window, sample t ∈ (0, 1], then replace every eligible token with `[MASK]` with probability
m(t) = t. Eligible = every position except position 0 (`<BOS>`) and PAD.

**Files:** `mdlm/masking.py`, `tests/test_masking.py`

**Run:** `python -m pytest tests\test_masking.py -v`

**Result:** ✅ 6 tests pass: BOS at position 0 is never masked; t = 1 masks every eligible position including
`<EOS>`; the masked fraction matches t (within 1% for t = 0.2, 0.5, 0.8 over 2,000 windows); at least one position
is masked per window; only masked positions change; PAD is never masked; same seed → same masks.

---

## Phase 5 — Bidirectional Transformer denoiser

**Learn:** the same block design as Project 1's GPT (pre-LayerNorm, learned positions, GELU MLP, tied embeddings),
but **no causal mask**: every position attends to the whole window. A padding mask is supported.

**Files:** `mdlm/model.py`, `tests/test_attention.py`

**Run:** `python -m pytest tests\test_attention.py -v`

**Result:** ✅ 4 tests pass: logits have shape [batch, length, 8193]; changing a token to the *right* of a masked
position changes its prediction (uses right context); padded positions are never attended to; the loss uses only
masked positions and gradients reach every parameter. **Mutation check:** forcing a causal mask makes the
right-context test fail, as it should. Parameters: 5,322,496 (small) and 13,891,968 (base) — the AR models plus one
embedding row (+256 / +384).

---

## Phase 6 — Masked-token training loss

**Learn:** L_mask = −mean over masked positions of log p(x₀ᵢ | xₜ). Unmasked tokens are context, not targets.
For monitoring we also compute the diffusion bound on the negative log-likelihood,
`NELBO = E_t[(1/t) · Σ masked CE] / (number of eligible tokens)`; `exp(NELBO)` is an upper bound on perplexity.

**Files:** `mdlm/model.py` (`masked_ce`), `mdlm/train.py`

**Run:** `python -m mdlm.train --config configs\smoke.yaml` (5 s)

**Result:** ✅ 60 steps of a 1.5M-parameter model: validation masked-token loss 6.23 → 6.01, finite loss, gradients
in every parameter.

---

## Phase 7 — Training loop

**Learn:** AdamW, warm-up + cosine decay, gradient clipping 1.0, checkpoints (best, last, every 1,000 steps),
`--resume`. Validation uses a **fixed** protocol (same 800 windows, same t, same masks each time), so curves are
comparable. Logged: train loss, val loss, val accuracy at masked positions, NELBO, perplexity bound, lr, tokens/s,
RAM, time.

**Files:** `configs/small.yaml`, `configs/base.yaml`, `mdlm/train.py`, `scripts/run_training.ps1`

**Run:**
```
powershell -ExecutionPolicy Bypass -File scripts\run_training.ps1      # small (~1.4 h), then base (~10 h)
```

**Check:** validation loss decreases; `checkpoints/mdlm_small/model_best.pt` and `checkpoints/mdlm_base/model_best.pt`.

**Result:**

*mdlm_small* (4L/4H/256, 5,322,496 params, 3,000 steps = 49.15M tokens) finished in **1.32 h** (4,737 s,
~10,400–11,000 tok/s, peak 1.9 GB RAM). Validation went down at every check:

| Step | Val masked CE | Val accuracy | Perplexity bound |
|---|---|---|---|
| 250 | 5.703 | 9.2% | 304.6 |
| 1,000 | 4.951 | 16.9% | 114.8 |
| 2,000 | 4.004 | 29.3% | 33.8 |
| 3,000 | **3.862** | **31.0%** | **28.6** |

There is a plateau for the first ~750 steps (the model first learns word frequencies, ~10% accuracy), then a
sharp drop between steps 750 and 1,500, when it starts to use the context. The curve is still falling at the
end, so the small model is under-trained at this budget (same as Project 2's AR student). Its perplexity
*bound* (28.6) is far above the AR student's *exact* perplexity at the same size and data (9.63). Phase 12
compares the two properly on the same test windows. Plot: `results/figures/curves.png`; table:
`results/metrics/curves_table.md`.

*mdlm_base* (6L/6H/384, 13,891,968 params, 10,000 steps = 163.84M tokens) finished in **9.54 h** (34,353 s,
4,000–4,800 tok/s, peak 2.2 GB RAM). It was slowed by two other jobs on the same CPU (the small model's evaluation
on 4 threads for ~1.4 h, and the user's separate experiment), so its time is not comparable with Projects 1–2.

| Step | Val masked CE | Val accuracy | Perplexity bound |
|---|---|---|---|
| 500 | 5.560 | 9.4% | 257.3 |
| 1,000 | 4.603 | 22.0% | 72.5 |
| 2,000 | 3.768 | 32.2% | 25.3 |
| 5,000 | 3.287 | 38.5% | 14.4 |
| 10,000 | **3.053** | **41.8%** | **11.15** |

Same shape as the small model (plateau, sharp drop, then slow gains), but the drop starts earlier (step ~500–1,500)
and the final bound is 2.6× lower than the small model's. It still improves at the end (−0.004 CE over the last
500 steps), so more training would help a little. Its bound (11.15) is about 2× the Project 1 teacher's exact
validation perplexity at the same size and data (5.90; exact AR values, not bounds).

---

## Phase 8 — Verify denoising before free generation

**Learn:** mask 20%, 50%, 80% and 100% of real test windows and predict them in one pass. High accuracy at low
masking does not guarantee good generation from an all-mask start.

**Run:** `python -m mdlm.evaluate --stage recon --run mdlm_base` (and `mdlm_small`)

**Result:** (1,000 test windows; CE and accuracy over the masked positions)

| Masked | mdlm_small CE | mdlm_small accuracy | mdlm_base CE | mdlm_base accuracy |
|---|---|---|---|---|
| 20% | 2.457 | 51.5% | 1.326 | 69.3% |
| 50% | 3.141 | 40.4% | 2.037 | 56.3% |
| 80% | 4.301 | 24.0% | 3.562 | 32.0% |
| 100% | 5.812 | 9.1% | 5.797 | 9.1% |

With most words visible the small model restores half of them; with nothing visible it can only guess frequent
words (9%). The base model is much better whenever some context is visible (69% at 20% masked) but exactly as
helpless with nothing visible (9.1% — the most frequent words). This is the gap generation has to cross: it starts
at 100% masked. Both models pass the check: they denoise real text well, so generation problems are not caused by
a broken denoiser.

---

## Phase 9 — Iterative non-autoregressive generation

**Learn:** start from `<BOS>` + optional prompt + `[MASK]`×(rest of 256). Each step predicts every masked position
in parallel; some are revealed (fixed for good), the rest stay masked.

**Files:** `mdlm/generate.py`, `tests/test_generation.py`

**Result:** ✅ code: 19 tests pass (no masks left for every schedule and step budget 1–64; at most S forward passes;
`<BOS>` and the prompt are kept; reveals follow the schedule exactly, e.g. 31 masks in 4 linear steps → 8, 8, 8, 7;
same seed → same output; the story is cut at the first `<EOS>`). Generation results: Phases 10–12.

---

## Phase 10 — Generation schedules

**Learn:** how many positions to reveal per step: **linear**, **cosine** (few early, many late), **confidence**
(linear quota, plus every position whose confidence ≥ 0.9, so it can finish early). Which positions: the most
confident first. Steps: 4, 8, 16, 32.

**Run:** `python -m mdlm.evaluate --stage grid --run mdlm_base` (all schedules × sampling rules × steps, dev prompts)

**Result (mdlm_small, 40 dev stories per setting = 8 prompts × 5 seeds):**

*Which positions to reveal matters most.* With **most-confident-first** ordering every non-greedy setting at 16–32
steps had teacher perplexity 27–65 but **72–95% repetitive stories** and Distinct-2 0.33–0.42: the model fills the
easy slots first with safe phrases ("went to the park", "with her toys") and later slots copy them. With **random**
ordering repetition disappears (0–3%), Distinct-2 rises to 0.65–0.88, but the text is much less fluent (teacher
perplexity 85–1,200). No setting passed the rule with confidence order; with random order 18 did.

*More steps help.* At the selected sampling rule (top-k 50, random order, linear): teacher perplexity
444 → 228 → 155 → 135 at 4 → 8 → 16 → 32 steps.

*Schedules barely differ.* Cosine vs linear changes perplexity by < 10% in most cells. The **confidence schedule is
identical to linear** for this model: almost no position ever reaches probability 0.9, so the adaptive extra
reveals never happen and it never finishes early.

**Selected on dev (main rule):** linear schedule, top-k 50, random order, 32 steps (perplexity 134.9,
Distinct-2 0.717, 0% repetitive). Full table: `results/metrics/mdlm_small/grid_dev.json`.

*Without a prompt* (same setting): perplexity 385 / 202 / 139 / 120 at 4 / 8 / 16 / 32 steps, Distinct-2 0.67–0.72.

**Result (mdlm_base):** the same picture, with better numbers.

| Schedule (random order, top-k 20) | 4 steps | 8 steps | 16 steps | 32 steps |
|---|---|---|---|---|
| linear | 132.4 | 45.3 | 28.2 | 21.2 |
| cosine | 115.7 | 40.0 | 25.6 | 19.5 |
| confidence (τ 0.9) | 132.4 | 45.2 | 27.5 | 23.2 (31.6 passes on average) |

Cosine is slightly better than linear at every step count (fewer tokens revealed early, when the model knows
least). The confidence schedule now does trigger sometimes (it finishes early in some stories) but saves < 2% of
the passes and is not better. Confidence-first ordering again gives 28–90% repetitive stories.

**Selected on dev (main rule, 63 of 168 settings passed):** confidence schedule, top-k 20, **random order**, 32 steps
(teacher perplexity 23.2, Distinct-2 0.706, 0% repetitive). The cosine setting at 32 steps had lower perplexity
(19.5) but Distinct-2 0.69, just below the bar.

*Without a prompt:* perplexity 123 / 42 / 28 / 26 at 4 / 8 / 16 / 32 steps, Distinct-2 0.66–0.71 — close to the
prompted numbers, so the model does not need a prompt to write a story.

---

## Phase 11 — Sampling experiments

**Learn:** what token to place at a revealed position: argmax, temperature 0.7 / 1.0, top-k 20 / 50,
top-p 0.90 / 0.95. Separate from the schedule (which positions).

**Result (mdlm_small, linear, 32 steps):**

| Sampling | Confidence order: ppl / D2 / repetitive | Random order: ppl / D2 / repetitive |
|---|---|---|
| argmax | 95.5 / 0.20 / 100% | 21.4 / 0.31 / 93% |
| temperature 0.7 | 32.6 / 0.34 / 95% | 84.7 / 0.66 / 0% |
| temperature 1.0 | 27.1 / 0.37 / 88% | 454 / 0.85 / 0% |
| top-k 20 | 31.7 / 0.34 / 95% | 100.4 / 0.66 / 0% |
| top-k 50 | 28.7 / 0.36 / 80% | **134.9 / 0.72 / 0%** (selected) |
| top-p 0.90 | 28.0 / 0.38 / 90% | 210 / 0.79 / 0% |
| top-p 0.95 | 29.0 / 0.37 / 90% | 276 / 0.82 / 0% |

The usual trade-off: sharper sampling (argmax, low temperature, small k) gives lower perplexity but more repetition;
wider sampling gives more variety and less fluency. Argmax always collapses into loops. Teacher perplexity alone
would pick the repetitive settings — that is why Distinct-2 and the repetition flag are part of the rule.

**mdlm_base (linear, 32 steps):**

| Sampling | Confidence order: ppl / D2 / repetitive | Random order: ppl / D2 / repetitive |
|---|---|---|
| argmax | 75.8 / 0.40 / 88% | 7.0 / 0.41 / 85% |
| temperature 0.7 | 19.5 / 0.50 / 63% | 17.2 / 0.67 / 0% |
| temperature 1.0 | 12.8 / 0.50 / 63% | 74.1 / 0.83 / 0% |
| top-k 20 | 17.7 / 0.49 / 60% | 21.2 / 0.70 / 0% |
| top-k 50 | 15.2 / 0.51 / 65% | 26.3 / 0.74 / 0% |
| top-p 0.90 | 13.9 / 0.51 / 60% | 32.7 / 0.78 / 0% |
| top-p 0.95 | 12.9 / 0.52 / 63% | 44.9 / 0.80 / 0% |

Same pattern as the small model. With random order, temperature 0.7 and top-k 20 are the best fluency/variety
balance; argmax still loops.

---

## Phase 12 — Compare with the autoregressive model

**Learn:** diffusion masked-token loss and AR perplexity are different quantities. We compare (1) the diffusion
*bound* with the AR *exact* NLL on the same test windows and the same 255 predicted tokens, and (2) generated
stories with identical measures, prompts, seeds and the same 256-token window.

**Run:**
```
python -m mdlm.evaluate --stage nelbo --run mdlm_base
python -m mdlm.evaluate --stage final --run mdlm_base
python scripts\benchmark_generation.py          # idle PC
```

**Result (mdlm_small vs. the AR model of the same size and data, Project 2's student from scratch):**

*(1) Likelihood on the same 1,000 test windows (255 predicted tokens each, bootstrap 95% CI):*

| Model | Perplexity | |
|---|---|---|
| mdlm_small | ≤ **29.5** [28.7, 30.2] | upper bound, exp(NELBO), 16-point t grid |
| AR small (Project 2 scratch) | **8.76** [8.55, 8.99] | exact |
| AR teacher (Project 1) | 5.24 [5.12, 5.37] | exact |
| mdlm_base | ≤ **11.69** [11.41, 11.98] | upper bound |

The bound is not the exact likelihood, but the gaps (3.4× for small, 2.2× for base vs. the AR model of the same size
and data) are larger than the looseness one would expect from the bound. mdlm_base (13.9M) is still worse than the
AR model with 2.6× fewer parameters (AR small, 8.76).

*(2) Generated stories, 8 test prompts × 5 seeds, same 256-token window, scored by the teacher:*

| Model | Teacher ppl ↓ | Distinct-2 | Repetitive | Ended with EOS | Forward passes |
|---|---|---|---|---|---|
| mdlm_small, 4 steps | 498.6 | 0.745 | 0% | 40% | 4 |
| mdlm_small, 8 steps | 233.2 | 0.733 | 0% | 40% | 8 |
| mdlm_small, 16 steps | 184.9 | 0.732 | 0% | 50% | 16 |
| mdlm_small, 32 steps | 154.7 | 0.745 | 0% | 38% | 32 |
| AR small | **10.6** | 0.762 | 5% | 65% | 183 |
| AR teacher | 3.7* | 0.730 | 7.5% | 68% | 189 |

\* the teacher scores its own text, so its number is biased low; it is shown only as a reference.

At the same size and data, the small diffusion model writes stories that are about **15× less fluent** (by teacher
perplexity) than the AR model, with similar variety, while using 6–46× fewer forward passes. Example (32 steps):
*"Timmy liked to play with toy swings with toys and play. One day, he went over to the slide, even the car, but
Timmy and fell on his toys, but was sad."* — local phrases are right, sentences do not hold together.

**mdlm_base vs. the AR teacher (same size, same data), same test protocol:**

| Model | Teacher ppl ↓ | Distinct-2 | Repetitive | Ended with EOS | Forward passes |
|---|---|---|---|---|---|
| mdlm_base, 4 steps | 123.2 | 0.737 | 0% | 48% | 4 |
| mdlm_base, 8 steps | 45.5 | 0.716 | 2.5% | 48% | 8 |
| mdlm_base, 16 steps | 27.8 | 0.709 | 5% | 45% | 16 |
| mdlm_base, 32 steps | **22.5** | 0.702 | 2.5% | 50% | 32 |
| AR small (5.3M) | 10.6 | 0.762 | 5% | 65% | 183 |
| AR teacher (13.9M) | 3.7* | 0.730 | 7.5% | 68% | 189 |

\* biased low (the teacher scores its own text). Even against AR small, scored fairly by a different model, the
best diffusion setting is about 2× less fluent.

**Speed** (`scripts/benchmark_generation.py`, idle CPU — load check passed for every row; prompt "Once upon a
time", one 256-token window, batch 1, mean of 5 runs after a warm-up; neither implementation uses a key-value cache):

| Model | Setting | Latency (s) | Speed-up vs AR teacher | RAM added |
|---|---|---|---|---|
| mdlm_base | 4 steps | 0.121 | 12.9× | 127 MB |
| mdlm_base | 8 steps | 0.228 | 6.8× | 127 MB |
| mdlm_base | 16 steps | 0.432 | 3.6× | 127 MB |
| mdlm_base | 32 steps | 0.843 | 1.85× | 127 MB |
| mdlm_small | 4 / 32 steps | 0.096 / 0.723 | 16.2× / 2.15× | 58 MB |
| AR small | full window | 0.850 | 1.83× | 100 MB |
| AR teacher | full window | 1.557 | 1.00× | 132 MB |

**The trade-off.** Diffusion is faster only when it uses few steps, and few steps is exactly where its stories are
worst (4 steps: 13× faster, but teacher perplexity 123). At 32 steps it is about as fast as AR small (0.84 s vs
0.85 s) and AR small writes stories 2× more fluent with 2.6× fewer parameters. Caveats: (1) an AR implementation
with a key-value cache would be several times faster, which would remove the diffusion speed advantage at 16–32
steps; (2) the diffusion model always computes the full 256-token window, while AR could stop at `<EOS>` (here it
was forced to fill the window, for a fair comparison of equal output length).
Files: `results/metrics/mdlm_base/{nelbo,final_test}.json`, `results/metrics/generation_benchmark.{json,md}`.

---

## Phase 13 — Error analysis

**Learn:** 60 stories from the diffusion model (selected setting, 32 steps) and 60 from the AR teacher (top-p 0.95,
same window), mixed and labelled blind with Project 1's rules plus one new category, *local incoherence* (a token
that does not fit its neighbours, e.g. doubled or broken words — a possible "masking artifact").

**Run:** `python -m mdlm.evaluate --stage errors --run mdlm_base`, then label, then `python scripts\error_counts.py`

**Result:** 60 diffusion stories (mdlm_base, selected setting: confidence schedule, top-k 20, random order,
32 steps, seed 1344) and 60 AR-teacher stories (top-p 0.95), same 8 test prompts, shuffled (seed 2026) into
`scripts/error_labels/blind_stories.txt`. **The labels were produced by an AI assistant (Claude), not by a human
annotator, and have not been checked by a person.** Blinding is imperfect: broken diffusion text is often easy to
recognise. Labels with one-line reasons: `scripts/error_labels/labels.py`.

| Error (60 stories each) | Diffusion (mdlm_base) | AR teacher | Fisher p |
|---|---|---|---|
| contradiction | 60 (100%) | 59 (98%) | 1 |
| grammar | 60 (100%) | 16 (27%) | 2e-19 |
| entity | 55 (92%) | 37 (62%) | 2e-4 |
| local incoherence | **58 (97%)** | 4 (7%) | 2e-26 |
| repetition (automatic) | 2 (3%) | 4 (7%) | 0.68 |
| incomplete (automatic) | 25 (42%) | 21 (35%) | 0.57 |
| no error at all | 0 | 0 | |

- **Local incoherence is the diffusion model's signature error** (97% vs 7%): doubled words ("the the", "ball ball",
  "carrot carrot"), stray punctuation ("Suddenly,,", "proud..") and words in impossible slots ("go home to When
  they arrived"). These come from positions revealed in the same step that cannot see each other's choice.
- **Grammar** follows from the same cause (100% vs 27%).
- **Entities drift more** (92% vs 62%): names, pronouns and species change mid-story ("Ben and Ben", "the dragon
  caught caught the dragon", a boy who "meows").
- **Contradiction is at the ceiling for both** with this strict rule (any clash with an earlier fact), so it does
  not separate the models; it did not in Project 2 either.
- **Repetition is not a problem for the selected setting** (random order); with confidence order it would have
  been (Phases 10–11).

Files: `results/metrics/error_counts.{json,md}`, `results/generations/mdlm_base/error_analysis_stories.json`.

---

## Phase 14 — Research experiments and ablations

| Experiment | Status |
|---|---|
| Number of refinement steps (4–32) | in Phases 10–12 |
| Confidence vs random reveal | in the grid stage |
| Prompted vs unprompted generation | in the grid stage |
| Model depth/width (small vs base) | two trained sizes |
| Mask schedule m(t) during training | not run (needs a retrain, ~1.4 h small) |
| Time-conditioning | not run (needs a retrain) |

**Result:**

| Question | Answer (dev prompts unless noted) |
|---|---|
| More refinement steps? | Always better: base 132 → 45 → 28 → 21 teacher ppl at 4 → 8 → 16 → 32 steps (random, linear, top-k 20). Gains shrink after 16. |
| Which positions to reveal? | Confidence-first is more fluent per story but repetitive (28–90% flagged); random order removes repetition. The biggest single effect in the grid. |
| Schedule? | Cosine ≈ 10% better than linear; the adaptive confidence schedule is no better and rarely saves passes. |
| Prompt needed? | No: unprompted stories are as good as prompted ones (base, 32 steps: 26.3 vs 23.2). |
| Bigger model? | Large gain: small → base, test teacher ppl 155 → 22.5 (32 steps), likelihood bound 29.5 → 11.7, fill-in accuracy at 20% masked 52% → 69%. |
| Under-training? | Both curves still fall at the end (Phase 7), so part of the gap to AR is the fixed budget. |

Not run, with cost: the masking schedule during training and time conditioning (one retrain each, ~1.4 h for the
small model); a key-value-cache AR baseline for the speed comparison (code change, minutes to run); a human check of
the error labels.

---

## Phase 15 — Repository structure

See Reference below.

---

## Phase 16 — Final report

**Result:** ✅ `README.md` answers the main question with the numbers above.

---

## Reference

```
configs/      smoke.yaml, small.yaml, base.yaml
mdlm/         __init__.py (paths), tokens.py, data.py, masking.py, model.py, train.py, generate.py, evaluate.py
scripts/      run_training.ps1, benchmark_generation.py, plot_curves.py, error_counts.py, try_model.py
tests/        test_masking.py, test_attention.py, test_generation.py
tokenizer/    tokenizer.json (Project 1 BPE + [MASK])
checkpoints/  mdlm_small/, mdlm_base/                       (not in git)
data/         cached story-start indices                    (not in git)
results/      metrics/, generations/, figures/
docs/         Project_3_Small_Diffusion_Language_Model_Guide.docx   (not in git)
```

**Rules**
- Never change anything in `..\project1\` or `..\project2\`; they are only read. The teacher is hash-checked.
- Test prompts and test windows are used only for final numbers; settings are chosen on validation data.
- One job at a time for anything that is timed.
