"""Train the masked diffusion LM (Phases 6-7).

One step: batch of story-aligned windows x0 -> sample t ~ U(0,1) per window -> mask each eligible token with
probability m(t) -> bidirectional Transformer -> cross-entropy on the masked positions only -> AdamW.

Validation every eval_interval steps on a FIXED protocol (same windows, same t and same masks every time):
  val_loss   mean masked-token CE (the training objective)
  val_acc    accuracy of the argmax prediction at masked positions
  val_nelbo  Monte-Carlo estimate of the diffusion bound on the negative log-likelihood, in nats per token:
             (1/t) * sum of masked CE / number of eligible tokens (linear schedule, continuous time).
             exp(val_nelbo) is an UPPER BOUND on perplexity; it is not the same quantity as an autoregressive
             model's perplexity (see INSTRUCTIONS.md, Phase 12).

Usage (from project3, .venv active):
    python -m mdlm.train --config configs/smoke.yaml
    python -m mdlm.train --config configs/small.yaml
    python -m mdlm.train --config configs/base.yaml [--resume]
Writes checkpoints/<run>/ and results/metrics/<run>/{run_info.json, log.csv, train_summary.json}.
"""
import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

import mdlm
from mdlm.data import StoryWindows, fixed_subset
from mdlm.masking import corrupt, eligible_positions, sample_t
from mdlm.model import MaskedDiffusionLM, MDLMConfig, count_parameters, masked_ce, save_checkpoint
from mdlm.tokens import load_tokenizer
from src.train import lr_at, make_optimizer
from src.utils import env_info, load_config, peak_memory_mb, set_seed

LOG_COLS = ["step", "train_loss", "val_loss", "val_acc", "val_nelbo", "val_ppl_bound", "lr", "tokens_per_sec",
            "mem_mb", "elapsed_s"]


class ValProtocol:
    """Fixed validation windows, masking levels and masks (identical at every evaluation)."""

    def __init__(self, n_windows: int, block: int, seed: int = 1234):
        self.x0 = fixed_subset("val", n_windows, block, seed=seed)
        g = torch.Generator().manual_seed(seed)
        self.t = sample_t(len(self.x0), g)
        self.xt, self.masked = corrupt(self.x0, self.t, g)

    @torch.no_grad()
    def evaluate(self, model, batch: int = 32) -> dict:
        model.eval()
        ce_sum = n_masked = correct = 0.0
        nelbo = []
        for i in range(0, len(self.x0), batch):
            x0, xt, m, t = (z[i:i + batch] for z in (self.x0, self.xt, self.masked, self.t))
            logits = model(xt, t)
            ce = torch.nn.functional.cross_entropy(logits.transpose(1, 2), x0, reduction="none")  # (B, T)
            ce = ce * m
            ce_sum += ce.sum().item()
            n_masked += m.sum().item()
            correct += ((logits.argmax(-1) == x0) & m).sum().item()
            n_elig = eligible_positions(x0).sum(1)
            nelbo.append((ce.sum(1) / t / n_elig).numpy())
        model.train()
        nel = float(np.concatenate(nelbo).mean())
        return {"val_loss": ce_sum / n_masked, "val_acc": correct / n_masked, "val_nelbo": nel,
                "val_ppl_bound": math.exp(nel)}


def batches(ds: StoryWindows, batch: int, seed: int, skip: int = 0):
    """Endless stream of batches in a fixed shuffled order (reproducible; `skip` batches for resuming)."""
    epoch = 0
    while True:
        perm = np.random.default_rng(seed + epoch).permutation(len(ds))
        for b in range(0, len(perm) - batch + 1, batch):
            if skip > 0:
                skip -= 1
                continue
            yield torch.stack([ds[int(i)] for i in perm[b:b + batch]])
        epoch += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc, mc = cfg["train"], cfg["model"]
    if args.max_steps:
        tc["max_steps"] = args.max_steps
    run = cfg["run_name"]
    set_seed(cfg["seed"])
    torch.set_num_threads(os.cpu_count() or 1)
    load_tokenizer()  # builds tokenizer/tokenizer.json with [MASK] if needed

    ck_dir = os.path.join(mdlm.PROJECT3, "checkpoints", run)
    m_dir = os.path.join(mdlm.RESULTS, "metrics", run)
    os.makedirs(ck_dir, exist_ok=True)
    os.makedirs(m_dir, exist_ok=True)

    model_cfg = MDLMConfig(**mc)
    model = MaskedDiffusionLM(model_cfg)
    opt = make_optimizer(model, tc)
    step, best_val = 0, float("inf")
    if args.resume and os.path.exists(os.path.join(ck_dir, "model_last.pt")):
        ck = torch.load(os.path.join(ck_dir, "model_last.pt"), weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, best_val = ck["step"], ck["best_val"]
        print(f"resumed at step {step}")

    params = count_parameters(model)
    train_ds = StoryWindows("train", mc["block_size"])
    val = ValProtocol(tc["eval_windows"], mc["block_size"])
    info = {"run": run, "config": cfg, "params": params, "train_windows_available": len(train_ds),
            "tokens_per_step": tc["batch_size"] * tc["grad_accum_steps"] * mc["block_size"], "env": env_info("cpu")}
    with open(os.path.join(m_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"{run}: {params['total']:,} parameters, {len(train_ds):,} story windows, "
          f"{info['tokens_per_step']:,} tokens/step, {tc['max_steps']:,} steps")

    log_path = os.path.join(m_dir, "log.csv")
    if not (args.resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLS)

    stream = batches(train_ds, tc["batch_size"], cfg["seed"], skip=step * tc["grad_accum_steps"])
    g_mask = torch.Generator().manual_seed(cfg["seed"] + 7 + step)
    model.train()
    t0 = time.time()
    tok_count, loss_acc, n_acc = 0, 0.0, 0
    elapsed0 = 0.0
    if args.resume and os.path.exists(os.path.join(m_dir, "train_summary_partial.json")):
        elapsed0 = json.load(open(os.path.join(m_dir, "train_summary_partial.json")))["elapsed_s"]
    while step < tc["max_steps"]:
        lr = lr_at(step, tc)
        for gr in opt.param_groups:
            gr["lr"] = lr
        opt.zero_grad(set_to_none=True)
        for _ in range(tc["grad_accum_steps"]):
            x0 = next(stream)
            t = sample_t(len(x0), g_mask)
            xt, masked = corrupt(x0, t, g_mask, tc.get("mask_schedule", "linear"))
            loss = masked_ce(model(xt, t), x0, masked)
            (loss / tc["grad_accum_steps"]).backward()
            loss_acc += loss.item()
            n_acc += 1
            tok_count += x0.numel()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        step += 1

        if step % tc["log_interval"] == 0 or step == tc["max_steps"]:
            el = time.time() - t0
            print(f"step {step:6d} | loss {loss_acc / n_acc:.4f} | lr {lr:.2e} | {tok_count / el:,.0f} tok/s")
        if step % tc["eval_interval"] == 0 or step == tc["max_steps"]:
            v = val.evaluate(model)
            el = time.time() - t0
            row = [step, round(loss_acc / max(n_acc, 1), 4), round(v["val_loss"], 4), round(v["val_acc"], 4),
                   round(v["val_nelbo"], 4), round(v["val_ppl_bound"], 2), f"{lr:.2e}", round(tok_count / el),
                   round(peak_memory_mb("cpu")), round(elapsed0 + el, 1)]
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            print("  " + ", ".join(f"{k}={x}" for k, x in zip(LOG_COLS, row)))
            loss_acc, n_acc = 0.0, 0
            if v["val_loss"] < best_val:
                best_val = v["val_loss"]
                save_checkpoint(os.path.join(ck_dir, "model_best.pt"), model, None, step, best_val, cfg, info)
            save_checkpoint(os.path.join(ck_dir, "model_last.pt"), model, opt, step, best_val, cfg, info)
            with open(os.path.join(m_dir, "train_summary_partial.json"), "w") as f:
                json.dump({"step": step, "elapsed_s": elapsed0 + el}, f)
        if step % tc["ckpt_interval"] == 0:
            save_checkpoint(os.path.join(ck_dir, f"model_step_{step}.pt"), model, None, step, best_val, cfg, info)

    summary = {"run": run, "steps": step, "tokens": step * info["tokens_per_step"],
               "best_val_loss": best_val, "wall_clock_s": round(elapsed0 + time.time() - t0, 1),
               "peak_mem_mb": round(peak_memory_mb("cpu"), 1), "params": params}
    with open(os.path.join(m_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
