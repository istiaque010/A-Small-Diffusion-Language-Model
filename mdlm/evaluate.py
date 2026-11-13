"""Evaluation of the diffusion LM and the autoregressive (AR) baselines (Phases 8, 10-12).

Stages (python -m mdlm.evaluate --stage <name> ...):
  recon   Phase 8: mask 20/50/80/100% of 1,000 fixed test windows, predict in ONE pass; masked-token CE and
          accuracy per rate, plus example reconstructions.
  nelbo   Diffusion bound on the negative log-likelihood (nats/token) on 1,000 fixed test windows, with the
          masking level on a fixed grid of 16 values. The AR models' exact NLL is computed on the SAME windows and
          the SAME 255 predicted positions, so the two numbers refer to identical tokens. The diffusion number is an
          upper bound (ELBO), the AR number is exact; the comparison is therefore conservative for diffusion.
  grid    Phases 10-11 on DEV prompts (beginnings of validation stories): every schedule x steps x sampling rule,
          plus random-vs-confidence selection and unprompted generation at the selected setting.
  final   On the 8 TEST prompts of Project 1 (5 seeds each): the diffusion model at the setting selected on dev,
          at every step count, and the AR baselines with Project 1's best decoding (top-p 0.95), all limited to the
          same 256-token window.
  errors  60 stories each from the diffusion model (selected setting, 32 steps) and from the AR teacher, for the
          error annotation (Phase 13).

Quality measures for generated stories (identical code for every model):
  gen_ppl     perplexity of the generated continuation (incl. <EOS> if produced) under the Project 1 teacher
              (an independent AR scorer; lower = more typical text by the teacher's standard). Low values can
              also come from repetitive text, so it is always read together with the diversity measures.
  distinct_1/2, rep4_rate, repetition_flag_rate (Rep-4 above the 95th percentile of real stories), eos_rate,
  madeup_rate (words never seen in the first 300k training stories), avg_new_tokens.
"""
import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import mdlm
from mdlm.data import fixed_subset
from mdlm.generate import SCHEDULES, generate_stories
from mdlm.masking import eligible_positions
from mdlm.model import count_parameters, load_checkpoint
from mdlm.tokens import BOS_ID, EOS_ID, MASK_ID, load_tokenizer
from src.evaluate import auto_flags, grouped_distinct, repeated_ngram_rate, words
from src.generate import generate as ar_generate
from src.model import load_checkpoint as load_ar_checkpoint

RESULTS = os.path.join(mdlm.RESULTS, "metrics")
GEN = os.path.join(mdlm.RESULTS, "generations")
TEST_PROMPTS = [l.strip() for l in open(os.path.join(mdlm.TEACHER_DIR, "configs", "prompts.txt"), encoding="utf-8")
                if l.strip()]
REP_THRESHOLD = 0.05693333568232272  # 95th percentile of Rep-4 on real test stories (Project 1 eval.json)
REAL = {"distinct_1": 0.3308, "distinct_2": 0.7641, "rep4_rate": 0.0164, "eos_rate": 0.987, "madeup_rate": 0.0002}
SAMPLING = {"argmax": dict(method="greedy"), "temp0.7": dict(method="temperature", temperature=0.7),
            "temp1.0": dict(method="temperature", temperature=1.0), "topk20": dict(method="top_k", top_k=20),
            "topk50": dict(method="top_k", top_k=50), "topp0.90": dict(method="top_p", top_p=0.90),
            "topp0.95": dict(method="top_p", top_p=0.95)}
STEPS = (4, 8, 16, 32)
SELECTS = ("confidence", "random")
N_GROUPS = 5
GROUP_SEED = 1343  # same base seed as Project 1's top-p 0.95 stories
# Selection rule, fixed BEFORE looking at results: among all settings evaluated on DEV prompts, choose the one with
# the lowest teacher perplexity among those with Distinct-2 >= 0.70 and at most 15% repetitive stories.
MIN_D2, MAX_REP = 0.70, 0.15


# ---------------------------------------------------------------- models
def teacher_hash_ok() -> bool:
    man = dict(l.split()[::-1] for l in open(os.path.join(mdlm.TEACHER_DIR, "MANIFEST.sha256")) if l.strip())
    h = hashlib.sha256(open(os.path.join(mdlm.TEACHER_DIR, "teacher.pt"), "rb").read()).hexdigest()
    return man.get("teacher.pt", man.get("./teacher.pt")) == h


def load_teacher():
    assert teacher_hash_ok(), "teacher.pt does not match Project 1's MANIFEST.sha256"
    from load_teacher import load_teacher as p1_load_teacher  # Project 1's own loader (teacher/load_teacher.py)
    model = p1_load_teacher("cpu")[0]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_ar_small():
    model, _ = load_ar_checkpoint(os.path.join(mdlm.PROJECT2, "checkpoints", "student_scratch", "model_best.pt"))
    return model.eval()


# ---------------------------------------------------------------- recon / nelbo
@torch.no_grad()
def reconstruction(model, tok, n=1000, rates=(0.2, 0.5, 0.8, 1.0), seed=4321) -> dict:
    x0 = fixed_subset("test", n, model.cfg.block_size, seed=seed)
    out, examples = {}, {}
    for r in rates:
        g = torch.Generator().manual_seed(seed + int(r * 100))
        elig = eligible_positions(x0)
        u = torch.where(elig, torch.rand(x0.shape, generator=g), torch.full(x0.shape, 2.0))
        k = round(r * elig[0].sum().item())
        order = u.argsort(1)
        masked = torch.zeros_like(elig)
        masked.scatter_(1, order[:, :k], True)
        xt = torch.where(masked, torch.full_like(x0, MASK_ID), x0)
        ce_sum = acc = cnt = 0.0
        for i in range(0, n, 50):
            lg = model(xt[i:i + 50])
            m = masked[i:i + 50]
            ce_sum += F.cross_entropy(lg[m], x0[i:i + 50][m], reduction="sum").item()
            acc += (lg.argmax(-1)[m] == x0[i:i + 50][m]).sum().item()
            cnt += m.sum().item()
            if i == 0:
                pred = torch.where(m, lg.argmax(-1), x0[i:i + 50])
                examples[str(r)] = [{"masked": tok.decode([t if t != MASK_ID else 8192 for t in xt[j, :80].tolist()],
                                                          skip_special=False).replace("[MASK]", " ___"),
                                     "reconstructed": tok.decode(pred[j, :80].tolist()),
                                     "original": tok.decode(x0[j, :80].tolist())} for j in range(3)]
        out[str(r)] = {"masked_tokens": int(cnt), "ce": ce_sum / cnt, "accuracy": acc / cnt}
    return {"rates": out, "examples": examples, "n_windows": n}


@torch.no_grad()
def nelbo(model, n=1000, n_t=16, seed=999) -> dict:
    """Diffusion bound: mean over t-grid of (1/t) * masked CE sum / eligible tokens (nats/token)."""
    x0 = fixed_subset("test", n, model.cfg.block_size, seed=seed)
    g = torch.Generator().manual_seed(seed)
    ts = (torch.arange(n_t) + 0.5) / n_t
    per_window = torch.zeros(n)
    for t in ts:
        u = torch.rand(x0.shape, generator=g)
        elig = eligible_positions(x0)
        masked = (u < t) & elig
        xt = torch.where(masked, torch.full_like(x0, MASK_ID), x0)
        for i in range(0, n, 50):
            lg = model(xt[i:i + 50])
            ce = F.cross_entropy(lg.transpose(1, 2), x0[i:i + 50], reduction="none") * masked[i:i + 50]
            per_window[i:i + 50] += ce.sum(1) / t / elig[i:i + 50].sum(1) / n_t
    v = per_window.numpy()
    rng = np.random.default_rng(0)
    boots = [v[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return {"nelbo": float(v.mean()), "ppl_bound": math.exp(v.mean()),
            "ppl_bound_ci95": [math.exp(np.percentile(boots, 2.5)), math.exp(np.percentile(boots, 97.5))],
            "n_windows": n, "n_t": n_t}


@torch.no_grad()
def ar_nll_same_windows(model, n=1000, seed=999, block=256) -> dict:
    x0 = fixed_subset("test", n, block, seed=seed)
    per = []
    for i in range(0, n, 50):
        x = x0[i:i + 50]
        lg, _ = model(x[:, :-1])
        per.append(F.cross_entropy(lg.transpose(1, 2), x[:, 1:], reduction="none").mean(1))
    v = torch.cat(per).numpy()
    rng = np.random.default_rng(0)
    boots = [v[rng.integers(0, n, n)].mean() for _ in range(2000)]
    return {"nll": float(v.mean()), "ppl": math.exp(v.mean()),
            "ppl_ci95": [math.exp(np.percentile(boots, 2.5)), math.exp(np.percentile(boots, 97.5))]}


# ---------------------------------------------------------------- generation quality
KNOWN = None


def madeup_rate(texts):
    global KNOWN
    if KNOWN is None:
        KNOWN = set(json.load(open(os.path.join(mdlm.PROJECT2, "results", "metrics", "known_words_300000.json"))))
    ws = [w for t in texts for w in words(t) if w.isalpha()]
    return sum(w not in KNOWN for w in ws) / max(1, len(ws))


@torch.no_grad()
def teacher_ppl(teacher, tok, samples) -> float:
    nll = cnt = 0.0
    for s in samples:
        p = tok.encode(s["prompt"])
        new = tok.encode(s["completion"]) if s["completion"] else []
        ids = [BOS_ID] + p + new + ([EOS_ID] if s["ended_with_eos"] else [])
        ids = ids[:256]
        if len(ids) <= 1 + len(p):
            continue
        x = torch.tensor([ids])
        lg, _ = teacher(x[:, :-1])
        ce = F.cross_entropy(lg[0], x[0, 1:], reduction="none")[len(p):]
        nll += ce.sum().item()
        cnt += len(ce)
    return math.exp(nll / max(cnt, 1))


def gen_metrics(groups, teacher, tok) -> dict:
    flat = [s for g in groups for s in g]
    d1, d1s = grouped_distinct([[s["completion"] for s in g] for g in groups], 1)
    d2, d2s = grouped_distinct([[s["completion"] for s in g] for g in groups], 2)
    flags = [auto_flags(s, REP_THRESHOLD) for s in flat]
    return {"n": len(flat), "gen_ppl": teacher_ppl(teacher, tok, flat), "distinct_1": d1, "distinct_2": d2,
            "distinct_2_std": d2s, "rep4_rate": float(np.mean([repeated_ngram_rate(s["completion"]) for s in flat])),
            "repetition_flag_rate": float(np.mean([f["repetition"] for f in flags])),
            "eos_rate": float(np.mean([s["ended_with_eos"] for s in flat])),
            "madeup_rate": madeup_rate([s["completion"] for s in flat]),
            "avg_new_tokens": float(np.mean([s["new_tokens"] for s in flat])),
            "avg_nfe": float(np.mean([s.get("nfe", s["new_tokens"]) for s in flat]))}


def diffusion_groups(model, tok, prompts, steps, schedule, sampling, select="confidence", seed=GROUP_SEED,
                     n_groups=N_GROUPS):
    kw = dict(SAMPLING[sampling])
    return [generate_stories(model, tok, prompts, seed=seed + g, steps=steps, schedule=schedule, select=select,
                             length=model.cfg.block_size, **kw) for g in range(n_groups)]


@torch.no_grad()
def ar_groups(model, tok, prompts, seed=GROUP_SEED, n_groups=N_GROUPS, window=256, **kw):
    groups = []
    for g in range(n_groups):
        torch.manual_seed(seed + g)
        out = []
        for p in prompts:
            ids = [BOS_ID] + tok.encode(p)
            x, n_new = ar_generate(model, torch.tensor([ids]), window - len(ids), eos_id=EOS_ID, **kw)
            new = x[0, len(ids):].tolist()
            ended = EOS_ID in new
            new = new[:new.index(EOS_ID)] if ended else new
            out.append({"prompt": p, "completion": tok.decode(new), "text": tok.decode(ids[1:] + new),
                        "new_tokens": len(new), "ended_with_eos": ended, "nfe": n_new})
        groups.append(out)
    return groups


def dev_prompts(tok, n=8, seed=2026) -> list[str]:
    """Beginnings (first 5 words) of random validation stories, without duplicates (most TinyStories start
    "Once upon a time"), used only for choosing generation settings. Test prompts are never used for that."""
    x = fixed_subset("val", 400, 256, seed=seed)
    out = []
    for row in x.tolist():
        text = tok.decode(row[1:row.index(EOS_ID)] if EOS_ID in row else row[1:])
        p = " ".join(text.split()[:5])
        if p not in out and p not in TEST_PROMPTS:
            out.append(p)
        if len(out) == n:
            break
    return out


# ---------------------------------------------------------------- stages
def save(obj, *path):
    p = os.path.join(*path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["recon", "nelbo", "grid", "final", "errors"])
    ap.add_argument("--run", required=True, help="diffusion run name, e.g. mdlm_base")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 1,
                    help="CPU threads (use few when another training run shares the machine)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    tok = load_tokenizer()
    model, ck = load_checkpoint(os.path.join(mdlm.PROJECT3, "checkpoints", args.run, "model_best.pt"))
    out_dir = os.path.join(RESULTS, args.run)
    print(f"{args.run}: step {ck['step']}, {count_parameters(model)['total']:,} params, stage {args.stage}")

    if args.stage == "recon":
        r = reconstruction(model, tok)
        save(r, out_dir, "reconstruction.json")
        for k, v in r["rates"].items():
            print(f"  mask {float(k):.0%}: CE {v['ce']:.3f}, accuracy {v['accuracy']:.1%}")

    elif args.stage == "nelbo":
        r = {"diffusion": nelbo(model), "ar_teacher_same_windows": ar_nll_same_windows(load_teacher()),
             "ar_small_same_windows": ar_nll_same_windows(load_ar_small())}
        save(r, out_dir, "nelbo.json")
        print(json.dumps(r, indent=1))

    elif args.stage == "grid":
        teacher = load_teacher()
        prompts = dev_prompts(tok)
        gpath = os.path.join(out_dir, "grid_dev.json")
        cache = {}
        if os.path.exists(gpath):  # generation is seeded, so rows already computed are reused
            old = json.load(open(gpath, encoding="utf-8"))
            if old.get("dev_prompts") == prompts:
                cache = {(r["schedule"], r["sampling"], r["steps"], r["select"]): r for r in old["rows"]}
        rows = []
        for select in SELECTS:
            for schedule in SCHEDULES:
                for sampling in SAMPLING:
                    for steps in STEPS:
                        key = (schedule, sampling, steps, select)
                        if key in cache:
                            rows.append(cache[key])
                            continue
                        t0 = time.time()
                        m = gen_metrics(diffusion_groups(model, tok, prompts, steps, schedule, sampling,
                                                         select=select), teacher, tok)
                        rows.append({"schedule": schedule, "sampling": sampling, "steps": steps, "select": select,
                                     **m, "seconds": time.time() - t0})
                        print(f"  {select:10s} {schedule:10s} {sampling:9s} {steps:2d}: ppl {m['gen_ppl']:7.2f} "
                              f"D2 {m['distinct_2']:.3f} rep {m['repetition_flag_rate']:.2f} "
                              f"eos {m['eos_rate']:.2f}")
        ok = [r for r in rows if r["distinct_2"] >= MIN_D2 and r["repetition_flag_rate"] <= MAX_REP]
        rule = "main"
        if not ok:  # fallback, fixed before any mdlm_base result existed (see INSTRUCTIONS.md, decisions)
            ok = [r for r in rows if r["repetition_flag_rate"] <= MAX_REP]
            rule = "fallback"
        best = dict(min(ok, key=lambda r: r["gen_ppl"]), rule=rule) if ok else None
        extra = []
        if best:
            for steps in STEPS:  # unprompted generation at the selected setting
                m = gen_metrics(diffusion_groups(model, tok, [""] * len(prompts), steps, best["schedule"],
                                                 best["sampling"], select=best["select"]), teacher, tok)
                extra.append({"schedule": best["schedule"], "sampling": best["sampling"], "steps": steps,
                              "select": best["select"], "unprompted": True, **m})
        save({"dev_prompts": prompts,
              "rule": f"main: min gen_ppl s.t. distinct_2>={MIN_D2}, rep_flag<={MAX_REP}; "
                      f"fallback if none: min gen_ppl s.t. rep_flag<={MAX_REP}",
              "selected": best, "rows": rows, "ablations": extra}, out_dir, "grid_dev.json")
        print("selected:", best)

    elif args.stage == "final":
        teacher = load_teacher()
        grid = json.load(open(os.path.join(out_dir, "grid_dev.json"), encoding="utf-8"))
        sel = grid["selected"]
        res = {"selected_on_dev": sel, "diffusion": {}, "ar": {}}
        samples = {}
        for steps in STEPS:  # selected schedule/sampling/selection, at every step count
            groups = diffusion_groups(model, tok, TEST_PROMPTS, steps, sel["schedule"], sel["sampling"],
                                      select=sel["select"])
            res["diffusion"][str(steps)] = gen_metrics(groups, teacher, tok)
            samples[f"diffusion_{steps}"] = groups
        for name, ar in (("ar_teacher", teacher), ("ar_small", load_ar_small())):
            groups = ar_groups(ar, tok, TEST_PROMPTS, method="top_p", top_p=0.95)
            res["ar"][name] = gen_metrics(groups, teacher, tok)
            samples[name] = groups
        save(res, out_dir, "final_test.json")
        save(samples, GEN, args.run, "final_test_samples.json")
        print(json.dumps(res, indent=1))

    elif args.stage == "errors":
        grid = json.load(open(os.path.join(out_dir, "grid_dev.json"), encoding="utf-8"))
        sel = grid["selected"]
        prompts = [TEST_PROMPTS[i % len(TEST_PROMPTS)] for i in range(60)]
        diff = generate_stories(model, tok, prompts, seed=1344, steps=sel["steps"], schedule=sel["schedule"],
                                select=sel["select"], length=model.cfg.block_size, **SAMPLING[sel["sampling"]])
        ar = ar_groups(load_teacher(), tok, prompts, seed=1344, n_groups=1, method="top_p", top_p=0.95)[0]
        for s in diff + ar:
            s.pop("window_ids", None)
        save({"setting": sel, "diffusion": diff, "ar_teacher": ar}, GEN, args.run, "error_analysis_stories.json")
        print("wrote", len(diff), "+", len(ar), "stories")


if __name__ == "__main__":
    main()
