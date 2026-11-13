"""See the diffusion model write a story, step by step.

Usage (from project3, .venv active, after training):
    python scripts/try_model.py                                         # default prompt, 16 steps
    python scripts/try_model.py --prompt "Tom found a red box" --steps 8 --show_steps
    python scripts/try_model.py --run mdlm_small --schedule cosine --compare_ar
--show_steps prints the window after every step ("_" = still masked), so you can watch tokens appear in parallel.
--compare_ar also writes the same beginning with the autoregressive teacher (Project 1).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

import mdlm  # noqa: E402
from mdlm.generate import SCHEDULES, generate, split_story  # noqa: E402
from mdlm.model import load_checkpoint  # noqa: E402
from mdlm.tokens import BOS_ID, EOS_ID, MASK_ID, load_tokenizer  # noqa: E402


def show(tok, ids, upto=60):
    out = []
    for t in ids[1:upto]:
        out.append("_" if t == MASK_ID else tok.decode([t]).strip() or "·")
    return " ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="mdlm_base")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--schedule", default="linear", choices=SCHEDULES)
    ap.add_argument("--select", default="random", choices=["random", "confidence"],
                    help="which masked positions to reveal first (random = selected setting; confidence repeats more)")
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show_steps", action="store_true")
    ap.add_argument("--compare_ar", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    tok = load_tokenizer()
    model, ck = load_checkpoint(os.path.join(mdlm.PROJECT3, "checkpoints", args.run, "model_best.pt"))
    p = tok.encode(args.prompt) if args.prompt else []
    t0 = time.perf_counter()
    x, info = generate(model, [p], model.cfg.block_size, steps=args.steps, schedule=args.schedule, select=args.select, method="top_p",
                       top_p=args.top_p, generator=torch.Generator().manual_seed(args.seed), trace=args.show_steps)
    dt = time.perf_counter() - t0
    if args.show_steps:
        for s, st in enumerate(info["states"]):
            print(f"step {s:2d}: {show(tok, st[0].tolist())} ...")
        print()
    sp = split_story(x[0].tolist(), len(p))
    print(f"--- Diffusion ({args.run}, {info['nfe']} steps, {args.schedule}, {args.select} order, {dt:.2f} s, "
          f"{'ended by itself' if sp['ended_with_eos'] else 'hit the 256-token window'})")
    print(tok.decode(sp["story_ids"]))
    if args.compare_ar:
        from mdlm.evaluate import load_teacher
        from src.generate import generate as ar_generate
        teacher = load_teacher()
        torch.manual_seed(args.seed)
        ids = torch.tensor([[BOS_ID] + p])
        t0 = time.perf_counter()
        y, n = ar_generate(teacher, ids, 256 - ids.size(1), eos_id=EOS_ID, method="top_p", top_p=args.top_p)
        dt = time.perf_counter() - t0
        new = y[0, ids.size(1):].tolist()
        new = new[:new.index(EOS_ID)] if EOS_ID in new else new
        print(f"\n--- Autoregressive teacher (Project 1, {n} steps, {dt:.2f} s)")
        print(tok.decode(p + new))


if __name__ == "__main__":
    main()
