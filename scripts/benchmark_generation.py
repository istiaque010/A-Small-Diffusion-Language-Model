"""Phase 12: generation speed and memory of the diffusion models and the autoregressive (AR) baselines on the CPU.

Each model is measured in its own subprocess (so peak memory belongs to that model). Protocol, identical for all:
  - prompt "Once upon a time", output window 256 tokens (= the diffusion model's fixed length), batch size 1
  - one untimed warm-up, then 5 timed generations; top-p 0.95
  - diffusion: S = 4, 8, 16, 32 refinement steps (S forward passes over the 256-token window)
  - AR: 251 new tokens without early stopping (fills the same window; 251 forward passes over a growing
    context), so both produce a full 256-token window. Neither implementation uses a key-value cache.
  - CPU load of other programs is measured before and during timing; above 15% the result is marked unreliable.
Usage (from project3, .venv active, idle PC):  python scripts/benchmark_generation.py
Writes results/metrics/generation_benchmark.json and .md
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mdlm  # noqa: E402

RUNS, STEPS, WINDOW, PROMPT, LIMIT = 5, (4, 8, 16, 32), 256, "Once upon a time", 15.0
MODELS = ["mdlm_small", "mdlm_base", "ar_small", "ar_teacher"]


def measure(which: str) -> dict:
    import psutil
    import torch
    from mdlm.evaluate import load_ar_small, load_teacher
    from mdlm.generate import generate
    from mdlm.model import load_checkpoint
    from mdlm.tokens import BOS_ID, load_tokenizer
    from src.evaluate import OtherCpuLoad
    from src.generate import generate as ar_generate
    from src.utils import peak_memory_mb
    torch.set_num_threads(os.cpu_count() or 1)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2**20
    tok = load_tokenizer()
    p = tok.encode(PROMPT)
    if which.startswith("mdlm"):
        model, _ = load_checkpoint(os.path.join(mdlm.PROJECT3, "checkpoints", which, "model_best.pt"))
        configs = {f"S={s}": (lambda s=s: generate(model, [p], WINDOW, steps=s, schedule="linear",
                                                    method="top_p", top_p=0.95)) for s in STEPS}
    else:
        model = load_teacher() if which == "ar_teacher" else load_ar_small()
        ids = torch.tensor([[BOS_ID] + p])
        configs = {"full window": lambda: ar_generate(model, ids, WINDOW - ids.size(1), eos_id=None,
                                                      method="top_p", top_p=0.95)}
    rss1 = proc.memory_info().rss / 2**20
    out = {"model": which, "params": sum(q.numel() for q in model.parameters()), "ram_added_by_model_mb": rss1 - rss0}
    load = OtherCpuLoad()
    others = []
    with torch.no_grad():
        for name, fn in configs.items():
            torch.manual_seed(0)
            fn()  # warm-up
            load.start()
            time.sleep(0.5)
            others.append(load.stop())
            load.start()
            t0 = time.perf_counter()
            for _ in range(RUNS):
                fn()
            dt = (time.perf_counter() - t0) / RUNS
            others.append(load.stop())
            out[name] = {"latency_s": dt, "window_tokens_per_s": (WINDOW - 1 - len(p)) / dt}
    out["peak_ram_mb"] = peak_memory_mb("cpu")
    out["other_cpu_max_pct"] = max(others)
    out["reliable"] = max(others) <= LIMIT
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--single":
        print(json.dumps(measure(sys.argv[2])))
        return
    res = []
    for m in MODELS:
        r = subprocess.run([sys.executable, __file__, "--single", m], capture_output=True, text=True,
                           cwd=mdlm.PROJECT3, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        if r.returncode != 0:
            print(r.stderr[-2000:])
            raise SystemExit(1)
        res.append(json.loads(r.stdout.strip().splitlines()[-1]))
        print(res[-1])
    ar = {r["model"]: r["full window"]["latency_s"] for r in res if "full window" in r}
    L = ["| Model | Parameters | Setting | Latency, 256-token window (s) | Tokens/s | Speedup vs AR teacher | "
         "RAM added (MB) | Reliable |", "|---|---|---|---|---|---|---|---|"]
    for r in res:
        for k, v in r.items():
            if isinstance(v, dict):
                L.append(f"| {r['model']} | {r['params']:,} | {k} | {v['latency_s']:.3f} | {v['window_tokens_per_s']:.0f} | "
                         f"{ar['ar_teacher'] / v['latency_s']:.2f}x | {r['ram_added_by_model_mb']:.0f} | "
                         f"{'yes' if r['reliable'] else 'NO'} |")
    os.makedirs(os.path.join(mdlm.RESULTS, "metrics"), exist_ok=True)
    with open(os.path.join(mdlm.RESULTS, "metrics", "generation_benchmark.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(mdlm.RESULTS, "metrics", "generation_benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
