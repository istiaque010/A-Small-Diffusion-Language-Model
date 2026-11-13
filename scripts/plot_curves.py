"""Phase 7: learning curves of the diffusion runs (validation masked-token loss, accuracy and perplexity bound).

Usage: python scripts/plot_curves.py --run mdlm_small --run mdlm_base   -> results/figures/curves.png
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mdlm  # noqa: E402

COLORS = {"mdlm_small": "#eb6834", "mdlm_base": "#2a78d6"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True)
    args = ap.parse_args()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in args.run:
        rows = list(csv.DictReader(open(os.path.join(mdlm.RESULTS, "metrics", r, "log.csv"))))
        s = [int(x["step"]) for x in rows]
        c = COLORS.get(r, None)
        axes[0].plot(s, [float(x["val_loss"]) for x in rows], "o-", ms=3, color=c, label=r)
        axes[1].plot(s, [100 * float(x["val_acc"]) for x in rows], "o-", ms=3, color=c, label=r)
        axes[2].plot(s, [float(x["val_ppl_bound"]) for x in rows], "o-", ms=3, color=c, label=r)
    axes[0].set_ylabel("validation masked-token CE (nats)")
    axes[1].set_ylabel("accuracy at masked positions (%)")
    axes[2].set_ylabel("perplexity upper bound, exp(NELBO)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Diffusion LM training (fixed validation protocol: 800 windows, fixed t and masks)", x=0.01,
                 ha="left")
    fig.tight_layout()
    os.makedirs(os.path.join(mdlm.RESULTS, "figures"), exist_ok=True)
    fig.savefig(os.path.join(mdlm.RESULTS, "figures", "curves.png"), dpi=150)
    with open(os.path.join(mdlm.RESULTS, "metrics", "curves_table.md"), "w") as f:
        f.write("| Run | Step | Val masked CE | Val accuracy | Perplexity bound | Elapsed (h) |\n|---|---|---|---|---|---|\n")
        for r in args.run:
            rows = list(csv.DictReader(open(os.path.join(mdlm.RESULTS, "metrics", r, "log.csv"))))
            for x in rows:
                f.write(f"| {r} | {x['step']} | {x['val_loss']} | {float(x['val_acc']):.1%} | {x['val_ppl_bound']} | "
                        f"{float(x['elapsed_s']) / 3600:.2f} |\n")
    print("wrote results/figures/curves.png and results/metrics/curves_table.md")


if __name__ == "__main__":
    main()
