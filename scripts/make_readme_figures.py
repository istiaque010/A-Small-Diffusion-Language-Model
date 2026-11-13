"""Figures for README.md, made only from saved result files.

Usage: python scripts/make_readme_figures.py
  -> results/figures/tradeoff.png  (generation latency vs. story quality, diffusion and AR)
  -> results/figures/errors.png    (error rates, diffusion vs. AR teacher)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mdlm  # noqa: E402

M = os.path.join(mdlm.RESULTS, "metrics")
FIG = os.path.join(mdlm.RESULTS, "figures")
C = {"mdlm_base": "#2a78d6", "mdlm_small": "#8fb8ea", "ar_teacher": "#1f1f1f", "ar_small": "#eb6834"}
INK, MUTED = "#222222", "#666666"


def load(*p):
    return json.load(open(os.path.join(M, *p), encoding="utf-8"))


def tradeoff():
    bench = {b["model"]: b for b in load("generation_benchmark.json")}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for run, label in (("mdlm_base", "Diffusion, 13.9M"), ("mdlm_small", "Diffusion, 5.3M")):
        fin = load(run, "final_test.json")
        xs = [bench[run][f"S={s}"]["latency_s"] for s in (4, 8, 16, 32)]
        ys = [fin["diffusion"][str(s)]["gen_ppl"] for s in (4, 8, 16, 32)]
        ax.plot(xs, ys, "-o", color=C[run], lw=2, ms=8, label=label)
        for s, x, y in zip((4, 8, 16, 32), xs, ys):
            ax.annotate(f"{s} steps", (x, y), textcoords="offset points", xytext=(7, 4), fontsize=8, color=MUTED)
    fin = load("mdlm_base", "final_test.json")
    for name, label in (("ar_small", "Autoregressive, 5.3M"), ("ar_teacher", "Autoregressive, 13.9M*")):
        x = bench[name]["full window"]["latency_s"]
        ax.plot([x], [fin["ar"][name]["gen_ppl"]], "s", color=C[name], ms=9, label=label)
    ax.set_xlabel("time to write one 256-token window on the CPU (s), lower = faster")
    ax.set_ylabel("teacher perplexity of the story (log), lower = better")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Diffusion is faster only with few steps, where its stories are worst", loc="left", fontsize=11,
                 color=INK)
    fig.text(0.01, 0.01, "* the 13.9M AR model is also the scorer, so its own value is biased low. "
             "No key-value cache in either implementation.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(FIG, "tradeoff.png"), dpi=150)


def errors():
    c = load("error_counts.json")["counts"]
    cats = ["local_incoherence", "grammar", "entity", "contradiction", "repetition", "incomplete"]
    names = ["local\nincoherence", "grammar", "entity", "contradiction", "repetition", "incomplete"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.38
    for k, (m, label) in enumerate((("diffusion", "Diffusion (13.9M)"), ("ar_teacher", "Autoregressive (13.9M)"))):
        vals = [100 * c[m][x] / c[m]["n"] for x in cats]
        xs = [i + (k - 0.5) * w for i in range(len(cats))]
        ax.bar(xs, vals, w - 0.04, color=C["mdlm_base"] if m == "diffusion" else C["ar_teacher"], label=label)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(cats)), names, fontsize=9)
    ax.set_ylabel("% of 60 stories")
    ax.set_ylim(0, 112)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Errors in 60 stories per model (labels by an AI assistant, not human-verified)", loc="left",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "errors.png"), dpi=150)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    tradeoff()
    errors()
    print("wrote results/figures/tradeoff.png and errors.png")
