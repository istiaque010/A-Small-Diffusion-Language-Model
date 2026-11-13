"""Phase 13: error rates of the diffusion model and the AR teacher, from the labelled stories.

Stories: results/generations/mdlm_base/error_analysis_stories.json (made by `mdlm.evaluate --stage errors`).
Labels:  scripts/error_labels/labels.py  (LABELS: blind id -> (contradiction, grammar, entity, local_incoherence,
         note)) and scripts/error_labels/key.json (blind id -> [model, index]). The 120 stories were shuffled with
         a fixed seed and labelled without knowing which model wrote which. The labels were produced with an AI
         assistant (Claude) and are not human-verified.
Rules (Project 1, plus one category):
  contradiction      something clashes with an earlier fact in the story or does not follow from it
  grammar            at least one broken sentence or invented word
  entity             a character changes name, gender or species, or the wrong character speaks or acts
  local_incoherence  a token or phrase that does not fit its immediate neighbours (doubled/garbled words,
                     a word in an impossible slot) -- the "masking artifact" category of the guide
  repetition, incomplete: automatic (Project 1 definitions)
Usage: python scripts/error_counts.py   -> results/metrics/error_counts.json and .md
"""
import importlib.util
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mdlm  # noqa: E402
from mdlm.evaluate import REP_THRESHOLD  # noqa: E402
from src.evaluate import auto_flags  # noqa: E402

HAND = ["contradiction", "grammar", "entity", "local_incoherence"]
D = os.path.join(mdlm.PROJECT3, "scripts", "error_labels")


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def fisher(a, n1, b, n2):
    K, N = a + b, n1 + n2
    pr = lambda x: math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)  # noqa: E731
    p0 = pr(a)
    return min(1.0, sum(pr(x) for x in range(max(0, K - n2), min(n1, K) + 1) if pr(x) <= p0 * (1 + 1e-9)))


def main():
    stories = json.load(open(os.path.join(mdlm.RESULTS, "generations", "mdlm_base", "error_analysis_stories.json"),
                             encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("labels", os.path.join(D, "labels.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    key = json.load(open(os.path.join(D, "key.json")))
    counts = {m: {k: 0 for k in HAND + ["repetition", "incomplete", "no_error"]} | {"n": 0}
              for m in ("diffusion", "ar_teacher")}
    for b, (model, i) in key.items():
        lab = mod.LABELS[int(b)]
        s = stories[model][i]
        f = auto_flags(s, REP_THRESHOLD)
        c = counts[model]
        c["n"] += 1
        for k, v in zip(HAND, lab[:4]):
            c[k] += bool(v)
        c["repetition"] += f["repetition"]
        c["incomplete"] += f["incomplete"]
        c["no_error"] += not (any(lab[:4]) or f["repetition"] or f["incomplete"])
    out = {"counts": counts, "wilson95": {m: {k: wilson(v, c["n"]) for k, v in c.items() if k != "n"}
                                          for m, c in counts.items()},
           "fisher_p": {k: fisher(counts["diffusion"][k], counts["diffusion"]["n"], counts["ar_teacher"][k],
                                  counts["ar_teacher"]["n"]) for k in HAND + ["repetition", "incomplete"]}}
    os.makedirs(os.path.join(mdlm.RESULTS, "metrics"), exist_ok=True)
    json.dump(out, open(os.path.join(mdlm.RESULTS, "metrics", "error_counts.json"), "w"), indent=1)
    L = ["| Error (60 stories each) | Diffusion (mdlm_base) | AR teacher | Fisher p |", "|---|---|---|---|"]
    for k in HAND + ["repetition", "incomplete", "no_error"]:
        a, b = counts["diffusion"], counts["ar_teacher"]
        L.append(f"| {k} | {a[k]}/{a['n']} ({a[k] / a['n']:.0%}) | {b[k]}/{b['n']} ({b[k] / b['n']:.0%}) | "
                 f"{out['fisher_p'].get(k, float('nan')):.3g} |")
    open(os.path.join(mdlm.RESULTS, "metrics", "error_counts.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
