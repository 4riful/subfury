"""Paired bootstrap: the retrieval architecture vs the beam-search model, on the identical harness split.

Overlapping marginal CIs do not settle whether one method beats another; both
methods see the same 545 apexes, so the paired per-apex difference removes
between-apex variance and answers the question directly.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "research"))

from research.harness import (load_groups, make_cases, evaluate_ranker,
                              paired_diff_ci, DEFAULT_SEED, DEFAULT_NS)
from baselines.neural import NeuralRanker

CONTENDERS = [
    ("deepsets-full",    "retriever"),
    ("deepsets-full",    "hybrid"),
    ("settrans-full",    "retriever"),
    ("settrans-noprior", "retriever"),
]

cases = make_cases(load_groups(), seed=DEFAULT_SEED)
print(f"{len(cases)} apexes", flush=True)

res = {}
# NeuralRanker imports subfury/predict.py lazily, which does `from model
# import GPTConfig`. research/model/model.py answers to the same name, so the beam-search model must
# be fully constructed before research/model ever reaches sys.path.
beam = NeuralRanker(num_beams=64, max_n=200)
res["subfury-beam"] = evaluate_ranker(beam, cases)
print(f"  scored subfury-beam in {res['subfury-beam'].seconds:.0f}s", flush=True)

for _stale in ("model", "predict"):
    sys.modules.pop(_stale, None)
sys.path.insert(0, HERE)
from rank import V3Ranker                                            # noqa: E402

for tag, src in CONTENDERS:
    ck = f"results/runs/{tag}/best.pt"
    if not os.path.exists(ck):
        print(f"  skip {tag}: no checkpoint", flush=True); continue
    r = V3Ranker(ckpt=ck, source=src); r.name = f"{tag}/{src}"
    res[r.name] = evaluate_ranker(r, cases)
    print(f"  scored {r.name} in {res[r.name].seconds:.0f}s", flush=True)

base = res["subfury-beam"]
out = {"harness": "research/harness.py", "seed": DEFAULT_SEED,
       "apexes": len(cases), "reference": "subfury-beam", "comparisons": []}

lines = [f"{'variant vs beam-search model':<44}" +
         "".join(f"{'@'+str(n):>26}" for n in DEFAULT_NS),
         "-" * (34 + 26 * len(DEFAULT_NS))]
for name, r in res.items():
    if name == "subfury-beam":
        continue
    row, cells = {"name": name, "budgets": {}}, []
    for n in DEFAULT_NS:
        d = paired_diff_ci(r.recall[n], base.recall[n], seed=DEFAULT_SEED)
        row["budgets"][str(n)] = d
        lo, hi = d["ci95"]
        star = "*" if d["excludes_zero"] else " "
        cells.append(f"{d['mean_diff']:+.3f} [{lo:+.3f},{hi:+.3f}]{star}".rjust(26))
    out["comparisons"].append(row)
    lines.append(f"{name:<44}" + "".join(cells))
lines.append("")
lines.append("* = paired 95% CI excludes zero.  Positive favours the retrieval architecture.")

table = "\n".join(lines)
out["table"] = table
os.makedirs("results/research", exist_ok=True)
json.dump(out, open("results/research/paired_vs_beam.json", "w"), indent=1, default=float)
print("\n" + table)
