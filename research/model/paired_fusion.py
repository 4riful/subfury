"""Did A2's pre-registered kill criterion fire?

METHODOLOGY.md §7 states: "Hybrid <= max(channels) at every N => the fusion
scorer is broken, not the idea." The marginal CIs overlap heavily, so this
answers it with the paired test the harness provides.
"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research"))
sys.path.insert(0, HERE)
from research.harness import (load_groups, make_cases, evaluate_ranker,
                              paired_diff_ci, DEFAULT_SEED, DEFAULT_NS)
from rank import V3Ranker

cases = make_cases(load_groups(), seed=DEFAULT_SEED)
out = {"question": "hybrid vs its own best single channel, paired",
       "criterion": "METHODOLOGY.md sec 7, axis A2",
       "apexes": len(cases), "seed": DEFAULT_SEED, "runs": {}}
lines = [f"{'comparison':<44}" + "".join(f"{'@'+str(n):>26}" for n in DEFAULT_NS),
         "-" * (44 + 26 * len(DEFAULT_NS))]

for tag in ("deepsets-full", "settrans-full"):
    ck = f"results/runs/{tag}/best.pt"
    if not os.path.exists(ck):
        print("skip", tag); continue
    scored = {}
    for src in ("hybrid", "retriever", "generator"):
        r = V3Ranker(ckpt=ck, source=src); r.name = f"{tag}/{src}"
        scored[src] = evaluate_ranker(r, cases)
        print(f"  {r.name} {scored[src].seconds:.0f}s", flush=True)
    for other in ("retriever", "generator"):
        row, cells = {}, []
        for n in DEFAULT_NS:
            d = paired_diff_ci(scored["hybrid"].recall[n], scored[other].recall[n],
                               seed=DEFAULT_SEED)
            row[str(n)] = d
            lo, hi = d["ci95"]
            star = "*" if d["excludes_zero"] else " "
            cells.append(f"{d['mean_diff']:+.4f} [{lo:+.4f},{hi:+.4f}]{star}".rjust(26))
        out["runs"][f"{tag}: hybrid - {other}"] = row
        lines.append(f"{tag+': hybrid - '+other:<44}" + "".join(cells))

lines += ["", "* = paired 95% CI excludes zero.  Negative means the hybrid is",
          "    WORSE than that single channel alone."]
out["table"] = "\n".join(lines)
json.dump(out, open("results/research/fusion.json", "w"), indent=1, default=float)
print("\n" + out["table"])
