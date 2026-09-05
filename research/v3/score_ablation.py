"""Score every v3 run through the same harness the baselines used."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from research.harness import run_evaluation                          # noqa: E402
from rank import V3Ranker                                            # noqa: E402

runs = sys.argv[1:] or ["settrans-full", "deepsets-full", "settrans-gen", "settrans-noprior"]
rankers = []
for tag in runs:
    ck = f"results/v3/{tag}/best.pt"
    if not os.path.exists(ck):
        print(f"skip {tag}: no checkpoint"); continue
    src = "generator" if tag.endswith("-gen") else "hybrid"
    r = V3Ranker(ckpt=ck, source=src)
    r.name = f"{tag}/{src}"
    rankers.append(r)
    if src == "hybrid":                       # also score its two halves alone
        for s in ("generator", "retriever"):
            r2 = V3Ranker(ckpt=ck, source=s); r2.name = f"{tag}/{s}"
            rankers.append(r2)

res = run_evaluation(rankers, out_path="results/research/v3_ablation.json")
print(res if isinstance(res, str) else json.dumps(res, indent=1)[:2000])
