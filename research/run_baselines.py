"""Run every method — baselines and (optionally) the neural model — through
the shared harness, on one identical split.

    python3 research/run_baselines.py                 # baselines only
    python3 research/run_baselines.py --neural        # + subfury-v2

Writes results/research/baselines.json and prints:
  * the comparison table (macro recall@N with bootstrap 95% CIs, MAP)
  * paired bootstrap tests of neural vs the frequency prior at every budget
  * two subset analyses for neural + frequency prior:
      - "reachable" subset: apexes with >= 1 held-out label in the training
        vocabulary (the rest are unwinnable for any closed-vocab method)
      - micro-averaged recall (total hits / total held-out labels)

No DNS, no network.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)          # allow `python3 research/run_baselines.py`

from harness import (DEFAULT_NS, DEFAULT_OUT_DIR, DEFAULT_SEED, DEFAULT_TEST,
                     format_table, paired_diff_ci, run_evaluation, summarize)
from baselines.frequency import FrequencyRanker
from baselines.markov import MarkovRanker
from baselines.wordlist import WordlistRanker


def build_rankers(orders=(3, 4, 5), pool_size: int = 512, beam_width: int = 2000,
                  neural: bool = False, num_beams: int = 64, max_n: int = 200):
    rankers = [WordlistRanker(), FrequencyRanker()]
    for o in orders:
        rankers.append(MarkovRanker(order=o, pool_size=pool_size,
                                    beam_width=beam_width))
    if neural:
        from baselines.neural import NeuralRanker
        rankers.append(NeuralRanker(num_beams=num_beams, max_n=max_n))
    return rankers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default=DEFAULT_TEST)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--min-labels", type=int, default=6)
    ap.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    ap.add_argument("--orders", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--pool-size", type=int, default=512)
    ap.add_argument("--beam-width", type=int, default=2000)
    ap.add_argument("--neural", action="store_true",
                    help="include the SubFury v2 transformer (needs torch + checkpoint)")
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--out", default=os.path.join(DEFAULT_OUT_DIR, "baselines.json"))
    args = ap.parse_args()

    print("building methods (training on data/groups_train.jsonl)...", flush=True)
    rankers = build_rankers(tuple(args.orders), args.pool_size, args.beam_width,
                            neural=args.neural, num_beams=args.num_beams,
                            max_n=max(args.ns))
    for r in rankers:
        print(f"  {r.name}: {r.describe()}")
    print()

    doc, cases, results = run_evaluation(
        rankers, test_jsonl=args.test_jsonl, ns=args.ns, seed=args.seed,
        min_labels=args.min_labels, out_path=None, return_results=True)

    by_name = {r.name: r for r in results}
    for r in rankers:                     # descriptions after the run
        doc.setdefault("rankers", {})[r.name] = r.describe()
        if hasattr(r, "save_cache"):
            r.save_cache()

    # ---- paired tests: everything vs the frequency prior ------------------
    prior_name = "frequency-prior"
    ns = tuple(sorted(args.ns))
    if prior_name in by_name:
        prior = by_name[prior_name]
        doc["paired_vs_frequency_prior"] = {}
        lines = ["paired bootstrap: method minus frequency-prior "
                 "(same apexes, 2000 resamples)",
                 f"{'method':<26}" + "".join(f"{'Δ@'+str(n):>26}" for n in ns)]
        for res in results:
            if res.name == prior_name:
                continue
            entry = {}
            row = f"{res.name:<26}"
            for n in ns:
                d = paired_diff_ci(res.recall[n], prior.recall[n], args.seed + n)
                entry[str(n)] = d
                star = "*" if d["excludes_zero"] else " "
                row += (f"{d['mean_diff']:>+9.3f} "
                        f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]{star}")
            doc["paired_vs_frequency_prior"][res.name] = entry
            lines.append(row)
        lines.append("* = 95% CI of the paired difference excludes zero")
        doc["paired_table"] = "\n".join(lines)
        print("\n".join(lines) + "\n")

    # ---- subset analyses: neural + frequency prior ------------------------
    focus = [r for r in results
             if r.name == prior_name or "neural" in r.name or "subfury" in r.name]
    if focus:
        counts = next(r.counts for r in rankers if isinstance(r, FrequencyRanker))
        reachable = np.array([any(l in counts for l in c.holdout) for c in cases])
        doc["reachable_subset"] = {
            "definition": "apexes with >= 1 held-out label present in the "
                          "training-label vocabulary",
            "apexes": int(reachable.sum()),
            "excluded": int((~reachable).sum()),
        }
        subs = [summarize(r, args.seed, mask=reachable, name=r.name + " [reachable]")
                for r in focus]
        doc["reachable_results"] = subs
        table = format_table(subs, ns)
        doc["reachable_table"] = table
        print(f"reachable subset: {int(reachable.sum())}/{len(cases)} apexes "
              f"({int((~reachable).sum())} have no held-out label anywhere in "
              f"the training vocabulary)")
        print(table + "\n")

        micro_lines = [f"{'method':<26}" + "".join(f"{'@'+str(n)+' macro/micro':>22}" for n in ns)]
        doc["micro"] = {}
        for r in focus:
            s_all = summarize(r, args.seed)
            doc["micro"][r.name] = {str(n): {"macro": s_all["recall"][str(n)]["mean"],
                                             "micro": s_all["recall"][str(n)]["micro"]}
                                    for n in ns}
            micro_lines.append(f"{r.name:<26}" + "".join(
                f"{s_all['recall'][str(n)]['mean']:>11.3f}/"
                f"{s_all['recall'][str(n)]['micro']:<10.3f}" for n in ns))
        doc["micro_table"] = "\n".join(micro_lines)
        print("macro (mean of per-apex recall) vs micro (total hits / total "
              "held-out labels), full 545-apex set")
        print("\n".join(micro_lines) + "\n")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import json
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
