"""Does the model use K at all?

The window test showed that which 24 known labels the model sees does not change
its output. Two explanations remain: the labels within one organization are
interchangeable (fine), or the model largely ignores K and recites a global prior
(not fine).

This distinguishes them. For each held-out apex A the model is conditioned on:

  own      A's own known labels                    the real task
  swapped  a different organization's labels        conditioning on nonsense
  generic  the globally most common labels          an explicit prior prompt
  single   one label from A                         minimal true conditioning

and every variant is scored against A's held-out labels. If `own` does not beat
`swapped`, the model is not conditioning — it is reciting.
"""
import argparse, json, os, random, statistics, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "subfury"))
from predict import load_model, predict_labels                      # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="data/groups_test.jsonl")
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--model", default="results/subfury")
    ap.add_argument("--budgets", default="10,25,50,100")
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--max-apexes", type=int, default=120)
    ap.add_argument("--min-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/research/swap.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    maxN = max(budgets)
    rng = random.Random(args.seed)
    model, tok, device = load_model(args.model)

    c = Counter()
    with open(args.train_jsonl) as f:
        for line in f:
            for lab in set(json.loads(line)["labels"]):
                c[lab] += 1
    generic = [lab for lab, _ in c.most_common(24)]

    groups = []
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["labels"]) >= args.min_labels:
                groups.append(rec)
    rng.shuffle(groups)
    groups = groups[: args.max_apexes]
    print(f"swap test over {len(groups)} apexes; generic prompt = {', '.join(generic[:6])}…",
          flush=True)

    acc = {v: {n: [] for n in budgets} for v in ("own", "swapped", "generic", "single")}
    rows = []
    for i, rec in enumerate(groups):
        labels = rec["labels"][:]
        rng.shuffle(labels)
        half = len(labels) // 2
        K, H = labels[:half], set(labels[half:])
        other = groups[(i + len(groups) // 2) % len(groups)]        # someone else's org
        other_K = other["labels"][: max(1, len(other["labels"]) // 2)]

        variants = {"own": K, "swapped": other_K, "generic": generic, "single": K[:1]}
        row = {"apex": rec["apex"], "k": len(K), "h": len(H), "v": {}}
        for name, cond in variants.items():
            preds = [p for p, _ in predict_labels(model, tok, device, cond,
                                                  topn=maxN, num_beams=args.num_beams)]
            # never credit a variant for echoing labels the real task already knew
            preds = [p for p in preds if p not in set(K)]
            row["v"][name] = {str(n): len(set(preds[:n]) & H) / len(H) for n in budgets}
            for n in budgets:
                acc[name][n].append(row["v"][name][str(n)])
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(groups)}", flush=True)

    summary = {"apexes": len(rows), "budgets": budgets,
               "per_variant": {v: {str(n): round(statistics.mean(acc[v][n]), 4) for n in budgets}
                               for v in acc}}
    # paired: own minus swapped, per apex
    for n in budgets:
        d = [r["v"]["own"][str(n)] - r["v"]["swapped"][str(n)] for r in rows]
        boot = []
        rb = random.Random(7)
        for _ in range(2000):
            s = [d[rb.randrange(len(d))] for _ in range(len(d))]
            boot.append(sum(s) / len(s))
        boot.sort()
        summary.setdefault("own_minus_swapped", {})[str(n)] = {
            "mean": round(statistics.mean(d), 4),
            "ci95": [round(boot[50], 4), round(boot[1949], 4)],
            "apexes_own_better": sum(1 for x in d if x > 0),
            "apexes_swapped_better": sum(1 for x in d if x < 0)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)

    print("\n  conditioned on" + "".join(f"{('@'+str(n)):>9}" for n in budgets))
    for v in ("own", "single", "swapped", "generic"):
        s = summary["per_variant"][v]
        print(f"  {v:<14}" + "".join(f"{s[str(n)]:>9.3f}" for n in budgets))
    print("\n  own − swapped (paired, 95% CI):")
    for n in budgets:
        d = summary["own_minus_swapped"][str(n)]
        print(f"   @{n:<4} {d['mean']:+.4f}  [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]   "
              f"own better on {d['apexes_own_better']}/{len(rows)} apexes")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
