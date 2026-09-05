"""Is the bottleneck permutation, or capacity?

predict.py conditions on sorted(known)[-block_size//8:] — the 24 alphabetically
last labels. Sorting already makes the model permutation-invariant, so the
invariance argument for a set encoder is moot. The open question is what the
truncation costs.

For held-out apexes with |K| > 24 this feeds the same model the same budget from
different 24-label windows of the same known set:

  tail      sorted(K)[-24:]      what the model does today
  head      sorted(K)[:24]       the labels it currently throws away
  random    24 drawn at random   x R repeats, to get a spread
  spread    every 
th label     an even sample across the alphabet

If recall depends on which window is chosen, the truncation is discarding
signal and pooling the whole set is the fix. If they are identical, capacity is
not the bottleneck and the set-encoder argument is weaker.
"""
import argparse, json, os, random, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "subfury"))
from predict import load_model, predict_labels                      # noqa: E402


def windows(K, w, repeats, rng):
    s = sorted(K)
    out = {"tail": s[-w:], "head": s[:w]}
    step = max(1, len(s) // w)
    out["spread"] = s[::step][:w]
    for r in range(repeats):
        out[f"random{r+1}"] = rng.sample(s, w)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="data/groups_test_uncapped.jsonl")
    ap.add_argument("--model", default="results/subfury")
    ap.add_argument("--budgets", default="10,25,50,100")
    ap.add_argument("--window", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--min-known", type=int, default=25, help="only apexes where truncation bites")
    ap.add_argument("--max-apexes", type=int, default=85)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/research/capacity.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    maxN = max(budgets)
    model, tok, device = load_model(args.model)

    groups = []
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["labels"]) >= 2 * args.min_known:
                groups.append(rec)
    random.Random(args.seed).shuffle(groups)
    groups = groups[: args.max_apexes]
    print(f"{len(groups)} apexes with |K| >= {args.min_known} "
          f"(window {args.window}, so truncation is active)", flush=True)

    per_variant = {}
    rows = []
    for i, rec in enumerate(groups):
        rng = random.Random(args.seed + i)
        labels = rec["labels"][:]
        rng.shuffle(labels)
        half = len(labels) // 2
        K, H = labels[:half], set(labels[half:])

        row = {"apex": rec["apex"], "k": len(K), "h": len(H), "variants": {}}
        for name, win in windows(K, args.window, args.repeats, rng).items():
            preds = [p for p, _ in predict_labels(model, tok, device, win,
                                                  topn=maxN, num_beams=args.num_beams)]
            row["variants"][name] = {str(n): len(set(preds[:n]) & H) / len(H) for n in budgets}
            per_variant.setdefault(name, {n: [] for n in budgets})
            for n in budgets:
                per_variant[name][n].append(row["variants"][name][str(n)])
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(groups)}", flush=True)

    # how much does the choice of window move recall, per apex?
    spreads = []
    for r in rows:
        vals = [v[str(maxN)] for v in r["variants"].values()]
        spreads.append(max(vals) - min(vals))

    summary = {"apexes": len(rows), "window": args.window, "budgets": budgets,
               "per_variant": {k: {str(n): round(statistics.mean(v[n]), 4) for n in budgets}
                               for k, v in per_variant.items()},
               "window_spread_at_max_budget": {
                   "mean": round(statistics.mean(spreads), 4),
                   "median": round(statistics.median(spreads), 4),
                   "max": round(max(spreads), 4)}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)

    print("\nrecall by which 24 labels the model was allowed to see")
    hdr = "  variant   " + "".join(f"{('@'+str(n)):>9}" for n in budgets)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for name in ["tail", "head", "spread"] + [f"random{r+1}" for r in range(args.repeats)]:
        if name in summary["per_variant"]:
            v = summary["per_variant"][name]
            print(f"  {name:<9} " + "".join(f"{v[str(n)]:>9.3f}" for n in budgets))
    s = summary["window_spread_at_max_budget"]
    print(f"\n  per-apex spread between best and worst window @{maxN}: "
          f"mean {s['mean']:.3f}, median {s['median']:.3f}, max {s['max']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
