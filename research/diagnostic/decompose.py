"""How much of the model's recall is conditioning, and how much is the global prior?

The model is trained on P(y | K). The question this asks is whether it has learned
that, or whether it has mostly learned P(y) — the marginal popularity of labels in
the training corpus — and is reciting it regardless of K.

Method, per held-out apex:
  K, H          = deterministic half/half split of the apex's labels
  M_N           = model's top-N given K
  F_N           = top-N of a global frequency prior fit on groups_train.jsonl
  recall(M_N)   = |M_N ∩ H| / |H|                     what the model gets
  recall(F_N)   = |F_N ∩ H| / |H|                     what popularity alone gets
  M \ F         = the part of the model's list the prior would not have proposed
  lift          = |(M_N \ F_N) ∩ H| / |H|             recall that is conditioning-driven
  overlap       = |M_N ∩ F_N| / N                     how much of the list is just the prior

No DNS. Ground truth is the withheld real hostnames.
"""
import argparse, json, os, random, sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "subfury_v2"))
from predict import load_model, predict_labels                      # noqa: E402


def frequency_prior(train_jsonl, cap=None):
    """P(y): how many distinct organizations use each label."""
    c = Counter()
    with open(train_jsonl) as f:
        for i, line in enumerate(f):
            if cap and i >= cap:
                break
            for lab in set(json.loads(line)["labels"]):
                c[lab] += 1
    return [lab for lab, _ in c.most_common()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="data/groups_test.jsonl")
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--model", default="results/subfury_v2")
    ap.add_argument("--budgets", default="10,25,50,100,200")
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--max-apexes", type=int, default=1000)
    ap.add_argument("--min-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/research/decomposition.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    maxN = max(budgets)
    rng = random.Random(args.seed)

    print("building the global frequency prior from the training split…", flush=True)
    prior = frequency_prior(args.train_jsonl)
    print(f"  {len(prior):,} distinct labels; most common: {', '.join(prior[:8])}", flush=True)

    model, tok, device = load_model(args.model)

    groups = []
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["labels"]) >= args.min_labels:
                groups.append(rec)
    rng.shuffle(groups)
    groups = groups[: args.max_apexes]
    print(f"decomposing over {len(groups)} apexes, budgets {budgets}", flush=True)

    rows = []
    for i, rec in enumerate(groups):
        labels = rec["labels"][:]
        rng.shuffle(labels)
        half = len(labels) // 2
        K, H = labels[:half], set(labels[half:])
        known = set(K)

        preds = [p for p, _ in predict_labels(model, tok, device, K,
                                              topn=maxN, num_beams=args.num_beams)]
        # the prior must be given the same courtesy the model gets: skip what is known
        prior_list = [p for p in prior if p not in known][:maxN]

        row = {"apex": rec["apex"], "k": len(K), "h": len(H), "per_n": {}}
        for n in budgets:
            M, F = preds[:n], prior_list[:n]
            Ms, Fs = set(M), set(F)
            only_model = Ms - Fs
            row["per_n"][str(n)] = {
                "model": len(Ms & H) / len(H),
                "prior": len(Fs & H) / len(H),
                "union": len((Ms | Fs) & H) / len(H),
                "lift": len(only_model & H) / len(H),        # conditioning-driven recall
                "overlap": len(Ms & Fs) / max(len(Ms), 1),   # share of list that is prior
                "novel_frac": len(only_model) / max(len(Ms), 1),
            }
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(groups)}", flush=True)

    def mean(key, n):
        vals = [r["per_n"][str(n)][key] for r in rows]
        return sum(vals) / len(vals)

    summary = {"apexes": len(rows), "seed": args.seed, "budgets": budgets,
               "per_n": {str(n): {k: round(mean(k, n), 4) for k in
                                  ("model", "prior", "union", "lift", "overlap", "novel_frac")}
                         for n in budgets}}

    # does conditioning get stronger as the known set grows?  (RQ3)
    buckets = {"1-3": [], "4-7": [], "8-15": [], "16+": []}
    for r in rows:
        k = r["k"]
        b = "1-3" if k <= 3 else "4-7" if k <= 7 else "8-15" if k <= 15 else "16+"
        buckets[b].append(r)
    summary["by_known_size"] = {
        b: {"apexes": len(v),
            "model@100": round(sum(x["per_n"]["100"]["model"] for x in v) / len(v), 4),
            "prior@100": round(sum(x["per_n"]["100"]["prior"] for x in v) / len(v), 4),
            "lift@100": round(sum(x["per_n"]["100"]["lift"] for x in v) / len(v), 4)}
        for b, v in buckets.items() if v}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"summary": summary, "rows": rows}, open(args.out, "w"), indent=1)

    print("\n  N   model   prior   union    lift   overlap  novel")
    for n in budgets:
        s = summary["per_n"][str(n)]
        print(f"{n:>4}  {s['model']:.3f}  {s['prior']:.3f}  {s['union']:.3f}  "
              f"{s['lift']:.3f}   {s['overlap']:.3f}   {s['novel_frac']:.3f}")
    print("\nby size of the known set (budget 100):")
    for b, v in summary["by_known_size"].items():
        print(f"  |K| {b:>5}  n={v['apexes']:<4} model {v['model@100']:.3f}  "
              f"prior {v['prior@100']:.3f}  lift {v['lift@100']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
