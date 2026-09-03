"""Offline evaluation: recall@N vs a wordlist baseline at equal budget.

For each apex group in the held-out test split:
  - split its labels: half "seen" (model input), half "holdout"
  - model: beam-search N candidate labels from the seen half
  - baseline: first N labels of the n0kovo wordlist (frequency-ordered)
  - recall@N = |candidates ∩ holdout| / |holdout|

No DNS queries — ground truth is the held-out real hostnames.
"""

import argparse
import json
import random

from predict import load_model, predict_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="data/groups_test.jsonl")
    ap.add_argument("--wordlist", default="data/subdomains_tiny.txt")
    ap.add_argument("--model", default="results/subfury_v2")
    ap.add_argument("-n", "--topn", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--max-apexes", type=int, default=150)
    ap.add_argument("--min-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    model, tok, device = load_model(args.model)

    with open(args.wordlist) as f:
        baseline_all = [ln.strip().lower() for ln in f if ln.strip()]
    baseline_topn = baseline_all[: args.topn]

    groups = []
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["labels"]) >= args.min_labels:
                groups.append(rec)
    rng.shuffle(groups)
    groups = groups[: args.max_apexes]
    print(f"Evaluating on {len(groups)} apex groups, N={args.topn}")

    model_recalls, base_recalls = [], []
    for i, rec in enumerate(groups):
        labels = rec["labels"][:]
        rng.shuffle(labels)
        half = len(labels) // 2
        seen, holdout = labels[:half], set(labels[half:])

        preds = predict_labels(model, tok, device, seen, topn=args.topn,
                               num_beams=args.num_beams)
        pred_set = {p for p, _ in preds}
        m_rec = len(pred_set & holdout) / len(holdout)
        b_rec = len((set(baseline_topn) - set(seen)) & holdout) / len(holdout)
        model_recalls.append(m_rec)
        base_recalls.append(b_rec)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(groups)}  model={sum(model_recalls)/len(model_recalls):.3f} "
                  f"baseline={sum(base_recalls)/len(base_recalls):.3f}", flush=True)

    m = sum(model_recalls) / len(model_recalls)
    b = sum(base_recalls) / len(base_recalls)
    print(f"\n=== recall@{args.topn} over {len(groups)} apexes ===")
    print(f"SubFury v2 : {m:.3f}")
    print(f"n0kovo top-{args.topn}: {b:.3f}")
    print(f"relative improvement: {((m-b)/max(b,1e-9))*100:+.1f}%")


if __name__ == "__main__":
    main()
