"""Beam search ranks by summed log-probability, so shorter is always better.

On a real target this puts `s`, `a`, `e`, `i`, `t` at the top of the list and
spends the DNS budget on single characters. This sweeps the length penalty and a
minimum-label-length filter and measures what each does to recall, so the fix is
chosen by measurement rather than taste.

  alpha = 0    score = sum(logp)                     what ships today
  alpha > 0    score = sum(logp) / len(tokens)^alpha
  minlen       drop candidate labels shorter than this
"""
import argparse, json, os, random, statistics, sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "subfury_v2"))
from predict import LABEL_RE, load_model                            # noqa: E402


def generate(model, tok, device, known, topn, num_beams, alpha, max_new_tokens=16):
    sep, delim, end = (tok.token_to_id(t) for t in ("[SEP]", "[DELIM]", "[END]"))
    ids = []
    for i, lab in enumerate(sorted(known)[-model.cfg.block_size // 8:]):
        if i:
            ids.append(sep)
        ids.extend(tok.encode(lab).ids)
    ids.append(delim)
    ids = ids[-(model.cfg.block_size - max_new_tokens - 1):]
    prefix = torch.tensor(ids, device=device)
    specials = [tok.token_to_id(t) for t in ("[PAD]", "[SEP]", "[DELIM]")]
    res = model.beam_search(prefix, end_id=end, num_beams=num_beams, topn=topn,
                            max_new_tokens=max_new_tokens, length_penalty=alpha,
                            banned_first=specials)
    out, seen = [], set(known)
    for toks, score in res:
        lab = tok.decode(toks).replace(" ", "").lower()
        if lab and lab not in seen and LABEL_RE.match(lab):
            seen.add(lab)
            out.append(lab)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-jsonl", default="data/groups_test.jsonl")
    ap.add_argument("--model", default="results/subfury_v2")
    ap.add_argument("--alphas", default="0,0.6,0.8,1.0")
    ap.add_argument("--minlens", default="1,2,3")
    ap.add_argument("--budgets", default="10,25,50,100")
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--max-apexes", type=int, default=120)
    ap.add_argument("--min-labels", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="results/research/lengthbias.json")
    args = ap.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    minlens = [int(m) for m in args.minlens.split(",")]
    budgets = [int(b) for b in args.budgets.split(",")]
    maxN = max(budgets)
    rng = random.Random(args.seed)
    model, tok, device = load_model(args.model)

    groups = []
    with open(args.test_jsonl) as f:
        for line in f:
            rec = json.loads(line)
            if len(rec["labels"]) >= args.min_labels:
                groups.append(rec)
    rng.shuffle(groups)
    groups = groups[: args.max_apexes]
    print(f"{len(groups)} apexes · alphas {alphas} · minlens {minlens}", flush=True)

    acc = {(a, m): {n: [] for n in budgets} for a in alphas for m in minlens}
    shortshare = {a: [] for a in alphas}
    for i, rec in enumerate(groups):
        labels = rec["labels"][:]
        rng.shuffle(labels)
        half = len(labels) // 2
        K, H = labels[:half], set(labels[half:])
        for a in alphas:
            # ask for extra so a minlen filter still has a full budget to give
            preds = generate(model, tok, device, K, maxN * 2, args.num_beams, a)
            shortshare[a].append(sum(1 for p in preds[:50] if len(p) <= 2) / max(len(preds[:50]), 1))
            for m in minlens:
                kept = [p for p in preds if len(p) >= m][:maxN]
                for n in budgets:
                    acc[(a, m)][n].append(len(set(kept[:n]) & H) / len(H))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(groups)}", flush=True)

    table = {}
    for (a, m), v in acc.items():
        table[f"alpha{a}_minlen{m}"] = {str(n): round(statistics.mean(v[n]), 4) for n in budgets}
    summary = {"apexes": len(groups), "budgets": budgets, "table": table,
               "short_label_share_top50": {str(a): round(statistics.mean(v), 3)
                                           for a, v in shortshare.items()}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=1)

    print("\n  alpha  minlen" + "".join(f"{('@'+str(n)):>9}" for n in budgets))
    print("  " + "-" * (14 + 9 * len(budgets)))
    best = None
    for a in alphas:
        for m in minlens:
            row = table[f"alpha{a}_minlen{m}"]
            mark = ""
            if best is None or row[str(budgets[1])] > best[1]:
                best = (f"alpha={a} minlen={m}", row[str(budgets[1])]); mark = ""
            print(f"  {a:<6} {m:<6}" + "".join(f"{row[str(n)]:>9.3f}" for n in budgets))
    print("\n  share of top-50 that is 1-2 characters:")
    for a in alphas:
        print(f"    alpha={a}: {summary['short_label_share_top50'][str(a)]:.1%}")
    print(f"\n  best at @{budgets[1]}: {best[0]} → {best[1]:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
