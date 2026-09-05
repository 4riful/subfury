"""Trace one prediction end-to-end, exposing the internals at each stage.

    python subfury/explain.py --known api,dev,staging

Shows: tokenization -> conditioned prefix -> beam search with probabilities
-> filtering. Purely illustrative; no DNS is performed.
"""

import argparse
import math

import torch

from predict import LABEL_RE, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", default="api,dev,staging")
    ap.add_argument("--model", default="results/subfury")
    ap.add_argument("-n", "--topn", type=int, default=10)
    args = ap.parse_args()

    known = [k.strip() for k in args.known.split(",") if k.strip()]
    model, tok, device = load_model(args.model)
    sep, delim, end = (tok.token_to_id(t) for t in ("[SEP]", "[DELIM]", "[END]"))

    print("=" * 66)
    print("STAGE 1 — known subdomains (what you already discovered)")
    print("=" * 66)
    print(f"  {known}")

    print()
    print("=" * 66)
    print("STAGE 2 — domain-specific BPE tokenization")
    print("=" * 66)
    for lab in known:
        e = tok.encode(lab)
        print(f"  {lab:12} -> {e.tokens}  ids={e.ids}")

    print()
    print("=" * 66)
    print("STAGE 3 — conditioned prefix fed to the model")
    print("=" * 66)
    ids = []
    for i, lab in enumerate(known):
        if i:
            ids.append(sep)
        ids.extend(tok.encode(lab).ids)
    ids.append(delim)
    pretty = []
    for t in ids:
        if t == sep:
            pretty.append("[SEP]")
        elif t == delim:
            pretty.append("[DELIM]")
        else:
            pretty.append(tok.id_to_token(t))
    print(f"  {' '.join(pretty)}")
    print(f"  ({len(ids)} tokens; model must now continue past [DELIM])")

    print()
    print("=" * 66)
    print("STAGE 4 — single-step next-token distribution after [DELIM]")
    print("=" * 66)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        logits, _ = model(x)
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_p, top_i = probs.topk(12)
    for p, i in zip(top_p.tolist(), top_i.tolist()):
        bar = "#" * max(1, int(p * 120))
        print(f"  {tok.id_to_token(i):>14}  {p:6.3%}  {bar}")

    print()
    print("=" * 66)
    print("STAGE 5 — beam search: full label sequences, ranked")
    print("=" * 66)
    specials = [tok.token_to_id(t) for t in ("[PAD]", "[SEP]", "[DELIM]")]
    prefix = torch.tensor(ids, device=device)
    results = model.beam_search(prefix, end_id=end, num_beams=64,
                                topn=args.topn, max_new_tokens=16,
                                banned_first=specials)
    known_set = set(known)
    shown = 0
    print(f"  {'rank':>4}  {'label':<18} {'logprob':>9}  {'prob':>8}  status")
    for toks, score in results:
        label = tok.decode(toks).replace(" ", "").lower()
        if not label:
            continue
        if label in known_set:
            status = "skip (already known)"
        elif not LABEL_RE.match(label):
            status = "skip (invalid DNS label)"
        else:
            status = "-> DNS queue"
            known_set.add(label)
        shown += 1
        print(f"  {shown:>4}  {label:<18} {score:9.3f}  {math.exp(score):8.4f}  {status}")
        if shown >= args.topn:
            break

    print()
    print("=" * 66)
    print("STAGE 6 — what happens next (predict.py, not run here)")
    print("=" * 66)
    print("  queued labels -> concurrent DNS A-record lookups")
    print("  resolved hits -> appended to the known set -> back to STAGE 3")
    print("  (recursion, up to --max-recursion rounds)")


if __name__ == "__main__":
    main()
