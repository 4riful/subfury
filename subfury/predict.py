"""SubFury beam-search inference: known subdomains in -> validated new subdomains out.

    python predict.py example.com --known known.txt -n 200

Pipeline:
  1. Encode known labels: k1 [SEP] k2 ... [DELIM]
  2. Beam search the N most likely new labels (deterministic, like subwiz)
  3. Filter: valid DNS labels, not already known
  4. Resolve concurrently via DNS (unless --no-resolve)
  5. Recursion: resolved hits are added to the known set and inference
     re-runs, up to --max-recursion times

Only run resolution against domains you are authorized to test.
"""

import argparse
import json
import math
import re

import torch
from tokenizers import Tokenizer

from model import GPTConfig, SubFuryGPT
from dns_validation import (QueryBudget, normalize_domain,
                            preferred_answer, resolve_with_evidence)

LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")

def load_model(model_dir, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"{model_dir}/best.pt", map_location=device)
    cfg = GPTConfig(**ckpt["config"])
    model = SubFuryGPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    return model, tok, device


def predict_labels(model, tok, device, known, topn=100, num_beams=64,
                   max_new_tokens=16):
    """Beam-search `topn` new labels given known labels (no DNS here)."""
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
    results = model.beam_search(prefix, end_id=end, num_beams=num_beams,
                                topn=topn, max_new_tokens=max_new_tokens,
                                banned_first=specials)
    known_set = set(known)
    out = []
    for toks, score in results:
        label = tok.decode(toks).replace(" ", "").lower()
        if label and label not in known_set and LABEL_RE.match(label):
            known_set.add(label)  # dedup across beams
            out.append((label, score))
        if len(out) >= topn:
            break
    return out


def resolve_all(labels, domain, workers=16, query_budget=None, max_qps=50.0,
                wildcard_probes=2, budget=None, detailed=False,
                wildcard_cache=None, answer_cache=None):
    """Resolve labels with a global cap and wildcard controls.

    The legacy mapping return remains available for callers that only need
    non-wildcard resolutions. New code should request the detailed report.
    """
    labels = list(labels)
    budget = budget or QueryBudget(query_budget or max(1, len(labels) * 3 + 6),
                                   max_qps=max_qps)
    report = resolve_with_evidence(labels, domain, budget, workers=workers,
                                   wildcard_probes=wildcard_probes,
                                   wildcard_cache=wildcard_cache,
                                   answer_cache=answer_cache)
    if detailed:
        return report
    return {label: preferred_answer(item)[1]
            for label, item in report["results"].items()
            if item["status"] == "resolved" and preferred_answer(item)[1]}


def run(domain, known, topn=100, resolve=True, max_recursion=3,
        model_dir="results/subfury", num_beams=64, quiet=False,
        query_budget=1000, max_qps=50.0, wildcard_probes=2,
        authorized=False, evidence_out=None):
    domain = normalize_domain(domain)
    if resolve and not authorized:
        raise ValueError("DNS resolution requires explicit authorization acknowledgement")
    model, tok, device = load_model(model_dir)
    known = set(known)
    all_hits = {}
    evidence = []
    wildcard_cache = {}
    answer_cache = {}
    budget = QueryBudget(query_budget, max_qps=max_qps) if resolve else None

    for depth in range(max_recursion):
        preds = predict_labels(model, tok, device, known, topn=topn,
                               num_beams=num_beams)
        labels = [p for p, _ in preds]
        if not quiet:
            print(f"[round {depth+1}] {len(labels)} candidates")
        if not resolve:
            return labels, {}
        report = resolve_all(labels, domain, wildcard_probes=wildcard_probes,
                             budget=budget, detailed=True,
                             wildcard_cache=wildcard_cache,
                             answer_cache=answer_cache)
        evidence.append(report)
        hits = {label: preferred_answer(item)[1]
                for label, item in report["results"].items()
                if item["status"] == "resolved" and preferred_answer(item)[1]}
        new = {k: v for k, v in hits.items() if k not in known}
        if not quiet:
            for k, v in sorted(new.items()):
                print(f"  [+] {k}.{domain} -> {v}")
            print(f"[round {depth+1}] {len(new)} new resolved "
                  f"({len(hits)}/{len(labels)} hit rate, "
                  f"{budget.used}/{budget.limit} DNS queries)")
            wildcards = report["counts"].get("probable_wildcard", 0)
            if wildcards:
                print(f"[round {depth+1}] excluded {wildcards} probable wildcard responses")
        all_hits.update(new)
        if not new or budget.remaining == 0:
            break
        known |= set(new)
    if evidence_out:
        with open(evidence_out, "w") as f:
            json.dump({"domain": domain, "query_budget": budget.limit,
                       "queries_used": budget.used, "rounds": evidence}, f, indent=2)
    return sorted(all_hits), all_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--known", required=True, help="file: one known label or FQDN per line")
    ap.add_argument("-n", "--topn", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--no-resolve", action="store_true")
    ap.add_argument("--max-recursion", type=int, default=3)
    ap.add_argument("--query-budget", type=int, default=1000,
                    help="hard cap on all DNS questions, including wildcard controls")
    ap.add_argument("--max-qps", type=float, default=50.0,
                    help="maximum DNS query start rate")
    ap.add_argument("--wildcard-probes", type=int, default=2)
    ap.add_argument("--authorized", action="store_true",
                    help="acknowledge that the domain is authorized and in scope")
    ap.add_argument("--evidence-out",
                    help="write DNS questions, outcomes, and wildcard controls as JSON")
    ap.add_argument("--model", default="results/subfury")
    args = ap.parse_args()
    if not args.no_resolve and not args.authorized:
        ap.error("--authorized is required when DNS resolution is enabled")
    if not math.isfinite(args.max_qps) or args.max_qps < 0.1:
        ap.error("--max-qps must be finite and at least 0.1")
    if not 1 <= args.wildcard_probes <= 10:
        ap.error("--wildcard-probes must be between 1 and 10")

    try:
        domain = normalize_domain(args.domain)
    except ValueError as exc:
        ap.error(str(exc))

    with open(args.known) as f:
        known = []
        for ln in f:
            ln = ln.strip().lower()
            if not ln:
                continue
            if ln.endswith("." + domain):
                ln = ln[: -len(domain) - 1]
            known.append(ln)

    labels, hits = run(domain, known, topn=args.topn,
                       resolve=not args.no_resolve,
                       max_recursion=args.max_recursion,
                       model_dir=args.model, num_beams=args.num_beams,
                       query_budget=args.query_budget, max_qps=args.max_qps,
                       wildcard_probes=args.wildcard_probes,
                       authorized=args.authorized,
                       evidence_out=args.evidence_out)
    if args.no_resolve:
        print("\n".join(labels))
    else:
        print(f"\nTotal new non-wildcard DNS resolutions: {len(hits)}")


if __name__ == "__main__":
    main()
