"""SubFury v2 inference: known subdomains in -> validated new subdomains out.

    python predict.py example.com --known known.txt -n 200

Pipeline:
  1. Encode known labels: k1 [SEP] k2 ... [DELIM]
  2. Beam search the N most likely new labels (deterministic, like subwiz)
  3. Filter: valid DNS labels, not already known
  4. Resolve concurrently via DNS (unless --no-resolve)
  5. Recursion: resolved hits are added to the known set and inference
     re-runs, up to --max-recursion times (replaces v1's "RL loop")

Only run resolution against domains you are authorized to test.
"""

import argparse
import concurrent.futures
import re

import torch
from tokenizers import Tokenizer

from model import GPTConfig, SubFuryGPT

LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")

try:
    import dns.resolver
except ImportError:
    dns = None


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


def resolve_all(labels, domain, workers=64):
    if dns is None:
        raise SystemExit("dnspython not installed")
    resolver = dns.resolver.Resolver()
    resolver.lifetime = 3.0

    def one(label):
        fqdn = f"{label}.{domain}"
        try:
            ans = resolver.resolve(fqdn, "A")
            return label, str(ans[0])
        except Exception:
            return label, None

    hits = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for label, ip in pool.map(one, labels):
            if ip:
                hits[label] = ip
    return hits


def run(domain, known, topn=100, resolve=True, max_recursion=3,
        model_dir="results/subfury_v2", num_beams=64, quiet=False):
    model, tok, device = load_model(model_dir)
    known = set(known)
    all_hits = {}

    for depth in range(max_recursion):
        preds = predict_labels(model, tok, device, known, topn=topn,
                               num_beams=num_beams)
        labels = [p for p, _ in preds]
        if not quiet:
            print(f"[round {depth+1}] {len(labels)} candidates")
        if not resolve:
            return labels, {}
        hits = resolve_all(labels, domain)
        new = {k: v for k, v in hits.items() if k not in known}
        if not quiet:
            for k, v in sorted(new.items()):
                print(f"  [+] {k}.{domain} -> {v}")
            print(f"[round {depth+1}] {len(new)} new resolved "
                  f"({len(hits)}/{len(labels)} hit rate)")
        all_hits.update(new)
        if not new:
            break
        known |= set(new)
    return sorted(all_hits), all_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--known", required=True, help="file: one known label or FQDN per line")
    ap.add_argument("-n", "--topn", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=64)
    ap.add_argument("--no-resolve", action="store_true")
    ap.add_argument("--max-recursion", type=int, default=3)
    ap.add_argument("--model", default="results/subfury_v2")
    args = ap.parse_args()

    with open(args.known) as f:
        known = []
        for ln in f:
            ln = ln.strip().lower()
            if not ln:
                continue
            if ln.endswith("." + args.domain):
                ln = ln[: -len(args.domain) - 1]
            known.append(ln)

    labels, hits = run(args.domain, known, topn=args.topn,
                       resolve=not args.no_resolve,
                       max_recursion=args.max_recursion,
                       model_dir=args.model, num_beams=args.num_beams)
    if args.no_resolve:
        print("\n".join(labels))
    else:
        print(f"\nTotal new validated subdomains: {len(hits)}")


if __name__ == "__main__":
    main()
