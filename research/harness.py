"""Shared evaluation harness for SubFury subdomain-prediction methods.

Every method — neural, statistical, wordlist — implements one interface:

    class Ranker:
        name: str
        def rank(self, apex: str, known: list[str], n: int) -> list[str]: ...

and is scored here, on exactly the same data, with exactly the same split.

Protocol (matches subfury_v2/evaluate.py, made deterministic and reusable):
  * load data/groups_test.jsonl, keep apexes with >= MIN_LABELS labels
  * split each apex's labels into K (shown to the ranker) and H (withheld
    ground truth), ~50/50, deterministically seeded *per apex* so that the
    split does not depend on iteration order, apex ordering, or which
    rankers are being run
  * every ranker gets a budget of N candidates and is scored by
    recall@N = |top-N ∩ H| / |H| for N in {10,25,50,100,200},
    plus MAP and the number of apexes with at least one hit
  * mean recall is reported with a bootstrap 95% CI over apexes

Never performs DNS or any other network I/O: the ground truth is the
withheld half of hostnames that were already observed in Common Crawl.

Prefix consistency
------------------
By default the harness calls rank(apex, known, N_max) once per apex and
scores prefixes of the returned list (top-10 = first 10 items, etc.).
That is the correct semantics for any ranker that produces a single score
ordering. A method whose output is *not* a prefix of its larger-N output
(e.g. a beam search whose beam width is tied to n) can set the class
attribute ``prefix_consistent = False``; the harness then calls rank()
separately for each N.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEST = os.path.join(REPO_ROOT, "data", "groups_test.jsonl")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "results", "research")

DEFAULT_SEED = 1337
DEFAULT_NS = (10, 25, 50, 100, 200)
MIN_LABELS = 6
BOOTSTRAP_ROUNDS = 2000


class Ranker(Protocol):
    """The one interface every SubFury method implements."""

    name: str

    def rank(self, apex: str, known: list[str], n: int) -> list[str]:
        """Return up to `n` candidate labels, best first, excluding `known`."""
        ...


# --------------------------------------------------------------------------
# data + split
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Case:
    apex: str
    known: tuple[str, ...]     # K, shown to the ranker
    holdout: frozenset[str]    # H, withheld ground truth


def load_groups(path: str = DEFAULT_TEST, min_labels: int = MIN_LABELS) -> list[dict]:
    groups = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            labels = sorted({str(x).strip().lower() for x in rec["labels"] if str(x).strip()})
            if len(labels) >= min_labels:
                groups.append({"apex": rec["apex"].strip().lower(), "labels": labels})
    groups.sort(key=lambda r: r["apex"])
    return groups


def _apex_rng(apex: str, seed: int) -> np.random.Generator:
    """A generator whose stream depends only on (seed, apex).

    Using a per-apex seed rather than one sequential RNG makes the K/H split
    invariant to apex ordering, to filtering, and to how many rankers run.
    """
    h = hashlib.sha256(f"{seed}:{apex}".encode()).digest()[:8]
    return np.random.default_rng(int.from_bytes(h, "big"))


def make_cases(groups: Sequence[dict], seed: int = DEFAULT_SEED) -> list[Case]:
    """Deterministic ~50/50 K/H split. Odd counts give the extra label to H,
    matching subfury_v2/evaluate.py (half = len // 2 goes to K)."""
    cases = []
    for rec in groups:
        labels = list(rec["labels"])
        rng = _apex_rng(rec["apex"], seed)
        perm = rng.permutation(len(labels))
        shuffled = [labels[i] for i in perm]
        half = len(shuffled) // 2
        cases.append(Case(apex=rec["apex"],
                          known=tuple(shuffled[:half]),
                          holdout=frozenset(shuffled[half:])))
    return cases


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _clean(preds: Iterable[str], known: Sequence[str], n: int) -> list[str]:
    """Lowercase, drop blanks/known/duplicates, truncate to n. Rankers are
    supposed to do this themselves; doing it here too means a sloppy ranker
    cannot accidentally buy recall by re-emitting a known label."""
    known_set = set(known)
    seen: set[str] = set()
    out: list[str] = []
    for p in preds:
        p = str(p).strip().lower()
        if not p or p in known_set or p in seen:
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= n:
            break
    return out


def _average_precision(ranked: Sequence[str], holdout: frozenset[str]) -> float:
    """AP truncated at len(ranked); denominator is |H| so that a ranker is
    penalised for relevant items it never surfaces inside the budget."""
    if not holdout:
        return 0.0
    hits = 0
    acc = 0.0
    for i, cand in enumerate(ranked, start=1):
        if cand in holdout:
            hits += 1
            acc += hits / i
    return acc / len(holdout)


@dataclass
class RankerResult:
    name: str
    ns: tuple[int, ...]
    recall: dict[int, np.ndarray]        # per-apex recall vectors
    hits: dict[int, np.ndarray]          # per-apex hit counts
    ap: np.ndarray                       # per-apex average precision (at max N)
    apexes: list[str] = field(default_factory=list)
    h_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0
    notes: dict = field(default_factory=dict)


def _bootstrap_ci(values: np.ndarray, seed: int, rounds: int = BOOTSTRAP_ROUNDS,
                  alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling apexes."""
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(rounds, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def evaluate_ranker(ranker: Ranker, cases: Sequence[Case],
                    ns: Sequence[int] = DEFAULT_NS,
                    progress_every: int = 0) -> RankerResult:
    ns = tuple(sorted(ns))
    n_max = ns[-1]
    prefix_consistent = getattr(ranker, "prefix_consistent", True)
    name = getattr(ranker, "name", type(ranker).__name__)

    recall = {n: np.zeros(len(cases)) for n in ns}
    hits = {n: np.zeros(len(cases)) for n in ns}
    ap = np.zeros(len(cases))
    short = 0
    t0 = time.time()

    for i, case in enumerate(cases):
        known = list(case.known)
        if prefix_consistent:
            full = _clean(ranker.rank(case.apex, known, n_max), known, n_max)
            per_n = {n: full[:n] for n in ns}
        else:
            per_n = {n: _clean(ranker.rank(case.apex, known, n), known, n) for n in ns}
            full = per_n[n_max]
        if len(full) < n_max:
            short += 1
        for n in ns:
            hit = len(set(per_n[n]) & case.holdout)
            hits[n][i] = hit
            recall[n][i] = hit / len(case.holdout)
        ap[i] = _average_precision(full, case.holdout)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [{name}] {i+1}/{len(cases)} "
                  f"recall@{n_max}={recall[n_max][:i+1].mean():.3f}", flush=True)

    return RankerResult(name=name, ns=ns, recall=recall, hits=hits, ap=ap,
                        apexes=[c.apex for c in cases],
                        h_sizes=np.array([len(c.holdout) for c in cases], dtype=float),
                        seconds=time.time() - t0,
                        notes={"apexes_short_of_budget": short})


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def paired_diff_ci(a: np.ndarray, b: np.ndarray, seed: int,
                   rounds: int = BOOTSTRAP_ROUNDS) -> dict:
    """Bootstrap CI of the *paired* per-apex difference a - b.

    Stronger than comparing two marginal CIs: both methods see the same
    apexes, so the paired difference removes between-apex variance. Two
    overlapping marginal CIs can still have a paired difference that
    excludes zero.
    """
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(rounds, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    p_le = float((means <= 0).mean())
    return {"mean_diff": float(d.mean()), "ci95": [lo, hi],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "boot_p_two_sided": float(2 * min(p_le, 1 - p_le))}


def summarize(result: RankerResult, seed: int, mask: np.ndarray | None = None,
              name: str | None = None) -> dict:
    """Summary over all apexes, or over the subset selected by `mask`.

    Reports macro-averaged recall (mean of per-apex recall, the headline
    metric) and micro-averaged recall (total hits / total held-out labels,
    which weights apexes by how many labels they actually withhold).
    """
    sel = slice(None) if mask is None else np.asarray(mask, dtype=bool)
    ap = result.ap[sel]
    h_sizes = result.h_sizes[sel] if len(result.h_sizes) else np.zeros(len(ap))
    out = {
        "name": name or result.name,
        "apexes": int(len(ap)),
        "held_out_labels": int(h_sizes.sum()),
        "seconds": round(result.seconds, 2),
        "map": float(ap.mean()),
        "map_ci95": list(_bootstrap_ci(ap, seed + 7919)),
        "recall": {},
        "notes": result.notes,
    }
    for n in result.ns:
        r = result.recall[n][sel]
        h = result.hits[n][sel]
        lo, hi = _bootstrap_ci(r, seed + n)
        out["recall"][str(n)] = {
            "mean": float(r.mean()),
            "ci95": [lo, hi],
            "micro": float(h.sum() / h_sizes.sum()) if h_sizes.sum() else 0.0,
            "apexes_with_hit": int((h > 0).sum()),
            "total_hits": int(h.sum()),
        }
    return out


def format_table(summaries: Sequence[dict], ns: Sequence[int]) -> str:
    width = max(len(s["name"]) for s in summaries) + 2
    lines = []
    head = f"{'method':<{width}}" + "".join(f"{'recall@'+str(n):>22}" for n in ns)
    head += f"{'MAP':>10}{'apex hit@max':>14}"
    lines.append(head)
    lines.append("-" * len(head))
    for s in summaries:
        row = f"{s['name']:<{width}}"
        for n in ns:
            e = s["recall"][str(n)]
            row += f"{e['mean']:>10.3f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}]"
        n_max = max(ns)
        row += f"{s['map']:>10.4f}"
        row += f"{s['recall'][str(n_max)]['apexes_with_hit']:>10d}/{s['apexes']}"
        lines.append(row)
    return "\n".join(lines)


def run_evaluation(rankers: Sequence[Ranker],
                   test_jsonl: str = DEFAULT_TEST,
                   ns: Sequence[int] = DEFAULT_NS,
                   seed: int = DEFAULT_SEED,
                   min_labels: int = MIN_LABELS,
                   out_path: str | None = None,
                   progress_every: int = 0,
                   extra_meta: dict | None = None,
                   return_results: bool = False):
    """Score every ranker on the same cases; print a table; write JSON."""
    ns = tuple(sorted(ns))
    groups = load_groups(test_jsonl, min_labels)
    cases = make_cases(groups, seed=seed)
    print(f"harness: {len(cases)} apexes (>= {min_labels} labels) from "
          f"{os.path.relpath(test_jsonl, REPO_ROOT)}, seed={seed}")
    k = np.array([len(c.known) for c in cases])
    h = np.array([len(c.holdout) for c in cases])
    print(f"harness: |K| mean {k.mean():.1f}  |H| mean {h.mean():.1f}  "
          f"total held-out labels {int(h.sum())}\n")

    summaries, results = [], []
    for ranker in rankers:
        res = evaluate_ranker(ranker, cases, ns=ns, progress_every=progress_every)
        results.append(res)
        summaries.append(summarize(res, seed))
        print(f"  scored {res.name} in {res.seconds:.1f}s", flush=True)

    table = format_table(summaries, ns)
    print("\n" + table + "\n")

    doc = {
        "harness": "research/harness.py",
        "protocol": ("50/50 K/H split of each test apex's labels; ranker sees K "
                     "and a budget of N; recall@N against H. No DNS/network."),
        "seed": seed,
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
        "ci": "percentile bootstrap 95% over apexes",
        "test_jsonl": os.path.relpath(test_jsonl, REPO_ROOT),
        "min_labels": min_labels,
        "apexes": len(cases),
        "ns": list(ns),
        "held_out_labels": int(h.sum()),
        "mean_known": float(k.mean()),
        "mean_holdout": float(h.mean()),
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": summaries,
        "table": table,
    }
    if extra_meta:
        doc.update(extra_meta)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {out_path}")
    if return_results:
        return doc, cases, results
    return doc
