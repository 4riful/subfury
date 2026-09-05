"""Build a TEMPORAL evaluation set from Certificate Transparency observations.

Reads the records ct_fetch.py emits (label -> earliest certificate not_before)
and splits each apex at a date T:

    known  = labels whose first CT observation is <= T
    future = labels whose first CT observation is >  T

An apex is kept only if it has at least --min-known known labels and at least
--min-future future labels. This is a genuine before/after split: nothing in
`future` was visible in CT at time T, so a model conditioned on `known` cannot
have seen it in the conditioning set.

Usage:
    # survival table across candidate split dates
    python research/data/build_temporal.py --in research/data/ct_observations.jsonl --sweep

    # write the set for one T
    python research/data/build_temporal.py --in research/data/ct_observations.jsonl \
        --T 2024-01-01 --out research/data/temporal.jsonl
"""

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_SWEEP = ["2021-01-01", "2022-01-01", "2023-01-01", "2023-07-01",
                 "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
                 "2026-01-01"]


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def _cut(T):
    """A date like '2024-01-01' becomes the instant that day begins, so
    timestamps sort against it lexicographically."""
    T = T.strip()
    return T if "T" in T else T + "T00:00:00"


def split_record(rec, T, min_known=5, min_future=2):
    cut = _cut(T)
    known, future = [], []
    for o in rec.get("observations", []):
        ts = o.get("first_seen")
        if not ts:
            continue
        (known if ts <= cut else future).append(o["label"])
    known.sort()
    future.sort()
    ok = len(known) >= min_known and len(future) >= min_future
    item = {"apex": rec["apex"], "known": known, "future": future, "T": T}
    # crt.sh silently cuts the full-history query short on very large apexes; the
    # flag rides along so a downstream eval can drop those apexes if it wants.
    if (rec.get("stats") or {}).get("truncated_history"):
        item["truncated_history"] = True
    return item, ok


def survival(recs, dates, min_known=5, min_future=2):
    rows = []
    for T in dates:
        kept = []
        for rec in recs:
            item, ok = split_record(rec, T, min_known, min_future)
            if ok:
                kept.append(item)
        rows.append({
            "T": T,
            "apexes": len(kept),
            "pct": 100.0 * len(kept) / max(1, len(recs)),
            "median_known": statistics.median([len(k["known"]) for k in kept]) if kept else 0,
            "median_future": statistics.median([len(k["future"]) for k in kept]) if kept else 0,
            "total_future": sum(len(k["future"]) for k in kept),
        })
    return rows


def print_survival(rows, n_recs, min_known, min_future):
    print("survival across candidate split dates "
          "(%d apexes in, keeping >=%d known and >=%d future)"
          % (n_recs, min_known, min_future))
    print("%-12s %8s %7s %14s %15s %14s"
          % ("T", "apexes", "%", "median |known|", "median |future|", "total future"))
    for r in rows:
        print("%-12s %8d %6.1f%% %14.0f %15.0f %14d"
              % (r["T"], r["apexes"], r["pct"], r["median_known"],
                 r["median_future"], r["total_future"]))


def describe(items):
    if not items:
        return
    k = sorted(len(i["known"]) for i in items)
    f = sorted(len(i["future"]) for i in items)

    def q(v, p):
        return v[min(len(v) - 1, int(p * (len(v) - 1)))]
    print("\n|known|  min=%d p25=%d median=%d p75=%d max=%d mean=%.1f total=%d"
          % (k[0], q(k, .25), q(k, .5), q(k, .75), k[-1], sum(k) / len(k), sum(k)))
    print("|future| min=%d p25=%d median=%d p75=%d max=%d mean=%.1f total=%d"
          % (f[0], q(f, .25), q(f, .5), q(f, .75), f[-1], sum(f) / len(f), sum(f)))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Temporal split from CT observations.")
    ap.add_argument("--in", dest="inp",
                    default=os.path.join(HERE, "ct_observations.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "temporal.jsonl"))
    ap.add_argument("--T", help="split date, e.g. 2024-01-01")
    ap.add_argument("--sweep", action="store_true",
                    help="print the survival table and write nothing")
    ap.add_argument("--dates", help="comma-separated candidate dates for the sweep")
    ap.add_argument("--min-known", type=int, default=5)
    ap.add_argument("--min-future", type=int, default=2)
    ap.add_argument("--exclude-truncated", action="store_true",
                    help="drop apexes whose crt.sh history came back truncated")
    args = ap.parse_args(argv)

    recs = load_records(args.inp)
    dates = args.dates.split(",") if args.dates else list(DEFAULT_SWEEP)
    if args.T and args.T not in dates:
        dates.append(args.T)
    dates.sort()

    rows = survival(recs, dates, args.min_known, args.min_future)
    print_survival(rows, len(recs), args.min_known, args.min_future)

    if args.sweep or not args.T:
        if not args.T:
            print("\n(no --T given: nothing written. Pick a T from the table above.)")
        return 0

    kept = []
    for r in recs:
        item, ok = split_record(r, args.T, args.min_known, args.min_future)
        if ok and not (args.exclude_truncated and item.get("truncated_history")):
            kept.append(item)
    with open(args.out, "w") as f:
        for item in kept:
            f.write(json.dumps(item) + "\n")
    print("\nT=%s: %d/%d apexes survive -> %s" % (args.T, len(kept), len(recs), args.out))
    describe(kept)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
