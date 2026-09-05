"""Fetch Certificate Transparency records for an apex, with issuance timestamps.

PASSIVE ONLY. This module never resolves DNS and never contacts the target
domain. It reads public CT aggregators (crt.sh, optionally CertSpotter) and
nothing else.

Emits, per apex:

    {"apex": "example.com",
     "observations": [{"label": "api", "first_seen": "2023-04-11T09:12:03"}, ...],
     "stats": {...}}

`first_seen` is the EARLIEST certificate not_before seen for that hostname, so
the record is a lower bound on when the host became publicly visible in CT.

Raw responses are cached to research/data/cache/<apex>.json, so re-runs are free.

Usage:
    python research/data/ct_fetch.py --apex example.com
    python research/data/ct_fetch.py --apexes-from data/groups_test.jsonl \
        --limit 40 --out research/data/ct_observations.jsonl

The HTTP layer (_http with retry, _loads for crt.sh's two response shapes, the
crt.sh/certspotter query shapes, the label grammar) is lifted from
webui/app.py's passive-seed implementation rather than rewritten.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE_DIR = os.path.join(HERE, "cache")

# same grammar the model's label vocabulary uses (subfury_v2/predict.py)
LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")

UA = "subfury-research/0.1 (+temporal CT eval; passive only)"


# ── HTTP (copied from webui/app.py) ─────────────────────────────────────────

def _http(url, headers=None, timeout=60, retries=2, backoff=4.0):
    """These archives 502 and 503 as a matter of course; retries with a growing
    pause turn a large share of those into answers."""
    hdrs = {"User-Agent": UA, "Accept": "application/json, text/plain, */*"}
    hdrs.update(headers or {})
    last = None
    for attempt in range(retries + 1):
        try:
            rq = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                raise                      # 404 or a hard rejection: do not retry
        except Exception as exc:           # noqa: BLE001 - urllib raises many shapes
            last = exc
        if attempt < retries:
            time.sleep(backoff * (attempt + 1))
    raise last


def _loads(body):
    """crt.sh has historically returned both a JSON array and a bare stream of
    concatenated objects. Accept either."""
    try:
        return json.loads(body)
    except ValueError:
        return json.loads("[" + body.strip().replace("}\n{", "},{").replace("}{", "},{") + "]")


# ── sources ─────────────────────────────────────────────────────────────────

def _crtsh(domain, timeout, exclude_expired):
    q = urllib.parse.quote("%." + domain)
    url = "https://crt.sh/?q=%s&output=json" % q
    if exclude_expired:
        url += "&exclude=expired"
    rows = _loads(_http(url, timeout=timeout))
    return [{"name_value": r.get("name_value"),
             "common_name": r.get("common_name"),
             "not_before": r.get("not_before")} for r in rows]


def src_crtsh(domain, timeout=60):
    """Full issuance history (no exclude=expired — expired certs are exactly the
    old observations a temporal split needs). Rows carry not_before."""
    return _crtsh(domain, timeout, exclude_expired=False)


def src_crtsh_unexpired(domain, timeout=45):
    """Currently-valid certs only. Roughly 10x smaller and always answers, and
    it is the guard against crt.sh silently truncating the full-history query on
    a large apex (see stats['truncated_history'])."""
    return _crtsh(domain, timeout, exclude_expired=True)


def src_certspotter(domain, timeout=45):
    """Fallback. Unauthenticated CertSpotter rate-limits hard (429 after a
    couple of calls per hour) and only returns a recent window, so it is used
    only when crt.sh fails outright."""
    rows = _loads(_http("https://api.certspotter.com/v1/issuances?domain=" + domain
                        + "&include_subdomains=true&expand=dns_names", timeout=timeout))
    return [{"name_value": "\n".join(r.get("dns_names") or []),
             "common_name": "",
             "not_before": r.get("not_before")} for r in rows]


SOURCES = {"crtsh": src_crtsh, "certspotter": src_certspotter}
CRTSH_GUARD = src_crtsh_unexpired


# ── record building ─────────────────────────────────────────────────────────

def _norm_ts(ts):
    """crt.sh: '2023-04-11T09:12:03'. certspotter: '2023-04-11T09:12:03Z' or with
    an offset. Normalise to a naive-UTC ISO string that sorts lexicographically."""
    if not ts:
        return None
    ts = str(ts).strip().replace(" ", "T")
    if ts.endswith("Z"):
        ts = ts[:-1]
    if len(ts) > 19 and (ts[19] in "+-"):
        ts = ts[:19]
    return ts[:19] if len(ts) >= 19 else None


def observations_from_rows(rows, domain):
    """Hostnames + issuance times → {label: earliest first_seen}.

    Label extraction follows webui/app.py's _labels_from: strip leading '*.',
    keep only names under the apex, drop the bare apex and anything the label
    grammar rejects. Wildcard-derived names are kept (a cert for
    '*.api.example.com' proves 'api.example.com' exists) but counted separately.
    """
    suffix = "." + domain
    earliest, stats = {}, {"rows": len(rows), "names": 0, "wildcard_names": 0,
                           "apex_wildcard_certs": 0, "wildcard_only_labels": 0,
                           "no_timestamp": 0}
    wildcard_labels, plain_labels = set(), set()
    for row in rows:
        ts = _norm_ts(row.get("not_before"))
        names = str(row.get("name_value") or "").split("\n")
        names.append(str(row.get("common_name") or ""))
        row_is_apex_wildcard = False
        for raw in names:
            name = raw.strip().lower().rstrip(".")
            if not name:
                continue
            stats["names"] += 1
            was_wild = name.startswith("*.")
            if was_wild:
                stats["wildcard_names"] += 1
            while name.startswith("*."):
                name = name[2:]
            if name == domain:
                if was_wild:
                    row_is_apex_wildcard = True
                continue
            if not name.endswith(suffix):
                continue
            label = name[: -len(suffix)]
            if not label or not LABEL_RE.match(label):
                continue
            (wildcard_labels if was_wild else plain_labels).add(label)
            if ts is None:
                stats["no_timestamp"] += 1
                continue
            if label not in earliest or ts < earliest[label]:
                earliest[label] = ts
        if row_is_apex_wildcard:
            stats["apex_wildcard_certs"] += 1
    stats["wildcard_only_labels"] = len(wildcard_labels - plain_labels)
    stats["labels"] = len(earliest)
    return earliest, stats


def cache_path(apex):
    return os.path.join(CACHE_DIR, apex + ".json")


def _max_nb(rows):
    ts = [_norm_ts(r.get("not_before")) for r in rows]
    ts = [t for t in ts if t]
    return max(ts) if ts else None


def fetch_apex(apex, source="crtsh", use_cache=True, timeout=60, retries=2,
               guard=True, delay=1.5):
    """Returns (record, status). status is 'cache', 'ok', 'empty' or 'error:...'.

    With guard=True (crt.sh only) a second, cheap unexpired-certs-only query is
    made and merged in. crt.sh silently truncates the full-history query on
    large apexes — intel.com came back with nothing newer than 2017 — so without
    this the recent half of the timeline can be missing entirely.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path, blob = cache_path(apex), None
    if use_cache and os.path.exists(path):
        try:
            with open(path) as f:
                blob = json.load(f)
        except (OSError, ValueError):
            blob = None
    want_guard = guard and source == "crtsh"
    status, dirty = "cache", False
    if blob is None:
        try:
            rows = _with_retry(SOURCES[source], apex, timeout, retries)
        except Exception as exc:           # noqa: BLE001
            return None, "error:%s:%s" % (type(exc).__name__, str(exc)[:60])
        blob = {"apex": apex, "source": source,
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "rows": rows}
        status, dirty = ("ok" if rows else "empty"), True
    if want_guard and "rows_unexpired" not in blob:
        if dirty and delay:
            time.sleep(delay)
        try:
            blob["rows_unexpired"] = _with_retry(CRTSH_GUARD, apex, timeout, retries)
            dirty = True
        except Exception as exc:           # noqa: BLE001
            blob["guard_error"] = "%s:%s" % (type(exc).__name__, str(exc)[:60])
            dirty = True
    if dirty:
        with open(path, "w") as f:
            json.dump(blob, f)
    if status == "cache" and not blob.get("rows") and not blob.get("rows_unexpired"):
        status = "cache-empty"

    full = blob.get("rows") or []
    fresh = blob.get("rows_unexpired") or []
    earliest, stats = observations_from_rows(full + fresh, apex)
    stats["rows_full"] = len(full)
    stats["rows_unexpired"] = len(fresh)
    mf, mu = _max_nb(full), _max_nb(fresh)
    stats["max_not_before_full"] = mf
    stats["max_not_before_unexpired"] = mu
    # a superset query that stops before the unexpired one did was cut short
    stats["truncated_history"] = bool(mf and mu and mu > mf)
    if blob.get("guard_error"):
        stats["guard_error"] = blob["guard_error"]
    obs = [{"label": lab, "first_seen": ts} for lab, ts in sorted(earliest.items())]
    rec = {"apex": apex, "observations": obs,
           "source": blob.get("source", source),
           "fetched_at": blob.get("fetched_at"), "stats": stats}
    return rec, status


def _with_retry(fn, apex, timeout, retries):
    last = None
    for attempt in range(retries + 1):
        try:
            return fn(apex, timeout=timeout)
        except Exception as exc:           # noqa: BLE001
            last = exc
            if attempt < retries:
                time.sleep(5.0 * (attempt + 1))
    raise last


def read_apexes(path, limit=None, offset=0):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line)["apex"])
            except (ValueError, KeyError):
                continue
    out = out[offset:]
    return out[:limit] if limit else out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fetch CT observations (passive only).")
    ap.add_argument("--apex", action="append", default=[])
    ap.add_argument("--apexes-from", help="jsonl with an 'apex' field per line")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "ct_observations.jsonl"))
    ap.add_argument("--source", default="crtsh", choices=sorted(SOURCES))
    ap.add_argument("--delay", type=float, default=2.5, help="seconds between apexes")
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--max-consecutive-failures", type=int, default=5)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the second unexpired-only crt.sh query")
    args = ap.parse_args(argv)

    apexes = list(args.apex)
    if args.apexes_from:
        apexes += read_apexes(args.apexes_from, args.limit, args.offset)
    if not apexes:
        ap.error("give --apex or --apexes-from")

    counts, consecutive, records = {}, 0, []
    for i, apex in enumerate(apexes, 1):
        hit_cache = os.path.exists(cache_path(apex)) and not args.no_cache
        rec, status = fetch_apex(apex, source=args.source, use_cache=not args.no_cache,
                                 timeout=args.timeout, retries=args.retries,
                                 guard=not args.no_guard, delay=args.delay)
        kind = status.split(":")[0]
        counts[kind] = counts.get(kind, 0) + 1
        if rec is None:
            consecutive += 1
            print("[%3d/%d] %-34s FAIL %s" % (i, len(apexes), apex, status), flush=True)
            if consecutive >= args.max_consecutive_failures:
                print("stopping: %d consecutive failures — the source is down or "
                      "rate-limiting us" % consecutive, file=sys.stderr)
                break
        else:
            consecutive = 0
            records.append(rec)
            s = rec["stats"]
            print("[%3d/%d] %-34s %-11s labels=%-5d rows=%-5d wildcard_names=%-5d%s"
                  % (i, len(apexes), apex, status, s["labels"], s["rows"],
                     s["wildcard_names"],
                     " TRUNCATED-HISTORY" if s.get("truncated_history") else ""),
                  flush=True)
        if not hit_cache and args.delay and i < len(apexes):
            time.sleep(args.delay)

    with open(args.out, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print("\nwrote %d records to %s" % (len(records), args.out))
    print("status counts:", json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
