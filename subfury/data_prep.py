"""Build grouped-by-apex training data from Common Crawl host-graph vertices.

Input: CC host vertices part files (gz), lines like:
    <id>\tcom.example.www
(reversed hostname notation, sorted — so apex groups are contiguous).

Output: JSONL, one apex per line:
    {"apex": "example.com", "labels": ["www", "mail", "dev.api", ...]}

Filters:
  - only apexes with >= MIN_LABELS distinct subdomain labels
  - valid DNS charset, no bare apexes, cap labels per apex so
    mega-hosters don't dominate the distribution
"""

import argparse
import glob
import gzip
import json
import random

import tldextract

MIN_LABELS = 4
MAX_LABELS = 40
MAX_LABEL_LEN = 60

# offline PSL snapshot bundled with tldextract
_extract = tldextract.TLDExtract(suffix_list_urls=())


def parse_vertices(paths):
    """Yield (apex, label) from reversed-hostname vertex files."""
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 2:
                    continue
                rev = parts[1]
                host = ".".join(reversed(rev.split(".")))
                ext = _extract(host)
                if not ext.suffix or not ext.domain or not ext.subdomain:
                    continue
                label = ext.subdomain.lower()
                if label == "www" or len(label) > MAX_LABEL_LEN:
                    continue  # www-only adds nothing to learn
                if not all(c.isalnum() or c in ".-" for c in label):
                    continue
                yield f"{ext.domain}.{ext.suffix}", label


def group_by_apex(pairs):
    """Group contiguous (apex, label) pairs. Input is sorted by rev-host,
    so all labels of an apex arrive together."""
    cur_apex, labels = None, set()
    for apex, label in pairs:
        if apex != cur_apex:
            if cur_apex and len(labels) >= MIN_LABELS:
                yield cur_apex, sorted(labels)
            cur_apex, labels = apex, set()
        labels.add(label)
    if cur_apex and len(labels) >= MIN_LABELS:
        yield cur_apex, sorted(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="data/cc/part-*.txt.gz")
    ap.add_argument("--out-train", default="data/groups_train.jsonl")
    ap.add_argument("--out-val", default="data/groups_val.jsonl")
    ap.add_argument("--out-test", default="data/groups_test.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(args.input_glob))
    print(f"Parsing {len(paths)} vertex files: {paths}")

    n_train = n_val = n_test = 0
    with open(args.out_train, "w") as ftr, open(args.out_val, "w") as fva, \
         open(args.out_test, "w") as fte:
        for apex, labels in group_by_apex(parse_vertices(paths)):
            if len(labels) > MAX_LABELS:
                labels = rng.sample(labels, MAX_LABELS)
            rec = json.dumps({"apex": apex, "labels": labels})
            r = rng.random()
            if r < args.test_frac:
                fte.write(rec + "\n"); n_test += 1
            elif r < args.test_frac + args.val_frac:
                fva.write(rec + "\n"); n_val += 1
            else:
                ftr.write(rec + "\n"); n_train += 1

    print(f"apex groups -> train: {n_train}  val: {n_val}  test: {n_test}")


if __name__ == "__main__":
    main()
