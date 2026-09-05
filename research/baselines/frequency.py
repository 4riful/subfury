"""Global-frequency prior: the P(y) control.

Ranks labels purely by how many training apexes use them, ignoring the known
set K entirely except to exclude labels already known for this apex. This is
the baseline every conditional model has to beat: if P(y | K) is not better
than P(y), the conditioning is doing nothing.

No network I/O.
"""

from __future__ import annotations

import json
import os
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(REPO_ROOT, "data", "groups_train.jsonl")


def label_document_frequency(train_path: str = DEFAULT_TRAIN) -> Counter:
    """label -> number of distinct training apexes that use it."""
    counts: Counter = Counter()
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for lab in {str(x).strip().lower() for x in rec["labels"] if str(x).strip()}:
                counts[lab] += 1
    return counts


class FrequencyRanker:
    """P(y) ranker. Deterministic: ties are broken alphabetically."""

    prefix_consistent = True

    def __init__(self, train_path: str = DEFAULT_TRAIN, min_count: int = 1,
                 name: str = "frequency-prior", limit: int = 4000):
        self.train_path = train_path
        self.name = name
        self.counts = label_document_frequency(train_path)
        ranked = sorted(((-c, lab) for lab, c in self.counts.items() if c >= min_count))
        # `limit` only bounds the cached prefix; it must comfortably exceed
        # the largest evaluated budget plus the largest possible |K|.
        self.ranking: list[str] = [lab for _, lab in ranked[:limit]]
        self.min_count = min_count

    def rank(self, apex: str, known: list[str], n: int) -> list[str]:
        known_set = set(known)
        out = []
        for lab in self.ranking:
            if lab in known_set:
                continue
            out.append(lab)
            if len(out) >= n:
                break
        return out

    def describe(self) -> dict:
        return {"train": os.path.basename(self.train_path),
                "distinct_labels": len(self.counts),
                "min_count": self.min_count,
                "cached_ranking": len(self.ranking),
                "top10": self.ranking[:10]}
