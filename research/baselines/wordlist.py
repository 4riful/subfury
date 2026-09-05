"""Static wordlist baseline: take the file in its own order.

This is what a bruteforcer actually does today (n0kovo / SecLists ordering),
so it is the practitioner-relevant control. It uses neither the apex nor K,
except to skip labels that are already known.

No network I/O.
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_WORDLIST = os.path.join(REPO_ROOT, "data", "subdomains_tiny.txt")


def load_wordlist(path: str = DEFAULT_WORDLIST, limit: int | None = None) -> list[str]:
    words: list[str] = []
    seen: set[str] = set()
    with open(path, errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if not w or w in seen:
                continue
            seen.add(w)
            words.append(w)
            if limit and len(words) >= limit:
                break
    return words


class WordlistRanker:
    """Ranks the wordlist in file order, excluding known labels."""

    prefix_consistent = True

    def __init__(self, path: str = DEFAULT_WORDLIST, name: str | None = None,
                 cache: int = 4000):
        self.path = path
        self.name = name or f"wordlist:{os.path.basename(path)}"
        self.words = load_wordlist(path, limit=cache)
        self.total_words = cache

    def rank(self, apex: str, known: list[str], n: int) -> list[str]:
        known_set = set(known)
        out = []
        for w in self.words:
            if w in known_set:
                continue
            out.append(w)
            if len(out) >= n:
                break
        return out

    def describe(self) -> dict:
        return {"wordlist": os.path.basename(self.path),
                "cached_words": len(self.words),
                "top10": self.words[:10]}
