"""Adapter that puts the beam-search SubFury transformer behind the shared Ranker API.

Wraps subfury/predict.py without modifying it. Beam search is by far the
slowest thing in the harness, so:
  * candidates are generated once per apex at the maximum budget and sliced
    for smaller N (the beam ordering is a single score ordering, so a
    prefix is the correct top-k), and
  * predictions are memoised to a JSON cache under results/research/cache/
    keyed by (model dir, num_beams, max_n, apex, known set), so re-runs and
    subset analyses are free.

No DNS: predict_labels() is the pure generation step; resolve_all() is never
called from here.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_V2 = os.path.join(REPO_ROOT, "subfury")
if _V2 not in sys.path:
    sys.path.insert(0, _V2)

DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "results", "subfury")
DEFAULT_CACHE = os.path.join(REPO_ROOT, "results", "research", "cache",
                             "neural_preds.json")


class NeuralRanker:
    """The beam-search model (set-conditioned transformer, beam search)."""

    prefix_consistent = True

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, num_beams: int = 64,
                 max_n: int = 200, name: str = "subfury-beam",
                 cache_path: str | None = DEFAULT_CACHE, verbose: bool = True):
        from predict import load_model, predict_labels  # noqa: E402

        self.name = name
        self.model_dir = model_dir
        self.num_beams = num_beams
        self.max_n = max_n
        self._predict_labels = predict_labels
        self.model, self.tok, self.device = load_model(model_dir)
        self.verbose = verbose
        self.calls = 0
        self.cache_hits = 0

        self.cache_path = cache_path
        self.cache: dict[str, list[str]] = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    doc = json.load(f)
                if doc.get("key") == self._config_key():
                    self.cache = doc.get("preds", {})
            except (OSError, ValueError):
                self.cache = {}

    def _config_key(self) -> str:
        return f"{os.path.basename(self.model_dir)}|beams={self.num_beams}|max_n={self.max_n}"

    @staticmethod
    def _case_key(apex: str, known: list[str]) -> str:
        h = hashlib.sha1("\x00".join(sorted(known)).encode()).hexdigest()[:16]
        return f"{apex}|{h}"

    def rank(self, apex: str, known: list[str], n: int) -> list[str]:
        key = self._case_key(apex, known)
        cached = self.cache.get(key)
        if cached is not None and len(cached) >= min(n, self.max_n):
            self.cache_hits += 1
            return cached[:n]
        preds = self._predict_labels(self.model, self.tok, self.device,
                                     list(known), topn=max(n, self.max_n),
                                     num_beams=self.num_beams)
        labels = [lab for lab, _ in preds]
        self.cache[key] = labels
        self.calls += 1
        if self.verbose and self.calls % 25 == 0:
            print(f"    [{self.name}] {self.calls} beam searches", flush=True)
        return labels[:n]

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump({"key": self._config_key(), "preds": self.cache}, f)

    def describe(self) -> dict:
        return {"model_dir": os.path.relpath(self.model_dir, REPO_ROOT),
                "device": str(self.device),
                "num_beams": self.num_beams,
                "max_n": self.max_n,
                "beam_searches": self.calls,
                "cache_hits": self.cache_hits}
