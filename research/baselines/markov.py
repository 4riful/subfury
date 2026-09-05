"""SDBF-style character n-gram Markov baseline.

Trains an order-n character Markov model on every label in
data/groups_train.jsonl, then enumerates the highest-probability strings the
model can emit (beam search over characters, terminating on an end symbol)
and returns the top candidates. Like the frequency prior it is unconditional
-- it models P(y), not P(y | K) -- which is exactly what SDBF-style
generators do; K is used only to exclude already-known labels.

`order` is the constructor argument: order=3 means a 2-character context
(trigram), order=5 a 4-character context. Probabilities are Jelinek-Mercer
interpolated down through every lower order to a uniform floor, so the model
is defined for unseen contexts without any backoff heuristics.

Deterministic (no sampling anywhere) and no network I/O.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_TRAIN = os.path.join(REPO_ROOT, "data", "groups_train.jsonl")

# a generated string is only a usable candidate if it is a legal DNS label
# (or dotted chain of labels, which the dataset also contains)
LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")


def iter_train_labels(train_path: str = DEFAULT_TRAIN):
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for lab in json.loads(line)["labels"]:
                lab = str(lab).strip().lower()
                if lab:
                    yield lab


class CharMarkov:
    """Interpolated character n-gram model over label strings."""

    def __init__(self, order: int = 3, lam: float = 0.8,
                 train_path: str = DEFAULT_TRAIN, max_train_labels: int | None = None):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.lam = lam
        self.ctx_len = order - 1
        self.train_path = train_path

        labels = []
        for i, lab in enumerate(iter_train_labels(train_path)):
            if max_train_labels and i >= max_train_labels:
                break
            labels.append(lab)
        self.n_train_labels = len(labels)

        chars = sorted({c for lab in labels for c in lab})
        self.chars = chars
        self.V = len(chars)             # emitting alphabet
        self.EOS = self.V               # end-of-label symbol
        self.BOS = self.V + 1           # left padding symbol (never emitted)
        self.n_sym = self.V + 2
        self.n_out = self.V + 1         # emittable targets: chars + EOS
        self.c2i = {c: i for i, c in enumerate(chars)}

        # one flat int array of BOS-padded, EOS-terminated label sequences
        pad = [self.BOS] * self.ctx_len
        flat: list[int] = []
        target_pos: list[int] = []
        for lab in labels:
            start = len(flat)
            flat.extend(pad)
            flat.extend(self.c2i[c] for c in lab)
            flat.append(self.EOS)
            target_pos.extend(range(start + self.ctx_len, len(flat)))
        self.S = np.asarray(flat, dtype=np.int64)
        T = np.asarray(target_pos, dtype=np.int64)

        # per level j (context length j), a sorted table of
        # key = ctx_id * n_out + target  ->  count
        self.levels = []
        for j in range(self.ctx_len + 1):
            cid = np.zeros(len(T), dtype=np.int64)
            base = 1
            for i in range(1, j + 1):
                cid += self.S[T - i] * base
                base *= self.n_sym
            key = cid * self.n_out + self.S[T]
            keys, counts = np.unique(key, return_counts=True)
            self.levels.append({
                "keys": keys,
                "ctx": keys // self.n_out,
                "tgt": (keys % self.n_out).astype(np.int64),
                "counts": counts.astype(np.float64),
            })
        self._cache: dict[tuple[int, ...], np.ndarray] = {}

    # -- probability -------------------------------------------------------
    def _ml(self, level: int, ctx_id: int) -> np.ndarray | None:
        lv = self.levels[level]
        lo = int(np.searchsorted(lv["ctx"], ctx_id, "left"))
        hi = int(np.searchsorted(lv["ctx"], ctx_id, "right"))
        if hi <= lo:
            return None
        vec = np.zeros(self.n_out)
        vec[lv["tgt"][lo:hi]] = lv["counts"][lo:hi]
        total = vec.sum()
        return vec / total if total else None

    def logprobs(self, ctx: tuple[int, ...]) -> np.ndarray:
        """log P(next symbol | ctx) as a vector over chars + EOS."""
        hit = self._cache.get(ctx)
        if hit is not None:
            return hit
        p = np.full(self.n_out, 1.0 / self.n_out)   # uniform floor
        for level in range(0, self.ctx_len + 1):
            if level == 0:
                ctx_id = 0
            else:
                ctx_id, base = 0, 1
                for i in range(1, level + 1):
                    ctx_id += int(ctx[-i]) * base
                    base *= self.n_sym
            ml = self._ml(level, ctx_id)
            if ml is not None:
                p = self.lam * ml + (1.0 - self.lam) * p
        out = np.log(np.maximum(p, 1e-300))
        self._cache[ctx] = out
        return out

    # -- generation --------------------------------------------------------
    def top_strings(self, k: int, beam_width: int = 2000, max_len: int = 24,
                    min_len: int = 1) -> list[tuple[str, float]]:
        """Beam search for the k most probable strings under the model.

        Ordering is by total log-probability (no length normalisation), which
        is the honest MAP ordering of the generative model and does favour
        short strings.
        """
        start_ctx = tuple([self.BOS] * self.ctx_len)
        beam_ctx = [start_ctx]
        beam_str: list[str] = [""]
        beam_lp = np.zeros(1)
        finished: list[tuple[float, str]] = []

        for _ in range(max_len):
            if not beam_ctx:
                break
            uniq: dict[tuple[int, ...], int] = {}
            rows = []
            for c in beam_ctx:
                if c not in uniq:
                    uniq[c] = len(rows)
                    rows.append(self.logprobs(c))
            P = np.asarray(rows)                     # [n_uniq, n_out]
            idx = np.fromiter((uniq[c] for c in beam_ctx), dtype=np.int64,
                              count=len(beam_ctx))
            M = beam_lp[:, None] + P[idx]            # [n_beam, n_out]

            # completed strings
            eos_lp = M[:, self.EOS]
            for i, lp in enumerate(eos_lp):
                if len(beam_str[i]) >= min_len:
                    finished.append((float(lp), beam_str[i]))

            cont = M[:, : self.V]
            flat = cont.ravel()
            take = min(beam_width, flat.size)
            part = np.argpartition(-flat, take - 1)[:take]
            part = part[np.argsort(-flat[part])]
            b_i, c_i = np.divmod(part, self.V)

            new_ctx, new_str = [], []
            for bi, ci in zip(b_i.tolist(), c_i.tolist()):
                ctx = beam_ctx[bi]
                new_ctx.append((ctx + (ci,))[-self.ctx_len:] if self.ctx_len else ())
                new_str.append(beam_str[bi] + self.chars[ci])
            beam_ctx, beam_str = new_ctx, new_str
            beam_lp = flat[part]

        finished.sort(key=lambda t: (-t[0], t[1]))
        out, seen = [], set()
        for lp, s in finished:
            if s in seen or not LABEL_RE.match(s):
                continue
            seen.add(s)
            out.append((s, lp))
            if len(out) >= k:
                break
        return out


class MarkovRanker:
    """Ranker wrapper: `order` (3, 4, 5, ...) is a constructor argument."""

    prefix_consistent = True

    def __init__(self, order: int = 3, train_path: str = DEFAULT_TRAIN,
                 lam: float = 0.8, pool_size: int = 512, beam_width: int = 2000,
                 max_len: int = 24, name: str | None = None,
                 max_train_labels: int | None = None):
        self.order = order
        self.name = name or f"markov-{order}gram"
        self.model = CharMarkov(order=order, lam=lam, train_path=train_path,
                                max_train_labels=max_train_labels)
        self.pool = self.model.top_strings(pool_size, beam_width=beam_width,
                                           max_len=max_len)
        self.candidates = [s for s, _ in self.pool]

    def rank(self, apex: str, known: list[str], n: int) -> list[str]:
        known_set = set(known)
        out = []
        for cand in self.candidates:
            if cand in known_set:
                continue
            out.append(cand)
            if len(out) >= n:
                break
        return out

    def describe(self) -> dict:
        return {"order": self.order,
                "train_labels": self.model.n_train_labels,
                "alphabet": len(self.model.chars),
                "pool": len(self.candidates),
                "top10": self.candidates[:10]}


if __name__ == "__main__":  # quick smoke test
    for o in (3, 4, 5):
        r = MarkovRanker(order=o, pool_size=40, beam_width=800)
        print(o, r.candidates[:20])
