"""SubFury — dataset and batching for set-conditioned subdomain prediction.

The corpus is apex-grouped JSONL, one organisation per line:

    {"apex": "example.com", "labels": ["api", "blog", "dev.internal", ...]}

Every training example is a random split of one organisation's labels into a
known set K and a held-out set H.  The split point is drawn uniformly, so the
model sees |K| = 1 as often as |K| = 39 — the beam-search model only ever trained on <= 12 known
labels serialised into a context window, and refusing to do that is the whole
the point of the architecture.

Two supervision signals come out of one split:

    generator   one held-out label, as BPE tokens, teacher-forced.  Open
                vocabulary: this is the only head that can emit a name nobody
                has ever registered before.
    retriever   one held-out label as the positive index into a fixed candidate
                vocabulary, against n_neg sampled negatives drawn mostly from
                the globally popular head.  A model that just memorises "every
                org has www/blog/api" gets no credit here: the negatives ARE
                the popular labels, and the ranking loss subtracts a popularity
                prior on top of that.

The corpus is positive-unlabeled.  A label absent from an organisation's list
is not evidence the host does not exist, it is evidence Common Crawl did not
see it.  So negatives are only ever *sampled* — never "everything not in K" —
and a sampled negative is rejected if it appears anywhere in K or H.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
from array import array
from dataclasses import dataclass

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import V3Config, SubFuryV3  # noqa: E402


# --------------------------------------------------------------------------
# candidate vocabulary
# --------------------------------------------------------------------------
class LabelVocab:
    """The retrieval candidate set: the `size` most widely-used labels.

    Ranked by *document frequency* — how many distinct apexes use the label —
    not by raw occurrence count, so an apex with forty numbered hosts cannot
    vote forty times for its own naming scheme.

    `prior_logits` is the popularity baseline the ranking loss subtracts.  It is
    log P(label) centred to zero mean, so subtracting it changes the *ordering*
    of candidates without adding a global offset: the model is scored on how far
    above chance popularity it puts a label, which is the only part of the
    prediction that is worth anything to a user who could have run a wordlist.
    """

    def __init__(self, labels, df):
        assert len(labels) == len(df)
        self.labels = list(labels)
        self.df = list(int(x) for x in df)
        self.stoi = {s: i for i, s in enumerate(self.labels)}
        self._prior = None
        self._cum_hard = None

    # ---- construction ----------------------------------------------------
    @classmethod
    def build(cls, jsonl_path, size=20000, min_df=1, limit=None):
        """One streaming pass over the corpus; never holds the file in memory."""
        df = {}
        n_docs = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                rec = json.loads(line)
                n_docs += 1
                for lab in set(rec["labels"]):        # distinct apexes only
                    df[lab] = df.get(lab, 0) + 1
        items = [(l, c) for l, c in df.items() if c >= min_df]
        # deterministic: frequency desc, then lexicographic
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        items = items[:size]
        v = cls([l for l, _ in items], [c for _, c in items])
        v.n_docs = n_docs
        v.n_unique_seen = len(df)
        return v

    def __len__(self):
        return len(self.labels)

    def index(self, label):
        """label -> index, or -1 when the label is outside the vocabulary."""
        return self.stoi.get(label, -1)

    def label(self, i):
        return self.labels[i]

    # ---- popularity prior -------------------------------------------------
    @property
    def prior_logits(self) -> torch.Tensor:
        if self._prior is None:
            df = torch.tensor(self.df, dtype=torch.float32)
            lp = torch.log(df / df.sum())
            self._prior = (lp - lp.mean()).contiguous()
        return self._prior

    # ---- negative sampling ------------------------------------------------
    def _cum_weights(self, alpha):
        """Cumulative df**alpha, for O(log V) weighted draws without numpy."""
        if self._cum_hard is None or self._cum_hard[0] != alpha:
            cum, tot = [], 0.0
            for c in self.df:
                tot += float(c) ** alpha
                cum.append(tot)
            self._cum_hard = (alpha, cum)
        return self._cum_hard[1]

    def sample_negatives(self, rng, exclude, n, hard_frac=0.75, alpha=1.0):
        """`n` vocabulary indices absent from `exclude` (the org's K and H).

        `hard_frac` of them are drawn with probability proportional to
        df**alpha, i.e. from the popular head — those are the labels a
        popularity baseline would guess and this organisation does not have.
        The rest are uniform over the vocabulary, so the tail keeps getting
        gradient too.
        """
        V = len(self.labels)
        if V == 0:
            return []
        cum = self._cum_weights(alpha)
        total = cum[-1]
        n_hard = int(round(n * hard_frac))
        out, seen = [], set()
        for want_hard, need in ((True, n_hard), (False, n - n_hard)):
            got, tries, budget = 0, 0, max(64, need * 64)
            while got < need and tries < budget:
                tries += 1
                if want_hard:
                    j = bisect.bisect_right(cum, rng.random() * total)
                    if j >= V:
                        j = V - 1
                else:
                    j = rng.randrange(V)
                if j in exclude or j in seen:
                    continue      # false negative, or a duplicate
                seen.add(j)
                out.append(j)
                got += 1
        # pathological orgs (huge label set, tiny vocab) may starve the loop;
        # top up with a linear scan so the tensor shape is always (n_neg,)
        j = 0
        while len(out) < n and j < V:
            if j not in exclude and j not in seen:
                seen.add(j)
                out.append(j)
            j += 1
        return out[:n]

    # ---- persistence ------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"labels": self.labels, "df": self.df}, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["labels"], d["df"])


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Knobs that belong to the data, not the network."""
    n_neg: int = 32
    hard_frac: float = 0.75
    neg_alpha: float = 1.0     # 1.0 = negatives ~ document frequency
    min_labels: int = 2        # need at least one known and one held out
    seed: int = 0


class OrgSetDataset(Dataset):
    """One organisation per item; the K/H split is resampled every epoch.

    Memory: labels are interned once into a shared string table and each group
    is stored as an array of int32 indices into it, so the 1.04M label
    occurrences in groups_train.jsonl cost ~4MB rather than a second copy of
    the file's strings.  BPE encodings for the candidate vocabulary (which
    covers half of all occurrences) are precomputed; everything else is encoded
    on demand by the Rust tokenizer.
    """

    def __init__(self, jsonl_path, tokenizer: Tokenizer, vocab: LabelVocab,
                 cfg: V3Config, dcfg: DataConfig = None, limit=None):
        self.cfg = cfg
        self.dcfg = dcfg or DataConfig()
        self.tok = tokenizer
        self.vocab = vocab
        self.epoch = 0

        self.bos = tokenizer.token_to_id("[DELIM]")   # no [BOS] in the shared BPE
        self.eos = tokenizer.token_to_id("[END]")
        assert self.bos is not None and self.eos is not None

        # -- one streaming pass: intern strings, store groups as int arrays --
        self.strings = []
        sid = {}
        self.groups = []        # list[array('i')] of string-table indices
        self.apexes = []
        n_skipped = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit is not None and i >= limit:
                    break
                rec = json.loads(line)
                labs = rec["labels"]
                if len(labs) < self.dcfg.min_labels:
                    n_skipped += 1
                    continue
                a = array("i")
                for lab in labs:
                    j = sid.get(lab)
                    if j is None:
                        j = len(self.strings)
                        sid[lab] = j
                        self.strings.append(lab)
                    a.append(j)
                self.groups.append(a)
                self.apexes.append(rec["apex"])
        self.n_skipped = n_skipped

        # -- string-table index -> candidate-vocab index (-1 outside) --------
        self.to_cand = array("i", (vocab.index(s) for s in self.strings))

        # -- BPE cache for the candidate head; the tail is encoded lazily -----
        T = cfg.max_label_tokens
        self._enc = {}
        head = [i for i, c in enumerate(self.to_cand) if c >= 0]
        if head:
            encs = tokenizer.encode_batch([self.strings[i] for i in head])
            for i, e in zip(head, encs):
                self._enc[i] = tuple(e.ids[:T]) or (self.bos,)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.groups)

    def set_epoch(self, epoch):
        """Change the split-sampling stream without touching worker RNGs."""
        self.epoch = int(epoch)

    def _rng(self, idx):
        """Per-item RNG: deterministic under (seed, epoch, idx) and therefore
        identical no matter how many DataLoader workers are used."""
        import random
        h = hashlib.blake2b(
            f"{self.dcfg.seed}:{self.epoch}:{idx}".encode(), digest_size=8
        ).digest()
        return random.Random(int.from_bytes(h, "big"))

    def _ids(self, sidx):
        e = self._enc.get(sidx)
        if e is None:
            e = tuple(self.tok.encode(self.strings[sidx]).ids[:self.cfg.max_label_tokens])
            if not e:
                e = (self.bos,)
            if len(self._enc) < 400_000:
                self._enc[sidx] = e
        return e

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        rng = self._rng(idx)
        g = list(self.groups[idx])
        rng.shuffle(g)
        n = len(g)

        # split point uniform in [1, min(n-1, max_set)] -> every set size,
        # from a single known label to the whole organisation
        k = rng.randint(1, min(n - 1, self.cfg.max_set))
        K, H = g[:k], g[k:]

        # -- retrieval positive must exist in the candidate vocabulary -------
        h_cand = [s for s in H if self.to_cand[s] >= 0]
        cand_valid = True
        if not h_cand:
            k_cand = [i for i, s in enumerate(K) if self.to_cand[s] >= 0]
            if k_cand:
                # move one in-vocab label across the split rather than dropping
                # the example: |K| and |H| are preserved, only membership moves
                i = rng.choice(k_cand)
                j = rng.randrange(len(H))
                K[i], H[j] = H[j], K[i]
                h_cand = [s for s in H if self.to_cand[s] >= 0]
            else:
                cand_valid = False          # org has no in-vocab label at all

        # -- generator target: any held-out label (open vocabulary) ----------
        tgt_s = rng.choice(H)
        # -- retrieval positive: prefer a *different* held-out label ---------
        if cand_valid:
            others = [s for s in h_cand if s != tgt_s]
            pos_s = rng.choice(others) if others else rng.choice(h_cand)
            cand_pos = self.to_cand[pos_s]
            exclude = {self.to_cand[s] for s in g if self.to_cand[s] >= 0}
            neg = self.vocab.sample_negatives(
                rng, exclude, self.dcfg.n_neg,
                hard_frac=self.dcfg.hard_frac, alpha=self.dcfg.neg_alpha)
        else:
            # No valid positive.  Point the positive AND every negative at the
            # same candidate: the softmax is uniform, the gradients cancel to
            # exactly zero, so this row costs a constant in the reported rank
            # loss and contributes nothing to the update.  `cand_valid` is
            # carried in the batch so train.py can mask it out properly.
            cand_pos = 0
            neg = [0] * self.dcfg.n_neg

        toks = list(self._ids(tgt_s))
        return {
            "set": [self._ids(s) for s in K],
            "tgt_in": [self.bos] + toks,
            "tgt_out": toks + [self.eos],
            "cand_pos": cand_pos,
            "cand_neg": neg,
            "cand_valid": cand_valid,
            "n_known": k,
            "apex": self.apexes[idx],
            # for assertions / eval only; collate drops it unless keep_meta
            "_org_cand": tuple(sorted({self.to_cand[s] for s in g
                                       if self.to_cand[s] >= 0})),
            "_k_cand": tuple(sorted({self.to_cand[s] for s in K
                                     if self.to_cand[s] >= 0})),
            "_h_cand": tuple(sorted({self.to_cand[s] for s in H
                                     if self.to_cand[s] >= 0})),
        }


# --------------------------------------------------------------------------
# collate
# --------------------------------------------------------------------------
def collate(batch, cfg: V3Config, keep_meta=False):
    """Exactly the tensor dict SubFuryV3.loss() reads.

    The set dimension is padded to the batch maximum, never to cfg.max_set, so
    a batch of four-label organisations costs four slots and not 512.  Padded
    slots are all-pad_id and set_mask is False there; the model's LabelEncoder
    zeroes those vectors and the set attention masks them out.
    """
    B = len(batch)
    P = cfg.pad_id
    T = cfg.max_label_tokens
    L = max(1, max(len(b["set"]) for b in batch))
    S = max(len(b["tgt_in"]) for b in batch)
    Kn = len(batch[0]["cand_neg"])

    set_ids = torch.full((B, L, T), P, dtype=torch.long)
    set_mask = torch.zeros(B, L, dtype=torch.bool)
    tgt_in = torch.full((B, S), P, dtype=torch.long)
    tgt_out = torch.full((B, S), P, dtype=torch.long)   # P is loss ignore_index
    cand_pos = torch.zeros(B, 1, dtype=torch.long)
    cand_neg = torch.zeros(B, Kn, dtype=torch.long)
    cand_valid = torch.zeros(B, dtype=torch.bool)
    n_known = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        for j, ids in enumerate(b["set"]):
            set_ids[i, j, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        set_mask[i, :len(b["set"])] = True
        tgt_in[i, :len(b["tgt_in"])] = torch.tensor(b["tgt_in"], dtype=torch.long)
        tgt_out[i, :len(b["tgt_out"])] = torch.tensor(b["tgt_out"], dtype=torch.long)
        cand_pos[i, 0] = b["cand_pos"]
        cand_neg[i] = torch.tensor(b["cand_neg"], dtype=torch.long)
        cand_valid[i] = b["cand_valid"]
        n_known[i] = b["n_known"]

    out = {"set_ids": set_ids, "set_mask": set_mask,
           "tgt_in": tgt_in, "tgt_out": tgt_out,
           "cand_pos": cand_pos, "cand_neg": cand_neg,
           "cand_valid": cand_valid, "n_known": n_known}
    if keep_meta:
        out["meta"] = [{"apex": b["apex"], "org_cand": b["_org_cand"],
                        "k_cand": b["_k_cand"], "h_cand": b["_h_cand"]}
                       for b in batch]
    return out


def make_loader(dataset, cfg, batch_size=32, shuffle=True, num_workers=0,
                keep_meta=False, drop_last=False):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=lambda b: collate(b, cfg, keep_meta=keep_meta))


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------
def _selftest(argv=None):
    import random
    import time

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--tokenizer", default="results/subfury/tokenizer.json")
    ap.add_argument("--limit", type=int, default=20000, help="apex slice")
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-neg", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print("=" * 72)
    print("SubFury data self-test")
    print("=" * 72)

    tok = Tokenizer.from_file(args.tokenizer)
    print(f"tokenizer            {args.tokenizer}")
    print(f"  vocab_size         {tok.get_vocab_size()}")
    print("  specials           " + "  ".join(
        f"{s}={tok.token_to_id(s)}" for s in ("[PAD]", "[DELIM]", "[END]", "[SEP]")))

    t0 = time.time()
    vocab = LabelVocab.build(args.train_jsonl, size=args.vocab_size,
                             limit=args.limit)
    t_vocab = time.time() - t0
    print(f"\nLabelVocab           built from {vocab.n_docs} apexes in "
          f"{t_vocab:.2f}s")
    print(f"  distinct labels    {vocab.n_unique_seen}")
    print(f"  kept               {len(vocab)}")
    print("  top 10             " + ", ".join(
        f"{vocab.label(i)}({vocab.df[i]})" for i in range(min(10, len(vocab)))))
    print(f"  df range           {vocab.df[0]} .. {vocab.df[-1]}")
    pl = vocab.prior_logits
    print(f"  prior_logits       shape={tuple(pl.shape)} dtype={pl.dtype} "
          f"min={pl.min():.4f} max={pl.max():.4f} mean={pl.mean():.2e} "
          f"std={pl.std():.4f}")

    cfg = V3Config(vocab_size=tok.get_vocab_size(), cand_vocab=len(vocab),
                   pad_id=tok.token_to_id("[PAD]"), encoder="settrans",
                   max_set=512, max_label_tokens=16)
    dcfg = DataConfig(n_neg=args.n_neg, hard_frac=0.75, seed=args.seed)

    t0 = time.time()
    ds = OrgSetDataset(args.train_jsonl, tok, vocab, cfg, dcfg, limit=args.limit)
    t_ds = time.time() - t0
    print(f"\nOrgSetDataset        {len(ds)} orgs in {t_ds:.2f}s "
          f"(skipped {ds.n_skipped} with <{dcfg.min_labels} labels)")
    print(f"  interned strings   {len(ds.strings)} "
          f"({sum(len(g) for g in ds.groups)} label occurrences)")
    print(f"  in-vocab strings   {sum(1 for c in ds.to_cand if c >= 0)}")
    print(f"  BPE cache          {len(ds._enc)} entries")

    dl = make_loader(ds, cfg, batch_size=args.batch_size, shuffle=True,
                     num_workers=0, keep_meta=True)
    t0 = time.time()
    batch = next(iter(dl))
    t_batch = time.time() - t0
    meta = batch.pop("meta")

    print(f"\nbatch                pulled in {t_batch*1000:.1f}ms")
    for k in ("set_ids", "set_mask", "tgt_in", "tgt_out",
              "cand_pos", "cand_neg", "cand_valid", "n_known"):
        v = batch[k]
        print(f"  {k:11s} shape={str(tuple(v.shape)):16s} dtype={v.dtype}")

    B, L, T = batch["set_ids"].shape
    print(f"\n  |K| per row        {batch['n_known'].tolist()}")
    print(f"  set padded to      L={L} (max_set={cfg.max_set}, "
          f"waste={100*(1 - batch['set_mask'].float().mean().item()):.1f}%)")
    print(f"  tgt padded to      S={batch['tgt_in'].shape[1]} "
          f"(ceiling {cfg.max_label_tokens + 1})")
    print(f"  cand_valid         {int(batch['cand_valid'].sum())}/{B} rows")

    apex0 = meta[0]["apex"]
    known0 = [tok.decode([i for i in row.tolist() if i != cfg.pad_id])
              for row, m in zip(batch["set_ids"][0], batch["set_mask"][0]) if m]
    tgt0 = tok.decode([i for i in batch["tgt_out"][0].tolist()
                       if i not in (cfg.pad_id, ds.eos)])
    print(f"\n  example row 0      apex={apex0}")
    print(f"    K ({len(known0)})           {known0}")
    print(f"    generator target  {tgt0!r}")
    print(f"    cand_pos          {int(batch['cand_pos'][0,0])} "
          f"-> {vocab.label(int(batch['cand_pos'][0,0]))!r}")
    print("    cand_neg[:8]      " + ", ".join(
        f"{vocab.label(j)}" for j in batch["cand_neg"][0, :8].tolist()))

    # ---- assertions --------------------------------------------------
    print("\nassertions")

    # 1. no sampled negative may be a label the organisation actually has
    n_checked = 0
    for i, m in enumerate(meta):
        if not bool(batch["cand_valid"][i]):
            continue
        org = set(m["org_cand"])
        negs = batch["cand_neg"][i].tolist()
        assert len(set(negs)) == len(negs), f"row {i}: duplicate negative"
        bad = org & set(negs)
        assert not bad, f"row {i}: false negatives {sorted(bad)}"
        assert int(batch["cand_pos"][i, 0]) in set(m["h_cand"]), \
            f"row {i}: cand_pos is not a held-out label"
        assert int(batch["cand_pos"][i, 0]) not in set(m["k_cand"]) - set(m["h_cand"]), \
            f"row {i}: cand_pos leaked from K"
        n_checked += len(negs)
    print(f"  negatives disjoint from K and H          PASS "
          f"({n_checked} negatives over "
          f"{int(batch['cand_valid'].sum())} valid rows)")

    # 2. set_mask must match the real known-label counts
    counts = batch["set_mask"].sum(1)
    assert torch.equal(counts, batch["n_known"]), \
        f"set_mask counts {counts.tolist()} != n_known {batch['n_known'].tolist()}"
    pad_rows = batch["set_ids"][~batch["set_mask"]]
    assert bool((pad_rows == cfg.pad_id).all()), "masked-out slot holds real tokens"
    real_rows = batch["set_ids"][batch["set_mask"]]
    assert bool((real_rows != cfg.pad_id).any(-1).all()), "real slot is all padding"
    print(f"  set_mask == real label counts            PASS "
          f"({counts.tolist()})")
    print(f"  padded slots are pure pad_id             PASS "
          f"({pad_rows.shape[0]} slots)")

    # 3. teacher forcing is shifted by one
    for i in range(B):
        a = batch["tgt_in"][i].tolist()
        b = batch["tgt_out"][i].tolist()
        n = sum(1 for t in b if t != cfg.pad_id)
        assert a[0] == ds.bos and b[n - 1] == ds.eos
        assert a[1:n] == b[:n - 1], f"row {i}: tgt_in/tgt_out not shifted"
    print("  tgt_in/tgt_out shifted by one            PASS")

    # ---- negative-sampling distribution -------------------------------
    allneg = batch["cand_neg"][batch["cand_valid"]].flatten().tolist()
    if allneg:
        in_head100 = sum(1 for j in allneg if j < 100) / len(allneg)
        in_head1k = sum(1 for j in allneg if j < 1000) / len(allneg)
        mean_df = sum(vocab.df[j] for j in allneg) / len(allneg)
        uni_df = sum(vocab.df) / len(vocab)
        print(f"\nnegative mix (hard_frac={dcfg.hard_frac}, "
              f"alpha={dcfg.neg_alpha})")
        print(f"  in top-100 labels  {in_head100*100:.1f}%   "
              f"(uniform would be {100*100/len(vocab):.2f}%)")
        print(f"  in top-1000 labels {in_head1k*100:.1f}%   "
              f"(uniform would be {1000*100/len(vocab):.2f}%)")
        print(f"  mean df of negs    {mean_df:.1f}   "
              f"(uniform draw would be {uni_df:.1f})")

    # ---- determinism ---------------------------------------------------
    a = ds[7]
    b = ds[7]
    assert a["set"] == b["set"] and a["cand_neg"] == b["cand_neg"], "not deterministic"
    ds.set_epoch(1)
    c = ds[7]
    ds.set_epoch(0)
    d = ds[7]
    assert d["set"] == a["set"], "set_epoch did not restore the stream"
    print(f"\ndeterminism          same idx twice -> identical            PASS")
    print(f"  epoch 0 |K|={a['n_known']}  epoch 1 |K|={c['n_known']}  "
          f"(resampled: {a['set'] != c['set']})")

    # ---- split-point coverage over many items --------------------------
    ks = [ds[i]["n_known"] for i in range(min(4000, len(ds)))]
    ns = [len(ds.groups[i]) for i in range(min(4000, len(ds)))]
    print(f"\nsplit coverage over {len(ks)} items")
    print(f"  |K| min/mean/max   {min(ks)} / {sum(ks)/len(ks):.2f} / {max(ks)}")
    print(f"  |labels| max       {max(ns)}   "
          f"(data_prep caps apexes at 40 labels)")
    buckets = {"1": 0, "2-4": 0, "5-9": 0, "10-19": 0, "20+": 0}
    for k in ks:
        key = "1" if k == 1 else "2-4" if k < 5 else "5-9" if k < 10 else \
              "10-19" if k < 20 else "20+"
        buckets[key] += 1
    print("  |K| histogram      " + "  ".join(
        f"{k}:{100*v/len(ks):.1f}%" for k, v in buckets.items()))
    nval = sum(1 for i in range(min(4000, len(ds))) if ds[i]["cand_valid"])
    print(f"  cand_valid rate    {100*nval/len(ks):.1f}%  "
          f"(orgs with no label in the top-{len(vocab)} vocabulary)")

    # ---- end-to-end through the model ----------------------------------
    print("\nend-to-end through SubFuryV3.loss()")
    model = SubFuryV3(cfg)
    print(f"  params             {model.num_params()/1e6:.2f}M "
          f"(encoder={cfg.encoder}, cand_vocab={cfg.cand_vocab})")
    tensors = {k: v for k, v in batch.items() if torch.is_tensor(v)}
    out = model.loss(tensors, lambda_rank=1.0, prior_logits=vocab.prior_logits)
    print(f"  gen                {out['gen'].item():.4f}   "
          f"(uniform over {cfg.vocab_size} = {math.log(cfg.vocab_size):.4f})")
    print(f"  rank               {out['rank'].item():.4f}   "
          f"(chance over 1+{dcfg.n_neg} = {math.log(1+dcfg.n_neg):.4f})")
    print(f"  total              {out['total'].item():.4f}")
    for k, v in out.items():
        assert torch.isfinite(v), f"{k} is not finite"
    out["total"].backward()
    gnorm = math.sqrt(sum(float((p.grad ** 2).sum()) for p in model.parameters()
                          if p.grad is not None))
    ngrad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"  backward           grad_norm={gnorm:.4f} over {ngrad} tensors")

    # the no-positive fallback (pos == every neg) must be gradient-free:
    # dL/ds_i = p_i - y_i sums to zero when every entry is the same candidate
    model.zero_grad(set_to_none=True)
    ref = float(torch.autograd.grad(
        model.loss(tensors, prior_logits=vocab.prior_logits)["rank"],
        model.cand.weight)[0].abs().max())
    deg = {k: v.clone() for k, v in tensors.items()}
    deg["cand_pos"].zero_()
    deg["cand_neg"].zero_()
    o2 = model.loss(deg, lambda_rank=1.0, prior_logits=vocab.prior_logits)
    gr = float(torch.autograd.grad(o2["rank"], model.cand.weight)[0].abs().max())
    print(f"  degenerate rows    rank={o2['rank'].item():.4f} "
          f"(== ln(1+{dcfg.n_neg})={math.log(1+dcfg.n_neg):.4f}); "
          f"max |d rank/d cand| = {gr:.2e} vs {ref:.2e} for a real batch "
          f"({gr/ref:.1e}x, fp32 roundoff)")
    assert gr < 1e-4 * ref, "the no-positive fallback is leaking gradient"

    # ---- throughput ------------------------------------------------------
    t0 = time.time()
    seen = 0
    it = iter(make_loader(ds, cfg, batch_size=args.batch_size, shuffle=True))
    for _ in range(20):
        b = next(it)
        seen += b["set_ids"].shape[0]
    dt = time.time() - t0
    print(f"\nthroughput           {seen/dt:.0f} examples/s single-process "
          f"({seen} in {dt:.2f}s)")
    print(f"  projected epoch    {84144/(seen/dt):.0f}s over 84144 apexes\n")

    print("SELF-TEST OK")


if __name__ == "__main__":
    _selftest()
