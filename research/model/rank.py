"""SubFury as a Ranker, so it is scored by exactly the harness every baseline used.

Candidates come from two places and are merged by score:

  generation  beam search over BPE tokens, cross-attending the set memory.
              Open vocabulary — it can propose names no corpus contains.
  retrieval   the candidate-vocabulary head, scored against the organisation
              vector, with the frequency prior subtracted.  The measurements say
              generation runs dry past rank ~50 while a prior keeps climbing, so
              this is what fills the tail.

`source=` selects generator / retriever / hybrid, which is the ablation.
"""
import os, sys
import torch
import torch.nn.functional as F

import re

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import V3Config, SubFuryV3                                # noqa: E402

# subfury/predict.py also lives behind a module called `model`, so importing
# it here would shadow this one's. This is the one thing needed from it.
LABEL_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")


class V3Ranker:
    name = "subfury"

    def __init__(self, ckpt="results/runs/settrans-rank1-prior1/best.pt",
                 tokenizer="results/subfury/tokenizer.json",
                 source="hybrid", num_beams=64, device=None, mix=0.5):
        from tokenizers import Tokenizer
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        blob = torch.load(ckpt, map_location=self.dev, weights_only=False)
        self.cfg = V3Config(**blob["cfg"])
        self.model = SubFuryV3(self.cfg).to(self.dev).eval()
        self.model.load_state_dict(blob["model"])
        self.tok = Tokenizer.from_file(tokenizer)
        self.cand = blob["vocab"]
        self.source, self.num_beams, self.mix = source, num_beams, mix
        self.name = f"subfury-{self.cfg.encoder}-{source}"

    def _memory(self, known):
        ids = []
        for lab in list(known)[: self.cfg.max_set]:
            t = self.tok.encode(lab).ids[: self.cfg.max_label_tokens]
            ids.append(t + [self.cfg.pad_id] * (self.cfg.max_label_tokens - len(t)))
        if not ids:
            ids = [[self.cfg.pad_id] * self.cfg.max_label_tokens]
        x = torch.tensor([ids], device=self.dev)
        m = torch.ones(1, x.size(1), dtype=torch.bool, device=self.dev)
        return self.model.org(x, m)

    @torch.no_grad()
    def _generate(self, mem, n, known):
        """Beam search over the decoder, conditioned on the set memory."""
        end = self.tok.token_to_id("[END]") or 0
        beams = [([], 0.0)]
        finished = []
        for _ in range(self.cfg.max_label_tokens):
            if not beams:
                break
            inp = torch.tensor([[self.cfg.pad_id] + b[0] for b in beams], device=self.dev)
            logits = self.model.generate_logits(mem.expand(len(beams), -1, -1), inp)
            lp = F.log_softmax(logits[:, -1, :], -1)
            k = min(self.num_beams, lp.size(-1))
            top_lp, top_ix = lp.topk(k, -1)
            cand = []
            for bi, (toks, sc) in enumerate(beams):
                for l, i in zip(top_lp[bi].tolist(), top_ix[bi].tolist()):
                    cand.append((toks + [i], sc + l))
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for toks, sc in cand:
                if toks[-1] == end:
                    finished.append((toks[:-1], sc))
                elif len(beams) < self.num_beams:
                    beams.append((toks, sc))
            if len(finished) >= n * 3:
                break
        out, seen = [], set(known)
        for toks, sc in sorted(finished, key=lambda c: c[1], reverse=True):
            lab = self.tok.decode(toks).replace(" ", "").lower()
            if lab and lab not in seen and LABEL_RE.match(lab):
                seen.add(lab)
                out.append((lab, sc))
        return out

    @torch.no_grad()
    def _retrieve(self, mem, n, known):
        if not self.cfg.cand_vocab:
            return []
        s = self.model.retrieve_scores(mem)[0]
        order = torch.argsort(s, descending=True).tolist()
        out, seen = [], set(known)
        for i in order:
            lab = self.cand[i]
            if lab not in seen:
                out.append((lab, float(s[i])))
                if len(out) >= n * 2:
                    break
        return out

    def rank(self, apex, known, n):
        mem = self._memory(known)
        if self.source == "generator":
            return [l for l, _ in self._generate(mem, n, known)][:n]
        if self.source == "retriever":
            return [l for l, _ in self._retrieve(mem, n, known)][:n]
        # hybrid: interleave by normalised rank so neither source starves the other
        g = [l for l, _ in self._generate(mem, n, known)]
        r = [l for l, _ in self._retrieve(mem, n, known)]
        out, seen = [], set()
        gi = ri = 0
        while len(out) < n and (gi < len(g) or ri < len(r)):
            take_g = (ri >= len(r)) or (gi < len(g) and (gi / max(len(g), 1)) <=
                                        (ri / max(len(r), 1)) * (self.mix / (1 - self.mix)))
            lab = g[gi] if take_g else r[ri]
            if take_g:
                gi += 1
            else:
                ri += 1
            if lab not in seen:
                seen.add(lab)
                out.append(lab)
        return out[:n]
