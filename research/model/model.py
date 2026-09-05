"""SubFury — set-conditioned, config-switched so the ablations are honest.

Every variant below shares one label encoder, one decoder and one training loop.
The only thing that changes is how the known set is pooled into an organization
representation, so a difference in the results table is a difference in that
choice and not in forty incidental details.

    encoder=concat   the beam-search baseline: labels serialised into the decoder context,
                     truncated to whatever fits.  Not a set model.
    encoder=deepsets mean/max pooling over per-label vectors.  Permutation
                     invariant by construction, unbounded |K|, no interaction
                     between labels.
    encoder=settrans self-attention across the set, then pooling by learned
                     queries.  Interaction, still permutation invariant.

Two heads sit on top of the organization vector:

    generator  autoregressive over BPE tokens, cross-attending the set memory.
               Supplies open-vocabulary names retrieval cannot know.
    retriever  scores every label in a fixed candidate vocabulary against the
               organization vector.  Supplies the tail, which the measurements
               say the generator cannot reach.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class V3Config:
    vocab_size: int = 4096
    d_model: int = 256
    n_head: int = 4
    label_layers: int = 2          # encodes one label's tokens -> one vector
    set_layers: int = 2            # mixes labels with each other
    dec_layers: int = 4
    n_seeds: int = 4               # learned pooling queries (set transformer)
    max_label_tokens: int = 16
    max_set: int = 512             # labels per organization; the beam-search model managed 24
    dropout: float = 0.1
    encoder: str = "settrans"      # concat | deepsets | settrans
    cand_vocab: int = 0            # size of the retrieval vocabulary, 0 = off
    pad_id: int = 0


class Block(nn.Module):
    """Pre-LN transformer block; cross-attends when memory is supplied."""
    def __init__(self, cfg, causal=False, cross=False):
        super().__init__()
        self.causal, self.cross = causal, cross
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        if cross:
            self.ln_x = nn.LayerNorm(cfg.d_model)
            self.xattn = nn.MultiheadAttention(cfg.d_model, cfg.n_head,
                                               dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model),
                                 nn.Dropout(cfg.dropout))

    def forward(self, x, mem=None, key_padding_mask=None, mem_padding_mask=None):
        h = self.ln1(x)
        mask = None
        if self.causal:
            t = x.size(1)
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        a, _ = self.attn(h, h, h, attn_mask=mask, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        if self.cross and mem is not None:
            h = self.ln_x(x)
            a, _ = self.xattn(h, mem, mem, key_padding_mask=mem_padding_mask,
                              need_weights=False)
            x = x + a
        return x + self.mlp(self.ln2(x))


class LabelEncoder(nn.Module):
    """One label's token ids -> one vector.  Shared by every variant."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_label_tokens, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.label_layers)])
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, ids):                       # (B, L, T)
        B, L, T = ids.shape
        flat = ids.view(B * L, T)
        pad = flat.eq(self.cfg.pad_id)
        # a slot padded to its full width would mask every key and make softmax
        # return NaN, which then survives the set-level mask.  Leave one key
        # attendable and zero the result instead.
        dead = pad.all(dim=1)
        pad = pad.clone()
        pad[dead, 0] = False
        x = self.tok(flat) + self.pos(torch.arange(T, device=ids.device))[None]
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad)
        x = self.ln(x)
        keep = (~pad).float().unsqueeze(-1)
        vec = (x * keep).sum(1) / keep.sum(1).clamp(min=1)     # mean over real tokens
        vec = vec.masked_fill(dead.unsqueeze(-1), 0.0)
        return vec.view(B, L, -1)


class SetEncoder(nn.Module):
    """Known labels -> organization memory.  This is the variable under study."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.labels = LabelEncoder(cfg)
        if cfg.encoder == "settrans":
            self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.set_layers)])
            self.seeds = nn.Parameter(torch.randn(1, cfg.n_seeds, cfg.d_model) * 0.02)
            self.pool = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        elif cfg.encoder == "deepsets":
            self.rho = nn.Sequential(nn.Linear(2 * cfg.d_model, cfg.d_model), nn.GELU(),
                                     nn.Linear(cfg.d_model, cfg.d_model))
        self.ln = nn.LayerNorm(cfg.d_model)

    def forward(self, set_ids, set_mask):         # (B, L, T), (B, L) True = real
        vecs = self.labels(set_ids)               # (B, L, D)
        pad = ~set_mask
        if self.cfg.encoder == "settrans":
            x = vecs
            for blk in self.blocks:
                x = blk(x, key_padding_mask=pad)
            q = self.seeds.expand(x.size(0), -1, -1)
            mem, _ = self.pool(q, x, x, key_padding_mask=pad, need_weights=False)
            return self.ln(mem)                   # (B, S, D)
        keep = set_mask.float().unsqueeze(-1)
        mean = (vecs * keep).sum(1) / keep.sum(1).clamp(min=1)
        mx = vecs.masked_fill(pad.unsqueeze(-1), -1e4).max(1).values
        return self.ln(self.rho(torch.cat([mean, mx], -1))).unsqueeze(1)   # (B, 1, D)


class SubFuryV3(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.set_enc = SetEncoder(cfg)
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_label_tokens + 1, cfg.d_model)
        self.dec = nn.ModuleList([Block(cfg, causal=True, cross=True)
                                  for _ in range(cfg.dec_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight                 # tied
        if cfg.cand_vocab:
            self.cand = nn.Embedding(cfg.cand_vocab, cfg.d_model)
            self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def org(self, set_ids, set_mask):
        return self.set_enc(set_ids, set_mask)

    def generate_logits(self, mem, tgt_ids):
        x = self.tok(tgt_ids) + self.pos(torch.arange(tgt_ids.size(1),
                                                      device=tgt_ids.device))[None]
        for blk in self.dec:
            x = blk(x, mem=mem)
        return self.head(self.ln_f(x))

    def retrieve_scores(self, mem):
        """One score per candidate label — the retrieval head."""
        q = self.proj(mem.mean(1))                         # (B, D)
        return q @ self.cand.weight.t()                    # (B, cand_vocab)

    def loss(self, batch, lambda_rank=1.0, prior_logits=None):
        mem = self.org(batch["set_ids"], batch["set_mask"])
        out = {}
        logits = self.generate_logits(mem, batch["tgt_in"])
        out["gen"] = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                     batch["tgt_out"].reshape(-1),
                                     ignore_index=self.cfg.pad_id)
        if self.cfg.cand_vocab and "cand_pos" in batch:
            s = self.retrieve_scores(mem)
            # positives are the held-out labels; unobserved labels are NOT known
            # negatives, so the loss only pushes down explicitly sampled ones
            if prior_logits is not None:
                s = s - prior_logits                       # score above the prior
            neg = s.gather(1, batch["cand_neg"])           # (B, Kneg)
            cp = batch["cand_pos"]                          # (B, P), -1 = padding
            valid_p = cp.ge(0)
            pos = s.gather(1, cp.clamp(min=0))              # (B, P)
            # one softmax per positive, each against the shared negative pool:
            # a split with 30 held-out labels supplies 30 gradients, not one
            per = F.cross_entropy(
                torch.cat([pos.unsqueeze(-1), neg.unsqueeze(1).expand(-1, pos.size(1), -1)], -1)
                     .reshape(-1, 1 + neg.size(1)),
                torch.zeros(pos.numel(), dtype=torch.long, device=s.device),
                reduction="none").view_as(pos)
            # rows with no label in the candidate vocabulary must not contribute a
            # constant ln(1+Kneg) to the reported loss
            keep = valid_p
            if "cand_valid" in batch:
                keep = keep & batch["cand_valid"].unsqueeze(1)
            out["rank"] = (per * keep).sum() / keep.sum().clamp(min=1)
        else:
            out["rank"] = torch.zeros((), device=logits.device)
        out["total"] = out["gen"] + lambda_rank * out["rank"]
        return out
