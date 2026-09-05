"""Compact GPT for subdomain prediction (nanoGPT-style, from scratch).

Architecture follows Karpathy's nanoGPT (MIT) as used by subwiz:
pre-LN transformer decoder, learned positional embeddings, weight-tied
LM head. Sized ~10M params for a 4096 BPE vocab.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    # Right-sized for ~3M-token corpus (84k apexes, ~1M labels): a deeper/
    # wider net overfits despite known-subset augmentation. ~6M params.
    block_size: int = 192
    vocab_size: int = 4096
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 300
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SubFuryGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction="none",
            )
            if loss_mask is not None:
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1)
            else:
                loss = loss.mean()
        return logits, loss

    @torch.no_grad()
    def beam_search(self, prefix_ids, end_id, num_beams=64, topn=50,
                    max_new_tokens=16, length_penalty=0.0, banned_first=None):
        """Return up to `topn` (token_list, logprob) completions of prefix.

        Deterministic beam search: expand all beams by full vocab each
        step, keep the best `num_beams` unfinished plus all finished.
        """
        device = prefix_ids.device
        beams = [(prefix_ids.tolist(), 0.0)]  # (tokens, logprob)
        finished = []
        prefix_len = len(beams[0][0])

        for step in range(max_new_tokens):
            if not beams:
                break
            batch = torch.tensor([b[0] for b in beams], device=device)
            logits, _ = self(batch)
            logp = F.log_softmax(logits[:, -1, :], dim=-1)  # (nbeams, V)
            if step == 0 and banned_first is not None:
                logp[:, banned_first] = float("-inf")
            k = min(num_beams, logp.size(-1))
            top_lp, top_ix = logp.topk(k, dim=-1)
            cand = []
            for bi, (toks, score) in enumerate(beams):
                for lp, ix in zip(top_lp[bi].tolist(), top_ix[bi].tolist()):
                    cand.append((toks + [ix], score + lp))
            cand.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for toks, score in cand:
                if toks[-1] == end_id:
                    norm = score / ((len(toks) - prefix_len) ** length_penalty
                                    if length_penalty else 1.0)
                    finished.append((toks[prefix_len:-1], norm))
                elif len(beams) < num_beams:
                    beams.append((toks, score))
                if len(beams) >= num_beams:
                    break
            if len(finished) >= topn * 3:
                break

        finished.sort(key=lambda c: c[1], reverse=True)
        return finished[: topn * 3]
