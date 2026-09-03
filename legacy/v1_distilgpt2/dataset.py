"""Dataset handling for SubFury.

The n0kovo_subdomains wordlists contain one subdomain *label* per line
(e.g. "mail", "dev.api"). We fine-tune DistilGPT-2 as a causal LM over
these lines so it learns the character/token patterns of real-world
subdomains.
"""

import random

import torch
from torch.utils.data import Dataset


class SubdomainDataset(Dataset):
    """Tokenizes one subdomain per line for causal-LM fine-tuning."""

    def __init__(self, subdomains, tokenizer, max_length=32):
        self.subdomains = subdomains
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.subdomains)

    def __getitem__(self, idx):
        # EOS terminates each subdomain so generation learns where to stop.
        text = self.subdomains[idx] + self.tokenizer.eos_token
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


def load_wordlist(path, limit=None, seed=42):
    """Load a subdomain wordlist, optionally sampling `limit` lines."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip().lower() for ln in f if ln.strip()]
    # Basic hygiene: valid DNS label charset only.
    lines = [ln for ln in lines if all(c.isalnum() or c in ".-_" for c in ln)]
    if limit and limit < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, limit)
    return lines
