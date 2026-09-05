"""Train a domain-specific BPE tokenizer on subdomain labels.

Rationale (subwiz methodology): a generic English BPE wastes vocabulary
on natural language. A small BPE trained on subdomain labels captures
units like "api", "dev", "staging", "vpn", "-prod" natively.
"""

import argparse
import json

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIALS = ["[PAD]", "[DELIM]", "[END]", "[SEP]"]


def iter_labels(jsonl_path):
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            yield from rec["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="results/tokenizer.json")
    args = ap.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    # split on . and - but keep them as tokens (they carry structure)
    tokenizer.pre_tokenizer = pre_tokenizers.Split(
        pattern=r"[.-]", behavior="isolated"
    )
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIALS,
        initial_alphabet=list("abcdefghijklmnopqrstuvwxyz0123456789.-"),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_labels(args.train_jsonl), trainer=trainer)
    tokenizer.save(args.out)
    print(f"vocab size: {tokenizer.get_vocab_size()} -> {args.out}")


if __name__ == "__main__":
    main()
