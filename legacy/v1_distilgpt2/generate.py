"""Generate candidate subdomains with the fine-tuned model.

Sampling parameters follow the original notebook (top_k=50, top_p=0.95,
temperature=0.7) with regex post-processing to keep only realistic
DNS labels.
"""

import argparse
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Valid DNS label(s): alnum + hyphen, dot-separated, no leading/trailing hyphen
SUBDOMAIN_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")

# Common single-char seeds cover the distribution of first characters
SEEDS = list("abcdefghijklmnopqrstuvwxyz0123456789")


def load_model(model_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def generate_subdomains(model, tokenizer, device, count=100, temperature=0.7,
                        top_k=50, top_p=0.95, max_length=20, batch_per_seed=16):
    """Sample subdomain labels from the model. Returns a deduped list."""
    results = set()
    seed_idx = 0
    while len(results) < count and seed_idx < len(SEEDS) * 4:
        seed = SEEDS[seed_idx % len(SEEDS)]
        seed_idx += 1
        inputs = tokenizer(seed, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=batch_per_seed,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        for out in outputs:
            text = tokenizer.decode(out, skip_special_tokens=True).strip().lower()
            text = text.split()[0] if text.split() else ""
            if text and SUBDOMAIN_RE.match(text):
                results.add(text)
    return sorted(results)[:count] if count < len(results) else sorted(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/final_model")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default=None, help="write results to file")
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    subs = generate_subdomains(model, tokenizer, device, count=args.count,
                               temperature=args.temperature)
    for s in subs:
        print(s)
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(subs) + "\n")


if __name__ == "__main__":
    main()
