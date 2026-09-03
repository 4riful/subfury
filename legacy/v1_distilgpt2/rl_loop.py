"""Reinforcement-style feedback loop (SubFury methodology step 6-7).

Loop:
  1. Generate candidate subdomain labels with the current model.
  2. Validate them via live DNS against the target domain.
  3. Reward: labels that resolved become new fine-tuning data
     (repeated `reward_weight` times); failures are dropped.
  4. Continue fine-tuning the model on the reward set at a low LR.

This mirrors the original notebook's approach of periodically re-training
on validated results rather than a policy-gradient RL algorithm — the
"reward" is inclusion (weighted) in the next fine-tuning corpus.

Only run against domains you are authorized to test.
"""

import argparse

import torch
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from dataset import SubdomainDataset
from dns_validate import validate
from generate import generate_subdomains, load_model


def rl_finetune(model, tokenizer, positives, output_dir, reward_weight=4, lr=1e-5, epochs=2):
    """Fine-tune on DNS-validated labels, weighted by repetition."""
    corpus = positives * reward_weight
    train_ds = SubdomainDataset(corpus, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=output_dir + "_ckpt",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", help="target domain (must be authorized)")
    ap.add_argument("--model", default="results/final_model")
    ap.add_argument("--output", default="results/final_model_finetuned")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--candidates", type=int, default=200, help="candidates per round")
    ap.add_argument("--reward-weight", type=int, default=4)
    args = ap.parse_args()

    model, tokenizer, device = load_model(args.model)
    all_hits = {}

    for rnd in range(1, args.rounds + 1):
        print(f"\n=== Round {rnd}/{args.rounds} ===")
        labels = generate_subdomains(model, tokenizer, device, count=args.candidates)
        print(f"Generated {len(labels)} candidate labels")

        results = validate(labels, args.domain)
        hits = {k: v for k, v in results.items() if v}
        all_hits.update(hits)
        rate = len(hits) / max(len(results), 1)
        print(f"DNS success rate: {len(hits)}/{len(results)} ({rate:.1%})")
        for fqdn, ip in sorted(hits.items()):
            print(f"  [+] {fqdn} -> {ip}")

        positives = [fqdn[: -len(args.domain) - 1] for fqdn in hits]
        if not positives:
            print("No positive rewards this round; skipping fine-tune step.")
            continue

        print(f"Reinforcing on {len(positives)} validated labels (weight={args.reward_weight})")
        rl_finetune(model, tokenizer, positives, args.output,
                    reward_weight=args.reward_weight)

    print(f"\nTotal validated subdomains: {len(all_hits)}")
    print(f"Final model saved to {args.output}")


if __name__ == "__main__":
    main()
