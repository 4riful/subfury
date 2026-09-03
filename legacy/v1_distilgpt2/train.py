"""Fine-tune DistilGPT-2 on the n0kovo_subdomains wordlist.

Methodology (from the original SubFury notebook / README):
  - architecture: distilgpt2, causal-LM objective
  - dataset: n0kovo_subdomains
  - mixed precision (fp16), save_total_limit=2
  - train/val split + early stopping
  - optional Weights & Biases tracking (offline unless WANDB_API_KEY is set)
"""

import argparse
import os

import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from dataset import SubdomainDataset, load_wordlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wordlist", default="data/subdomains_tiny.txt")
    ap.add_argument("--limit", type=int, default=50_000, help="max training lines")
    ap.add_argument("--model", default="distilgpt2", help="base model or checkpoint to continue from")
    ap.add_argument("--output", default="results/final_model")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wandb", action="store_true", help="enable W&B logging (needs WANDB_API_KEY env var)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # W&B: never hardcode API keys — read from the environment if enabled.
    if args.wandb and os.environ.get("WANDB_API_KEY"):
        report_to = ["wandb"]
        os.environ.setdefault("WANDB_PROJECT", "subdomain-prediction")
    else:
        report_to = []
        os.environ["WANDB_DISABLED"] = "true"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    subdomains = load_wordlist(args.wordlist, limit=args.limit)
    train_lines, val_lines = train_test_split(subdomains, test_size=0.05, random_state=42)
    print(f"Train: {len(train_lines)}  Val: {len(val_lines)}")

    train_ds = SubdomainDataset(train_lines, tokenizer)
    val_ds = SubdomainDataset(val_lines, tokenizer)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output + "_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        fp16=device == "cuda",
        eval_strategy="steps",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=report_to,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()
