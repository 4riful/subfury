"""Train SubFury v2: set-conditioned subdomain prediction.

Each training example is built dynamically from an apex group:

    known_1 [SEP] known_2 [SEP] ... known_k [DELIM] target [END]

with cross-entropy loss computed only on the target + [END] positions.
Random subsetting of the known set each epoch acts as augmentation and
teaches the model to work from few or many known subdomains.
"""

import argparse
import json
import math
import os
import random
import time

import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset

from model import GPTConfig, SubFuryGPT


class GroupDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, block_size=256, max_known=12, seed=0):
        self.groups = []
        with open(jsonl_path) as f:
            for line in f:
                rec = json.loads(line)
                if len(rec["labels"]) >= 2:
                    self.groups.append(rec["labels"])
        self.tok = tokenizer
        self.block_size = block_size
        self.max_known = max_known
        self.pad = tokenizer.token_to_id("[PAD]")
        self.sep = tokenizer.token_to_id("[SEP]")
        self.delim = tokenizer.token_to_id("[DELIM]")
        self.end = tokenizer.token_to_id("[END]")
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        labels = self.groups[idx][:]
        self.rng.shuffle(labels)
        n = len(labels)
        k = self.rng.randint(1, min(n - 1, self.max_known))
        known, target = labels[:k], labels[k]

        ids = []
        for i, lab in enumerate(known):
            if i:
                ids.append(self.sep)
            ids.extend(self.tok.encode(lab).ids)
        ids.append(self.delim)
        ctx_len = len(ids)
        ids.extend(self.tok.encode(target).ids)
        ids.append(self.end)
        ids = ids[-self.block_size:]
        ctx_len = min(ctx_len, len(ids) - 1)

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        mask = torch.zeros(len(y))
        mask[ctx_len - 1:] = 1.0  # predict target tokens + [END]
        return x, y, mask


def collate(batch, pad_id):
    maxlen = max(len(x) for x, _, _ in batch)
    X = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    Y = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)
    M = torch.zeros(len(batch), maxlen)
    for i, (x, y, m) in enumerate(batch):
        X[i, : len(x)] = x
        Y[i, : len(y)] = y
        M[i, : len(m)] = m
    return X, Y, M


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype, max_batches=100):
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y, m) in enumerate(loader):
        if i >= max_batches:
            break
        x, y, m = x.to(device), y.to(device), m.to(device)
        with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
            _, loss = model(x, y, m)
        tot += loss.item()
        n += 1
    model.train()
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--val-jsonl", default="data/groups_val.jsonl")
    ap.add_argument("--tokenizer", default="results/tokenizer.json")
    ap.add_argument("--out", default="results/subfury_v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-embd", type=int, default=300)
    ap.add_argument("--block-size", type=int, default=192)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"device={device} autocast={autocast_dtype}")

    tok = Tokenizer.from_file(args.tokenizer)
    pad_id = tok.token_to_id("[PAD]")

    train_ds = GroupDataset(args.train_jsonl, tok, args.block_size, seed=1)
    val_ds = GroupDataset(args.val_jsonl, tok, args.block_size, seed=2)
    print(f"apex groups: train={len(train_ds)} val={len(val_ds)}")

    coll = lambda b: collate(b, pad_id)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=coll, num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=coll)

    cfg = GPTConfig(block_size=args.block_size, vocab_size=tok.get_vocab_size(),
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = SubFuryGPT(cfg).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device)["model"])
        print(f"resumed from {args.resume}")
    print(f"params: {model.num_params()/1e6:.1f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1,
                            betas=(0.9, 0.95))
    total_steps = len(train_dl) * args.epochs

    def lr_at(step):
        if step < args.warmup:
            return args.lr * step / args.warmup
        p = (step - args.warmup) / max(total_steps - args.warmup, 1)
        return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * p))

    os.makedirs(args.out, exist_ok=True)
    # stage tokenizer alongside checkpoints so predict.py/evaluate.py find it
    import shutil
    shutil.copy(args.tokenizer, os.path.join(args.out, "tokenizer.json"))
    best_val = float("inf")
    step = 0
    t0 = time.time()
    model.train()
    for epoch in range(args.epochs):
        for x, y, m in train_dl:
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y, m = x.to(device), y.to(device), m.to(device)
            with torch.autocast(device_type=device, dtype=autocast_dtype, enabled=device == "cuda"):
                _, loss = model(x, y, m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} "
                      f"loss {loss.item():.4f} lr {lr:.2e} "
                      f"({(time.time()-t0)/step:.3f}s/step)", flush=True)
            if step % args.eval_every == 0:
                vl = evaluate(model, val_dl, device, autocast_dtype)
                print(f"eval_loss={vl:.4f} (best {best_val:.4f})", flush=True)
                if vl < best_val:
                    best_val = vl
                    torch.save({"model": model.state_dict(), "config": vars(cfg),
                                "step": step, "val_loss": vl},
                               os.path.join(args.out, "best.pt"))
        torch.save({"model": model.state_dict(), "config": vars(cfg), "step": step},
                   os.path.join(args.out, "last.pt"))

    vl = evaluate(model, val_dl, device, autocast_dtype)
    print(f"FINAL eval_loss={vl:.4f} best={min(best_val, vl):.4f}")
    if vl < best_val:
        torch.save({"model": model.state_dict(), "config": vars(cfg),
                    "step": step, "val_loss": vl}, os.path.join(args.out, "best.pt"))
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
