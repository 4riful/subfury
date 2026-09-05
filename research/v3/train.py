"""Train SubFury v3.  One loop, config-switched, so ablations differ in one thing.

    python research/v3/train.py --encoder settrans --steps 4000
    python research/v3/train.py --encoder deepsets --lambda-rank 0   # generator only

Every run writes results/v3/<tag>/{best.pt,config.json,log.jsonl} so the harness
can score them side by side.
"""
import argparse, json, math, os, sys, time

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model import V3Config, SubFuryV3                                # noqa: E402


def build(args, vocab_size, cand_vocab):
    cfg = V3Config(vocab_size=vocab_size, d_model=args.d_model, n_head=args.n_head,
                   label_layers=args.label_layers, set_layers=args.set_layers,
                   dec_layers=args.dec_layers, n_seeds=args.n_seeds,
                   max_label_tokens=args.max_label_tokens, max_set=args.max_set,
                   dropout=args.dropout, encoder=args.encoder,
                   cand_vocab=cand_vocab if args.lambda_rank > 0 else 0)
    return cfg, SubFuryV3(cfg)


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.1 * base + 0.45 * base * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="data/groups_train.jsonl")
    ap.add_argument("--val-jsonl", default="data/groups_val.jsonl")
    ap.add_argument("--tokenizer", default="results/subfury_v2/tokenizer.json")
    ap.add_argument("--encoder", default="settrans", choices=["concat", "deepsets", "settrans"])
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--label-layers", type=int, default=2)
    ap.add_argument("--set-layers", type=int, default=2)
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--max-label-tokens", type=int, default=16)
    ap.add_argument("--max-set", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cand-vocab", type=int, default=20000)
    ap.add_argument("--n-neg", type=int, default=32)
    ap.add_argument("--hard-frac", type=float, default=0.75)
    ap.add_argument("--lambda-rank", type=float, default=1.0)
    ap.add_argument("--use-prior", type=int, default=1,
                    help="subtract prior logits so the head must beat popularity")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out-root", default="results/v3")
    args = ap.parse_args()

    tag = args.tag or f"{args.encoder}-rank{args.lambda_rank:g}-prior{args.use_prior}"
    out = os.path.join(args.out_root, tag)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from data import DataConfig, LabelVocab, OrgSetDataset, make_loader   # noqa: E402
    from tokenizers import Tokenizer                                      # noqa: E402
    tok = Tokenizer.from_file(args.tokenizer)

    print("building the candidate vocabulary…", flush=True)
    vocab = LabelVocab.build(args.train_jsonl, size=args.cand_vocab)
    prior = vocab.prior_logits.to(dev) if args.use_prior else None

    cfg, model = build(args, tok.get_vocab_size(), len(vocab))
    model.to(dev)
    print(f"{tag}: {model.num_params()/1e6:.1f}M params on {dev}", flush=True)

    dcfg = DataConfig(n_neg=args.n_neg, hard_frac=args.hard_frac, seed=args.seed)
    train_ds = OrgSetDataset(args.train_jsonl, tok, vocab, cfg, dcfg)
    val_ds = OrgSetDataset(args.val_jsonl, tok, vocab, cfg, dcfg)
    print(f"data: {len(train_ds)} train orgs, {len(val_ds)} val", flush=True)
    train_dl = make_loader(train_ds, cfg, args.batch, True, args.workers, drop_last=True)
    val_dl = make_loader(val_ds, cfg, args.batch, False, 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))
    logf = open(os.path.join(out, "log.jsonl"), "a")
    json.dump({k: v for k, v in vars(args).items()},
              open(os.path.join(out, "config.json"), "w"), indent=1)

    def evaluate():
        model.eval()
        tot = {"gen": 0.0, "rank": 0.0, "n": 0}
        with torch.no_grad():
            for i, b in enumerate(val_dl):
                if i >= 40:
                    break
                b = {k: v.to(dev) for k, v in b.items()}
                o = model.loss(b, args.lambda_rank, prior)
                tot["gen"] += o["gen"].item(); tot["rank"] += o["rank"].item(); tot["n"] += 1
        model.train()
        n = max(tot["n"], 1)
        return tot["gen"] / n, tot["rank"] / n

    best, step, t0, epoch = float("inf"), 0, time.time(), 0
    model.train()
    while step < args.steps:
        train_ds.set_epoch(epoch); epoch += 1
        for b in train_dl:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
            with torch.amp.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = model.loss(b, args.lambda_rank, prior)
            opt.zero_grad(set_to_none=True)
            scaler.scale(o["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            step += 1

            if step % 100 == 0:
                print(f"  step {step:>5}  gen {o['gen'].item():.3f}  "
                      f"rank {o['rank'].item():.3f}  {(time.time()-t0)/step:.2f}s/step", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                vg, vr = evaluate()
                rec = {"step": step, "val_gen": round(vg, 4), "val_rank": round(vr, 4),
                       "lr": opt.param_groups[0]["lr"]}
                print(f"  eval  step {step}: gen {vg:.4f}  rank {vr:.4f}", flush=True)
                logf.write(json.dumps(rec) + "\n"); logf.flush()
                score = vg + args.lambda_rank * vr
                if score < best:
                    best = score
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                                "vocab": vocab.labels, "step": step, "val": rec},
                               os.path.join(out, "best.pt"))
    print(f"done: {tag}  best val {best:.4f}  -> {out}/best.pt")


if __name__ == "__main__":
    main()
