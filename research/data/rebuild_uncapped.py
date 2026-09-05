"""Replay data_prep's split, but keep every label instead of capping at 40.

Why: MAX_LABELS=40 means |K| never exceeds 20 in evaluation, so the truncation
the model applies at inference — sorted(known)[-block_size//8:], i.e. the 24
alphabetically-last labels — is never exercised by the test set. In deployment a
passive seed supplies 500-700 labels and the truncation always bites. To measure
that, the held-out apexes are needed at their true size.

The split is reproduced exactly: same seed, same stream order, and the same
rng.sample() call is consumed when an apex exceeds the cap, so the RNG state
stays in lockstep with the original run and the test set is identical.
"""
import argparse, glob, json, os, random, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "subfury_v2"))
from data_prep import MAX_LABELS, group_by_apex, parse_vertices     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="data/cc/part-*.txt.gz")
    ap.add_argument("--val-frac", type=float, default=0.01)
    ap.add_argument("--test-frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/groups_test_uncapped.jsonl")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--cap", type=int, default=0, help="0 = keep every label")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(args.input_glob))
    print(f"replaying the split over {len(paths)} vertex files…", flush=True)

    n_test = kept = 0
    sizes = []
    with open(args.out, "w") as out:
        for apex, labels in group_by_apex(parse_vertices(paths)):
            full = labels                       # every label, before the cap
            if len(labels) > MAX_LABELS:
                labels = rng.sample(labels, MAX_LABELS)   # consume identically
            r = rng.random()
            where = ("test" if r < args.test_frac else
                     "val" if r < args.test_frac + args.val_frac else "train")
            if where == args.split:
                n_test += 1
                if len(full) >= 6:
                    labs = full
                    if args.cap and len(labs) > args.cap:
                        labs = random.Random(abs(hash(apex)) & 0xffffffff).sample(labs, args.cap)
                    out.write(json.dumps({"apex": apex, "labels": labs}) + "\n")
                    kept += 1
                    sizes.append(len(labs))

    sizes.sort()
    print(f"{args.split} apexes: {n_test}, kept (>=6 labels): {kept} -> {args.out}")
    if sizes:
        q = lambda p: sizes[min(int(len(sizes) * p), len(sizes) - 1)]
        print(f"  labels/apex  median {q(.5)}  p75 {q(.75)}  p90 {q(.9)}  p99 {q(.99)}  max {sizes[-1]}")
        for t in (24, 48, 100, 200):
            print(f"  apexes with >{t} labels: {sum(1 for s in sizes if s > t)}")


if __name__ == "__main__":
    main()
