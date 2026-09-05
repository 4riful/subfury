#!/bin/bash
# One axis at a time, not a grid. Each run writes results/runs/<tag>/.
set -u
COMMON="--train-jsonl data/groups_train_uncapped.jsonl --val-jsonl data/groups_val_uncapped.jsonl \
        --steps ${STEPS:-3000} --batch 24 --workers 2 --max-set 64 --eval-every 500"
run () { echo "=== $1"; python3 research/model/train.py --tag "$1" $2 $COMMON 2>&1 | tail -6; }

# axis 1: how the known set is turned into an organisation representation
run settrans-full   "--encoder settrans --lambda-rank 1 --use-prior 1"
run deepsets-full   "--encoder deepsets --lambda-rank 1 --use-prior 1"
# axis 2: does the ranking head matter, and does beating the prior matter
run settrans-gen    "--encoder settrans --lambda-rank 0"
run settrans-noprior "--encoder settrans --lambda-rank 1 --use-prior 0"
