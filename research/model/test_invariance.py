"""The set encoders must be permutation invariant and size-unbounded.

The beam-search model achieves invariance by sorting and then truncating to 24 labels, which throws
information away. These assertions are what separates this claim from that.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from model import V3Config, SubFuryV3                                # noqa: E402

torch.manual_seed(0)
B, L, T = 2, 64, 16
ids = torch.randint(1, 4096, (B, L, T))
mask = torch.ones(B, L, dtype=torch.bool)

for enc in ("deepsets", "settrans"):
    m = SubFuryV3(V3Config(encoder=enc)).eval()
    with torch.no_grad():
        a = m.org(ids, mask)
        perm = torch.randperm(L)
        b = m.org(ids[:, perm], mask[:, perm])
    delta = (a - b).abs().max().item()
    print(f"{enc:9s} permutation delta {delta:.2e}  {'PASS' if delta < 1e-4 else 'FAIL'}")

# and it must accept sets far larger than the beam-search model's 24-label ceiling
m = SubFuryV3(V3Config(encoder="settrans")).eval()
for n in (24, 128, 512):
    with torch.no_grad():
        out = m.org(torch.randint(1, 4096, (1, n, T)), torch.ones(1, n, dtype=torch.bool))
    print(f"|K|={n:<4} org shape {tuple(out.shape)}  ok")

# masked padding must not leak: padded labels change nothing
m2 = SubFuryV3(V3Config(encoder="settrans")).eval()
small_ids = torch.randint(1, 4096, (1, 8, T))
pad_ids = torch.cat([small_ids, torch.zeros(1, 24, T, dtype=torch.long)], 1)
pad_mask = torch.cat([torch.ones(1, 8, dtype=torch.bool),
                      torch.zeros(1, 24, dtype=torch.bool)], 1)
with torch.no_grad():
    x = m2.org(small_ids, torch.ones(1, 8, dtype=torch.bool))
    y = m2.org(pad_ids, pad_mask)
d = (x - y).abs().max().item()
print(f"padding leak    {d:.2e}  {'PASS' if d < 1e-4 else 'FAIL'}")
