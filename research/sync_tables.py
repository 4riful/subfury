"""Re-embed measured tables into the docs from the artifacts that produced them.

Anchored on the prose that introduces each block, NOT on the table's own header:
baselines.json and ablation.json share the identical `method  recall@10 ...`
header line, so header matching silently puts one table under the other's
heading. Each anchor below is the last line of prose before the fence.
"""
import io, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "results", "research")

# (doc, anchor substring that precedes the fence, artifact, key)
BINDINGS = [
    ("README.md", "percentile-bootstrap 95% CIs over apexes, 2,000 resamples.",
     "baselines.json", "table"),
    ("README.md", "2,000 resamples (`paired_vs_frequency_prior` in the same artifact):",
     "baselines.json", "paired_table"),
    ("README.md", "prior subtraction.",
     "ablation.json", "table"),
    ("README.md", "(`results/research/paired_vs_beam.json`):",
     "paired_vs_beam.json", "table"),
    ("research/METHODOLOGY.md", "seed-1337 split and budgets as\nevery baseline in §2.",
     "ablation.json", "table"),
    ("research/METHODOLOGY.md", "Paired against the beam-search model — same apexes, so between-apex variance cancels:",
     "paired_vs_beam.json", "table"),
]

def table_of(fn, key):
    p = os.path.join(ART, fn)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    t = d.get(key)
    return t.rstrip("\n") if isinstance(t, str) else None

fence = re.compile(r"```\n.*?\n```", re.S)
changed = 0
for doc in sorted({b[0] for b in BINDINGS}):
    p = os.path.join(ROOT, doc)
    s = io.open(p, encoding="utf-8").read()
    for d, anchor, fn, key in BINDINGS:
        if d != doc:
            continue
        t = table_of(fn, key)
        if t is None:
            print(f"  {doc}: {fn}:{key} absent, skipped"); continue
        i = s.find(anchor)
        if i < 0:
            print(f"  {doc}: anchor not found for {fn}:{key}"); continue
        m = fence.search(s, i)
        if not m:
            print(f"  {doc}: no fence after anchor for {fn}:{key}"); continue
        new = "```\n" + t + "\n```"
        if m.group(0) != new:
            s = s[:m.start()] + new + s[m.end():]
            changed += 1
            print(f"  {doc}: resynced {fn}:{key}")
    io.open(p, "w", encoding="utf-8").write(s)
print(f"{changed} block(s) updated")
