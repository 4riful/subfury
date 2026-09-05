# Product

<!-- impeccable:product-schema 1 -->

# PRODUCT.md — SubFury

Durable product context. Facts here are verified, not aspirational. Every
performance number is traceable to a JSON artifact under `results/research/`
(and, where noted, to the data files it was derived from); claims that are not
are labelled as such.

## Platform

web

## What it is

A subdomain prediction engine. A 7.79M-parameter transformer, trained from
scratch on subdomain data, predicts an organization's unknown subdomains from
the ones already discovered, ranks them by probability, validates them over DNS,
and feeds confirmed hits back as context.

## The one idea

**Organizations name infrastructure consistently.** A host called `inventory`
makes `inventoryapp` likely. That per-organization signal is invisible to a
static wordlist — which is identical for every target. The signal is real and
measured (swapping in another org's labels collapses recall@100 from 0.288 to
0.059), but it buys a lead only while the candidate budget is small; see Evidence
on Hand, which is measured rather than asserted.

## Users

Security researchers, bug bounty hunters, pentesters, and red teamers doing
reconnaissance against authorized targets. Two postures, one surface:

- **Operator** — wants results fast: paste known subdomains, get validated new
  ones, pipe them into `httpx` / `nuclei` / `dnsx`.
- **Evaluator** — deciding whether to trust it: needs to see which model runs,
  how it works, and evidence against a real baseline — including the budgets at
  which it does not win.

Both run it locally on their own machine, usually mid-engagement with a scope
list beside them.

## Product Purpose

Spend a finite DNS budget on the candidates most likely to exist for *this*
target, instead of the same wordlist everyone fires at every target. Success is
a validated hostname the operator did not already have, found within the query
budget they were willing to spend.

## Positioning

A wordlist is target-independent by construction. SubFury conditions generation
on the set you already hold, so the candidate ranking changes per organization —
and it is measured at an equal candidate budget against five controls, including
a frequency prior fit on its own training data, rather than described. That prior
is the honest bar: SubFury clears it at small budgets and falls behind it at
N=200. No CT-log scraper or brute-forcer makes a per-target prediction.

## Operating Context

- Runs locally: FastAPI on `127.0.0.1:8000`, no auth, no telemetry, no accounts.
- Model runs on a laptop GPU or CPU; the shipped checkpoint was trained on an
  RTX 4060 and loads in seconds.
- The console is a five-section app — Run, Results, History, Model, Methodology
  — plus a live run monitor that opens while the pipeline streams.
- Composes with the operator's existing chain: passive sources in, validated
  hostnames out as list / JSON / CSV.

## Capabilities and Constraints

**Pipeline.** Known set → tokenize & condition → beam search → DNS resolve →
recurse (hits rejoin the known set) → export. Streamed to the browser over SSE.

**Search budget: Auto or Manual.** Auto sizes candidates, beam width and rounds
from the known set — its size and how many distinct name families it spans —
and states its reasoning in the UI. Manual unlocks the three fields and still
reports what Auto would have chosen. DNS off forces a single round.

**Passive seeding.** crt.sh, certspotter, hackertarget and the Wayback CDX index
are queried concurrently under a 20s budget and merged; OTX joins when
`OTX_API_KEY` is set. A seed fails only when every source fails, and per-source
status is always reported — individual sources fail often (crt.sh 502s and
stalls routinely). Passive sources send **nothing to the target**.

**Authorization is mandatory.** DNS validation is the only thing that emits
traffic toward the target, and every surface that can emit it must say so.

**Results belong to the user.** Export is one click; run history stays in the
browser's localStorage and is never transmitted.

MIT licensed, self-hostable.

### Open decisions

None outstanding. *(Resolved 2026-09-06: the landing surface addresses a
visitor who already has it running locally — primary CTA opens the console, the
install block serves repo readers second.)*

## Model

| | |
|---|---|
| Architecture | nanoGPT-style causal decoder (pre-LN, SDPA, tied head) |
| Size | 6 layers · 6 heads · `n_embd` 300 · `block_size` 192 → 7,790,400 params |
| Tokenizer | BPE, 4096 vocab, trained on subdomain labels |
| Objective | `known_1 [SEP] … [DELIM] target [END]`, loss on target only |
| Decoding | deterministic beam search |
| Data | 84,144 apex groups (~1.04M hostnames), Common Crawl host webgraph |
| Trained | val loss 5.0339 @ step 3504, RTX 4060 |

Rationale for every choice: `MODEL_SELECTION.md`.

## Evidence on Hand

### Measured — `results/research/baselines.json`

One shared split, produced by `research/harness.py`: the 885-apex held-out test
file, restricted to the **545 apexes with at least 6 labels** (4,839 held-out
labels; mean 8.5 known, 8.9 withheld). Each apex is split ~50/50 into known and
withheld by an RNG seeded from `sha256("1337:<apex>")`, so the split is
independent of apex ordering, filtering, and which methods run. Equal candidate
budget for every method; no DNS. Percentile-bootstrap 95% CIs over apexes, 2,000
resamples. Regenerate with `python3 research/run_baselines.py --neural`.

Mean per-apex recall@N:

| Method | @10 | @25 | @50 | @100 | @200 |
|---|---:|---:|---:|---:|---:|
| SubFury v2 | **0.114** | **0.172** | **0.207** | **0.216** | 0.217 |
| frequency prior (train split) | 0.052 | 0.100 | 0.134 | 0.179 | **0.236** |
| Markov 5-gram | 0.057 | 0.098 | 0.132 | 0.175 | 0.224 |
| Markov 4-gram | 0.059 | 0.090 | 0.122 | 0.154 | 0.194 |
| Markov 3-gram | 0.029 | 0.047 | 0.073 | 0.117 | 0.145 |
| n0kovo wordlist | 0.018 | 0.027 | 0.058 | 0.097 | 0.138 |

Paired bootstrap, SubFury minus the frequency prior on the same apexes:
**+0.062** [+0.048,+0.077] @10 · **+0.071** [+0.055,+0.089] @25 · **+0.073**
[+0.056,+0.090] @50 · **+0.037** [+0.023,+0.053] @100 · **−0.019**
[−0.030,−0.007], p = 0.002 @200.

**The result is a regime, not a number — copy must always state the budget with
the number.** At N=10 the model more than doubles the prior (2.2x) with cleanly
separated intervals. At N=100 the marginal CIs overlap (0.216 [0.190,0.244] vs
0.179 [0.158,0.202]) and only the paired test separates them. At N=200 the prior
is better. Model recall is flat past N≈50 — beam search exhausts distinct
plausible labels while the prior keeps enumerating the global head. The product
claim that survives this: SubFury earns its place exactly where the DNS budget is
tight, and does not where it is loose.

The frequency prior is the baseline that matters — same training data, zero
parameters. The n0kovo wordlist is the weakest control on the board.

### Superseded — `results/subfury_v2/eval.json`

The "recall@100 0.220 vs 0.097" headline this file carries was measured against
the n0kovo wordlist alone, and was produced by `subfury_v2/evaluate.py`, whose
K/H split comes from one sequential `random.Random(seed)` applied in iteration
order — it changes with `--max-apexes`, the min-label filter and file order, so
numbers from different runs of it are not comparable. **Do not quote it.** The
web UI still reads this file (`webui/app.py` → `MODEL_DIR/eval.json`); repointing
it at `results/research/baselines.json` is outstanding work, and until then the
Model tab shows a superseded comparison.

### Measured — conditioning is real, and shallow

**Swap test** (`results/research/swap.json`, 100 held-out apexes, recall@N). The
same model conditioned four ways, scored against the same withheld set:

| Conditioned on | @10 | @25 | @50 | @100 |
|---|---:|---:|---:|---:|
| its own known set | 0.173 | 0.243 | 0.279 | 0.288 |
| a different org's labels | 0.017 | 0.035 | 0.054 | 0.059 |
| the globally most common labels | 0.020 | 0.035 | 0.050 | 0.050 |
| one label from its own set | 0.145 | 0.213 | 0.253 | 0.259 |

Own beats swapped by 0.230 [0.166,0.298] at N=100; own is better on 49 of 100
apexes, swapped on 4. Conditioning is not decoration — but a **single** known
label already reaches 0.259 of the 0.288 the full set reaches, so the rest of the
set is worth 0.029. `results/research/capacity.json` agrees from the other side:
feeding six different 24-label windows of the same known set (tail, head, evenly
spread, three random draws) to the 30 apexes with >24 known labels moves
recall@100 only between 0.125 and 0.129, median per-apex spread 0.000.

Copy may say conditioning is measured and real. Copy may **not** imply the model
reads deeply into a large known set; one label carries most of the effect.

### Measured — the closed-vocabulary ceiling

`reachable_subset` in `baselines.json`: 75 of the 545 apexes have **zero**
held-out labels present in the training vocabulary — unwinnable for any
closed-vocabulary method. Averaged over apexes, 57.6% of an apex's held-out
labels are in-vocabulary (55.3% pooled). On the 470 reachable apexes the figures
become 0.251 vs 0.207 at N=100 and 0.252 vs 0.274 at N=200: the ordering and the
crossover are unchanged.

### Measured — divergence between seed types (`divergence*.json`)

Re-measured 2026-09-06; these replaced an earlier unsourced "Jaccard 0.18 /
28–66% seed-driven" claim that did not survive measurement. Real held-out apexes
(`divergence_real.json`): pairwise Jaccard 0.010–0.786, mean 0.200. Hand-made
topical seeds (`divergence.json`): 0.408–0.786, mean 0.565, only 24–32%
seed-driven. Conditioning is uneven — distinctive label sets diverge sharply,
generic English web seeds converge. Never imply uniform per-target divergence.

### Unverified — do not cite as fact

- "Seeded with `www` and `blog` on a real domain, round 1 found `shop`; round 2
  found `learn`." Plausible, never reproduced under measurement here (no DNS run
  was performed). Removed from `README.md` on 2026-09-06. Re-run against an
  authorized domain before publishing it.
- The `grafana` → `prometheus` and `inventory` → `inventoryapp` examples do
  **not** reproduce against the shipped checkpoint. Removed from `README.md` on
  2026-09-06; do not reintroduce them.

### Absences future work must not fabricate

No users, no testimonials, no adoption numbers, no press, no case studies, no
pricing, no hosted service. None exist.

## Brand Commitments

- Name **SubFury**, version 2. Author **4riful**, repo
  `https://github.com/4riful/subfury`, MIT.
- The wordmark is `webui/static/subfury-mark.png` — green "Sub", white "Fury",
  skull between them. On dark surfaces the skull is lifted from its original
  `#44475a`, which was drawn for a light page; the original is kept at
  `webui/static/subfury.png`.
- Voice: measured, technical, evidence-first. State the limitation before
  anyone asks. Never oversell — the limitation section is load-bearing, not a
  disclaimer.

## Product Principles

1. **Every number on screen is traceable to an artifact.** If it was not
   measured, it is labelled as unmeasured or it does not appear.
2. **Say what leaves the machine.** Passive sources and DNS validation are
   different acts; the interface must never blur them.
3. **The budget is the scarce resource.** Rank for a finite DNS spend, not for
   diversity or volume.
4. **Expose the mechanism.** An evaluator must be able to see the tokenization,
   the ranking, the beam probabilities and the module map without reading code.
5. **The operator's data stays theirs.** Local execution, browser-local history,
   frictionless export, no telemetry.

## Honest limitation

**The corpus is mismatched with the use case, and the evaluation rewards the
mismatch.** Training *and* evaluation data are both the Common Crawl **host
webgraph** — crawlable public sites. This is stronger than the earlier framing
("internal infrastructure is underrepresented"): the benchmark itself scores the
model on the wrong name distribution.

- Held-out Common Crawl labels are **18.3%** one or two characters (884 of
  4,839); the most frequent are `m`, `blog`, `de`, `my`, `nl`, `es`, `no`, `pt`,
  `fr`, `ru`, `it` — language and mobile variants of a public website. Derived
  from `data/groups_test.jsonl` under the `research/harness.py` split.
- Certificate Transparency on the same kind of target
  (`research/data/ct_observations.jsonl`, 40 apexes, 27 of them from this test
  split) is **1.3%** short labels, and the labels shared across the most
  organizations are `www`, `test`, `api`, `mail`, `support`, `demo`, `status`,
  `cdn`, `docs`, `dev`, `webmail`, `staging` — the recon surface.
- The scoring pays for short labels. `results/research/lengthbias.json` (80
  apexes): refusing labels shorter than 3 characters drops recall@100 from
  **0.265 to 0.117**, a 56% loss, while a beam length penalty that cuts the
  short-label share of the top-50 from 48.6% to 23.7% costs almost nothing
  (0.265 → 0.259).

So the tool can score 0.22 on this benchmark and find very little on a live
target: it is being graded on a distribution it will never meet in an engagement.
Retraining on Certificate Transparency or passive DNS — **and re-benchmarking on
the same** — is the highest-value next step; a CT-trained model measured against
Common Crawl would just move the mismatch. Until that is done, every recall
number in this document measures Common-Crawl mimicry, not recon value, and must
be presented that way. Stated publicly in `README.md`, not hidden.

## Anti-goals

- Not a SaaS. No signup, no pricing, no dashboard-for-teams.
- Not a general enumeration suite — it predicts; it does not replace `amass`,
  `subfinder`, or CT-log scraping. It composes with them.
- Never oversell.

## Accessibility & Inclusion

WCAG 2.2 AA is the floor for every surface:

- text contrast ≥ 4.5:1, UI and graphical boundaries ≥ 3:1;
- every control keyboard-reachable in a meaningful order, with a visible focus
  ring that is never removed;
- `prefers-reduced-motion` honored — all animation collapses;
- live pipeline output announced through `aria-live`, not colour alone.
