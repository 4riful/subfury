# Budgeted Set-Conditioned Subdomain Discovery: Problem, Measurements, and Protocol

**Status:** research spine. Every quantitative claim below carries the artifact it came from.
Numbers were re-derived from the artifacts for this document rather than copied from prior summaries;
where a re-derivation disagreed with a circulating figure, the disagreement is stated (see §2.4).

**Scope:** this describes what has been measured on SubFury v2 and the baselines, what those
measurements do to the original research hypothesis, and the protocol for the work that follows.
§7.1 carries measured v3 results for axes A1 (partial), A2 and A3 (partial);
A4, A5 and A6 have not been run, and every v3 claim outside §7.1 is a design
rationale rather than a result. One pre-registered falsifier has fired — the
fusion scorer — and is recorded as such in §7.1 and §8.

---

## 1. Problem statement

### 1.1 Formalisation

An organisation owns a set of hostnames under an apex domain `d`. Write each hostname as
`ℓ.d` where `ℓ` is a *label* (which may itself contain dots: `api.staging` under `example.com`).
Let `L(d) ⊆ Σ*` be the complete, unobservable set of labels the organisation actually operates,
over the DNS label alphabet `Σ = [a-z0-9-.]` under the grammar in `subfury_v2/predict.py:LABEL_RE`.

A discovery process observes a subset `K ⊂ L(d)` — from passive sources: Common Crawl, Certificate
Transparency, passive DNS. `K` is **unordered**: it is a set of strings with no canonical sequence,
no timestamps at query time, and no internal structure a sequence model could legitimately exploit.

Given `K` and an integer **query budget** `N`, a method must emit an ordered list of `N` candidate
labels `C = (c₁, …, c_N)`, `cᵢ ∉ K`, chosen to maximise

    hits(C) = |{c ∈ C : c ∈ L(d) \ K}|

Each candidate costs one DNS query (or one probe), so `N` is the real, budgeted resource. The two
metrics that follow are

    recall@N = |C ∩ H| / |H|              H = a held-out subset of L(d) \ K
    hit rate = hits(C) / N                queries per discovery = N / hits(C)

This is the same shape as budgeted IPv6 target generation — 6Gen's "sole parameter: probe budget"
(`research/related-work.md` §1) — transposed from a numeric address space to a discrete label space
with linguistic structure and an **open vocabulary**: the correct answer is frequently a string no
corpus contains.

### 1.2 What makes it non-trivial

Three properties, each of which one class of prior method violates:

1. **Unordered, variable-size conditioning.** `|K|` ranges from 1 to several hundred in deployment.
   Any method that imposes an order on `K` is modelling an artifact.
2. **Open vocabulary.** Wordlists and retrieval-only methods are capped by their vocabulary. §2.6
   quantifies that cap on our data at **57.6%**.
3. **A strong unconditional prior.** Label popularity across organisations (`www`, `api`, `mail`,
   `dev`) is a very good predictor on its own. Any claim that conditioning helps must be measured
   *against a fitted prior at equal budget*, not against a wordlist. This is the single most
   important methodological point in this document, and §2.1 is what happens when you do it.

### 1.3 The original hypothesis, as stated

> Permutation-invariant set conditioning on an organisation's known hostnames beats sequence
> conditioning on the same information, and a hybrid retrieval + open-vocabulary generator beats
> either channel alone, under an equal query budget.

§3 records that the first clause was **falsified as a framing** and what replaced it.

---

## 2. What was measured

All offline results use the shared harness `research/harness.py`. Protocol, verbatim from
`results/research/baselines.json`:

> 50/50 K/H split of each test apex's labels; ranker sees K and a budget of N; recall@N against H.
> No DNS/network.

Configuration (`research/harness.py`, echoed into `results/research/baselines.json`):
seed **1337**; the split for each apex is seeded by `SHA256(seed:apex)` so it is invariant to apex
ordering, filtering, and which methods are being run; apexes kept if they have ≥ **6** labels;
`N ∈ {10, 25, 50, 100, 200}`; **2000**-round percentile bootstrap over apexes for every CI.
Corpus: `data/groups_test.jsonl` — **545 apexes**, **4,839 held-out labels**, mean `|K|` **8.48**,
mean `|H|` **8.88** (`results/research/baselines.json`, fields `apexes`, `held_out_labels`,
`mean_known`, `mean_holdout`).

The system under test is SubFury v2 (`subfury_v2/model.py`, `results/subfury_v2/best.pt`):
a 6-layer, 6-head, `n_embd=300`, `block_size=192`, 4096-BPE decoder-only transformer,
**9,019,200 parameters**, checkpoint at step 3504, `val_loss` 5.0339. It conditions by serialising
`sorted(K)[-24:]` into the causal context (`subfury_v2/predict.py:predict_labels`,
`block_size // 8 = 24`) and beam-searching with 64 beams and no length penalty.

### 2.1 The budget crossover: the model wins small budgets and loses large ones

`results/research/baselines.json`, field `table` (macro recall, bootstrap 95% CI):

| method | r@10 | r@25 | r@50 | r@100 | r@200 | MAP | apexes with ≥1 hit @200 |
|---|---|---|---|---|---|---|---|
| wordlist `subdomains_tiny.txt` | 0.018 [0.013,0.024] | 0.027 | 0.058 | 0.097 | 0.138 | 0.0083 | 284/545 |
| **frequency-prior** | 0.052 [0.042,0.062] | 0.100 | 0.134 | 0.179 | **0.236** | 0.0361 | 306/545 |
| markov-3gram | 0.029 | 0.047 | 0.073 | 0.117 | 0.145 | 0.0202 | 200/545 |
| markov-4gram | 0.059 | 0.090 | 0.122 | 0.154 | 0.194 | 0.0356 | 256/545 |
| markov-5gram | 0.057 | 0.098 | 0.132 | 0.175 | 0.224 | 0.0369 | 296/545 |
| **subfury-v2 (neural)** | **0.114** [0.095,0.134] | **0.172** | **0.207** | **0.216** | 0.217 | **0.0968** | 264/545 |

The frequency prior is fitted on `data/groups_train.jsonl` over **494,934** distinct labels, ranked by
number of distinct organisations using each label, with the apex's own known labels removed before
ranking (`research/baselines/frequency.py`, described in `results/research/baselines.json:rankers`).
Its top-10 is `blog, m, app, es, fr, de, support, it, ru, my` — not a hand-curated wordlist, a
corpus-fitted popularity model. It is the baseline that matters; the wordlist is not.

Paired bootstrap over the same 545 apexes (`results/research/baselines.json`,
`paired_vs_frequency_prior["subfury-v2 (neural)"]`), model **minus** prior:

| N | Δ recall | 95% CI | bootstrap p (two-sided) |
|---|---|---|---|
| 10 | **+0.0623** | [+0.0483, +0.0772] | 0.000 |
| 25 | +0.0713 | [+0.0553, +0.0888] | 0.000 |
| 50 | +0.0731 | [+0.0556, +0.0900] | 0.000 |
| 100 | +0.0373 | [+0.0227, +0.0528] | 0.000 |
| 200 | **−0.0192** | [−0.0305, −0.0072] | **0.002** |

**The finding.** At N=10 the model is **2.19×** the prior (0.1143 / 0.0520). By N=200 it is
**significantly worse** than the prior it is supposed to beat. The model's own curve is flat past
N=50: 0.207 → 0.216 → 0.217. The prior's keeps climbing: 0.134 → 0.179 → 0.236.

This is the central measurement in the project. A generative model that saturates is not a better
ranker with a smaller constant — it is a *different shape of method*, useful only where the budget is
tight. Some of the flattening is literal exhaustion: the neural ranker returned fewer than 200
distinct valid candidates on **7 of 545** apexes (`results/research/baselines.json`,
`results[5].notes.apexes_short_of_budget`); the rest is the beam collapsing onto near-duplicates.

The end-to-end run in `results/subfury_v2/eval.json` (the same 545 apexes but the older
`subfury_v2/evaluate.py` split, seed 42, against the n0kovo wordlist rather than a fitted prior)
reports model 0.175 / 0.210 / 0.220 / 0.221 at N = 25/50/100/200, agreeing with the harness to within
0.004 and showing the same saturation. **Its baseline column (0.028 → 0.142) is the wordlist, not the
prior, and should not be quoted as evidence the model beats popularity.** That comparison is why the
project believed the model was 1.6× better at N=200, when against a fitted prior it is worse.

### 2.2 Conditioning is real

`results/research/swap.json` (100 apexes, `research/diagnostic/swap.py`). Same model, same budget,
four conditioning sets, all scored against the *same* apex's held-out labels; candidates that echo the
real apex's `K` are stripped from every variant so no variant can buy recall by copying:

| conditioned on | r@10 | r@25 | r@50 | r@100 |
|---|---|---|---|---|
| **own** — the apex's own `K` | 0.1726 | 0.2433 | 0.2794 | **0.2882** |
| **single** — one label from `K` | 0.1454 | 0.2132 | 0.2533 | **0.2593** |
| **swapped** — a *different* org's labels | 0.0168 | 0.0347 | 0.0536 | **0.0585** |
| **generic** — the 24 globally commonest labels | 0.0203 | 0.0350 | 0.0497 | 0.0497 |

Paired `own − swapped` @100 = **+0.2297** [+0.1661, +0.2983]; own is better on 49/100 apexes and worse
on 4 (`summary.own_minus_swapped`). Recall collapses **0.288 → 0.059**, a 4.9× drop, when the model is
fed someone else's hostnames. The model is genuinely conditioning; it is not reciting a memorised list.

### 2.3 …but conditioning saturates after roughly one label

From the same table: **`single` reaches 0.2593 of `own`'s 0.2882 at N=100 — 90.0%** of the full
conditioning benefit from *one* known label. At N=10 it is 0.1454 / 0.1726 = 84.2%.

The 2nd through 20th known hostnames are together worth **0.029 recall** — about a tenth of what the
first one is worth.

Two corroborating measurements:

**Window choice is irrelevant** (`results/research/capacity.json`,
`research/diagnostic/capacity.py`, 30 apexes with `|K| ≥ 25` drawn from
`data/groups_test_uncapped.jsonl`, so the 24-label truncation actually bites). Feeding the model six
different 24-label windows of the same known set:

| window | r@10 | r@25 | r@50 | r@100 |
|---|---|---|---|---|
| tail (`sorted(K)[-24:]`, what ships) | 0.0355 | 0.0730 | 0.1185 | 0.1280 |
| head (`sorted(K)[:24]`, what it discards) | 0.0381 | 0.0771 | 0.1189 | 0.1249 |
| spread (every ⌊|K|/24⌋-th) | 0.0339 | 0.0773 | 0.1202 | 0.1287 |
| random ×3 | 0.0369 / 0.0373 / 0.0377 | 0.0770 / 0.0742 / 0.0778 | 0.1195 / 0.1207 / 0.1204 | 0.1273 / 0.1272 / 0.1272 |

Per-apex spread between the best and worst window at N=100: **mean 0.0153, median 0.0000**, max 0.0755
(`summary.window_spread_at_max_budget`). On more than half the apexes, *which* 24 of the organisation's
labels the model sees changes its output not at all.

**Lift does not grow with `|K|`** (`results/research/decomposition.json`,
`summary.by_known_size`, N=100). "Lift" = recall from candidates the frequency prior would not have
proposed:

| `|K|` | apexes | model@100 | prior@100 | lift@100 |
|---|---|---|---|---|
| 1–3 | 151 | 0.2114 | 0.1937 | 0.0353 |
| 4–7 | 192 | 0.2361 | 0.2080 | 0.0599 |
| 8–15 | 82 | 0.2479 | 0.2198 | 0.0547 |
| 16+ | 120 | 0.1851 | 0.1177 | 0.0756 |

Lift is non-monotone and never exceeds 0.076. There is no regime in which more known hostnames buy
proportionally more conditioning-driven recall.

**Consequence for architecture.** v2 sorts `K` before truncating
(`subfury_v2/predict.py`: `sorted(known)[-block_size // 8:]`) and SubWiz sorts before comma-joining
(`research/related-work.md` §2: `",".join(sorted(subs))`). **Both incumbent methods are already
order-canonical.** A permutation-invariant encoder therefore cannot claim an empirical win over them
on the grounds of invariance, and the truncation it would remove costs, on this evidence, nothing.
This is stated again as a falsification in §3.

*Caveat, stated because it limits the claim:* the capacity test covers 30 apexes at `|K|` between 25
and ~50. Deployment passive seeds reach 500–700 labels (`research/data/rebuild_uncapped.py` docstring).
The measurement says truncation is harmless at 2× the window; it does not say it is harmless at 25×.

### 2.4 The corpus the model was trained on does not look like the corpus it will be deployed against

Training and evaluation data come from the Common Crawl host graph via `subfury_v2/data_prep.py`.
Deployment seeds come from Certificate Transparency (`webui/app.py`, `research/data/ct_fetch.py`).
Recomputed from the artifacts:

| | Common Crawl (`data/groups_test.jsonl`) | Certificate Transparency (`research/data/ct_observations.jsonl`) |
|---|---|---|
| label instances | 10,937 (885 apexes) | 7,382 (40 apexes) |
| **1–2 character labels** | **17.73%** | **0.66%** (1.29% among dot-free labels) |
| 1-character labels | 2.50% | 0.07% |
| top labels by #orgs | `blog, m, app, es, fr, de, support, it, ru, my, cdn, pt, shop, help, ar, docs, nl, api, tr, go` | `www, test, api, mail, support, demo, status, cdn, docs, dev, webmail, staging, shop, vpn, s, social, admin, analytics, autodiscover, pay` |

(Training split `data/groups_train.jsonl`: 1,044,524 label instances, **16.35%** 1–2 characters,
494,934 distinct.)

Two lists that barely overlap. Common Crawl's head is dominated by **language and country codes**
(`es, fr, de, it, ru, pt, ar, nl, tr`) — a crawler artifact of localised marketing sites. CT's head is
**operational infrastructure** (`www, test, api, mail, demo, status, staging, vpn, admin`) — what a
security engineer is actually looking for.

**And `www` cannot be predicted at all.** `subfury_v2/data_prep.py` drops it by construction:

```python
if label == "www" or len(label) > MAX_LABEL_LEN:
    continue  # www-only adds nothing to learn
```

The single most common label in CT is absent from the training corpus by design.

*Discrepancy, stated plainly:* the figure previously circulated for the CT side of this comparison
was 1.9%. Recomputing over every definition I could construct — all 7,382 label observations (0.66%),
distinct labels (0.60%), dot-free only (1.29%), macro-averaged per apex (2.93% over 37 apexes with
observations, 2.71% over all 40), the 57-apex `research/data/cache/` pull (0.84%), the 21-apex
temporal split (0.55% micro / 2.72% macro) — none reproduces 1.9%. **The number reported here is
0.66% under the definition that matches the Common Crawl side exactly** (all label instances in the
pool). The direction and order of magnitude of the mismatch are unchanged and if anything larger than
previously stated; the definition should be quoted with the number in any paper.

### 2.5 Length-penalty and short-label filtering both hurt — and §2.4 explains why

Beam search ranks by summed log-probability, so short strings score high by construction. On a real
target this puts `s`, `a`, `e` at the top of the list and spends DNS budget on single characters. The
obvious fixes were measured rather than assumed (`results/research/lengthbias.json`,
`research/diagnostic/lengthbias.py`, 80 apexes; `α` is the beam length-normalisation exponent,
`score = Σlogp / len^α`; `minlen` drops candidates shorter than that):

| α | minlen | r@10 | r@25 | r@50 | r@100 |
|---|---|---|---|---|---|
| **0.0** | **1** | **0.1653** | **0.2282** | **0.2599** | **0.2654** |
| 0.6 | 1 | 0.1653 | 0.2269 | 0.2574 | 0.2654 |
| 0.8 | 1 | 0.1653 | 0.2269 | 0.2553 | 0.2654 |
| 1.0 | 1 | 0.1616 | 0.2210 | 0.2520 | 0.2593 |
| 0.0 | 2 | 0.1514 | 0.2044 | 0.2298 | 0.2346 |
| 0.0 | 3 | 0.0809 | 0.1028 | 0.1173 | 0.1173 |
| 1.0 | 3 | 0.0789 | 0.1014 | 0.1105 | 0.1159 |

Share of the top-50 that is 1–2 characters, by α (`short_label_share_top50`):
**α=0 → 48.6%**, α=0.6 → 40.8%, α=0.8 → 32.5%, α=1.0 → 23.7%.

The length penalty does exactly what it was designed to do — it halves the short-label share — and
**costs recall at every budget**. `minlen=2` costs 10.4% of recall@25 relative; `minlen=3` costs
**55%**. Nothing beats shipping configuration (α=0, no filter).

The reason is §2.4: **on this benchmark, 17.7% of the ground truth genuinely is 1–2 characters.**
Filtering short labels is not removing beam-search noise; it is removing correct answers. The
benchmark is rewarding a behaviour that will be wrong in deployment. This is the sharpest instance of
the corpus mismatch: an inference-time fix that a practitioner would call obviously correct is
punished by the offline metric, and both facts are true.

### 2.6 The vocabulary ceiling and the 75 unwinnable apexes

Of the 4,839 held-out labels, **2,674 (55.3%) appear anywhere** in the 494,934-label training
vocabulary. Macro-averaged over apexes — which is the headline metric — the mean per-apex fraction of
held-out labels that exist in the training vocabulary is **57.58%** (recomputed from
`data/groups_test.jsonl` + `data/groups_train.jsonl` under `research/harness.py`'s seed-1337 split).

**57.6% is the hard ceiling on macro recall for any closed-vocabulary method** — every wordlist, the
frequency prior, an *n*-gram model restricted to observed strings, and any retrieval-only channel. The
remaining 42.4% is reachable only by open-vocabulary generation. That number is the entire
justification for keeping a generator at all, and it should be printed at the top of every results
table so no method is ever compared against 100%.

**75 of 545 apexes have zero held-out labels anywhere in the training vocabulary**
(`results/research/baselines.json:reachable_subset` — 470 reachable, 75 excluded). They are
unwinnable for any closed-vocabulary method and contribute a hard zero to its macro average. Scoring
on the reachable subset (`reachable_table`):

| method | r@10 | r@25 | r@50 | r@100 | r@200 | MAP |
|---|---|---|---|---|---|---|
| frequency-prior [reachable] | 0.060 | 0.116 | 0.155 | 0.207 | 0.274 | 0.0418 |
| subfury-v2 [reachable] | 0.133 | 0.199 | 0.240 | 0.251 | 0.252 | 0.1123 |

The crossover survives restriction to reachable apexes; the prior still overtakes between N=100 and
N=200. Excluding the unwinnable apexes raises both methods ~15% relative and changes no conclusion.

Micro vs macro (`micro_table`) — micro weights apexes by how many labels they withhold:

| method | @10 macro/micro | @50 macro/micro | @200 macro/micro |
|---|---|---|---|
| frequency-prior | 0.052 / 0.038 | 0.134 / 0.111 | 0.236 / 0.219 |
| subfury-v2 | 0.114 / 0.093 | 0.207 / 0.194 | 0.217 / 0.205 |

Micro is uniformly lower for both: **both methods do relatively worse on the apexes with the most to
find**, which are exactly the ones an operator cares about. Report both, always.

### 2.7 Model and prior are complementary, which is a finding in its own right

`results/research/decomposition.json` (545 apexes, seed 42, `research/diagnostic/decompose.py`):

| N | model | prior | **union** | lift | overlap | novel_frac |
|---|---|---|---|---|---|---|
| 10 | 0.1186 | 0.0577 | 0.1314 | 0.0738 | 0.2787 | 0.7213 |
| 25 | 0.1752 | 0.1043 | 0.1858 | 0.0815 | 0.2975 | 0.7025 |
| 50 | 0.2100 | 0.1389 | 0.2206 | 0.0817 | 0.3183 | 0.6817 |
| 100 | 0.2198 | 0.1859 | 0.2417 | 0.0558 | 0.2318 | 0.7682 |
| 200 | 0.2214 | 0.2399 | **0.2623** | 0.0225 | 0.1977 | 0.8023 |

`overlap` = share of the model's list the prior also proposed; `novel_frac` = share it would not have.
**68–80% of the model's list is novel relative to the prior**, and the union beats both channels at
every budget — by 0.041 over the prior and 0.041 over the model at N=200. Neither channel is
redundant. That is the empirical case for a hybrid, and it is a stronger case than the original
architectural intuition was.

### 2.8 Divergence: conditioning changes the output, but not enough of it

`results/subfury_v2/divergence.json` — six synthetic 24-label "profile" prompts (dev, monitoring,
ecommerce, infra, media, neutral), top-50 each, 128 beams. Pairwise Jaccard between output lists:
**min 0.408, max 0.786, mean 0.565**. `seed_driven_share` — the fraction of the top-50 that is not in
every other profile's list — is 0.24–0.32. Two organisations with completely different infrastructure
profiles receive lists that are more than half identical.

`results/subfury_v2/divergence_real.json` — the same measurement on six real apexes with real seeds
(`|K|` 7–20): pairwise Jaccard **min 0.010, max 0.786, mean 0.200**, and `seed_driven_share` spanning
**0.16 (leafphp.dev) to 0.94 (hanke100.com)**.

Real seeds diverge far more than synthetic profiles (mean Jaccard 0.200 vs 0.565), which is the
expected direction and confirms the synthetic prompts understate conditioning. But the per-apex spread
is enormous: on `leafphp.dev` only 16% of the top-50 is seed-driven, on `hanke100.com` 94%. Whatever
conditioning the model has is applied very unevenly, and nothing currently predicts which regime a
given apex falls into. n=6; this is an observation, not a result.

---

## 3. What this does to the hypothesis

### 3.1 Falsified as a framing: "permutation-invariant set conditioning beats sequence conditioning"

The question presumed that the incumbents impose a *meaningful* order on `K` that a set encoder would
remove. They do not:

- v2 conditions on `sorted(known)[-24:]` (`subfury_v2/predict.py`) — sorted, hence a canonical order,
  hence already invariant to the input's presentation order.
- SubWiz builds `apex + "[DELIM]" + ",".join(sorted(subs)) + "[DELIM]"`
  (`research/related-work.md` §2, verified against `subwiz/main.py`) — also sorted.

Both baselines are permutation-invariant *in effect*. A set encoder cannot beat them by being
invariant, because there is no order-sensitivity left to beat. And the residual argument — that
sorting-plus-truncation discards information a set encoder would keep — is contradicted by
`results/research/capacity.json`: median per-apex spread across six different 24-label windows is
**0.000**.

`research/v3/test_invariance.py` verifies the v3 encoders are invariant to machine precision
(deepsets max Δ **7.15e-07**, settrans **8.34e-07**, padding leak **9.54e-07**, `|K|` accepted up to
512). That is a correctness property of the implementation. **It is not a research contribution and
must not be reported as one.** Any paper claiming a set-encoder win must show it against a
*sorted-concatenation* baseline on identical data — which is what `encoder="concat"` in
`research/v3/model.py` exists to provide — and must expect the difference to be small.

The residual, defensible version of the claim: v2's context is capped at 24 labels; deployment seeds
reach 500–700; a set encoder removes that cap architecturally (`V3Config.max_set = 512`). That is an
**engineering** justification with a measured cost of approximately zero at the sizes tested. Ship it,
do not headline it.

### 3.2 The replacement question

> **Conditioning saturates after roughly one known label. What architecture and training objective make
> the 20th known hostname contribute as much as the 1st?**

Restated as testable sub-questions:

- **RQ1 (marginal value of set size).** Does recall@N increase monotonically in `|K|` when the encoder
  can see all of `K`? Measured by conditioning on `K` truncated to 1, 2, 4, 8, 16, 32, 64, 128 labels
  and reading the curve. v2's curve is flat after 1; a set encoder that does not bend it has failed.
- **RQ2 (the objective, not the architecture).** Cross-entropy over labels rewards predicting
  *frequent* labels. There is no term in `L = CE(y | K)` that rewards predicting a label the global
  prior would have missed — and §2.7 shows 68–80% of the model's value lies precisely there. Does a
  ranking loss computed *against prior logits* change the marginal value of set size, holding the
  encoder fixed? This is a stronger candidate explanation for saturation than the encoder is.
- **RQ3 (the budget shape).** Given that generation saturates by N=50 and the prior keeps climbing to
  N=200 (§2.1), can a single calibrated scorer over a generation channel and a retrieval channel
  dominate both at *every* budget? §2.7's union column (0.2623 at N=200 vs 0.2399 prior, 0.2214 model)
  is the target to beat; it is an oracle-free upper reference, since the union is 2N candidates at
  budget N and is therefore not itself a legal method.

**What the project can honestly claim, from `research/related-work.md` §6:** a reproducible recall@N
benchmark at equal candidate budget across 545 organisations; a temporal-split evaluation (nobody in
this literature does one); the first budget-aware ranking *for hostname discovery* (budget-aware
evaluation itself is established in IPv6 target generation and in Marchal et al. 2012); a hybrid
retrieval + open-vocabulary generator for this task. **What it cannot claim:** that a permutation-
invariant encoder is the differentiator, that recall@N without DNS is discovery, or that using a
transformer LM for subdomains is new (SubWiz, 2024).

---

## 4. The v3 design, with the measurement each part answers to

`research/v3/model.py`. Every component is config-switched off one shared label encoder, decoder and
training loop, so an ablation moves exactly one variable.

| Component | Config | The measurement that requires it |
|---|---|---|
| **Set encoder over unbounded `|K|`** (`deepsets` = mean/max pool; `settrans` = self-attention + learned-query pooling; `concat` = the v2 sorted-context baseline) | `encoder`, `max_set=512`, `n_seeds=4` | §2.3: v2 sees 24 of up to 700 labels. Median window spread 0.000 says the cap is *currently* free — but the cap is why `|K|` cannot be a variable at all, and RQ1 needs it to be. `concat` exists so the comparison is against v2's actual conditioning, not a strawman. |
| **Retrieval head** — scores a fixed candidate vocabulary against the org vector (`retrieve_scores`, `cand_vocab`) | `cand_vocab > 0` | §2.1: generation is flat past N=50 (0.207 → 0.217) while the prior climbs to 0.236 at N=200 and 0.274 on the reachable subset. Generation cannot fill a large budget; retrieval can — **confirmed by A2** (§7.1), which also **refuted** the second half of this rationale: §2.7's union-beats-both result did not survive, the fused hybrid scores at or below the retriever alone at every budget. |
| **Generator head** — autoregressive over BPE, cross-attending set memory | always on | §2.6: **42.4%** of held-out labels are outside the training vocabulary. Retrieval alone is capped at 57.6% macro recall. Removing the generator forfeits that headroom. |
| **Ranking loss against prior logits** — `s = s − prior_logits` before the contrastive term (`loss(..., prior_logits=...)`) | `lambda_rank` | §2.1 + §2.7: cross-entropy has no pressure to beat popularity; a model that perfectly learns `P(y)` scores well under CE and loses to the prior at N=200. Subtracting prior logits makes the objective *"score above popularity"* — the quantity §2.7 measures as `lift`, and the one the model is losing on. |
| **Sampled negatives only** — unobserved labels are not treated as negatives | `cand_neg` in the batch | The ground truth is a *withheld half*; an unobserved label is unknown, not absent. Treating it as a negative would train the model to suppress correct answers, and would make the 42.4% out-of-vocabulary tail actively harder. |

Note what is *not* in the list: nothing is justified by permutation invariance. Invariance is a
property the implementation must have and `research/v3/test_invariance.py` asserts; it is not a reason
for any design decision above.

---

## 5. Evaluation protocol going forward

### 5.1 The harness is the contract

Every method — neural, statistical, wordlist, hybrid — implements
`rank(apex, known, n) -> list[str]` and is scored by `research/harness.py`. Nothing else is a result.
Guarantees the harness provides and that must not be weakened:

- **Per-apex seeded split** (`_apex_rng` = `SHA256(seed:apex)`): the K/H split does not depend on apex
  ordering, on filtering, or on which methods are in the run. Adding a method cannot move a baseline.
- **Ranker-independent cleaning** (`_clean`): known labels, blanks and duplicates are stripped by the
  harness as well as by the method, so a sloppy ranker cannot buy recall by re-emitting `K`.
- **Prefix consistency** is opt-out. A method whose output at budget N is not a prefix of its output at
  a larger budget sets `prefix_consistent = False` and is re-invoked per budget. Beam searches whose
  width is tied to N must set this.
- **AP denominator is `|H|`**, not the number of retrieved relevant items, so a method is penalised for
  correct answers it never surfaces inside the budget.
- **No network.** Ground truth is withheld real hostnames. Nothing in the offline harness resolves DNS.

### 5.2 Budgets

Report `N ∈ {10, 25, 50, 100, 200}`. **The discriminating budgets are 10, 25 and 50** — that is where
the model's advantage lives (Δ +0.062 / +0.071 / +0.073, all p < 0.001) and where a security operator
with a rate-limited resolver actually sits. N=100 and N=200 must be reported anyway, because that is
where the method loses (§2.1), and a results table that stops at 50 is a results table designed to
hide the crossover.

### 5.3 Mandatory companion reporting

Every headline table carries, without exception:

1. **Paired bootstrap against the fitted frequency prior** at every budget — mean difference, 95% CI,
   two-sided bootstrap p (`research/harness.py:paired_diff_ci`, 2000 rounds). Marginal CIs are not
   sufficient: both methods see the same apexes, and the paired difference removes between-apex
   variance. Never compare against a wordlist alone (§2.1).
2. **Macro and micro recall side by side.** Macro is the headline; micro reveals that both methods do
   worse on label-rich apexes (§2.6).
3. **The reachable subset** (470/545) alongside the full set, with the 75 unwinnable apexes named as
   such, and **57.6%** printed as the closed-vocabulary ceiling.

### 5.4 The three diagnostics become standard reporting, not one-off investigations

Any model claiming to be set-conditioned reports all three. They are cheap and they are the only
things that distinguish conditioning from recitation.

| Diagnostic | Script | Reports | Failure condition |
|---|---|---|---|
| **Swap** | `research/diagnostic/swap.py` | recall under own / single / swapped / generic conditioning; paired `own − swapped` | `own ≈ swapped` ⇒ the model is reciting a prior. `own ≈ single` ⇒ conditioning saturates at one label (v2's current state, §2.3). |
| **Decomposition** | `research/diagnostic/decompose.py` | model / prior / union / lift / overlap / novel_frac per budget, and lift bucketed by `|K|` | lift not increasing with `|K|` ⇒ RQ1 unmet. union > model ⇒ the hybrid is leaving recall on the table. |
| **Capacity** | `research/diagnostic/capacity.py` | recall across tail / head / spread / random windows of `K`; per-apex spread | Large spread ⇒ truncation is discarding signal. Zero spread at large `|K|` ⇒ the encoder is not using set size, whatever its architecture. |

Length-bias (`research/diagnostic/lengthbias.py`) is run whenever the decoding configuration changes,
because §2.5 shows decoding knobs that look obviously right are measurably wrong on this benchmark.

### 5.5 The temporal split, and exactly what is wrong with it

`research/data/ct_fetch.py` → `research/data/build_temporal.py` → `research/data/temporal.jsonl`.
For each apex, Certificate Transparency gives every label a `first_seen` = the earliest `not_before`
across all certificates containing it. A split date `T` partitions:

    known  = labels first seen ≤ T        (the conditioning set)
    future = labels first seen > T        (the ground truth)

Current build: **T = 2024-07-01**, `min_known=5`, `min_future=2`, **21 apexes**, 4,969 known and 1,996
future labels. Nothing in `future` was visible in CT at time `T`, so a model conditioned on `known`
cannot have had it in the conditioning set. This is a genuine before/after split, and per
`research/related-work.md` §5 **no prior work in this space does one** — SubWiz publishes neither its
corpus nor a date cutoff, so training-set contamination of its benchmark cannot be ruled out.

The collection is **passive only**: `ct_fetch.py` reads public CT aggregators and never resolves DNS or
contacts the target.

**Quantified limitations. All of these must appear beside any temporal number.**

- **Wildcards hide labels.** Across the 40 fetched apexes: 111,705 SAN names, **3,444 wildcard
  (3.08%)**. `*.api.example.com` still proves `api.example.com` exists, so wildcard-derived names are
  kept; 3 apexes have labels that appear *only* through a wildcard.
- **Three of 40 apexes are entirely wildcard-hidden.** `hingekin.com` (66 apex-wildcard certs),
  `hongyufish.com` (46) and `sa-verdun.com` (13) yield **zero** recoverable labels: every certificate
  is `*.apex`. An organisation that issues only apex wildcards is invisible to CT. This is a
  **selection bias against exactly the better-run organisations**, and it is not fixable from CT alone.
- **crt.sh silently truncates full-history queries on large apexes.** `ct_fetch.py` runs a second
  cheap unexpired-certs-only query as a guard and flags the apex when the guard returns something
  newer than the full-history query did (`stats.truncated_history`). **3 of 40 apexes tripped it**:
  `drew.edu`, `intel.com`, `sdmdev.dk`. Without the guard, `intel.com` came back with nothing newer
  than 2017 — the entire recent half of its timeline missing. `build_temporal.py` propagates the flag
  and `--exclude-truncated` drops those apexes. Any temporal result must state whether it was used.
- **One collection failure**: `sa-verdun.com` guard query returned HTTP 502 (`stats.guard_error`).
- **`first_seen` is a lower bound on existence, not a birth date.** A host may run for years before it
  is first put behind a certificate. The split measures *first public CT visibility*, and that phrase,
  not "first existed", is what belongs in a paper.
- **n = 21 apexes.** Too small for a headline number. It is currently a *protocol demonstration*.
  `research/data/crawl_queue.txt` holds 299 further apexes for expansion.
- **Domain shift.** The model is trained on Common Crawl (§2.4). Evaluating it on CT measures
  conditioning *and* corpus transfer simultaneously and cannot separate them. Reporting a
  Common-Crawl-trained model's temporal-CT recall without also reporting its Common-Crawl recall
  confounds the two.

### 5.6 What is still missing from the protocol

- **A live-resolution experiment.** Every prior hostname system resolves live
  (`research/related-work.md` §4). "Recall@N without DNS is not discovery" is the third-most-likely
  reviewer objection. At minimum: one authorised run reporting **queries per discovery**, comparable
  to Marchal et al.'s ≈1/200 and Regulator's ≈1/100.
- **SubWiz as a baseline, run by us.** It is pip-installable, MIT, weights public. Its absence will be
  noticed. Note the regime difference when doing so: SubWiz's benchmark has **median 5** seed
  hostnames across 369 apexes; our `min_labels=6` filter puts us in an easier regime, and that must be
  stated rather than quietly enjoyed.
- **Apex-level disjointness statement** between `data/groups_train.jsonl` and every evaluation set,
  asserted in code and printed by the harness.

---

## 6. Threats to validity

**Construct.** Recall@N against a withheld half of Common Crawl hostnames is *set completion*, not
discovery. It measures whether a method can reconstruct the half of an organisation's already-crawled
footprint it was not shown. It does not measure whether the method finds hosts nobody has crawled,
which is the deployment task. `subfury_v2/evaluate.py`'s docstring is correct on this; the framing
must not drift toward "predicting future hostnames" — that claim requires §5.5's temporal split.

**Corpus.** §2.4. The training distribution is language/country-code-heavy and `www`-free; the
deployment distribution is operational-infrastructure-heavy and `www`-led. §2.5 shows this is not
cosmetic: it inverts the sign of an inference-time fix. Every number in §2 should be read as *"on
Common Crawl"*, and a CT-trained replication is the correct control, not an extension.

**Ceiling.** 42.4% of held-out labels are outside the training vocabulary and 75/545 apexes are
entirely unwinnable for closed-vocabulary methods (§2.6). Macro means are depressed by a floor of hard
zeros that is a property of the corpus, not of the methods. Comparing any absolute recall figure
against 100% is meaningless.

**Statistical.** CIs are percentile bootstraps over apexes, which handles between-apex variance but
not the two model-side sources: a single training run (no seed variance is reported anywhere in this
document — every neural number comes from one checkpoint, `results/subfury_v2/best.pt`) and beam
search's determinism-but-brittleness. The diagnostics use smaller samples: swap n=100, lengthbias
n=80, capacity n=30, divergence n=6. Capacity's median-0.000 result is robust to n; divergence's is
not, and is labelled an observation.

**Split leakage.** The split is by apex, so no organisation appears in both train and test. But
labels are shared across organisations by design — that is what the frequency prior exploits — so
"unseen apex" does not mean "unseen labels". This is correct for the task and should be stated so a
reviewer does not mistake it for leakage.

**Multiple comparisons.** §2.1 reports 25 paired tests (5 methods × 5 budgets) without correction. The
headline results (Δ@10 p<0.001, Δ@200 p=0.002) survive a Bonferroni factor of 25; the markov-5gram
results at N=10 (p=0.027) do not, and should not be described as significant.

**Metric choice.** Macro recall treats a 6-label apex and a 100-label apex equally. Micro is reported
alongside (§2.6) and is uniformly lower. Neither is wrong; reporting only one is.

**Diagnostic construction.** The swap test pairs apex `i` with apex `(i + n/2) mod n` after a seeded
shuffle — an arbitrary pairing that does not control for the swapped organisation's *size* or sector.
A more adversarial version would swap with the most similar other organisation, which would shrink
`own − swapped` and is the harder test the current 0.2297 has not faced.

---

## 7. Ablation plan, and the four runs that have been executed

One axis at a time. A grid over encoder × retrieval × loss × decoding is 4 confounded variables and
produces a table nobody can read a mechanism out of. Every row below changes **one** thing against the
reference configuration, on the same 545 apexes, the same seed-1337 split, the same budgets, with the
full §5.3 companion reporting. Order is by information gained per GPU-hour.

**Reference configuration:** `encoder=settrans`, retrieval on, ranking loss with prior logits on,
α=0, no minlen filter.

| # | Axis | Arms | Status | Question it answers | What would kill the design |
|---|---|---|---|---|---|
| **A1** | Encoder | `concat` (v2-equivalent) / `deepsets` / `settrans` | **partial** — `deepsets` and `settrans` run; `concat` not run | Does a set encoder beat sorted concatenation on identical data and budget? (§3.1) | `concat ≈ settrans` at every N ⇒ the encoder is not the differentiator; report it as such and move the claim to A3. |
| **A2** | Channels | generation only / retrieval only / hybrid | **run** — all three arms | Does the hybrid beat both channels? Retrieval-only is bounded at **57.6%** macro (§2.6); generation-only saturates by N=50 (§2.1). | Hybrid ≤ max(channels) at every N ⇒ the fusion scorer is broken, not the idea. |
| **A3** | Objective | CE only / CE + ranking loss / CE + ranking loss **against prior logits** | **partial** — prior-relative and non-prior-relative run; CE-only not run | Is saturation an objective problem rather than an architecture problem? (RQ2) | No change in `lift` or in the `|K|` slope ⇒ the prior-relative objective is not the mechanism; look at the data. |
| **A4** | Set size | `\|K\|` truncated to 1 / 2 / 4 / 8 / 16 / 32 / 64 / 128 | **not run** — still the decisive test | **The decisive test.** Does recall increase monotonically in `\|K\|`? v2's curve is flat after 1 (§2.3). | Flat curve for v3 ⇒ the replacement hypothesis (§3.2) is falsified and the project's premise is wrong. |
| **A5** | Decoding | α ∈ {0, 0.6, 0.8, 1.0} × minlen ∈ {1, 2, 3} | **not run** on v3 | Re-run of §2.5 on the new model. Does a corpus with a realistic short-label share change the answer? | If α>0 still hurts on Common Crawl but helps on CT, that is the corpus-mismatch finding confirmed, and it belongs in the paper. |
| **A6** | Corpus | train on Common Crawl / train on CT / train on both | **not run** — CT corpus still being fetched | Isolates §2.4. Is the model's weakness the architecture or the training distribution? | CT-trained ≈ CC-trained on CT eval ⇒ the mismatch is not load-bearing and §2.4 is over-weighted here. |

**A4 runs first if compute is scarce.** It is the cheapest — it needs no retraining, only re-evaluation
at truncated `|K|` — and it is the only one that can falsify the project's premise rather than tune it.


### 7.1 The four runs that have been executed

`results/research/v3_ablation.json`, `results/research/v3_vs_v2_paired.json`.
Trained on the uncapped Common Crawl corpus (52,508 apexes; 13,272 with more
than 24 labels, so `max_set` is exercised by real data), scored through
`research/harness.py` on the same 545 apexes, seed-1337 split and budgets as
every baseline in §2.

```
method                                   recall@10             recall@25             recall@50            recall@100            recall@200       MAP  apex hit@max
------------------------------------------------------------------------------------------------------------------------------------------------------------------
settrans-full/hybrid             0.081 [0.068,0.096]     0.131 [0.112,0.152]     0.172 [0.151,0.195]     0.216 [0.194,0.241]     0.253 [0.229,0.280]    0.0556       317/545
settrans-full/generator          0.046 [0.033,0.059]     0.073 [0.058,0.090]     0.100 [0.082,0.119]     0.100 [0.083,0.120]     0.100 [0.083,0.120]    0.0340       137/545
settrans-full/retriever          0.086 [0.072,0.101]     0.136 [0.117,0.156]     0.177 [0.156,0.201]     0.219 [0.196,0.244]     0.252 [0.228,0.278]    0.0638       320/545
deepsets-full/hybrid             0.096 [0.080,0.112]     0.141 [0.121,0.162]     0.184 [0.162,0.208]     0.227 [0.202,0.254]     0.258 [0.233,0.286]    0.0656       317/545
deepsets-full/generator          0.060 [0.047,0.075]     0.093 [0.076,0.111]     0.119 [0.101,0.140]     0.122 [0.104,0.143]     0.123 [0.104,0.143]    0.0428       170/545
deepsets-full/retriever          0.100 [0.084,0.116]     0.145 [0.125,0.166]     0.184 [0.164,0.208]     0.232 [0.207,0.259]     0.260 [0.235,0.286]    0.0759       322/545
settrans-gen/generator           0.026 [0.019,0.034]     0.075 [0.061,0.091]     0.103 [0.086,0.122]     0.107 [0.089,0.127]     0.107 [0.089,0.126]    0.0171       162/545
settrans-noprior/hybrid          0.076 [0.062,0.092]     0.113 [0.094,0.134]     0.138 [0.118,0.161]     0.161 [0.140,0.185]     0.184 [0.159,0.210]    0.0537       216/545
settrans-noprior/generator       0.048 [0.035,0.061]     0.091 [0.074,0.108]     0.119 [0.100,0.140]     0.122 [0.103,0.142]     0.122 [0.103,0.143]    0.0382       164/545
settrans-noprior/retriever       0.082 [0.066,0.098]     0.117 [0.096,0.138]     0.134 [0.114,0.156]     0.155 [0.133,0.179]     0.180 [0.157,0.207]    0.0631       211/545
```

Paired against v2 — same apexes, so between-apex variance cancels:

```
v3 variant vs subfury-v2                                 @10                       @25                       @50                      @100                      @200
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
deepsets-full/retriever              -0.015 [-0.024,-0.006]*   -0.026 [-0.037,-0.016]*   -0.023 [-0.033,-0.013]*   +0.016 [+0.005,+0.028]*   +0.043 [+0.031,+0.055]*
deepsets-full/hybrid                 -0.019 [-0.028,-0.010]*   -0.031 [-0.041,-0.021]*   -0.023 [-0.033,-0.014]*   +0.011 [-0.000,+0.021]    +0.041 [+0.030,+0.053]*
settrans-full/retriever              -0.028 [-0.039,-0.018]*   -0.036 [-0.048,-0.023]*   -0.030 [-0.043,-0.017]*   +0.003 [-0.010,+0.016]    +0.035 [+0.022,+0.048]*
settrans-noprior/retriever           -0.032 [-0.042,-0.024]*   -0.055 [-0.068,-0.043]*   -0.073 [-0.087,-0.060]*   -0.061 [-0.075,-0.048]*   -0.037 [-0.050,-0.024]*

* = paired 95% CI excludes zero.  Positive favours v3.
```

**A2 — the pre-registered kill criterion fired.** §7's A2 row states: *"Hybrid ≤
max(channels) at every N ⇒ the fusion scorer is broken, not the idea."* For
`deepsets-full` the hybrid is at or below the retriever alone at every budget —
0.096/0.100, 0.141/0.145, 0.184/0.184, 0.227/0.232, 0.258/0.260 — and the same
holds for `settrans-full`. The falsifier list in §8 says the same thing from the
other direction. **The fusion scorer is therefore reported as broken.** The
components are not: the retriever alone is the best v3 configuration measured.
This is a pre-registered negative result, not a tuning opportunity, and it must
not be renegotiated by searching for a fusion weight that reverses it.

**A2 — generation cannot fill a budget, and this is now measured on v3 as well
as v2.** Every generator arm is flat from N=50 onward (`deepsets-full`:
0.119 → 0.122 → 0.123). The retriever arms keep climbing to N=200. §2.1
attributed v2's saturation to beam exhaustion rather than to the model; a
second architecture reproducing it on the same split supports that reading.

**A3 — subtracting the prior improves recall while making the loss look worse.**
`settrans-noprior` recorded the *best* validation ranking loss of the four runs
(3.010 against 3.877 for `settrans-full`) and is the *worst* ranker at every
budget, losing to v2 at all five. This is the §4 argument confirmed by
measurement: cross-entropy rewards reproducing `P(y)`, and a model that does so
scores well on the loss and adds nothing over the prior. Had A3 not been run,
the better validation loss would have selected the worse model.

**A3, second effect — the retrieval head improves the generator sharing its
encoder.** `settrans-gen` (generator trained with no retrieval head) reaches
0.026 at N=10; the same generator inside `settrans-full` reaches 0.046. The
ranking objective is doing work on the shared encoder that the generation loss
alone does not.

**A1 — Deep Sets beats the Set Transformer at every budget**, and the A1
falsifier cannot yet be evaluated: `concat`, the arm that would show whether a
set encoder beats v2's sorted concatenation *at all*, has not been run. Until it
is, no claim in §4 about the set encoder is supported by an ablation.

**Cost.** Retriever-only evaluation of 545 apexes takes 7–10s; every arm that
decodes takes 78–89s (`seconds`, same artifact). The best-scoring configuration
is also the cheapest by an order of magnitude.

### 7.2 What these four runs do not show

- They train on Common Crawl, so they inherit §2.4 in full. They are an
  architecture result on a benchmark already established as mismatched to the
  use case, and they say nothing about the operational failure that motivated
  v3.
- **A4 has not been run.** The `|K|` curve is the test that can falsify §3.2,
  and no v3 number here bears on it. v3 currently has an unbounded set encoder
  whose benefit over one label is unmeasured.
- No v3 number here was produced against live DNS, CT, or a temporal split.

---

## 8. Open questions, and what would falsify the hypothesis

**Open**

1. **Why does one label carry 90% of the conditioning benefit?** Three candidate mechanisms, currently
   indistinguishable: (i) intra-organisation labels are genuinely near-interchangeable as evidence, so
   there is little to extract; (ii) the objective gives no gradient for extracting more (RQ2);
   (iii) the capacity is there but the training regime never demands it — `subfury_v2/train.py` samples
   `k ~ U[1, min(n−1, 12)]` known labels, so the model spends most of training on small `K` and rarely
   sees a large set at all. **(iii) is testable immediately** by retraining v2 with a `|K|`-biased
   sampler and re-running the swap diagnostic; if `single` drops relative to `own`, the saturation was
   a training-regime artifact and not an architectural limit.
2. **A train/inference mismatch nobody has priced.** Training shuffles `K` and caps it at 12
   (`GroupDataset(max_known=12)`); inference sorts `K` and takes the alphabetically-last 24
   (`predict.py`). The model never saw a sorted context during training and never saw more than 12
   labels. This may be why window choice is irrelevant (§2.3) — not because sets are interchangeable,
   but because the conditioning format at inference is out of distribution. **This confound must be
   resolved before the capacity result is cited as evidence about set encoders at all.**
3. **What predicts an apex's `seed_driven_share`?** It ranges 0.16 to 0.94 across six real apexes
   (`results/subfury_v2/divergence_real.json`). If it is predictable from `K`, the system can tell an
   operator when to trust the model over the prior — which is worth more operationally than a recall
   point.
4. **Is the crossover budget a property of this model or of generation-plus-beam-search generally?**
   If every generative decoder saturates around N=50 regardless of size and training, the correct
   architecture is retrieval-led with a generation channel bounded at ~50, not a hybrid balanced
   across the whole budget.
5. **Does the 57.6% ceiling move with corpus size?** It is a property of `data/groups_train.jsonl`
   (494,934 distinct labels). If 10× the corpus raises it to 75%, the case for open-vocabulary
   generation weakens considerably and the project should become a retrieval project.

**Resolved by the A1/A2/A3 runs (§7.1)**

6. **Is saturation an objective problem or an architecture problem?** Partly the objective: the
   prior-relative ranking loss is worth +0.077 recall@200 over the same architecture without it
   (0.260 against 0.180 on the retriever arm). But the generator saturates in v3 exactly as it did
   in v2, so the *decoder* half of the saturation is architectural and the fix was to stop asking a
   decoder to fill a 200-slot budget.
7. **Does the hybrid beat its channels?** No. A2's kill criterion fired; see §7.1. Question 4 below
   is correspondingly sharpened rather than answered.

**Falsifiers — stated in advance, so the result is not renegotiated after it arrives**

- **A4 shows a flat `|K|` curve for v3.** If a set encoder with unbounded `|K|`, a prior-relative
  ranking loss, and a retrieval channel *still* gets 90% of its benefit from one label, then the
  hypothesis in §3.2 is wrong: the information is not in the set. The correct conclusion is that a
  well-fitted global prior plus a small generative channel for the top-50 is the right system, and the
  paper is a negative result about set conditioning for hostname discovery — which is publishable and
  more useful than a 2-point recall improvement.
- **A1 shows `concat ≈ settrans` everywhere.** Then the set encoder is engineering (it removes the
  24-label cap) and not science, and every claim about it must be reworded.
- ~~**The hybrid never beats its own union-of-channels reference** (0.2623 at N=200,
  `results/research/decomposition.json`). Then the fusion scorer, not the components, is the
  problem.~~ **This falsifier fired** — see §7.1, axis A2. The fusion scorer is the problem.
- **Temporal-CT recall collapses to prior-level at every budget.** Then the offline Common Crawl
  numbers were measuring corpus reconstruction, not transferable conditioning, and §2 must be
  re-reported as a study of the benchmark rather than of the method.
- **A live-resolution run yields queries-per-discovery worse than Marchal et al.'s ≈200 or Regulator's
  ≈100.** Then the budget-efficiency claim fails against 2012 and 2022 prior art, offline recall
  notwithstanding.

---

## Artifact index

| Artifact | Produced by | Contents |
|---|---|---|
| `results/research/baselines.json` | `research/run_baselines.py` | 6 methods × 5 budgets, bootstrap CIs, paired tests vs the prior, reachable subset, micro/macro |
| `results/research/decomposition.json` | `research/diagnostic/decompose.py` | 545 rows: model / prior / union / lift / overlap / novel_frac per budget; lift by `\|K\|` |
| `results/research/swap.json` | `research/diagnostic/swap.py` | 100 rows: own / single / swapped / generic conditioning |
| `results/research/capacity.json` | `research/diagnostic/capacity.py` | 30 rows: six 24-label windows of the same `K` |
| `results/research/lengthbias.json` | `research/diagnostic/lengthbias.py` | 12-cell α × minlen sweep, 80 apexes; short-label share |
| `results/subfury_v2/eval.json` | `subfury_v2/evaluate.py` | End-to-end recall@N vs the n0kovo wordlist (**wordlist baseline, not the prior**) |
| `results/subfury_v2/divergence.json` | — | 6 synthetic profile prompts, pairwise Jaccard |
| `results/subfury_v2/divergence_real.json` | — | 6 real apexes with real seeds, pairwise Jaccard |
| `research/data/ct_observations.jsonl` | `research/data/ct_fetch.py` | 40 apexes, CT `first_seen` per label, wildcard/truncation stats |
| `research/data/temporal.jsonl` | `research/data/build_temporal.py` | T=2024-07-01 split, 21 apexes, 4,969 known / 1,996 future |
| `research/related-work.md` | literature lane | 14-work matrix, SubWiz's real numbers, novelty assessment, verification log |
| `research/harness.py` | — | The evaluation contract (§5.1) |
| `results/research/v3_ablation.json` | `research/v3/score_ablation.py` | 4 v3 runs × 3 channels × 5 budgets, bootstrap CIs, wall-clock per run |
| `results/research/v3_vs_v2_paired.json` | `research/v3/paired_v2_v3.py` | Paired per-apex bootstrap of each v3 variant against subfury-v2 |
| `research/v3/model.py`, `research/v3/test_invariance.py` | — | v3 architecture; invariance/size/padding assertions |
