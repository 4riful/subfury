# Related Work: Budgeted Set-Conditioned Subdomain Discovery

**Compiled:** 2026-09-06 · **Lane:** literature · **Status:** first pass, all citations verified against a primary
source unless explicitly flagged.

**Problem we are positioning against.** Given an unordered set `K` of known hostnames belonging to one
organization, rank or generate the most probable *undiscovered* hostnames under a DNS query budget `N`,
maximizing discoveries per query.

**Three claimed contributions being tested for novelty:**
- **(a)** permutation-invariant conditioning on the organization's known-hostname *set*
- **(b)** hybrid retrieval + open-vocabulary generation
- **(c)** budget-aware ranking / evaluation

**How this was searched.** arXiv API (`abs:` field queries), DBLP publication API, OpenAlex citation graphs,
Semantic Scholar Graph API, Crossref, plus open-web search and direct source-code inspection of the two
closest tools. Two negative results are load-bearing and are reported as findings, not as absence of effort:
arXiv returns **0 papers** with `"subdomain enumeration"` or `"DNS enumeration"` in the abstract, and DBLP
returns exactly **1** publication for `subdomain enumeration` in its entire index (Degani et al., SAC '22).
The academic literature on this problem is genuinely thin; the state of the art lives in tools and blog posts.

---

## 1. The matrix

| Work | Year | Venue | Method | Conditioning | Evaluation | Reported result | What it does NOT do |
|---|---|---|---|---|---|---|---|
| **SubWiz** (Hadrian Security) | 2024–25 | GitHub / HuggingFace, **not peer-reviewed** | 17.3M-param nanoGPT decoder-only LM; custom 8192-token tokenizer; beam search with pruning; recursive re-feeding | Apex domain + known subdomains **lexicographically sorted and comma-joined into one causal token string**, left-truncated to context | 369 apexes; subfinder seeds; live DNS resolution; metric = mean resolved subdomains per apex vs subfinder-alone | +5.18 % (v0.4.1), **+6.74 %** (v1.0.1 default), +14.95 % (`max_recursion=50`) over subfinder's 24.96/apex | No permutation invariance (canonical sort ≠ invariance); no retrieval component; no query-cost accounting; no recall@N; no equal-budget baseline; no temporal split; no published paper |
| **SDBF** (Wagner et al.) | 2012 | IEEE/IFIP NOMS, pp. 1001–1007, DOI 10.1109/NOMS.2012.6212021 | Per-DNS-label-level character *n*-gram Markov chain + distributions over #labels, label length, first character; samples names, then probes | **Global** passive-DNS corpus statistics; apex supplied as a fixed `--suffix`, optional `--prefix` | Configured to emit as many labels as DNSenum (266,930); compared to Fierce (1,895-word dict) and DNSenum by count of resolving names | "SDBF and Fierce provide the best results, but all of them are complementary" (as summarized in Marchal et al. 2012) | Not conditioned on the target's own known set; pure sampling, **no ranking**; no retrieval; no budget optimization; no notion of held-out ground truth |
| **Semantic Exploration of DNS** (Marchal et al.) | 2012 | IFIP Networking, LNCS 7289 | Distributional/semantic similarity (DISCO) over label words + word splitter + numeric incrementer, iterated | **Yes — conditioned on the target's already-discovered hostname list** (horizontal: words similar to observed labels; vertical: iterate on new finds) | 24 domains (19 Alexa top-50 + 5 Luxembourg); metric `%Imp = |New| / |Init|`; **overhead = # additional DNS probes** | Mean improvement 84–102 % over 24 domains; +30 % over the *union* of SDBF∪Fierce∪DNSenum; **≈1 new hostname per 200 probes**; <100k probes vs >250k for SDBF/DNSenum | Not learned end-to-end; no permutation-invariant encoder; no open-vocabulary generation (only dictionary/semantic-neighbour expansion); budget is *measured*, not *optimized against* |
| **Regulator** (cramppet) | 2022 | Personal blog + GitHub, **not peer-reviewed** | Levenshtein-closure clustering of observed hostnames → heuristic regex induction (`closure_to_regex`) → exhaustive enumeration of the induced regular language via `dank`; 25× ratio filter | **Yes — target-specific**, induced from ≥~100 observed hostnames of that org | One worked target (adobe.com, 1,960 amass seeds); resolved with puredns; dedup against seeds; plus unreported "more trials" in an image | 124,821 guesses → **1,242 new hostnames** (≈1 %); altdns: 3,619,334 guesses → 362; author states it "still loses in general to `dnsgen`" | Closed vocabulary — can only recombine observed tokens; no learning across organizations; no probabilistic ranking (enumeration is unordered); fails on <100 seeds or unstructured zones; single-target, non-scientific evaluation (author's own words) |
| **GAN subdomain enumeration** (Degani et al.) | 2022 | ACM SAC '22, pp. 1636–1645, DOI 10.1145/3477314.3506967 | GAN samples unseen candidates from the distribution of valid subdomain names | **Global** — distribution learned from public datasets; *not* conditioned on the target's set | 15 bug-bounty domains + ground truth of 1,164 other targets; metrics = candidate validity and sample uniqueness | Traditional workflow performance "increased by up to 61 %"; GAN guessed **on average 32 % of subdomains** in the ground-truth experiment | No per-organization conditioning at all; no retrieval; no budget-aware ranking; no temporal split; only 5 citations to date |
| **Deep Sets** (Zaheer et al.) | 2017 | NIPS 2017, pp. 3391–3401, arXiv:1703.06114 | Permutation-invariant architecture: elementwise φ → sum-pool → ρ; theory of permutation-invariant/equivariant functions | Unordered set, by construction | **Set expansion** = text concept set retrieval on LDA-1k/3k/5k (17k/38k/61k vocab, 50 words/topic); also image tagging, set anomaly detection | Recall@10/@100/@1k, MRR, median rank vs Random, Bayes Set, w2v-Near, NN-max/-sum-con/-max-con | Not a security/DNS application; closed vocabulary (ranks a fixed vocabulary, cannot emit unseen strings); no notion of query cost |
| **Set Transformer** (Lee et al.) | 2019 | ICML 2019, pp. 3744–3753, arXiv:1810.00825 | SAB / **ISAB** (inducing points → linear complexity) / **PMA** (pooling by multi-head attention) | Unordered set, by construction | 3D shape recognition, few-shot classification, multiple-instance learning, amortized clustering | SOTA on set-input tasks at the time | No generation of novel discrete strings; no security application; no budget notion |
| **TSPN / Conditional Set Generation with Transformers** (Kosiorek et al.) | 2020 | arXiv:2006.16841; ICML 2020 OOL workshop | Permutation-**equivariant** set generator built on transformers, extending DSPN (Zhang et al., NeurIPS 2019); explicitly predicts set cardinality | Conditioned on a fixed-size vector (image/latent), generates a set | SET-MNIST point-cloud generation; CLEVR object detection | Outperforms DSPN on element quality and predicted set size | Generates *continuous* elements (points/boxes), not discrete open-vocabulary strings; not set→set; no ranking under budget |
| **S2SRec2** (Cao et al.) | 2025 | arXiv:2507.09101 | Set-to-set basket completion; Set Transformer encoder, multi-task (retrieve missing items + assess completeness) | Unordered partial basket | Large-scale recipe datasets; Precision@k, Recall@k, F1@k, MSE@k | Significantly beats single-target baselines; **"The Bi-directional LSTM baseline, which imposes a sequential structure on unordered ingredient sets, performs worst overall"** | Closed item vocabulary (retrieval only); no generation of novel items; no query budget |
| **SetExpan** (Shen et al.) | 2017 | ECML-PKDD 2017 (LNCS 10534), arXiv:1910.08192 | Corpus-based entity set expansion: context-feature selection + rank-ensemble, iterative | Unordered **seed set** of entities | 3 datasets; MAP@K; recall@k (k=10,20,50) against an unordered ground-truth set | Robust, beats prior SOTA in MAP | Retrieval from a fixed corpus vocabulary; no generation; no cost model |
| **6Gen** (Murdock et al.) | 2017 | ACM IMC 2017, pp. 242–253, DOI 10.1145/3131365.3131405 | Agglomerative clustering of seeds by Hamming distance → dense-region range expansion → candidate targets | **Unordered seed set** of IPv6 addresses; **"sole parameter: probe budget"** | Live ICMP/port scanning; hit rate = newly responsive addresses ÷ budget | Recovers **1–8×** as many addresses as Entropy/IP for the same input | IPv6 address space, not hostnames; no learned encoder; no language model; no retrieval |
| **Entropy/IP** (Foremski et al.) | 2016 | ACM IMC 2016, pp. 167–181 | Entropy analysis of address segments + clustering + Bayesian network generative model | Unordered seed set of IPv6 addresses | Live scanning, hit rate under budget | Superseded by 6Gen (1–8× worse) | Same as above |
| **Target Acquired?** (Steger et al.) | 2023 | IFIP TMA 2023 | Systematic comparison of **10** IPv6 TGAs under a common protocol | Seed sets from the IPv6 Hitlist, categorized by network type | Fixed **10 M** output/scanning budget per algorithm; dedup against seed set; measure responsiveness by protocol | TGAs show "vastly differing responsiveness levels" | IPv6; no temporal train/test split; hostnames out of scope |
| **SEAT-FA** (Hegde & Chattaraj) | 2025 | IEEE CISCON 2025, DOI 10.1109/CISCON66933.2025.11337258 | *Unverified — no accessible full text* | Unknown | Unknown | Unknown | Cannot be assessed; listed only so it is not silently missed |

---

## 2. Per-work notes

### SubWiz (Hadrian Security) — the closest prior art
A 17.3M-parameter decoder-only transformer copied from Karpathy's nanoGPT, trained on 26M tokens of
subdomain lists from passive sources with a custom 8,192-token tokenizer; MIT-licensed, weights on
HuggingFace (`HadrianSecurity/subwiz`, ~177k downloads/month). Inference is a custom beam search
(`pruning_ratio=4`, `pruning_offset=5`, `max_new_tokens ≤ 20`, default `n=500`), with recursive re-feeding of
newly resolved names up to `max_recursion=5`. **The conditioning detail matters for our claim:** reading
`subwiz/main.py`, the prompt is built as `apex + "[DELIM]"` followed by `",".join(sorted(subs)) + "[DELIM]"`,
then left-truncated to fit `block_size`. So the known set is *canonically ordered*, not *permutation-invariant*
— an important distinction, because the left-truncation means that when `|K|` exceeds the context window the
model sees only the alphabetically-late tail of the organization's hostnames. Its real evaluation is the
repo's `benchmark/benchmark.ipynb`: 369 apex domains, seeded by subfinder (9,210 hostnames total, mean 24.96
per apex, **median 5**), wildcard- and outlier-filtered, ground truth by live DNS resolution. It measures
*mean resolved subdomains per apex* against subfinder alone: +5.18 % / +6.74 % / +14.95 %. The marketing
claim of "up to a 10 % increase in discovered subdomains on customer environments" is not the same number as
the reproducible +6.74 %. There is no paper, no recall@N, no equal-budget baseline, and no accounting of how
many DNS queries were spent to obtain the lift.

### SDBF: Smart DNS Brute-Forcer (Wagner, François, State, Engel, Wagener, Dulaunoy)
IEEE/IFIP NOMS 2012, pp. 1001–1007. The first DNS scanner built on statistical language modelling. Two
modules: a *Processor* that mines a passive-DNS corpus into distributions (number of labels, label length per
level, first-character per level, and per-level character *n*-gram transition matrices) and a *Generator* that
walks the resulting Markov chain to synthesize names, which are then probed live. The released package
(`jfrancois/SDBF`, GPLv3, 2012-07-15) exposes `markov.pl <domains> <n-gram size> …` and
`sdbf.py -n <count> -p <prefix> -s <suffix> -w <levels>`; `-n` is effectively a probe budget knob but is not
optimized against. **Conditioning is global**, not per-organization: the model learns from a corpus, and the
target enters only as a fixed suffix. It samples rather than ranks, so there is no principled "top-N".
*Caveat:* the NOMS PDF itself is behind an Anubis challenge on HAL and IEEE Xplore; the method description
above is reconstructed from the released source README and from the detailed SDBF overview in the authors'
own companion paper (Marchal et al. 2012, §3.2), both primary sources. I could not read the NOMS
evaluation section directly and therefore do not quote numbers from it.

### Semantic Exploration of DNS (Marchal, François, Wagner, Engel) — the overlooked one
IFIP Networking 2012, LNCS 7289. This is the closest prior work on **query-efficiency**, and the lane's
briefing did not list it. It takes the hostnames already discovered for a target (by SDBF, Fierce, or
DNSenum) and expands them via (i) distributional semantic similarity of label words using DISCO trained on
Wikipedia, (ii) a word splitter, and (iii) an incremental numeric module, iterating on new discoveries. It
is explicitly **conditioned on the target organization's known set** and explicitly **measures cost**:
"The overhead is defined as the number of additional DNS requests (#probes)." Reported: mean `%Imp` of
84–102 % across 24 domains, +30 % beyond the union of all three seeding tools, <100k probes (vs >250k for
SDBF/DNSenum), 200–500 probes per initial name, and **roughly one new hostname per 200 probes** for the
similar-names module; 55–80 % of findings arrive in the first iteration and >95 % by the fourth. Any
"discoveries per query" claim we make will be compared to this.

### Regulator (cramppet)
A 2022 blog post and GitHub tool, not peer-reviewed, and honest about it ("Here's the non-scientific
methodology I used"). Method: build a table of pairwise Levenshtein distances over the observed hostnames,
form "Levenshtein closures", and heuristically fold each closure into a regex by merging shared token levels
into alternations and collapsing numeric runs into ranges (`(foo[1-5])-(dev|qa|prod).example.com`); a 25×
ratio test discards rules that would blow past ~1M guesses; `dank` then enumerates each language and set-
subtracts the observed names. Worked example on adobe.com: 1,960 amass seeds → 6,215 rules → 124,821 guesses
→ **1,242 new hostnames** (≈1 % hit rate, ≈100 queries per discovery), versus altdns at 3,619,334 guesses →
362. The author explicitly states Regulator "still loses in general to `dnsgen`". Limitations he names:
needs ≥~100 observed subdomains, fails when hostnames are unstructured, and the induction is "a simple
heuristic function", not L-star/RPNI. **Crucially it is closed-vocabulary**: it can only recombine tokens
already seen for that organization.

### GAN-based subdomain enumeration (Degani, Bergadano, Mirheidari, Martinelli, Crispo)
ACM SAC '22, pp. 1636–1645. The only peer-reviewed paper in DBLP on this exact task. A GAN samples unseen
candidates from the distribution of valid subdomain names, learned from public datasets, replacing the
dictionary in an otherwise standard generate-then-validate pipeline. Evaluated against 15 carefully chosen
bug-bounty domains plus a ground truth of 1,164 other targets, benchmarking "candidates' validity and sample
uniqueness". Reported: enumeration workflow performance "increased by up to 61 %", and the GAN "was able to
guess, on average, 32 % of subdomains" in the ground-truth experiment. The model is **unconditioned on the
specific organization** — it draws from a global distribution — which is precisely the gap our set
conditioning targets. It has 8 Semantic Scholar citations / 5 OpenAlex citing works, of which only one
(URLGEN, IEEE TNSM 2022, GAN-based URL generation by an overlapping author group) is methodologically close.

### Deep Sets (Zaheer, Kottur, Ravanbakhsh, Póczos, Salakhutdinov, Smola)
NIPS 2017, pp. 3391–3401. Characterizes permutation-invariant functions and gives the canonical
φ→sum-pool→ρ architecture. The relevant experiment for us is **set expansion**: "text concept set retrieval"
on LDA-1k/3k/5k (17k/38k/61k vocabularies, 50 top words per topic as ground-truth sets) — condition on a
subset of a concept's words, retrieve the rest — evaluated with **Recall@10, Recall@100, Recall@1k, MRR and
median rank**, against Random, Bayes Set, word2vec nearest neighbours, and ablated pooling variants
(NN-max, NN-sum-con, NN-max-con). It also does image tagging on ESPgame/IAPRTC-12.5/COCO-Tag (recall@K, MRR,
median rank). This is the direct precedent for our *metric*, in a non-security domain, and the direct
precedent for the architectural claim — but it is retrieval over a **fixed vocabulary**, never generation.

### Set Transformer (Lee, Lee, Kim, Kosiorek, Choi, Teh)
ICML 2019, pp. 3744–3753. Attention-based permutation-invariant encoder/decoder: SAB for self-attention
among set elements, ISAB using inducing points to cut quadratic to linear complexity, and PMA (pooling by
multi-head attention) for learned aggregation instead of a sum. Experiments: amortized clustering, 3D shape
recognition (point clouds), few-shot classification, multiple-instance learning. This is the encoder family
our contribution (a) would instantiate; nothing in the paper touches strings, security, or budgets.

### TSPN / Conditional Set Generation with Transformers (Kosiorek, Kim, Rezende)
arXiv:2006.16841, ICML 2020 Object-Oriented Learning workshop. Extends Deep Set Prediction Networks (Zhang
et al., NeurIPS 2019) into a transformer-based permutation-**equivariant** set generator that explicitly
learns set cardinality, letting it generalize to larger sets. Evaluated on SET-MNIST (point-cloud
generation) and CLEVR (object detection); it beats DSPN on both element quality and predicted set size.
The generated elements are continuous vectors (points, boxes), **not discrete open-vocabulary strings**, and
the conditioning is a single vector, not a set. It is the right citation for "generate a set, not a
sequence", and the wrong architecture to copy directly.

### Set expansion / complementary-item prediction where unordered beats sequential
- **SetExpan** (Shen et al., ECML-PKDD 2017): corpus-based entity set expansion from an unordered seed set
  via context-feature selection plus a rank-ensemble; evaluated with MAP@K and recall@k for k∈{10,20,50}
  against an unordered ground-truth set. **CGExpan** (BERT + Hearst patterns) is the standard follow-up —
  I did not independently verify its venue (commonly cited as ACL 2020), so treat that detail as unconfirmed.
- **S2SRec2** (Cao et al., arXiv:2507.09101, 2025): reformulates basket completion as set-to-set
  recommendation with a Set Transformer, and reports directly that the **Bi-LSTM baseline, "which imposes a
  sequential structure on unordered ingredient sets, performs worst overall"** among five baselines. This is
  the cleanest citable empirical evidence that permutation-invariant conditioning beats sequence conditioning
  on the same unordered context — the exact argument we need against SubWiz's sorted-concatenation prompt.
- Adjacent supporting evidence: work on next-basket / session recommendation reports unordered set models
  without positional encoding outperforming ordered sequence models; and **Set-Encoder** (arXiv:2404.06912)
  makes the same permutation-invariance argument for listwise passage re-ranking.

### IPv6 target generation — the strongest structural analogue (not in the brief; a reviewer will raise it)
**6Gen** (IMC 2017) formalizes exactly our problem shape in a different address space: given an unordered
**seed set** and a **probe budget** (the paper's "sole parameter"), generate candidates that maximize live
hits; it beats **Entropy/IP** (IMC 2016) by 1–8×. Hit rate is defined as newly responsive addresses (in the
candidate set, responsive, not in the seed set) divided by the budget. **Target Acquired?** (Steger, Kuang,
Zirngibl, Carle, Gasser; TMA 2023) is the community's systematic 10-algorithm comparison, using a fixed
10M-address scanning budget, deduplication against the seed set, and multi-protocol responsiveness scanning.
None of these do a temporal split, and none work on hostnames — but the *budgeted, seed-set-conditioned
discovery* framing is fully established there, so we cannot claim it as novel framing in general, only as
novel for the hostname/label space.

---

## 3. The critical question: does the (a)+(b)+(c) combination already exist?

**Short answer: no published work combines all three, and I found no work combining even two of them for
subdomain or asset discovery.** Searches run: arXiv API on `abs:subdomain`, `abs:"subdomain enumeration"`,
`abs:"DNS enumeration"`, `abs:"attack surface" AND abs:discovery AND cat:cs.CR`; DBLP on
`subdomain enumeration`, `subdomain discovery`; OpenAlex forward-citation graphs of both SDBF and Degani et
al.; Semantic Scholar and Crossref lookups; plus open-web searches for *attack surface discovery*, *asset
discovery*, *DNS enumeration*, *hostname prediction*, *subdomain generation*, *infrastructure completion*,
and combinations with *permutation-invariant*, *Deep Sets*, *Set Transformer*, *retrieval-augmented*,
*budget-aware*, *query-efficient*.

Component by component:

| Component | Occupied? | By whom, and how completely |
|---|---|---|
| **(a) Permutation-invariant conditioning on an org's known hostname SET** | **Unoccupied for hostnames.** | The *machinery* is fully established and heavily cited (Deep Sets, Set Transformer, TSPN, S2SRec2, SetExpan). Nobody has applied it to hostnames or DNS assets. SubWiz conditions on the set but as a **sorted concatenated token string** in a causal LM — a canonical ordering, and one that left-truncates the alphabetical tail; Regulator conditions on the set but through non-learned Levenshtein/regex machinery; Marchal et al. condition on the set through hand-built semantic modules. None is a learned permutation-invariant encoder. |
| **(b) Hybrid retrieval + open-vocabulary generation** | **Unoccupied for this task.** | Retrieval-only: Deep Sets set expansion, SetExpan/CGExpan, S2SRec2, and every wordlist tool. Generation-only: SDBF (Markov), SubWiz (LM), Degani (GAN), Regulator (regex enumeration — and closed-vocabulary, so not even open-vocab). Marchal et al. is arguably the closest to a hybrid (retrieve semantic neighbours + synthesize numeric increments), but it is not learned and not a unified scorer. No work I found combines a retrieval channel over a global label prior with an open-vocabulary generative channel under one calibrated ranking. |
| **(c) Budget-aware ranking** | **Partly occupied — flag this honestly.** | Budget-*aware evaluation* is standard in IPv6 target generation (6Gen's "sole parameter: probe budget"; TMA 2023's fixed 10M budget) and is measured explicitly in Marchal et al. 2012 (overhead = #probes; ~1 discovery per 200 probes). Budgeted top-K ranking is also textbook in IR. What is genuinely absent is budget-aware ranking **for hostname discovery**: SubWiz, Degani et al., and Regulator all report absolute discoveries without normalizing by queries spent, and no hostname work optimizes the ranking *for* a budget. So the honest claim is "first budget-aware ranking for hostname discovery", not "first budget-aware ranking". |

**Nearest misses, in order of closeness:**
1. **SubWiz** — same task, same input, same output; differs on (a) mechanism, (b) no retrieval, (c) no budget accounting.
2. **Marchal et al. 2012** — set-conditioned and cost-measured, but hand-engineered and pre-deep-learning.
3. **6Gen / TMA 2023** — the (a)+(c) framing done properly, but on IPv6 addresses, without a learned set encoder.
4. **Degani et al. 2022** — the only peer-reviewed hostname-generation paper, but unconditioned on the target.

---

## 4. Metrics and evaluation protocols used by prior work

So that our numbers are comparable, here is what each line of work actually reports.

**Hostname / subdomain discovery**
- *SubWiz*: mean resolved subdomains per apex, and % increase over a subfinder-only baseline, on 369 apexes.
  Ground truth is live DNS resolution at eval time. Wildcard filtering and top-k outlier filtering applied.
  **No** recall@N, no equal-budget comparison, no query cost.
- *Degani et al.*: candidate **validity** (fraction of generated candidates that resolve) and **sample
  uniqueness**; % improvement of an end-to-end enumeration workflow (up to 61 %); and fraction of a known
  ground-truth set recovered (avg 32 % over 1,164 targets). The last one is effectively *recall*, which is
  the closest peer-reviewed analogue to our recall@N.
- *Marchal et al. 2012*: `%Imp = |New_i| / |Init_i|` per seeding tool and against the union; **overhead =
  number of additional DNS probes**; probes-per-initial-name; **discoveries-per-probe** (~1/200); saturation
  curves over iteration depth.
- *Regulator*: raw counts — guesses issued vs new hostnames found — on a single target, resolved with
  puredns, deduplicated against the seed set. Implicitly a hit rate (1,242/124,821 ≈ 1.0 %).
- *SDBF*: number of resolving names found at a generation count matched to a dictionary tool's size.

**Set expansion / set-to-set (the metric family we are borrowing)**
- *Deep Sets*: **Recall@10 / @100 / @1k, MRR, median rank**.
- *SetExpan*: **MAP@K**, recall@{10,20,50}.
- *S2SRec2*: **Precision@k, Recall@k, F1@k** (+ MSE@k for cardinality).

**IPv6 target generation (the budget family)**
- **Hit rate** = |newly responsive candidates \ seed set| ÷ budget, under a fixed probe budget; candidate sets
  deduplicated and stripped of seed-set overlap before scoring; responsiveness measured per protocol.

**Recommended reporting for SubFury, to be comparable to all of the above:**
1. `recall@N` on held-out hostnames (comparable to Deep Sets / SetExpan / Degani's 32 %).
2. `hit rate @ budget N` = new resolving hostnames ÷ N queries (comparable to 6Gen and Marchal).
3. `queries per discovery` (comparable to Regulator's ≈100 and Marchal's ≈200).
4. `% lift over subfinder alone` at a stated query budget (the only way to compare with SubWiz's headline).
5. Report all of the above at several N so the budget curve is visible, not one point.

---

## 5. Temporal splits — searched for specifically; answer is no

**I found no prior work in this space that trains on hostnames observed before a time T and evaluates on
hostnames that first appeared after T.**

- *SubWiz*: the benchmark notebook uses a static snapshot of subfinder results per apex and resolves live at
  eval time. The 26M-token training corpus has no published date boundary, so **training-set contamination
  with benchmark hostnames cannot be ruled out** — the repo publishes neither the corpus nor a cutoff.
- *Degani et al.*: 15 targets + 1,164 ground-truth targets, no stated temporal boundary.
- *Regulator, SDBF, Marchal et al.*: all live-probe protocols; "new" means "not in the seed set", not
  "appeared later in time".
- *IPv6 TGAs (6Gen, Entropy/IP, TMA 2023)*: seed sets and scanning are contemporaneous; TMA 2023 studies
  address *lifetime* longitudinally but does not train-before-T / test-after-T.
- Certificate Transparency is used for *phishing-domain prediction before content goes live* (e.g.
  content-agnostic phishing detection from CT + passive DNS), which is temporal — but that is a
  classification task on registered domains, not organization-conditioned hostname discovery.

**Consequence for us:** a temporal split is a genuinely open evaluation contribution and is cheap
differentiation. Note that `subfury/evaluate.py` currently does a **random half-split per apex on
held-out apexes** — a set-completion split, not a temporal one. That is fine and matches Deep Sets' set
expansion protocol, but it is not the same claim, and it should not be described as predicting *future*
hostnames.

---

## 6. Novelty assessment

**Genuinely unoccupied**
- **Permutation-invariant set conditioning for hostname prediction.** No prior work, learned or otherwise,
  encodes an organization's known hostnames with a permutation-invariant encoder. The strongest supporting
  argument is external: S2SRec2 shows a Bi-LSTM over an unordered set "performs worst overall" against a Set
  Transformer on the same data. The claim is safe *if* framed as a transfer of established set-learning
  machinery (Deep Sets / Set Transformer) to a domain that has never used it — not as a new architecture.
- **Hybrid retrieval + open-vocabulary generation for hostname discovery.** Every prior system is purely one
  or the other, and the one closed-vocabulary system (Regulator) is explicitly limited by that.
- **Temporal-split evaluation for subdomain discovery.** Nobody does it.
- **A reproducible recall@N benchmark at equal candidate budget across many organizations.** Prior hostname
  work reports absolute counts, single targets, or lift over one specific tool.

**Partly occupied — must be attributed, not claimed clean**
- **Conditioning on the target's known set at all.** Occupied by Regulator (2022, regex induction), Marchal
  et al. (2012, semantic expansion), and SubWiz (2024, sorted-string prompt). Our contribution is the
  *mechanism*, not the idea.
- **Budget-aware ranking / query efficiency.** Occupied in spirit by 6Gen (probe budget as the sole
  parameter), TMA 2023 (fixed 10M budget protocol), and Marchal et al. (overhead in #probes; ~1 discovery per
  200 probes). Claim "first for hostname discovery", and cite all three.
- **Learned generative models over hostnames.** Occupied by SDBF (2012, Markov), Degani et al. (2022, GAN),
  SubWiz (2024, transformer LM). We are the fourth entrant, not the first.
- **Recall@N as the metric.** Standard in set expansion since Deep Sets (2017). Not novel; just correct.

**Not novel and should not be claimed**
- Using a transformer LM to generate subdomains (SubWiz).
- Seeding generation from passively discovered hostnames (SDBF → Marchal → Regulator → SubWiz, a 13-year line).
- Ranking candidates and cutting at N (basic IR).

**What a reviewer will most likely push back on**

1. **"Is your set encoder actually doing anything, or is it a sorted prompt with extra steps?"** The single
   most important ablation is same-data, same-budget: permutation-invariant encoder vs. sorted-concatenation
   causal LM (i.e. re-implement SubWiz's conditioning), plus a permutation-sensitivity test — shuffle `K`
   and show output stability, and show SubWiz's output changes when the set exceeds its context window
   (its left-truncation of a sorted list is an exploitable weakness, and a concrete figure).
2. **"Your baseline is a wordlist; theirs is subfinder."** SubWiz's only published number is lift over
   subfinder on 369 apexes with **median 5 seed hostnames**. If we evaluate on apexes with ≥6 labels
   (as `evaluate.py` does) we are in a *different, easier* regime. Either run the SubWiz benchmark protocol
   as-is, or state the regime difference explicitly. Running actual SubWiz as a baseline is cheap (pip
   install, MIT, weights public) and its absence will be noticed.
3. **"Recall@N without DNS is not discovery."** Our recall@N uses held-out real hostnames and no queries.
   Prior hostname work all resolves live. Expect a demand for at least one live-resolution experiment with
   queries-per-discovery reported, comparable to Marchal's ~1/200 and Regulator's ~1/100.
4. **"Training-set contamination."** If we train on a passive-DNS/CT corpus and evaluate on apexes present in
   it, recall@N is inflated. A temporal split plus an explicit apex-level disjointness statement kills this;
   without it, it is the first thing an adversarial reviewer attacks. Note we can also use this *against*
   SubWiz, which publishes neither corpus nor cutoff.
5. **"Novelty is a recombination of known parts."** True, and the defence is empirical: the set encoder must
   beat the sorted-sequence conditioning, and the hybrid must beat each channel alone, both at equal budget.
   Three ablations (set-encoder vs sequence; retrieval-only; generation-only) are non-optional.
6. **"You ignored IPv6 target generation."** 6Gen/Entropy/IP/TMA already formalize budgeted seed-conditioned
   discovery. Cite them up front and state the difference (discrete label space with linguistic structure and
   an open vocabulary, versus a numeric address space), rather than letting a reviewer find them.
7. **"n=369 / n=150 apexes is small."** Both SubWiz's benchmark and our `--max-apexes 150` default are small;
   report confidence intervals and per-apex distributions, since the median-5-seeds long tail dominates means.

---

## 7. Verification log

**Verified against a primary source:** SubWiz architecture, prompt construction and benchmark numbers
(GitHub source `subwiz/main.py`, `subwiz/model.py`, `benchmark/benchmark.ipynb` with stored outputs, and
`benchmark/benchmark_dataset.json` — 369 keys, 9,210 hostnames, mean 24.96, median 5); Regulator's method,
adobe.com numbers and the author's own caveats (cramppet.github.io writeup, scraped in full); Degani et al.
abstract, authors, venue, pages, DOI (Semantic Scholar Graph API + DBLP); SDBF authorship, venue, pages, DOI
(Crossref + DBLP) and method/CLI (released source README, `jfrancois/SDBF`); Marchal et al. method, metrics
and all quoted numbers (full-text PDF from `opendl.ifip-tc6.org`); Deep Sets set-expansion protocol, datasets,
metrics and baselines (ar5iv full text); Set Transformer and TSPN abstracts and components (arXiv);
S2SRec2 abstract, metrics and the Bi-LSTM sentence (arXiv HTML); 6Gen/Entropy/IP venue and pages (DBLP),
6Gen's "sole parameter: probe budget" (authors' IMC talk slides); TMA 2023 methodology and 10M budget
(full-text PDF).

**Explicitly NOT verified — do not cite these as established:**
- The NOMS 2012 SDBF paper's own evaluation section and numbers. HAL and IEEE Xplore both blocked; the method
  description here comes from the released code and the authors' companion paper.
- SubWiz's exact layer/head/embedding configuration (lives in `model_args` inside the `.pt` checkpoint; the
  HuggingFace `config.json` is literally `{}` and there is no model card). Only "17.3M parameters" is stated.
- SubWiz's training-corpus date cutoff, sources, and whether benchmark apexes appear in it. Unpublished.
- CGExpan's exact venue (commonly cited as ACL 2020; not independently confirmed here).
- SEAT-FA (CISCON 2025, DOI 10.1109/CISCON66933.2025.11337258): existence confirmed, content not accessible.
- Hadrian's "up to 10 % increase on customer environments" claim: marketing copy, no methodology published,
  and it does not match the reproducible +6.74 % in their own notebook.

## 8. Sources

- SubWiz: https://github.com/hadriansecurity/subwiz · https://huggingface.co/HadrianSecurity/subwiz ·
  https://hadrian.io/blog/how-ai-is-transforming-subdomain-enumeration-a-q-a-with-the-creators-of-subwiz ·
  https://hadrian.io/blog/can-llms-improve-subdomain-enumeration
- SDBF: https://dblp.org/rec/conf/noms/WagnerFSEWD12.html · DOI 10.1109/NOMS.2012.6212021 ·
  https://github.com/jfrancois/SDBF
- Semantic Exploration of DNS: https://opendl.ifip-tc6.org/db/conf/networking/networking2012-1/MarchalFWE12.pdf ·
  https://link.springer.com/chapter/10.1007/978-3-642-30045-5_28
- Regulator: https://cramppet.github.io/regulator/index.html · https://github.com/cramppet/regulator
- Degani et al.: https://dl.acm.org/doi/10.1145/3477314.3506967
- Deep Sets: https://arxiv.org/abs/1703.06114
- Set Transformer: https://arxiv.org/abs/1810.00825
- TSPN: https://arxiv.org/abs/2006.16841 · https://oolworkshop.github.io/program/ool_31.html
- S2SRec2: https://arxiv.org/abs/2507.09101
- SetExpan: https://arxiv.org/abs/1910.08192 · https://link.springer.com/chapter/10.1007/978-3-319-71249-9_18
- Set-Encoder: https://arxiv.org/abs/2404.06912
- 6Gen: https://dl.acm.org/doi/10.1145/3131365.3131405
- Entropy/IP: IMC 2016, pp. 167–181
- Target Acquired?: https://tma.ifip.org/2023/wp-content/uploads/sites/12/2023/06/tma2023-final50.pdf
