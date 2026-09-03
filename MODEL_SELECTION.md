# Model Selection — Research Notes

This documents *why* SubFury v2 uses the model it does, rather than defaulting
to the original notebook's DistilGPT-2. Every decision below is backed by
contemporary tooling or literature.

## Task, restated

Given a set of **known** subdomain labels of one organization, predict **other**
labels that organization is likely to run. This is set-conditioned generation of
short, non-linguistic strings — not natural-language modeling.

## Decision 1 — From scratch vs. fine-tune a pretrained LM → **from scratch**

| Evidence | Implication |
|---|---|
| HF empirical study (Sep 2025): from-scratch models beat pretrained on generalization below ~10 MB of data; pretrained only wins >10 MB. | Our corpus (~8 MB of labels, ~3M tokens) is squarely in from-scratch territory. |
| Subdomain labels are non-linguistic tokens (`vpn-gw2`, `k8s-prod`, `mx01`). | A pretrained English LM's linguistic priors are largely wasted; its English BPE fragments these strings. |
| subwiz (Hadrian) trains from scratch: 17.3M params, ~1000× smaller than ChatGPT, and reports ~10% more subdomains found than conventional methods. | A tiny purpose-built model is the validated state of the art here. |

Fine-tuning a pretrained model (ByT5-small, a small Qwen/Llama, GPT-2) would add
hundreds of MB, English-centric tokenization, and no accuracy upside at this data
scale.

## Decision 2 — Tokenization → **domain-specific BPE (4096)**

Candidates considered:

- **Character / byte-level** (ByT5, CANINE): zero OOV, robust to odd strings, but
  O(n²) longer sequences and no reusable subword units. Robustness matters less
  here because we *train the tokenizer on the same distribution* we generate.
- **Generic English BPE** (GPT-2's): fragments `staging`→`stag`+`ing`, wastes
  vocab on natural language. This was v1's mistake.
- **Domain-specific BPE** (chosen): trained on the subdomain corpus itself, so
  `api`, `dev`, `mail`, `-prod`, `vpn` become single/few tokens → shorter
  sequences, faster inference, and the model reasons over meaningful units.
  subwiz uses exactly this (8192-token subdomain BPE). We use 4096 because our
  corpus is smaller (~3M vs 26M tokens).

Pre-tokenizer splits on `.` and `-` (isolated) so structural separators are
first-class tokens.

## Decision 3 — Architecture → **decoder-only GPT (nanoGPT-style)**

- The task is naturally autoregressive: `known… [DELIM] target [END]`.
- Encoder-decoder (T5) would add a separate encoder for no clear benefit on
  short single-target generation, and complicate beam search.
- subwiz and the practical SOTA are decoder-only. We match it: pre-LN blocks,
  fused SDPA attention, learned positional embeddings, weight-tied LM head.

## Decision 4 — Size → **~6M params (6L / 6H / 300d)**

- Corpus ≈ 84k apexes, ~1.04M labels, ~8.2M chars ≈ **~3M BPE tokens**.
- Chinchilla's ~20 tokens/param would imply a *sub-million* param model; subwiz
  deliberately over-parameterizes (17.3M on 26M tokens ≈ 1.5 tok/param) because
  the task rewards memorizing organizational naming patterns.
- Our first cut (8L/8H/320d ≈ 11M) is too large for ~3M tokens and would overfit.
- **Known-subset augmentation** (re-sampling 1–k knowns per apex each epoch)
  multiplies effective training pairs well beyond the naive token count, which
  supports a mid-size model — but not an 11M one.
- Landing point: **6L / 6H / n_embd=300 / vocab=4096 ≈ 6M params**, dropout 0.1,
  AdamW + cosine schedule. Re-tune down if val loss diverges from train loss.

## Decision 5 — Decoding → **deterministic beam search**

v1 used temperature sampling (temp 0.7), which is the wrong objective: we want
the *N most probable* labels to minimize wasted DNS queries, not diverse samples.
Beam search (subwiz's choice) returns a ranked candidate list directly. Temperature
remains available for exploration but defaults to greedy/beam.

## What we did NOT do (and would, with more compute/data)

- Apex/TLD context conditioning (subwiz lists this as future work).
- Larger corpus: only 4 of 48 Common Crawl vertex parts were used here.
- Byte-level ablation to empirically confirm BPE > char on held-out recall.

## Sources
- https://huggingface.co/blog/RDTvlokip/from-scratch-vs-pre-trained
- https://github.com/hadriansecurity/subwiz
- https://hadrian.io/blog/how-ai-is-transforming-subdomain-enumeration-a-q-a-with-the-creators-of-subwiz
- https://github.com/cramppet/regulator
- https://dl.acm.org/doi/10.1145/3477314.3506967 (GANs for subdomain enumeration, ACM SAC '22)
- ByT5 / CANINE (byte/char-level tokenization trade-offs)
