# Landing surface — structural shape

Structure only: hierarchy, grid, alignment, density, order, whitespace, and
content relationships. **No visual direction, no decorative effects, no motion.**
Those come after this is confirmed.

Source of truth: `PRODUCT.md`. Visual authority is inherited from the shipped
console (`webui/static/index.html`), not reinvented — see §11.

---

## 1. Mode and job

**Mode: Persuade**, handing off to Operate.

A tool's landing page is still Persuade even though the tool itself is Operate.
The visitor's success is *deciding this is worth running* — then landing inside
the console with intent.

The surface serves two postures simultaneously:

| Visitor | Arrives believing | Must leave with |
|---|---|---|
| **Operator** | "another recon tool" | it runs locally, output pipes into my chain, one command |
| **Evaluator** | "AI-powered = marketing" | which model, how it works, numbers against the wordlist they use |

**Design consequence:** proof cannot live in a footnote, and the demo cannot be
a video. The evaluator's objection ("prove it isn't a wordlist with extra
steps") is the page's central problem to solve.

---

## 2. Information hierarchy

Strict priority. When space is contested, lower ranks yield.

```
1  The claim, stated as a measured result — not an adjective
2  Live demonstration — the real model, running, from the visitor's own input
3  Why a wordlist loses — the one idea
4  How it works — the six stages
5  Evidence — recall table + conditioning matrix
6  Limitation — stated plainly
7  Get started — one command
```

**Rank 1 and 2 are inseparable.** The headline makes a claim; the demo
substantiates it in the same viewport. A claim alone reads as marketing; a demo
alone reads as a toy.

**Rank 6 above rank 7 is deliberate.** Admitting the Common Crawl web bias
immediately before asking for install converts skepticism into trust. Burying it
would be both dishonest and less effective.

---

## 3. Section order and transitions

Transitions are *logical* connective tissue — what each section leaves the
visitor asking, which the next answers. Not visual effects.

| # | Section | Leaves the visitor asking | Answered by |
|---|---|---|---|
| 0 | Nav | — | persistent |
| 1 | Hero + live demo | "wait — it changed based on *my* input?" | §2 |
| 2 | The insight | "so how does it actually do that?" | §3 |
| 3 | How it works | "fine, but is it better than my wordlist?" | §4 |
| 4 | Evidence | "what's the catch?" | §5 |
| 5 | Limitation | "okay. how do I run it?" | §6 |
| 6 | Get started | — | terminal |
| 7 | Footer | — | — |

The chain is a single argument: **demonstrate → explain → prove → qualify →
enable.** No section is decorative; removing any one breaks the sequence.

---

## 4. Grid

```
Desktop ≥1200px    12 columns · 72px max gutter · content max-width 1200px
Tablet   768–1199   8 columns · 24px gutter · fluid, 32px page margin
Mobile     <768     4 columns · 16px gutter · 16px page margin
```

Console max-width is 1180px. Landing uses 1200px so the two feel like one
product without being pixel-identical.

**Alignment rule:** one dominant left edge runs the full page — headline, body,
section labels, and the demo input all share it. Centered text is used **only**
in the closing CTA, where the break signals the argument has ended.

**Whitespace rhythm** (vertical space *between* sections; extends the console's
4/8/12/16/24/32 scale):

```
--gap-section     desktop 128px   mobile 72px
--gap-block        desktop 64px   mobile 40px
--gap-element      desktop 24px   mobile 16px
```

More space above a heading than below it — the heading binds to its content,
not to the section above.

**Density:** landing runs at ~5/10; the console runs at 8/10. The landing must
breathe to persuade; the console must be dense to operate. The one exception is
the evidence table, which inherits console density (tabular, tight) because
that is what makes it read as *data* rather than as a marketing graphic.

---

## 5. Full-page composition — desktop

```
┌────────────────────────────────────────────────────────────────────────┐
│ NAV                                                              64px  │
│ ◆ SubFury               7.8M params · recall@200 0.260   [GitHub] [Console]│
├────────────────────────────────────────────────────────────────────────┤ ← sticky
│                                                                        │
│ 01 HERO + LIVE DEMO                                    grid: 6 / 6     │
│ ┌──────────────────────────┬─────────────────────────────────────────┐ │
│ │ cols 1–6                 │ cols 7–12                               │ │
│ │                          │                                         │ │
│ │ H1 — measured claim,     │ ┌─── LIVE DEMO ───────────────────────┐ │ │
│ │ two lines max            │ │ known subdomains  [api        ] [+] │ │ │
│ │                          │ │                   [dev        ]     │ │ │
│ │ Sub — one sentence:      │ │                   [staging    ]     │ │ │
│ │ what it does, plainly    │ │ ─────────────────────────────────── │ │ │
│ │                          │ │ predicted, ranked      ┌──────────┐ │ │ │
│ │ [Open console] [Install] │ │  1 app        -3.31 ▓▓▓│          │ │ │
│ │                          │ │  2 support    -3.52 ▓▓ │ swap the │ │ │
│ │ ┌─ proof strip ────────┐ │ │  3 docs       -3.64 ▓▓ │ inputs,  │ │ │
│ │ │ +482% recall@25      │ │ │  4 status     -3.76 ▓  │ list     │ │ │
│ │ │ vs the wordlist      │ │ │  5 cdn        -3.78 ▓  │ changes  │ │ │
│ │ └──────────────────────┘ │ │ ─────────────────────────────────── │ │ │
│ │                          │ │ model only · no DNS queries sent    │ │ │
│ │                          │ └─────────────────────────────────────┘ │ │
│ └──────────────────────────┴─────────────────────────────────────────┘ │
│                                                                        │
├────────────────────────────────────────────────────────── 128px ───────┤
│                                                                        │
│ 02 THE INSIGHT                                         grid: 5 / 7     │
│ ┌────────────────────┬─────────────────────────────────────────────┐   │
│ │ cols 1–5           │ cols 7–12                                   │   │
│ │ H2 — why a         │  wordlist  →  same 100 guesses, every target│   │
│ │ wordlist loses     │  SubFury   →  reads THIS org's naming       │   │
│ │                    │                                             │   │
│ │ 2 short paras:     │  ┌───────────────────────────────────────┐  │   │
│ │ orgs name things   │  │ known:  inventory                     │  │   │
│ │ consistently;      │  │ predicts: inventoryapp ← wordlist has │  │   │
│ │ a wordlist can't   │  │           no such entry               │  │   │
│ │ see it             │  └───────────────────────────────────────┘  │   │
│ └────────────────────┴─────────────────────────────────────────────┘   │
│                                                                        │
├────────────────────────────────────────────────────────── 128px ───────┤
│                                                                        │
│ 03 HOW IT WORKS                                    grid: full, 6 rows  │
│ H2 ── left aligned                                                     │
│                                                                        │
│ Six stages as a numbered vertical sequence — NOT a card row.           │
│ Each row: [n] │ name + one line │ its real artifact                    │
│                                                                        │
│  1 │ Known set     │ what you already found      │ api, dev, staging   │
│  2 │ Tokenize      │ subdomain BPE, not English  │ api [SEP] dev [DELIM]│
│  3 │ Beam search   │ N most probable, ranked     │ app -3.31 …         │
│  4 │ DNS resolve   │ concurrent validation       │ shop → 15.197.252.184│
│  5 │ Recurse       │ hits rejoin the known set   │ round 2: +learn     │
│  6 │ Export        │ list / JSON / CSV           │ pipes to httpx      │
│                                                                        │
│ Vertical, because the pipeline IS sequential. A 3×2 card grid would    │
│ misrepresent it as parallel — the same error the old console made.     │
│                                                                        │
├────────────────────────────────────────────────────────── 128px ───────┤
│                                                                        │
│ 04 EVIDENCE                                            grid: 7 / 5     │
│ ┌──────────────────────────────┬───────────────────────────────────┐   │
│ │ cols 1–7  RECALL TABLE       │ cols 9–12  CONDITIONING           │   │
│ │ (console density — tabular)  │                                   │   │
│ │ Budget  SubFury  Wordlist  Δ │  overlap of top-50 predictions    │   │
│ │ N=25     0.175    0.028 +525%│        dev  infra  ecom  monitor  │   │
│ │ N=50     0.210    0.060 +250%│  dev   1.00  0.35  0.59   0.18    │   │
│ │ N=100    0.220    0.097 +127%│  mon   0.18  0.43  0.20   1.00    │   │
│ │ N=200    0.221    0.142  +56%│                                   │   │
│ │                              │  dev seed and monitoring seed     │   │
│ │ 545 held-out apexes · equal  │  share only 18% — the input       │   │
│ │ budget · no DNS involved     │  steers the output                │   │
│ └──────────────────────────────┴───────────────────────────────────┘   │
│                                                                        │
│ Method line sits directly under the table, same left edge — the        │
│ caveat is part of the claim, not a footnote.                           │
│                                                                        │
├────────────────────────────────────────────────────────── 128px ───────┤
│                                                                        │
│ 05 LIMITATION                                      grid: cols 1–7      │
│ H2 — what it does not do well                                          │
│ Trained on the Common Crawl web graph, so internal infra names are     │
│ underrepresented; infra-flavored seeds drift generic. Narrow measure.  │
│ Deliberately narrow column — reads as a considered note, not a banner. │
│                                                                        │
├────────────────────────────────────────────────────────── 128px ───────┤
│                                                                        │
│ 06 GET STARTED                                  grid: cols 3–10, CENTER│
│                     H2 — run it locally                                │
│         ┌────────────────────────────────────────────────┐             │
│         │ $ pip install -r requirements.txt        [copy]│             │
│         │ $ python webui/app.py                          │             │
│         └────────────────────────────────────────────────┘             │
│                    [Open console]  [GitHub]                            │
│         MIT · runs on a laptop GPU or CPU · no accounts                │
│  ← the ONLY centered section: the break signals the argument has ended │
│                                                                        │
├────────────────────────────────────────────────────────────────────────┤
│ 07 FOOTER — repo · MODEL_SELECTION.md · license · authorized-use note  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Full-page composition — mobile (<768px)

Single column throughout. Every two-column pair collapses **content-first**:
the explanation precedes its artifact, never the reverse.

```
┌──────────────────────────┐
│ NAV  ◆ SubFury  [Console]│ 56px · sticky · spec strip drops to a
├──────────────────────────┤        single line: "7.8M params"
│ 01 HERO                  │
│ H1 (claim)               │  Order preserved: claim → demo → proof.
│ Sub                      │  Demo stays above the fold-ish: the
│ [Open console] full-width│  input sits within the first ~1.5
│ [Install]     full-width │  screens. It is the page's whole
│ ┌──────────────────────┐ │  argument — it does not get demoted
│ │ LIVE DEMO            │ │  below three sections of prose.
│ │ known: [api      ]   │ │
│ │        [dev      ]   │ │  Prediction list capped at 5 rows on
│ │ ──────────────────── │ │  mobile (8 on desktop) — enough to
│ │ 1 app      -3.31 ▓▓▓ │ │  show ranking, short enough to keep
│ │ 2 support  -3.52 ▓▓  │ │  the next section reachable.
│ │ 3 docs     -3.64 ▓▓  │ │
│ │ 4 status   -3.76 ▓   │ │
│ │ 5 cdn      -3.78 ▓   │ │
│ │ no DNS sent          │ │
│ └──────────────────────┘ │
│ ┌──────────────────────┐ │
│ │ +482% recall@25      │ │
│ └──────────────────────┘ │
├─────────────── 72px ─────┤
│ 02 INSIGHT               │  text block, then the contrast example
├─────────────── 72px ─────┤
│ 03 HOW IT WORKS          │  six rows stack; artifact column moves
│  1 Known set             │  BELOW each stage's name + line, indented
│    what you already found│  to the number's left edge
│    api, dev, staging     │
│  2 Tokenize …            │
├─────────────── 72px ─────┤
│ 04 EVIDENCE              │  recall table: 4 columns fit at 375px if
│  [recall table]          │  Δ column is dropped to a badge in-row.
│  [conditioning]          │  conditioning matrix → 2-row comparison
│                          │  (dev vs monitor only); full 4×4 needs
│                          │  horizontal scroll — CONTAINED in its own
│                          │  overflow-x wrapper, page never scrolls
├─────────────── 72px ─────┤
│ 05 LIMITATION            │
├─────────────── 72px ─────┤
│ 06 GET STARTED           │  command block scrolls horizontally inside
│  [command block]         │  itself; copy button always visible
│  [Open console] full-w   │
├──────────────────────────┤
│ 07 FOOTER                │
└──────────────────────────┘
```

**Mobile-specific rules**
- Touch targets ≥44×44px (landing) — stricter than the console's 34–36px
  secondary controls, because this surface is thumb-first.
- Tables and command blocks scroll inside their own `overflow-x` container.
  The page body never scrolls horizontally — the exact defect found and fixed
  in the console audit.
- Sticky nav collapses to wordmark + single primary CTA.

---

## 7. Product demonstration — the load-bearing decision

The hero demo **is the real model**, not a recording, not a screenshot.

```
input        3 editable known-subdomain fields, prefilled: api / dev / staging
action       debounced on edit — no "Run" button, no gate
output       top 5–8 predicted labels, ranked, with log-prob bars
endpoint     POST /api/predict  { resolve: false }   ← model only
```

**`resolve: false` is mandatory here.** A landing page that fires DNS at
arbitrary domains from anonymous input is an abuse vector. The hero demonstrates
*prediction*; validation is the console's job, behind an explicit authorization
notice. This constraint is not negotiable and shapes the section: there is no
domain field in the hero at all — only labels.

**Why it must be interactive:** the evaluator's core objection is "this is a
wordlist with extra steps." Static output cannot refute that. Letting them swap
`api,dev,staging` for `vpn,citrix,owa` and watch the list change is the single
most persuasive event on the page — it refutes the objection using their own
input, in about four seconds.

**Preset chips** below the input (`dev` · `infra` · `ecommerce` · `monitoring`)
lower the cost of that experiment to one tap. Without them most visitors never
edit anything and the demo degrades to a static list.

**Failure states**, all designed, none decorative:
- model still loading → the three inputs render disabled with a "warming up" line
- backend absent (page opened as a static file) → demo swaps to captured example
  output, labeled *recorded* — never silently faked
- no valid labels entered → prediction list holds its last result, input marked

---

## 8. CTA placement

Primary CTA is **Open console** — one job, repeated at exactly three depths.

| Position | CTA | Rationale |
|---|---|---|
| Nav (sticky) | `Open console` | always one click away once convinced |
| Hero | `Open console` + `Install` | the two postures, side by side |
| After Evidence | `Open console` | peak conviction — directly after proof |
| Get started | `Open console` + command block | terminal action |

Four placements, one verb. No competing CTAs (no "learn more", no newsletter,
no star-us interstitial). `Install` and `GitHub` are persistently secondary in
weight — never adjacent equals.

**Open decision affecting this** (`PRODUCT.md` §Open decision): if the page is
primarily read on GitHub rather than served locally, primary and secondary swap
— `Install` leads, `Open console` becomes the follow-on. The structure holds
either way; only emphasis changes.

---

## 9. Proof placement

Proof is distributed, not pooled in one section:

```
Nav          7.8M params · recall@100 0.220        ambient credibility
Hero         +482% recall@25 vs the wordlist       the claim's receipt
Demo         live ranked output                    proof by demonstration
How it works real artifacts per stage              proof of mechanism
Evidence     full table + conditioning matrix      the formal case
Limitation   what it does badly                    proof of honesty
```

**Rule: never state a number without its method.** Every recall figure is
adjacent to "545 held-out apexes · equal budget · no DNS." A number without
its conditions is a marketing claim; with them it is a result. This is the
difference between the page being believed and being discounted.

---

## 10. Content/product relationships

- **Nav spec strip ↔ console identity bar** — same facts, same order, so
  crossing from landing to console feels continuous, not like a second product.
- **Hero demo ↔ console stage 3** — literally the same endpoint and the same
  ranked-list structure. The landing shows one stage of the machine; the console
  shows all six. The demo is a *component*, not a mock.
- **How-it-works six stages ↔ console pipeline rail** — identical names and
  order. The landing teaches the vocabulary the console then uses.
- **Evidence table ↔ Engineering drawer table** — same numbers, same source
  (`/api/model`). One source of truth; no drift possible.

This is the structural payoff: the landing page is not a brochure *about* the
tool, it is a thin, honest slice *of* the tool.

---

## 11. Inherited visual authority

No DESIGN.md exists. Rather than invent a second visual world, the landing
inherits the console's shipped, audited system:

```
surfaces   --bg #0b1120 · --surface #0f172a · --raised #151f33 · --sunk #080d18
text       --fg #f8fafc · --fg-2 #a5b3c8 · --fg-3 #7b8ba3   (all ≥4.5:1, verified)
signal     --accent #22c55e · --info #5ea9ff · --warn #f5b545 · --err #f4626b
type       IBM Plex Sans (chrome) · JetBrains Mono (data, labels, commands)
```

Divergences the landing is permitted, and only these:
- lower density (5/10 vs 8/10) and the larger 48/64/96/128 spacing steps
- display type scale above the console's ceiling for H1/H2 only
- 44px touch targets rather than 34–36px

Everything else — color, tokens, table treatment, focus rings, scrollbars,
selection — is reused verbatim. `/impeccable document` should generate the real
DESIGN.md from the shipped console before implementation begins, so this is
recorded rather than described.

---

## 12. Open decisions — do not invent these

1. **Primary audience** (§8) — locally-served front door, or GitHub-read
   marketing page? Determines CTA emphasis. *Assumption: locally served.*
2. **Route** — landing at `/` with console moved to `/console`, or landing at
   `/about` leaving `/` as the console? *Assumption: landing at `/`.*
3. **Demo preset set** — are `dev / infra / ecommerce / monitoring` the four
   that best show conditioning, or should one be replaced by a real bug-bounty
   target shape?

---

## 13. Explicitly out of scope here

Color application, typography scale values, motion, imagery, iconography,
copywriting, and any decorative treatment. This document ends at structure.
