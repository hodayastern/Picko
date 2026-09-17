# PICKO — a tiny scientific tool-picker

PICKO fine-tunes **Needle** (Cactus's 26M-parameter JAX/Flax encoder–decoder tool-picker, a *Simple
Attention Network*) to select the right **scientific** tool and extract its arguments from a single
natural-language prompt. The research question: *how far can a tiny model go as a scientific tool-picker?*

- **Authors:** Hadar Bitan & Hodaya Stern Zuckerman — Tom Hope's *LLMs for Science* course.
- **Stack:** JAX/Flax (not PyTorch). Training/inference go through the upstream `needle` CLI + library.
- **Model I/O:** fine-tune consumes **JSONL** of `{"query", "tools", "answers"}`, where `tools` and
  `answers` are JSON-encoded strings. `needle finetune` auto-downloads base weights from
  `Cactus-Compute/needle` and reports base-vs-fine-tuned metrics.

> **Needle is not vendored here.** It is upstream Cactus code, pulled as a dependency at a pinned commit
> (see [Setup](#setup)). This repository is only the PICKO layer — data, catalogs, scripts, notebooks.

---

## The three research dimensions

| Notebook | Dimension | Question | Metric |
|---|---|---|---|
| `nb1_breadth_amount` | **Breadth** | How many tools can it choose among before it picks the wrong one? | tool-selection accuracy vs *k* offered tools |
| `nb2_depth_by_example` | **Depth** | Given the right tool, does argument extraction get harder as a call has more arguments? | `args_exact` / `param_f1` bucketed by **#arguments per example** |
| `nb3_separation_ambiguous` | **Separation** | Can it tell curated look-alike tools apart (same action across sources, same source across actions)? | per-group selection accuracy + confusion |
| `nb3_semantic_separation` | **Semantic separation** | Can it *learn* to route between look-alike tools by **domain meaning alone**, when the query never names the source? | before-vs-after selection accuracy on a held-out test |

The two `nb3` notebooks are complementary: **`nb3_separation_ambiguous`** probes disambiguation on the
existing name-containing data (queries often name the source), while **`nb3_semantic_separation`** removes
the source name and asks whether the distinction is *learnable* from domain meaning alone.

A recurring finding: the base model usually names a *plausible* tool but botches the **arguments** —
fine-tuning's main win is parameter extraction, not selection. `nb2` studies exactly that frontier, and
shows `param_f1` staying roughly flat with argument count while exact-match decays multiplicatively
(each argument is an independent chance to slip).

**Encoder limit.** Needle packs `[query, <tools>, schemas]` into a **1024-token** encoder and truncates
beyond it. Full schemas are large (~112 tokens median, up to ~271), so only ~8 fit; the notebooks use a
token-aware trim and, for Breadth, compact (name+description) schemas so more tool names fit. That
ceiling — not model capacity — is what caps Breadth, and the notebooks annotate it with `n_visible`.

---

## Repository layout

```
README.md                              this file
README_needle.md                       upstream Needle README (attribution)
LICENSE                                upstream MIT license
requirements.txt                       PICKO deps + how to install Needle
full_tools_53tools_11products.json     master tool catalog (75 tools, 11 families)
11tools_11products.json                one representative tool per family
tools_metadata.csv                     family -> category + argument counts
data/
  scientific_pools.json                the 40 focus tools by family — data-generation input
  semantic/cs.json                     nb3 source-free CS queries (-> arxiv)
  semantic/medicine.json               nb3 source-free medical queries (-> pubmed)
  semantic/probe.jsonl                 nb3 prebuilt fallback dataset
scripts/
  tool_catalog.py                      Catalog: load/slice tools, restrict_dataset()
  research_sets.py                     focus families, buckets, look-alike groups
  picko_eval.py                        evaluator: selection vs argument metrics, confusion
  picko_research.py                    single import facade the notebooks use
  generate_picko_data.py               Gemini data synthesis
  build_research_nbs.py                regenerates the 3 research notebooks
notebooks/
  01_generate_data.ipynb               build the training data with Gemini
  research/
    nb1_breadth_amount.ipynb
    nb2_depth_by_example.ipynb
    nb3_separation_ambiguous.ipynb     Separation (curated look-alike groups)
    nb3_semantic_separation.ipynb      Semantic separation (learn to route by domain)
results/                               figures for the write-up (one per notebook)
```

The 26 MB training pool `picko_training_pool.jsonl` is **not** committed — see [Data](#data).

---

## Setup

PICKO imports the upstream Needle package. Install it from the pinned commit that this project was
developed against (it contains the per-tool split the notebooks rely on):

```bash
git clone https://github.com/cactus-compute/needle.git
cd needle && git checkout 34861f39ae292429f80a62c96abe83218a852d57 && cd ..
pip install -e ./needle
pip install "jax[cuda12]==0.10.2" "jaxlib==0.10.2" "flax==0.12.8"   # GPU; drop [cuda12] for CPU
```

Then run the notebooks with this repo on the path (they add the repo root automatically).

**On Colab, none of this is manual:** cell 0 of each research notebook clones Needle at the pinned SHA,
installs it, clones this repo, pins JAX/Flax, mounts Drive, and verifies the Needle API — failing loudly
if upstream ever drifts. Set the two repo variables at the top of that cell to your fork if needed.

---

## Data

All experiments are scoped from one balanced pool, **`picko_training_pool.jsonl`** (75 tools × 120 positives +
120 negatives ≈ 9,120 lines, ~26 MB), built by `notebooks/01_generate_data.ipynb` (Gemini synthesis →
per-tool coverage → balance). It is git-ignored, so upload it once to Google Drive:

```
MyDrive/picko/
  picko_training_pool.jsonl        <- upload this (the only manual step)
  picko_out/                  <- auto-created: checkpoints, *_results.json, run.log, figures
    nb1/ nb2/ nb3_separation/ nb3/
```

Because `picko_out/` lives on Drive, a Colab restart **resumes** and skips finished work. The two
source-free query files for `nb3` (`data/semantic/cs.json`, `data/semantic/medicine.json`) are committed,
so they arrive with the clone — no upload needed.

---

## Running on Colab

1. Runtime → Change runtime type → **GPU** (L4 recommended; T4/A100 fine).
2. Open a research notebook, run **cell 0** (bootstrap), then top-to-bottom.
3. **Order:** run `nb1`, `nb2`, or `nb3_separation_ambiguous` before `nb3_semantic_separation` — the
   semantic notebook's baseline reuses a 40-focus checkpoint any of them writes to Drive (it logs
   "no baseline — skipping" if none is present).

Each notebook writes its checkpoint, `*_results.json`, and figures into its own `picko_out/` subfolder.

## Regenerating the notebooks

The three research notebooks are generated, not hand-edited:

```bash
python scripts/build_research_nbs.py
```

Edit `scripts/build_research_nbs.py` and re-run to change any of them.

---

## Attribution

Needle is © Cactus Compute, released under the MIT license (see `LICENSE` and `README_needle.md`). PICKO
uses it unmodified as a dependency. Weights: `Cactus-Compute/needle` on Hugging Face.
