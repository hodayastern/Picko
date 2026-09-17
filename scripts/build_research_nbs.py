#!/usr/bin/env python3
"""Build the 3 PICKO research notebooks (Breadth / Depth / Separation) in
notebooks/research/ with shared, restart-safe init cells.

Run:  python scripts/build_research_nbs.py   (regenerates all three notebooks)

Design goals baked into the generated cells:
  * Colab-aligned to Google Drive `MyDrive/picko/` — data in, checkpoints+results out
    (so a runtime restart loses nothing), tuned for an L4 GPU.
  * Run-and-forget: every sweep is resumable (finished sizes/iterations/groups are
    skipped from Drive) and fault-tolerant (one failure is logged and the loop goes on).
  * Observability: timestamped `log()` lines to screen AND a durable Drive run.log,
    per-step timing, checkpoint-write confirmation, and a one-glance env/GPU header.
"""
import json
import os

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "notebooks", "research")
os.makedirs(OUTDIR, exist_ok=True)


def md(s): return {"cell_type": "markdown", "metadata": {}, "id": None, "source": s}
def co(s): return {"cell_type": "code", "metadata": {}, "id": None, "execution_count": None, "outputs": [], "source": s}


# ---------- shared init cells (identical across the 3 notebooks) ----------
BOOTSTRAP_MD = md("""## 0 · Colab quick-start (GPU) — run & forget, restart-safe

**On Colab first: Runtime → Change runtime type → GPU (L4 recommended; T4/A100 also fine).**
This cell clones **upstream Needle** (Cactus, pinned commit) and installs it, clones this **PICKO** repo,
pins the exact JAX/Flax, mounts Drive, and points **both** the data (in) and the checkpoints+results (out)
at your **`MyDrive/picko/`** folder — so a runtime restart loses nothing.

**Prerequisite (one-time):** `picko_training_pool.jsonl` must be in `MyDrive/picko/`. **Running locally?**
This cell is a no-op — install Needle yourself (`pip install -e /path/to/needle`) and skip to cell 1.""")

BOOTSTRAP = co('''# --- Colab bootstrap (safe to re-run; no-op locally) ---
import os, sys
IN_COLAB = "google.colab" in sys.modules
# Upstream Needle (Cactus) — de-vendored: cloned + installed at a pinned commit.
NEEDLE_REPO = "https://github.com/cactus-compute/needle.git"
NEEDLE_SHA  = "34861f39ae292429f80a62c96abe83218a852d57"   # pinned; has _per_tool_split — update if upstream drifts
# This PICKO repo — the scripts/notebooks/data imported below.
PICKO_REPO   = "https://github.com/HadarBit/picko.git"     # <- set to your submission repo URL
PICKO_BRANCH = "main"
if IN_COLAB:
    if not os.path.exists("/content/needle"):
        !git clone -q {NEEDLE_REPO} /content/needle && cd /content/needle && git checkout -q {NEEDLE_SHA}
    if not os.path.exists("/content/picko"):
        !git clone -q -b {PICKO_BRANCH} {PICKO_REPO} /content/picko
    %pip install -q "jax[cuda12]==0.10.2" "jaxlib==0.10.2" "flax==0.12.8"
    %pip install -q -e /content/needle                          # install upstream Needle
    sys.path.insert(0, "/content/picko")
    from google.colab import drive; drive.mount("/content/drive")
    import shutil
    DRIVE = "/content/drive/MyDrive/picko"                      # <- everything lives here
    os.environ["PICKO_OUT_DIR"] = f"{DRIVE}/picko_out"          # checkpoints + results (durable)
    os.environ["PICKO_LOG"]     = f"{DRIVE}/picko_out/run.log"  # durable log across restarts
    os.makedirs(os.environ["PICKO_OUT_DIR"], exist_ok=True)
    dst = "/content/picko/data/picko_training_pool.jsonl"
    if not os.path.exists(dst):
        cands = [f"{DRIVE}/picko_training_pool.jsonl", "/content/drive/MyDrive/picko_training_pool.jsonl"]
        src = next((c for c in cands if os.path.exists(c)), None)
        if src is None:
            have = os.listdir(DRIVE) if os.path.isdir(DRIVE) else "(MyDrive/picko not found)"
            raise FileNotFoundError(
                "picko_training_pool.jsonl not found. Upload it to MyDrive/picko/. "
                f"Currently in {DRIVE}: {have}")
        os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy(src, dst)
        print("copied data from", src)
    try:                                                        # guard: upstream must expose the split PICKO uses
        from needle.training.finetune import _per_tool_split    # noqa: F401
    except Exception as e:
        raise ImportError(f"Upstream Needle @ {NEEDLE_SHA[:7]} lacks _per_tool_split ({e}). "
                          "Pin NEEDLE_SHA to a commit that has it, or re-run the install.")
    import jax
    print("GPU:");
    !nvidia-smi -L
    print("jax devices:", jax.devices())
    _plat = jax.devices()[0].platform
    assert _plat == "gpu", (
        f"JAX is running on '{_plat}', NOT the GPU — every finetune/eval will be ~30x slower "
        "(hours instead of minutes). FIX: Runtime > Change runtime type > GPU (L4), then "
        "Runtime > Restart session, and re-run this cell. If a GPU IS selected but this still "
        "fails, the CUDA plugin didn't load — re-run the %pip lines above, then restart.")
    print(f"bootstrap OK · GPU active · needle@{NEEDLE_SHA[:7]} · data =", dst, "· OUT_DIR =", os.environ["PICKO_OUT_DIR"])
else:
    print("Not on Colab — running locally. Install upstream Needle first: pip install -e /path/to/needle")''')

SETUP_MD = md("## 1 · Setup & data overview")

SETUP = co('''# ensure the repo root is importable (works from notebooks/research/, Colab, etc.)
import os, sys
_here = os.path.abspath(os.getcwd())
for _ in range(6):
    if os.path.exists(os.path.join(_here, "scripts", "picko_research.py")): break
    _here = os.path.dirname(_here)
if os.path.isdir("/content/picko"): _here = "/content/picko"
if _here not in sys.path: sys.path.insert(0, _here)

from scripts.picko_research import *
import json, time
import pandas as pd, numpy as np, matplotlib.pyplot as plt
try:
    import seaborn as sns; sns.set_theme(style="whitegrid")
except Exception:
    sns = None
from tqdm.auto import tqdm

cat, tok, raw, FOCUS, OUT_DIR = load_context()
env_report(OUT_DIR)   # jax devices + is OUT_DIR durable (Drive)?''')

DF1_MD = md("### The 40 focus tools\\nOne row per tool, with its family, category and **parameter count / bucket**.")
DF1 = co('display(tools_dataframe(cat, FOCUS))')

DF2_MD = md("### All examples for these 40 tools\\nOne row per training example (query → gold tool), tagged with the gold tool's **param bucket**.")
DF2 = co('''ex_df = examples_dataframe(cat, raw, FOCUS)
print("examples:", ex_df.shape[0], "| per param bucket:", ex_df["param_bucket"].value_counts().to_dict())
display(ex_df.head(10))''')


def init_cells(title_md):
    return [md(title_md), BOOTSTRAP_MD, BOOTSTRAP, SETUP_MD, SETUP, DF1_MD, DF1, DF2_MD, DF2]


# ======================================================================
# NB1 — Breadth (amount)
# ======================================================================
nb1 = init_cells("""# PICKO Research · NB1 — **Breadth**: how many tools before selection breaks?

We finetune **one** model on the 40 focus tools and then, on its **held-out test set**, offer each query a
random subset of `k` tools (its gold tool + `k-1` distractors) and measure tool-selection accuracy. Training
is fixed, so the curve isolates a single variable — *how many tools are offered at inference* — averaged
over `N_REPEATS` random subsets per `k` (mean ± std). The 1024-token encoder truncates long compact lists,
so past ~20 tools some are never seen; that ceiling is the result, annotated with `n_visible`.""")
nb1 += [
 md("""## 2 · Train / test split

Both the training subprocess and this notebook call the **same deterministic** `per_tool_split`
(`seed=42`, 10 test + 10 val per tool). The model trains **only on the train split**; the sweep below runs
**only on the held-out test split**, so no test query is ever seen in training."""),
 md("## 3 · Configure"),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb1"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
BREADTH_SIZES  = [3, 5, 10, 20, 30, 40]   # tools offered per query at inference
N_REPEATS      = 8                        # random subsets averaged per size (mean +/- std)
CAP_PER_TOOL   = 120                       # examples/tool -> 100 train / 10 val / 10 test
EPOCHS         = 1
EVAL_SUBSAMPLE = 100                       # test queries per (size, repeat), sampled across all tools; None = full
MAX_GEN_LEN    = 64
BATCH_SIZE     = 8                         # lower to 4 on OOM, raise to 16 if headroom
RUN_TRAIN      = True
FORCE_RETRAIN  = False
print("focus tools:", len(FOCUS), "| sizes:", BREADTH_SIZES, "| repeats/size:", N_REPEATS, "| out:", NB_DIR)'''),
 md("## 4 · Train the model (40 focus tools, compact)\\nTrained once to Drive and reused; the returned `test` set is the held-out split used by the sweep."),
 co('''FOCUS40 = finetune_and_eval(cat, raw, tok, FOCUS, "breadth_focus40", NB_DIR,
                            cap=CAP_PER_TOOL, epochs=EPOCHS, compact=True, offer_all=len(FOCUS),
                            eval_subsample=None, run_train=RUN_TRAIN, force_retrain=FORCE_RETRAIN,
                            max_gen_len=MAX_GEN_LEN, batch_size=BATCH_SIZE)
m40, p40, tk40 = FOCUS40["bundle"]
TEST = FOCUS40["test"]                     # held-out test queries (never trained on)
log(f"model ready · held-out test queries={len(TEST)}")'''),
 md("## 5 · Sweep: offer k tools per query, repeat, average\\nInference only, on the held-out test set. Resumable: finished `(k, repeat)` pairs persist to `nb1/breadth_results.json`."),
 co('''import contextlib, io
RES = os.path.join(NB_DIR, "breadth_results.json")
rows = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else []
done = {(r["k"], r["repeat"]) for r in rows}
if done: log(f"resumed {len(done)} finished (k,repeat) run(s)")

def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

t_all = time.time()
for k in BREADTH_SIZES:
    for rep in range(N_REPEATS):
        if (k, rep) in done and not FORCE_RETRAIN: continue
        try:
            test = TEST[:EVAL_SUBSAMPLE] if EVAL_SUBSAMPLE else TEST
            offered = [offer_subset(cat, gold_of(e), FOCUS, k, seed=k*100000+rep*1000+i)
                       for i, e in enumerate(test)]
            with contextlib.redirect_stdout(io.StringIO()):
                preds = predict(m40, p40, tk40, test, tools_override=offered, max_gen_len=MAX_GEN_LEN)
            met = evaluate(test, preds, family_of=family_of)
            vis = int(np.median([n_visible(e["query"], json.loads(o), tok) for e, o in zip(test, offered)]))
            rows = [r for r in rows if not (r["k"] == k and r["repeat"] == rep)] + [{
                "k": k, "repeat": rep, "selection_acc": met["selection_acc"],
                "name_f1": met["name_f1"], "parse_rate": met["parse_rate"],
                "n_visible": vis, "n_test": met["n"]}]
            done.add((k, rep)); json.dump(rows, open(RES, "w"), indent=2)
            log(f"k={k} rep={rep}: selection={met['selection_acc']:.3f} visible={vis}/{k}")
        except Exception as e:
            log(f"k={k} rep={rep}: FAILED ({type(e).__name__}: {e})")

log(f"ALL RUNS DONE in {time.time()-t_all:.0f}s · results={RES}")
runs = pd.DataFrame(rows)
breadth = (runs.groupby("k")
           .agg(selection_mean=("selection_acc", "mean"), selection_std=("selection_acc", "std"),
                name_f1_mean=("name_f1", "mean"), parse_rate=("parse_rate", "mean"),
                n_visible=("n_visible", "median"), n_repeats=("repeat", "nunique"),
                n_test=("n_test", "sum")).reset_index())
breadth["selection_std"] = breadth["selection_std"].fillna(0)
display(breadth.round(3))'''),
 md("## 6 · The Breadth curve"),
 co('''fig, ax = plt.subplots(figsize=(8,4.5))
ax.errorbar(breadth["k"], breadth["selection_mean"], yerr=breadth["selection_std"],
            fmt="o-", color="#4C72B0", capsize=4, label="selection_acc (mean +/- std)")
ax.scatter(runs["k"], runs["selection_acc"], s=12, color="#4C72B0", alpha=0.25, zorder=1)
wall = breadth[breadth["n_visible"] < breadth["k"]]
if len(wall):
    kw = int(wall["k"].iloc[0]); vw = int(wall["n_visible"].iloc[0])
    ax.axvline(kw, color="#C44E52", ls=":", lw=1.5)
    ax.text(kw, 0.06, f" truncation wall\\n (~{vw} of {kw} tools visible)", color="#C44E52", fontsize=9, va="bottom")
ax.set_xlabel("# tools offered at inference (k)"); ax.set_ylabel("tool-selection accuracy")
ax.set_ylim(0,1.02); ax.set_title(f"Breadth: selection vs #tools offered ({int(breadth['n_repeats'].max())} subsets/size)")
ax.legend()
plt.tight_layout(); save_fig("breadth_curve", out_dir=NB_DIR); plt.show()'''),
 md("""## 7 · Read-out

One model, probed on held-out queries with random subsets of `k` tools, so the curve reflects the *offered*
tool count alone — the error bars show the spread a single sample per size would hide. Selection stays high
for small `k` and falls as `k` grows past the red line, where the compact offered list overflows the
1024-token encoder and the extra tools are truncated away. **Takeaway:** one PICKO instance is bounded by
the context window it can offer, not by what it was trained on — beyond the wall, shard the tool set."""),
]

# ======================================================================
# NB2 — Depth (parameters), repeated stratified sampling
# ======================================================================
nb2 = init_cells("""# PICKO Research · NB2 — **Depth**: is parameter extraction harder with more parameters?

40 full-schema tools can't all be offered at once (token limit), and a single model gives one
unreplicated score per tool. Instead we **repeatedly sample a small, bucket-balanced set** (tools from
every param-count bucket, sized to fit the encoder), finetune, and measure argument extraction — over
several iterations — so each bucket gets many measurements and we can show **error bars**.

*Run & forget:* each iteration trains once to Drive; a restart **skips finished iterations** and keeps
the collected per-tool rows in `picko_out/depth_results.json`.""")
nb2 += [
 md("## 2 · Configure the repeated sampling\\nEach iteration draws `TOOLS_PER_BUCKET` tools from **each** bucket (0 / 1 / 2-3 / 4+) into one small model."),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb2"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
N_ITER          = 5     # <- number of independent (tool-sample + finetune) iterations
TOOLS_PER_BUCKET = 2     # tools drawn from each param bucket per iteration
CAP_PER_TOOL     = 120   # examples/tool -> 100 train / 10 val / 10 test
EPOCHS           = 1
EVAL_SUBSAMPLE   = 40
BATCH_SIZE       = 8     # lower to 4 on OOM, raise to 16 if headroom
RUN_TRAIN        = True
FORCE_RETRAIN    = False
print("param buckets available:", tools_dataframe(cat, FOCUS)["param_bucket"].value_counts().to_dict(), "| out:", NB_DIR)'''),
 md("## 3 · Run the iterations\\n*Resumable:* finished iterations are skipped; per-tool rows persist to `nb2/depth_results.json` after each iteration."),
 co('''RES = os.path.join(NB_DIR, "depth_results.json")
rows = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else []
done_iters = {r["iteration"] for r in rows}
if done_iters: log(f"loaded {len(done_iters)} finished iteration(s) from {RES}")

t_all = time.time()
for i in range(N_ITER):
    ckpt = os.path.join(NB_DIR, f"picko_depth_iter{i}_best.pkl")
    if (i in done_iters) and os.path.exists(ckpt) and not FORCE_RETRAIN:
        log(f"iter {i}: skip (already done)"); continue
    try:
        names = sample_stratified(cat, FOCUS, TOOLS_PER_BUCKET, seed=i)
        log(f"=== start iter {i} · tools={names} ===")
        R = finetune_and_eval(cat, raw, tok, names, f"depth_iter{i}", NB_DIR,
                              cap=CAP_PER_TOOL, epochs=EPOCHS, compact=False, token_aware=True,
                              eval_subsample=EVAL_SUBSAMPLE, run_train=RUN_TRAIN,
                              force_retrain=FORCE_RETRAIN, batch_size=BATCH_SIZE)
        new = []
        for tool, s in R["metrics"]["per_tool"].items():
            _, tot = cat.params_of(tool)
            new.append({"iteration": i, "tool": tool, "total_params": tot,
                        "param_bucket": param_bucket(tot), "n": s["n"],
                        "selection_acc": s["selection_acc"],
                        "args_exact_acc": s["args_exact_acc"], "param_f1": s["param_f1"]})
        rows = [r for r in rows if r["iteration"] != i] + new
        done_iters.add(i)
        json.dump(rows, open(RES, "w"), indent=2)   # persist each iteration
        log(f"=== done iter {i}: {len(new)} tools measured ===")
    except Exception as e:
        log(f"iter {i}: FAILED ({type(e).__name__}: {e}) — skipping; re-run to resume")

log(f"ALL ITERATIONS DONE in {time.time()-t_all:.0f}s · results={RES}")
depth = pd.DataFrame(rows)
print("collected", len(depth), "per-tool measurements across", depth["iteration"].nunique(), "iterations")
display(depth.head(12))'''),
 md("## 4 · Extraction accuracy per parameter bucket (mean ± std)"),
 co('''# per-iteration bucket means first (paired within iteration), then mean/std across iterations
per_iter = (depth.groupby(["iteration","param_bucket"])[["args_exact_acc","param_f1"]]
            .mean().reset_index())
agg = (per_iter.groupby("param_bucket")
       .agg(args_mean=("args_exact_acc","mean"), args_std=("args_exact_acc","std"),
            pf1_mean=("param_f1","mean"), pf1_std=("param_f1","std"),
            n_iter=("iteration","nunique"))
       .reindex([b for b in PARAM_BUCKET_ORDER if b in per_iter["param_bucket"].values]))
display(agg.round(3))

x = np.arange(len(agg)); w = 0.38
fig, ax = plt.subplots(figsize=(8,4.5))
ax.bar(x-w/2, agg["args_mean"], w, yerr=agg["args_std"].fillna(0), capsize=4, color="#4C72B0", label="args_exact_acc")
ax.bar(x+w/2, agg["pf1_mean"], w, yerr=agg["pf1_std"].fillna(0), capsize=4, color="#DD8452", label="param_f1")
# overlay each iteration's bucket mean as points
for _, r in per_iter.iterrows():
    xi = list(agg.index).index(r["param_bucket"]) if r["param_bucket"] in list(agg.index) else None
    if xi is not None: ax.scatter(xi-w/2, r["args_exact_acc"], color="#243b57", s=14, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(agg.index); ax.set_ylim(0,1)
ax.set_xlabel("# parameters (bucket)"); ax.set_ylabel("accuracy")
ax.set_title(f"Depth: parameter extraction vs #params ({int(agg['n_iter'].max())} iterations)"); ax.legend()
plt.tight_layout(); save_fig("depth_buckets", out_dir=NB_DIR); plt.show()'''),
 md("## 5 · Per-tool scatter (all iterations)"),
 co('''tool_mean = depth.groupby(["tool","total_params"])["args_exact_acc"].mean().reset_index()
plt.figure(figsize=(7.5,4.5))
plt.scatter(tool_mean["total_params"], tool_mean["args_exact_acc"], s=55, color="#4C72B0")
for _, r in tool_mean.iterrows():
    plt.annotate(r["tool"].split("_")[0], (r["total_params"], r["args_exact_acc"]), fontsize=7)
plt.xlabel("# parameters in tool"); plt.ylabel("mean args_exact_acc")
plt.title("Depth: per-tool extraction vs parameter count")
plt.tight_layout(); save_fig("depth_scatter", out_dir=NB_DIR); plt.show()'''),
 md("""## 6 · Read-out

Argument extraction is near-solved for **0–1 parameter** tools and **degrades for multi-parameter (4+)**
tools — the error bars show it's a consistent effect across independent tool samples, not one unlucky
model. This is where a small specialist model needs the most help (and where finetuning gains most)."""),
]

# ======================================================================
# NB3 — Separation (ambiguous)
# ======================================================================
nb3 = init_cells("""# PICKO Research · NB3 — **Separation**: can it tell look-alike tools apart?

We finetune one model on the 40 focus tools and probe curated groups of near-identical tools (same action
across sources, or same source across actions). Each group's queries are drawn from the model's **held-out
test set** and offered the full group in one shot, so we measure pure disambiguation + which tool it
confuses for which.""")
nb3 += [
 md("""## 2 · Train / test split

Both the training subprocess and this notebook call the **same deterministic** `per_tool_split`
(`seed=42`, 10 test + 10 val per tool). The model trains **only on the train split**; the group probes
below run **only on the held-out test split**, so no test query is ever seen in training."""),
 md("""## 3 · The ambiguous groups

Each group is a set of look-alike tools on one of two axes: **cross-source** (same action, different
source — the source word disambiguates) or **within-source** (same source, subtly different action)."""),
 co('''AXIS = {"cross_source_search": "cross-source", "single_item_summary": "cross-source",
        "hf_search_variants": "within-source", "wikipedia_retrieve_vs_summarize": "within-source",
        "get_paper_content": "within-source", "arxiv_latex": "within-source"}
grp_rows = []
for g, tools in SIMILAR_GROUPS.items():
    fams = sorted({family_of(t) for t in tools})
    grp_rows.append({"group": g, "axis": AXIS.get(g, ""), "n_tools": len(tools),
                     "families": ",".join(fams), "tools": ", ".join(tools)})
groups_df = pd.DataFrame(grp_rows).sort_values("axis")
display(groups_df)'''),
 md("### Sample queries per tool\\nThe actual requests the model must tell apart — one example query per tool, grouped."),
 co('''def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

sample_q = {}
for e in raw:
    g = gold_of(e)
    if g and g not in sample_q: sample_q[g] = e["query"]

ex_rows = [{"group": g, "axis": AXIS.get(g, ""), "tool": t, "example_query": sample_q.get(t, "")[:160]}
           for g, tools in SIMILAR_GROUPS.items() for t in tools]
display(pd.DataFrame(ex_rows))'''),
 md("## 4 · Train the model (40 focus tools)\\nTrained once to Drive and reused; the returned `test` set is the held-out split the group probes run on."),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb3"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
CAP_PER_TOOL, EPOCHS, BATCH_SIZE = 120, 1, 8   # examples/tool -> 100 train / 10 val / 10 test; BATCH_SIZE: lower to 4 on OOM
RUN_TRAIN, FORCE_RETRAIN = True, False
FOCUS40 = finetune_and_eval(cat, raw, tok, FOCUS, "focus40", NB_DIR,
                            cap=CAP_PER_TOOL, epochs=EPOCHS, compact=False, token_aware=True,
                            eval_subsample=None, run_train=RUN_TRAIN,
                            force_retrain=FORCE_RETRAIN, batch_size=BATCH_SIZE)
m40, p40, tk40 = FOCUS40["bundle"]
TEST = FOCUS40["test"]                     # held-out test queries (never trained on)
log(f"focus40 selection_acc={FOCUS40['metrics']['selection_acc']:.3f} · held-out test={len(TEST)}")'''),
 md("## 5 · Per-group disambiguation\\nFor each group we take the held-out queries whose gold tool is in the group and offer the full group. Resumable: finished groups persist to `nb3/separation_results.json`."),
 co('''import contextlib, io
RES = os.path.join(NB_DIR, "separation_results.json")
prev = json.load(open(RES)) if (os.path.exists(RES) and not FORCE_RETRAIN) else {"per_group": [], "confusion": {}}
sep_by = {r["group"]: r for r in prev.get("per_group", [])}
group_conf = prev.get("confusion", {})
if sep_by: log(f"resumed {len(sep_by)} finished group(s)")

def gold_of(ex):
    calls = json.loads(ex.get("answers", "[]"))
    return next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)

t_all = time.time()
for gname, gtools in SIMILAR_GROUPS.items():
    if gname in sep_by and gname in group_conf and not FORCE_RETRAIN:
        log(f"{gname}: skip (done)"); continue
    try:
        gset = set(gtools)
        gtest = [e for e in TEST if gold_of(e) in gset]                      # held-out queries for this group
        offered = [offer_subset(cat, gold_of(e), gtools, len(gtools), seed=0, compact=False) for e in gtest]
        with contextlib.redirect_stdout(io.StringIO()):
            gpreds = predict(m40, p40, tk40, gtest, tools_override=offered)
        gm = evaluate(gtest, gpreds, family_of=family_of)
        sep_by[gname] = {"group": gname, "n_tools": len(gtools), "n": gm["n"],
                         "selection_acc": gm["selection_acc"], "name_f1": gm["name_f1"]}
        group_conf[gname] = confusion(gtest, gpreds)
        json.dump({"per_group": list(sep_by.values()), "confusion": group_conf}, open(RES, "w"), indent=2)
        log(f"{gname}: selection={gm['selection_acc']:.3f} (n={gm['n']})")
    except Exception as e:
        log(f"{gname}: FAILED ({type(e).__name__}: {e})")

log(f"ALL GROUPS DONE in {time.time()-t_all:.0f}s · results={RES}")
if not sep_by:
    raise RuntimeError("No group succeeded — see the FAILED lines above.")
separation = pd.DataFrame(list(sep_by.values())).sort_values("selection_acc")
display(separation)

plt.figure(figsize=(8,4)); plt.barh(separation["group"], separation["selection_acc"], color="#4C72B0")
plt.xlim(0,1); plt.xlabel("tool-selection accuracy"); plt.title("Separation: hardest look-alike groups (lower = more confused)")
plt.tight_layout(); save_fig("separation_groups", out_dir=NB_DIR); plt.show()'''),
 md("## 6 · Confusion heatmaps (who gets mistaken for whom)"),
 co('''for gname, conf in group_conf.items():
    labels = sorted(set(conf) | {p for row in conf.values() for p in row})
    M = pd.DataFrame(0, index=sorted(conf), columns=labels)
    for r, row in conf.items():
        for p, n in row.items(): M.loc[r, p] = n
    plt.figure(figsize=(0.9*len(labels)+2, 0.5*len(M)+1.5))
    if sns: sns.heatmap(M, annot=True, fmt="d", cmap="Blues", cbar=False)
    else:
        plt.imshow(M.values, cmap="Blues"); plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(M)), M.index)
    plt.title(f"Separation · {gname}"); plt.xlabel("predicted"); plt.ylabel("reference")
    plt.tight_layout(); save_fig(f"separation_confusion_{gname}", out_dir=NB_DIR); plt.show()'''),
 md("""## 7 · Read-out

Residual selection errors concentrate inside these look-alike groups. The lowest-accuracy group is the
frontier for a tool-picker; the heatmaps show whether confusions are symmetric (two tools mutually
confused) or a sink (everything collapses to one generic tool)."""),
]


# ======================================================================
# NB2 V2 — Depth by EXAMPLE (#args), one focus40 model
# ======================================================================
nb2v2 = [
 md("""# PICKO Research · NB2 V2 — **Depth by example**: does extraction degrade with more *arguments*?"""),
 BOOTSTRAP_MD, BOOTSTRAP, SETUP_MD, SETUP, DF1_MD, DF1,
 md("""## 2 · What changed vs NB2, and why

NB2 bucketed **tools** by their *schema* size (`total_params`) and trained a fresh small model per
iteration. Two problems: (1) the metric grades the answer against the arguments the gold call **actually
supplies** (`n_args`), not the schema size — a 5-parameter tool whose queries fill only one argument is an
*easy* extraction, yet lands in the "4+" bucket; (2) 0/4+ hold only 4–5 tools, so their error bars reflect
a handful of resampled tools.

**V2 fixes both:**
- **Bucket per example by `n_args`** (`0 / 1 / 2 / 3+`) — exactly what the extraction metrics grade — so
  every bucket holds hundreds of examples and difficulty is measured honestly.
- **One focus40 model** (like NB1/NB3): the only variable across buckets is the number of arguments, not
  which tools were trained on.
- **Fixed small offered set** — each query is offered its **gold + 5 distractors** with **full schemas**,
  token-trimmed to fit the 1024-token encoder (gold is never dropped). Full schemas are large (~110 tok
  each, up to ~270), so even 6 tools can approach the limit; the trim guarantees no silent truncation and a
  scorable gold every time. The offering is identical for every example, so the offered-tool count is not a
  confound."""),
 md("""### Examples bucketed by #arguments (before finetune)

One row per focus example, tagged with `required` / `total` params **and** `n_args` (what the gold answer
supplies) + its bucket. `n_args` is the axis V2 studies — it sits between `required` (a floor) and `total`
(a ceiling that also counts optionals the query may never fill)."""),
 co('''rows = []
for ex in raw:
    try: calls = json.loads(ex["answers"])
    except (ValueError, TypeError): continue
    gold = next((c["name"] for c in calls if isinstance(c, dict) and c.get("name")), None)
    if gold not in set(FOCUS): continue
    req, tot = cat.params_of(gold)
    k = gold_n_args(ex)
    rows.append({"query": ex["query"][:80], "gold_tool": gold, "required_params": req,
                 "total_params": tot, "n_args": k, "n_args_bucket": nargs_bucket(k)})
ex_df = pd.DataFrame(rows)
print("focus examples:", len(ex_df), "| per n_args bucket:",
      ex_df["n_args_bucket"].value_counts().reindex(NARGS_BUCKET_ORDER).to_dict())
display(ex_df.head(12))'''),
 md("## 3 · Configure"),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb2"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
CAP_PER_TOOL  = 120    # examples/tool -> 100 train / 10 val / 10 test
OFFER_K       = 6      # tools offered per query at inference: gold + 5 distractors, full schema, token-trimmed
EPOCHS        = 1
MAX_GEN_LEN   = 256    # long enough for full argument dicts
BATCH_SIZE    = 8      # lower to 4 on OOM, raise to 16 if headroom
N_BOOT        = 1000   # bootstrap resamples for the per-bucket error bars
RUN_TRAIN     = True
FORCE_RETRAIN = False
print("focus tools:", len(FOCUS), "| offered/query:", OFFER_K, "| cap:", CAP_PER_TOOL, "| out:", NB_DIR)'''),
 md("""## 4 · Train / test split (no leakage)

Both the training subprocess and this notebook call the **same deterministic** `per_tool_split`
(`seed=42`, 10 test + 10 val per tool). The model trains **only on the train split**; every metric below is
computed **only on the held-out test split**, so no test query is ever seen in training."""),
 md("## 5 · Train one focus40 model (full schemas)\\nTrained once to Drive and reused; the returned held-out `test` + `preds` feed the per-example scoring."),
 co('''DEPTH = finetune_and_eval(cat, raw, tok, FOCUS, "depth_focus40", NB_DIR,
                          cap=CAP_PER_TOOL, epochs=EPOCHS, compact=False, token_aware=True,
                          offer_all=OFFER_K, eval_subsample=None, run_train=RUN_TRAIN,
                          force_retrain=FORCE_RETRAIN, max_gen_len=MAX_GEN_LEN, batch_size=BATCH_SIZE)
TEST, PREDS = DEPTH["test"], DEPTH["preds"]
log(f"model ready · held-out test queries={len(TEST)} · selection={DEPTH['metrics']['selection_acc']:.3f}")'''),
 md("""## 6 · Score the held-out test per example

`evaluate_per_example` returns one scored row per non-abstention query (`selected`, `args_exact`, and
per-parameter `p_tp/p_fp/p_fn`). We tag each row with the query's `n_args` bucket — computed with the same
abstention filter so the two align row-for-row."""),
 co('''scored = evaluate_per_example(TEST, PREDS)
n_args_list = []
for e in TEST:
    try: calls = json.loads(e.get("answers", "[]"))
    except (ValueError, TypeError): calls = []
    prim = next((c for c in calls if isinstance(c, dict) and c.get("name")), None)
    if prim is None: continue                        # abstention — evaluate_per_example skips it too
    a = prim.get("arguments", {})
    n_args_list.append(len(a) if isinstance(a, dict) else 0)
assert len(n_args_list) == len(scored), "alignment mismatch between scores and n_args"
per_ex = pd.DataFrame(scored)
per_ex["n_args"] = n_args_list
per_ex["bucket"] = per_ex["n_args"].map(nargs_bucket)
print("scored test examples:", len(per_ex), "| per bucket:",
      per_ex["bucket"].value_counts().reindex(NARGS_BUCKET_ORDER).to_dict())
display(per_ex.head(10))'''),
 md("""## 7 · Accuracy per #args bucket (bootstrap 95% CI)

A single model → the uncertainty is *which test queries we happened to draw*, so error bars come from
**bootstrapping the test examples**. `args_exact_acc` is the mean over selected queries; `param_f1` is a
**micro**-F1 (summed `tp/fp/fn`) — undefined at bucket 0 (no parameters), shown as **n/a**."""),
 co('''def _f1(tp, fp, fn): return 2*tp / max(2*tp + fp + fn, 1)

def _bucket_point(df):
    sel = df[df["selected"] == 1]
    ae = sel["args_exact"].mean() if len(sel) else np.nan
    pf = _f1(sel["p_tp"].sum(), sel["p_fp"].sum(), sel["p_fn"].sum()) if len(sel) else np.nan
    return float(df["selected"].mean()), float(ae), float(pf)

def _bucket_ci(df, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    sel = df["selected"].values; ax = df["args_exact"].values
    tp, fp, fn = df["p_tp"].values, df["p_fp"].values, df["p_fn"].values
    ae, pf = [], []
    for _ in range(n_boot):
        s = rng.integers(0, len(df), len(df)); m = sel[s] == 1
        if m.sum() == 0: continue
        ae.append(ax[s][m].mean())
        pf.append(_f1(tp[s][m].sum(), fp[s][m].sum(), fn[s][m].sum()))
    ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)
    return ci(ae), ci(pf)

agg = []
for b in NARGS_BUCKET_ORDER:
    d = per_ex[per_ex["bucket"] == b]
    if d.empty: continue
    sacc, ae, pf = _bucket_point(d); (ae_lo, ae_hi), (pf_lo, pf_hi) = _bucket_ci(d)
    if b == "0": pf, pf_lo, pf_hi = np.nan, np.nan, np.nan   # param_f1 undefined at 0 params
    agg.append({"bucket": b, "n": len(d), "n_selected": int(d["selected"].sum()),
                "selection_acc": round(sacc, 4), "args_exact_acc": round(ae, 4),
                "args_ci": [round(ae_lo, 4), round(ae_hi, 4)], "param_f1": None if np.isnan(pf) else round(pf, 4),
                "pf1_ci": [None, None] if np.isnan(pf) else [round(pf_lo, 4), round(pf_hi, 4)]})
RES = os.path.join(NB_DIR, "depth_by_example_results.json")
json.dump({"offer_k": OFFER_K, "n_test": len(per_ex), "buckets": agg}, open(RES, "w"), indent=2)
log(f"saved {RES}")
display(pd.DataFrame(agg))'''),
 md("## 8 · The Depth-by-example plot"),
 co('''ORDER = [r["bucket"] for r in agg]
x = np.arange(len(ORDER)); w = 0.38
def _yerr(rows, mkey, cikey):
    lo, hi = [], []
    for r in rows:
        m = r[mkey]; c = r[cikey]
        if m is None or c[0] is None: lo.append(0); hi.append(0)
        else: lo.append(max(0, m - c[0])); hi.append(max(0, c[1] - m))
    return np.array([lo, hi])
ae = [r["args_exact_acc"] for r in agg]
pf = [0 if r["param_f1"] is None else r["param_f1"] for r in agg]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
ax.bar(x-w/2, ae, w, yerr=_yerr(agg, "args_exact_acc", "args_ci"), capsize=3,
       color="#0072B2", edgecolor="white", lw=0.6, label="args_exact_acc (all-or-nothing)",
       error_kw=dict(lw=1, ecolor="#444"))
ax.bar(x+w/2, pf, w, yerr=_yerr(agg, "param_f1", "pf1_ci"), capsize=3,
       color="#E69F00", edgecolor="white", lw=0.6, label="param_f1 (partial credit)",
       error_kw=dict(lw=1, ecolor="#444"))
for xi, r in zip(x-w/2, agg): ax.text(xi, min(r["args_exact_acc"]+0.04, 1.04), f"{r['args_exact_acc']:.2f}", ha="center", fontsize=9, color="#0072B2")
for xi, r in zip(x+w/2, agg):
    if r["param_f1"] is None: ax.text(xi, 0.03, "n/a", ha="center", fontsize=9, color="#8a8a8a")
    else: ax.text(xi, min(r["param_f1"]+0.04, 1.04), f"{r['param_f1']:.2f}", ha="center", fontsize=9, color="#b07400")
ax.set_xticks(x); ax.set_xticklabels([f"{r['bucket']}\\n(n={r['n']})" for r in agg])
ax.set_ylim(0, 1.08); ax.set_xlabel("Number of arguments the gold answer supplies (n_args)")
ax.set_ylabel("Extraction accuracy")
ax.set_title("Impact of Argument Count on Parameter-Extraction Accuracy")
ax.legend(loc="lower left")
if sns: sns.despine(ax=ax)
ax.grid(axis="y", color="#cccccc", lw=0.6, alpha=0.6); ax.set_axisbelow(True)
plt.tight_layout(); save_fig("depth_by_example", out_dir=NB_DIR); plt.show()'''),
 md("""## 9 · Read-out

One focus40 model, probed on held-out queries each offered its gold tool plus five full-schema distractors,
scored **per example** by how many arguments the answer requires. Because the offered set is identical
across buckets, the trend isolates a single variable — argument count. Exact-match extraction falls as more
arguments must be produced, while `param_f1` degrades more gently (partial credit for the arguments that are
right). **Takeaway:** PICKO reliably picks the tool, but *fully* populating a multi-argument call is the
harder half of the task, and that difficulty scales with the number of arguments actually required."""),
]


# ======================================================================
# NB3 V2 — Semantic separation: routing by DOMAIN without the source name
# ======================================================================
nb3v2 = [
 md("""# PICKO Research · NB3 V2 — **Semantic separation**: can it *learn* to route by domain without the source name?"""),
 BOOTSTRAP_MD, BOOTSTRAP, SETUP_MD, SETUP,
 md("""## 2 · The question

NB3 found look-alike **search** tools easy to tell apart across sources — but most training queries **name
the source** (58-85% of `arxiv/pubmed/wikipedia` search queries contain the source word), so that result may
reflect **lexical matching**, not understanding.

**V2 asks a stronger question: can the model *learn* to route by domain meaning alone?** We take a pair that
does the same action in disjoint domains — `arxiv_search_papers` (computer science) vs
`pubmed_search_articles` (medicine) — with freshly written queries that **never name the source**, only the
topic, and **train a focused 2-tool model** on them.

- **Total train/test separation** — the same deterministic `per_tool_split` holds out 10 queries per tool
  (20 total), so the model can never memorise a test answer.
- **Baseline (before):** the already-trained focus40 model on the same held-out test — how well the
  name-trained model routes source-free queries *without* semantic training.
- **After:** the 2-tool model trained on the source-free difference, on the same test.

The **before→after** gap is the finding: if the baseline is near chance but training on the clean semantic
signal lifts it well above 50%, the distinction **is learnable** and the original data simply never taught
it."""),
 md("""## 3 · Build the source-free dataset

Reads the two domain query files (`data/semantic/cs.json` → arxiv, `data/semantic/medicine.json` → pubmed),
drops any query that still names a source (safety net), and bakes the offered pair (full schema, order
randomised) into each row — the standard `{query, tools, answers}` format. Only the **unnamed** (source-free)
queries are used here. Falls back to the prebuilt `data/semantic/probe.jsonl` if the raw files aren't present."""),
 co('''import re, collections, random as _rnd
BANNED = r"\\b(arxiv|arxiv\\.org|pubmed|medline|ncbi|pmid|biorxiv|medrxiv|preprint|wikipedia|wiki|hugging\\s*face|huggingface|semantic scholar)\\b"
DOMAIN = {"arxiv_search_papers": ("cs", "arXiv"), "pubmed_search_articles": ("medicine", "PubMed")}
RAWFILE = {"arxiv_search_papers": "semantic/cs.json", "pubmed_search_articles": "semantic/medicine.json"}
PAIR_TOOLS = list(DOMAIN)                       # the forced 2-way choice

def _find(name):
    for root in [os.path.join(ROOT, "data"), "/content/drive/MyDrive/picko",
                 "/content/drive/MyDrive/picko/data"]:
        p = os.path.join(root, name)
        if os.path.exists(p): return p
    return None

def _tools_json(seed):                          # offered pair (full schema), order randomised per row
    ts = [cat.by_name[n] for n in PAIR_TOOLS]
    _rnd.Random(seed).shuffle(ts)
    return json.dumps(ts, separators=(",", ":"), ensure_ascii=False)

raw_paths = {g: _find(f) for g, f in RAWFILE.items()}
if all(raw_paths.values()):
    DATA, seen, i = [], set(), 0
    for gold, (domain, src) in DOMAIN.items():
        ans = json.dumps([{"name": gold, "arguments": {}}])
        for q in json.load(open(raw_paths[gold])):
            q = q.strip(); key = re.sub(r"\\W+", " ", q.lower()).strip()
            if not q or key in seen or re.search(BANNED, q, re.I): continue
            seen.add(key)
            DATA.append({"query": q, "tools": _tools_json(i), "answers": ans,
                         "gold": gold, "domain": domain}); i += 1
    pp = os.path.join(ROOT, "data", "semantic", "train.jsonl")
    os.makedirs(os.path.dirname(pp), exist_ok=True)
    with open(pp, "w") as f:
        for r in DATA: f.write(json.dumps(r, ensure_ascii=False) + "\\n")
    log(f"built source-free dataset ({len(DATA)} rows) → {pp}")
else:
    pp = _find("semantic/probe.jsonl")
    if not pp: raise FileNotFoundError("need data/semantic/cs.json + medicine.json (or probe.jsonl) in data/ or Drive")
    DATA = [r for r in (json.loads(l) for l in open(pp) if l.strip()) if r.get("condition", "unnamed") == "unnamed"]
    log(f"loaded {len(DATA)} source-free rows from {pp}")

df = pd.DataFrame(DATA)
print("dataset:", len(DATA), "| per domain:", df["domain"].value_counts().to_dict())
display(df[["query", "gold", "domain"]].head(6))'''),
 md("## 4 · Configure"),
 co('''NB_DIR = os.path.join(OUT_DIR, "nb3"); os.makedirs(NB_DIR, exist_ok=True)   # this notebook's outputs
EPOCHS        = 3      # small 2-tool set -> a few passes
BATCH_SIZE    = 8
MAX_GEN_LEN   = 64     # we only score tool SELECTION
RUN_TRAIN     = True
FORCE_RETRAIN = False
BASELINE_CKPT = next((c for c in [os.path.join(OUT_DIR, "nb2", "picko_depth_focus40_best.pkl"),
                                  os.path.join(OUT_DIR, "nb1", "picko_breadth_focus40_best.pkl")]
                      if os.path.exists(c)), None)
print("pair:", PAIR_TOOLS, "| epochs:", EPOCHS, "| baseline:", BASELINE_CKPT, "| out:", NB_DIR)'''),
 md("""## 5 · Train / test split (total separation)

The same deterministic `per_tool_split` (`seed=42`) both notebook and training subprocess use: 10 test + 10
val per tool, the rest train. The test 20 are never seen in training."""),
 co('''train, val, TEST = per_tool_split(DATA)
print(f"train={len(train)}  val={len(val)}  test={len(TEST)}")
print("test per domain:", collections.Counter(e["domain"] for e in TEST))'''),
 md("## 6 · Baseline — the existing model on the held-out test (before semantic training)"),
 co('''import contextlib, io
if BASELINE_CKPT:
    bm, bp, btk = load_model(BASELINE_CKPT)
    with contextlib.redirect_stdout(io.StringIO()):
        base_preds = predict(bm, bp, btk, TEST, max_gen_len=MAX_GEN_LEN, batch=BATCH_SIZE)
    base_m = evaluate(TEST, base_preds, family_of=family_of)
    log(f"BASELINE (name-trained model, no semantic training): selection={base_m['selection_acc']:.3f}")
else:
    base_preds, base_m = None, None
    log("no baseline checkpoint found (run nb1/nb2 first) — skipping the 'before' comparison")'''),
 md("## 7 · Train the 2-tool semantic model\\nTrained on the source-free difference only; the returned held-out `test` + `preds` are scored below. Resumable to Drive."),
 co('''R = finetune_and_eval(cat, raw, tok, PAIR_TOOLS, "semantic_pair", NB_DIR,
                      dataset=DATA, epochs=EPOCHS, run_train=RUN_TRAIN,
                      force_retrain=FORCE_RETRAIN, max_gen_len=MAX_GEN_LEN, batch_size=BATCH_SIZE)
after_test, after_preds, after_m = R["test"], R["preds"], R["metrics"]
log(f"AFTER (trained on the semantic difference): selection={after_m['selection_acc']:.3f}")'''),
 md("## 8 · Before vs after, per domain"),
 co('''def _by_domain(examples, preds):
    sc = evaluate_per_example(examples, preds)
    d = pd.DataFrame({"domain": [e["domain"] for e in examples], "selected": [s["selected"] for s in sc]})
    return d.groupby("domain")["selected"].mean().round(4).to_dict()

rowsout = []
if base_m:
    rowsout.append({"phase": "before (baseline)", "overall": round(base_m["selection_acc"], 4), **_by_domain(TEST, base_preds)})
rowsout.append({"phase": "after (semantic)", "overall": round(after_m["selection_acc"], 4), **_by_domain(after_test, after_preds)})
summary = pd.DataFrame(rowsout)
RES = os.path.join(NB_DIR, "semantic_learn_results.json")
json.dump({"pair": PAIR_TOOLS, "summary": rowsout, "confusion_after": confusion(after_test, after_preds)},
          open(RES, "w"), indent=2)
log(f"saved {RES}")
display(summary)'''),
 md("## 9 · Plot"),
 co('''cats = ["overall", "cs", "medicine"]; x = np.arange(len(cats)); w = 0.8 / max(len(rowsout), 1)
palette = {"before (baseline)": "#8a8a8a", "after (semantic)": "#009E73"}
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
for j, r in enumerate(rowsout):
    vals = [r.get(c, np.nan) for c in cats]
    off = (j - (len(rowsout) - 1) / 2) * w
    ax.bar(x + off, vals, w, color=palette.get(r["phase"], "#0072B2"), edgecolor="white", lw=0.6, label=r["phase"])
    for k, v in enumerate(vals):
        if not np.isnan(v): ax.text(x[k] + off, min(v + 0.03, 1.03), f"{v:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, color="#C44E52", ls="--", lw=1.2, label="chance (2-way)")
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylim(0, 1.08); ax.set_ylabel("Tool-selection accuracy")
ax.set_title("Can PICKO Learn to Route by Domain Without the Source Name?"); ax.legend(loc="lower right")
if sns: sns.despine(ax=ax)
ax.grid(axis="y", color="#cccccc", lw=0.6, alpha=0.6); ax.set_axisbelow(True)

conf = confusion(after_test, after_preds)
M = pd.DataFrame(0, index=PAIR_TOOLS, columns=PAIR_TOOLS)
for rr, row in conf.items():
    for p, c in row.items():
        if rr in PAIR_TOOLS and p in PAIR_TOOLS: M.loc[rr, p] = c
short = lambda n: n.replace("_search_papers", "").replace("_search_articles", "")
if sns:
    sns.heatmap(M.div(M.sum(1).replace(0, 1), axis=0), cmap="Greens", vmin=0, vmax=1, cbar=False,
                annot=M.values, fmt="d", linewidths=0.5, linecolor="white", ax=ax2,
                xticklabels=[short(c) for c in M.columns], yticklabels=[short(r) for r in M.index])
ax2.set_title("After training: routing"); ax2.set_xlabel("predicted"); ax2.set_ylabel("true domain tool")
plt.tight_layout(); save_fig("semantic_learnability", out_dir=NB_DIR); plt.show()'''),
 md("""## 10 · Read-out

Offered only the two same-action tools on source-free queries, routing can come only from the domain
meaning. The **baseline** shows how the name-trained model copes without the source word; the **after** bar
shows the focused 2-tool model trained on the clean semantic difference, on a **held-out** test it never
saw. If after-training accuracy sits well above the 50% chance line — and above the baseline — PICKO **can**
learn domain-based routing; if it stays near chance, the distinction is beyond what this signal teaches the
26M model. The confusion panel shows whether residual errors are symmetric or collapse toward one domain."""),
]


def write(cells, name):
    for j, c in enumerate(cells):
        c["id"] = f"c{j:02d}"
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "PICKO (.venv)", "language": "python", "name": "picko"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    path = os.path.join(OUTDIR, name)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", path, "·", len(cells), "cells")


if __name__ == "__main__":
    write(nb1,   "nb1_breadth_amount.ipynb")
    write(nb2v2, "nb2_depth_by_example.ipynb")
    write(nb3v2, "nb3_semantic_separation.ipynb")
