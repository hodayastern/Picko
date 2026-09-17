#!/usr/bin/env python3
"""Shared building blocks for the PICKO research notebooks (Breadth / Depth /
Separation). Import this once and the notebooks stay thin:

    from scripts.picko_research import *
    cat, tok, raw, FOCUS, OUT_DIR = load_context()
    display(tools_dataframe(cat, FOCUS))
    display(examples_dataframe(cat, raw, FOCUS))

Works both locally (CPU) and on Colab (GPU) — ROOT is found robustly, and OUT_DIR
comes from PICKO_OUT_DIR (point it at Google Drive on Colab so checkpoints survive
a runtime restart).
"""
import glob
import json
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime

# ---- locate repo ROOT robustly (works from notebooks/research/, Colab, etc.) ----
_MARKER = "full_tools_53tools_11products.json"


def _find_root():
    if os.path.isdir("/content/picko") and os.path.exists(
            os.path.join("/content/picko", _MARKER)):
        return "/content/picko"
    d = os.path.abspath(os.getcwd())
    for _ in range(6):
        if os.path.exists(os.path.join(d, _MARKER)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    # fall back to two levels up from this file (scripts/ -> repo root)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = _find_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- JAX / GPU env (must be set BEFORE jax initializes, i.e. before the imports
# below and inherited by the finetune subprocess) ----
_ON_COLAB = "google.colab" in sys.modules
if not _ON_COLAB:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")   # local = CPU; Colab auto-detects GPU
# reuse compiled XLA kernels across the many finetune subprocesses (huge speedup —
# the train step has fixed shapes, so it compiles once and is reused for every size)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR",
                      "/content/jax_cache" if _ON_COLAB else os.path.join(ROOT, ".jax_cache"))
if _ON_COLAB:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")     # don't grab 75% of VRAM up front
    # cudaMalloc per allocation (releases between allocs) — critical because the notebook
    # KERNEL and the finetune SUBPROCESS share the one GPU, so a retained pool fragments and OOMs.
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from scripts.tool_catalog import Catalog, family_of                      # noqa: E402
from scripts.research_sets import (FOCUS_FAMILIES, focus_names, BREADTH_SIZES,   # noqa: E402,F401
                                   nested_sets, breadth_pool, size_sets,
                                   SIMILAR_GROUPS, param_bucket, PARAM_BUCKET_ORDER,
                                   nargs_bucket, NARGS_BUCKET_ORDER, gold_n_args)
from scripts.picko_eval import (load_model, predict, evaluate, confusion,   # noqa: E402,F401
                                evaluate_per_example, base_checkpoint,
                                tools_token_len, n_visible)
from needle.training.finetune import _per_tool_split                     # noqa: E402
from needle.dataset.dataset import get_tokenizer                         # noqa: E402

# public alias — `from scripts.picko_research import *` skips underscore names,
# so notebooks that need the split must see it under a non-underscore name.
per_tool_split = _per_tool_split


# ---- observability (works the same locally and on Colab) ----
def log(msg):
    """Timestamped, flushed print. Also appends to PICKO_LOG (point it at Drive on
    Colab) so a walk-away run leaves a durable trail that survives a runtime restart."""
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    path = os.environ.get("PICKO_LOG")
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def save_fig(name, out_dir=None, dpi=150):
    """Save the current matplotlib figure to OUT_DIR (Drive on Colab) and log where.
    `name` may omit the extension. Call it right before `plt.show()`."""
    import matplotlib.pyplot as plt
    out_dir = out_dir or os.environ.get("PICKO_OUT_DIR") or os.path.join(ROOT, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name if name.endswith(".png") else name + ".png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    log(f"saved plot → {path}")
    return path


def env_report(out_dir=None):
    """Print jax devices, the accelerator, and whether OUT_DIR is durable (Drive) or
    ephemeral — a one-glance header for a run-and-forget session."""
    import jax
    out_dir = out_dir or os.environ.get("PICKO_OUT_DIR") or os.path.join(ROOT, "checkpoints")
    durable = out_dir.startswith("/content/drive")
    devs = jax.devices()
    plat = devs[0].platform if devs else "?"
    log(f"jax {jax.__version__} · devices={devs} · platform={plat}")
    log(f"OUT_DIR={out_dir} ({'DURABLE — Google Drive' if durable else 'ephemeral — NOT Drive'})")
    log(f"compile-cache={os.environ.get('JAX_COMPILATION_CACHE_DIR', '(off)')}")
    if not durable and _ON_COLAB:
        log("WARNING: OUT_DIR is not on Drive — checkpoints/results will be LOST on a runtime restart.")
    if _ON_COLAB and plat != "gpu":
        log("WARNING: JAX is on CPU, not GPU — training/eval will be ~30x slower (hours vs minutes). "
            "Fix: Runtime > Change runtime type > GPU, then Restart session, and re-run the bootstrap cell.")
    return {"devices": [str(d) for d in devs], "platform": plat,
            "out_dir": out_dir, "durable": durable}


def _run_finetune(jsonl_path, epochs, batch_size=16):
    """Run `needle finetune` robustly. Prefer the console script if present, else
    invoke needle.cli.main in a subprocess with the SAME Python (works on Colab
    without `pip install -e .`, and inherits the GPU/compile-cache env).

    Captures output and re-raises with the real traceback tail on failure — a
    subprocess writes to OS-level stderr, which Colab/Jupyter does NOT mirror into
    the cell, so a plain check=True would hide the actual error (e.g. GPU OOM)."""
    os.makedirs(os.environ.get("JAX_COMPILATION_CACHE_DIR", ""), exist_ok=True) \
        if os.environ.get("JAX_COMPILATION_CACHE_DIR") else None
    args = ["finetune", jsonl_path, "--epochs", str(epochs), "--batch-size", str(batch_size)]
    exe = shutil.which("needle")
    if exe:
        cmd = [exe] + args
    else:
        code = (f"import sys; sys.argv=['needle']+{args!r}; "
                "from needle.cli import main; main()")
        cmd = [sys.executable, "-c", code]
    proc = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout or "").splitlines()[-45:])
        raise RuntimeError(f"`needle finetune` failed (exit {proc.returncode}):\n{tail}")
    for line in (proc.stdout or "").splitlines():   # brief summary on success
        if line.startswith(("BASE_EVAL", "  Base:", "Best checkpoint", "Training complete")):
            print("  " + line.strip())


# ---- context ----
def load_context():
    """Return (cat, tok, raw, FOCUS, OUT_DIR). Reads the balanced pool and resolves
    the persistent output dir (Drive on Colab via PICKO_OUT_DIR, else <ROOT>/checkpoints)."""
    cat = Catalog()
    tok = get_tokenizer()
    data_path = os.path.join(ROOT, "data", "picko_balanced.jsonl")
    raw = [json.loads(l) for l in open(data_path) if l.strip()]
    focus = focus_names(cat)
    out_dir = os.environ.get("PICKO_OUT_DIR") or os.path.join(ROOT, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    print(f"ROOT={ROOT} · {len(cat.tools)} tools · focus {len(focus)} · "
          f"{len(raw)} examples · OUT_DIR={out_dir}")
    return cat, tok, raw, focus, out_dir


# ---- intro dataframes (shared by all three notebooks) ----
def tools_dataframe(cat, names):
    """One clean row per tool: family, category, param counts + bucket, description."""
    import pandas as pd
    by = cat.by_name
    rows = []
    for n in names:
        t = by[n]
        req, tot = cat.params_of(n)
        rows.append({"tool": n, "family": family_of(n),
                     "category": cat.category_of(family_of(n)),
                     "required_params": req, "total_params": tot,
                     "param_bucket": param_bucket(tot),
                     "description": (t.get("description") or "")[:100]})
    return (pd.DataFrame(rows)
            .sort_values(["family", "tool"]).reset_index(drop=True))


def examples_dataframe(cat, raw, names):
    """One row per training example whose gold tool is in `names`: the query, its
    gold tool + param bucket, and how many arguments the answer supplies."""
    import pandas as pd
    keep = set(names)
    rows = []
    for ex in raw:
        try:
            calls = json.loads(ex["answers"])
        except (ValueError, TypeError):
            continue
        gold = next((c["name"] for c in calls
                     if isinstance(c, dict) and c.get("name")), None)
        if gold not in keep:
            continue
        _, tot = cat.params_of(gold)
        args = calls[0].get("arguments", {}) if isinstance(calls[0], dict) else {}
        rows.append({"query": ex["query"], "gold_tool": gold,
                     "family": family_of(gold),
                     "category": cat.category_of(family_of(gold)),
                     "total_params": tot, "param_bucket": param_bucket(tot),
                     "n_args": len(args) if isinstance(args, dict) else 0})
    return pd.DataFrame(rows).reset_index(drop=True)


# ---- sampling (Depth) ----
def sample_stratified(cat, names, tools_per_bucket, seed=0):
    """Pick `tools_per_bucket` tools from EACH param bucket present in `names`.
    Returns a flat list (a small, bucket-balanced set that fits the encoder window)."""
    rng = random.Random(seed)
    by_bucket = {}
    for n in names:
        _, tot = cat.params_of(n)
        by_bucket.setdefault(param_bucket(tot), []).append(n)
    chosen = []
    for b in PARAM_BUCKET_ORDER:
        pool = by_bucket.get(b, [])
        rng.shuffle(pool)
        chosen.extend(pool[:tools_per_bucket])
    rng.shuffle(chosen)
    return chosen


def offer_subset(cat, gold, pool, k, seed, compact=True):
    """Compact-tools JSON of k tools from `pool` (gold included), shuffled; no token-fit so the 1024-tok encoder still truncates — that truncation is the Breadth signal."""
    rng = random.Random(seed)
    others = [n for n in pool if n != gold]
    chosen = ([gold] if gold in pool else []) + rng.sample(others, min(max(k, 1) - (gold in pool), len(others)))
    rng.shuffle(chosen)
    tools = [({"name": t["name"], "description": t.get("description", "")} if compact else t)
             for t in (cat.by_name[n] for n in chosen)]
    return json.dumps(tools, separators=(",", ":"), ensure_ascii=False)


# ---- finetune + eval (shared by NB1/NB2/NB3 and run_research.py) ----
def finetune_and_eval(cat, raw, tok, names, tag, out_dir, *, cap=40, epochs=1,
                      compact=False, offer_all=None, token_aware=False,
                      eval_subsample=None, run_train=True, force_retrain=False,
                      max_gen_len=256, quiet_decode=True, batch_size=16,
                      dataset=None):
    """Re-scope -> finetune (gated/resumable) -> copy stable checkpoint -> eval.

    `max_gen_len` caps decode length — short is much faster because weak models
    otherwise generate to the cap. Selection is recovered by regex even if a short
    cap truncates the JSON (see picko_eval._salvage), so Breadth can use ~64.
    `quiet_decode` silences needle's constrained-decoder prints (thousands of lines
    that slow Colab). `dataset` (list of {query,tools,answers}) trains on a prebuilt
    set instead of `restrict_dataset(raw, names)` — the same deterministic
    per_tool_split still holds out its test rows. Returns {names, ckpt, bundle,
    test, preds, metrics, data}.
    """
    import contextlib
    import io
    if dataset is not None:
        data = list(dataset)
    else:
        kw = dict(cap_per_tool=cap, compact=compact, seed=0)
        if offer_all is not None:
            kw["offer_all_max"] = offer_all
        if token_aware:
            kw["tokenizer"] = tok
        data = cat.restrict_dataset(raw, names, **kw)
    path = os.path.join(ROOT, "data", f"picko_{tag}.jsonl")
    with open(path, "w") as f:
        for e in data:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    ckpt = os.path.join(out_dir, f"picko_{tag}_best.pkl")
    if run_train and (force_retrain or not os.path.exists(ckpt)):
        log(f"[{tag}] finetuning on {len(data)} examples ({len(names)} tools), "
            f"epochs={epochs} batch={batch_size}…")
        t0 = time.time()
        _run_finetune(path, epochs, batch_size=batch_size)
        newest = max(glob.glob(os.path.join(ROOT, "checkpoints", "needle_finetuned_*_best.pkl")),
                     key=os.path.getmtime)
        shutil.copy(newest, ckpt)
        mb = os.path.getsize(ckpt) / 1e6
        log(f"[{tag}] trained in {time.time()-t0:.0f}s · checkpoint saved → {ckpt} ({mb:.1f} MB)")
    else:
        log(f"[{tag}] resumed (skipped training) — using existing checkpoint {os.path.basename(ckpt)}")
    assert os.path.exists(ckpt), f"missing {ckpt} — run with run_train=True first"

    _, _, test = _per_tool_split(data)
    random.Random(0).shuffle(test)   # per_tool_split returns test grouped by tool; shuffle so eval_subsample spans all tools
    if eval_subsample:
        test = test[:eval_subsample]
    t1 = time.time()
    m, p, tk = load_model(ckpt)
    _ctx = contextlib.redirect_stdout(io.StringIO()) if quiet_decode else contextlib.nullcontext()
    with _ctx:
        preds = predict(m, p, tk, test, max_gen_len=max_gen_len)
    metrics = evaluate(test, preds, family_of=family_of)
    log(f"[{tag}] evaluated {len(test)} examples in {time.time()-t1:.0f}s · "
        f"selection={metrics['selection_acc']:.3f} name_f1={metrics['name_f1']:.3f}")
    return {"names": names, "ckpt": ckpt, "bundle": (m, p, tk),
            "test": test, "preds": preds, "metrics": metrics, "data": data}


def base_eval(test):
    """Base (un-finetuned) model metrics on `test` — for the baseline comparison."""
    m, p, tk = load_model(base_checkpoint())
    preds = predict(m, p, tk, test)
    return evaluate(test, preds, family_of=family_of)
