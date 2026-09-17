#!/usr/bin/env python3
"""Synthesize PICKO finetuning data over the scientific tool subset.

Reuses needle's Gemini generator (`needle/dataset/generate.py`) unchanged, but
monkeypatches its module globals so it operates on OUR scientific tools instead
of the built-in on-device assistant tools:

  * ALL_POOLS        -> scientific clusters from data/scientific_pools.json
  * _synthesize_tools-> disabled (never invent novel tools; stay on-catalog)
  * _OVERLAP_PAIRS   -> our own similar "search" tools, so PICKO is forced to
                        disambiguate between them (project dimension D3)
  * _CONTEXT_SEEDS   -> science-user personas (grad student, bioinformatician…)
  * CALL_TYPES       -> single-shot heavy + abstention negatives (PICKO scope is
                        one prompt -> one tool call, no multi-turn/multi-call)
  * LANGUAGES        -> English only (quick start)

Output is the exact {query, tools, answers} JSONL that `needle finetune` reads.

Prerequisites:
    export GEMINI_API_KEY=...        # from https://aistudio.google.com/apikey
    python scripts/make_scientific_pools.py   # writes data/scientific_pools.json

Usage:
    # main pass over all 8 tools
    python scripts/generate_picko_data.py --num-samples 1600 \
        --output-jsonl data/picko_subset.jsonl --workers 8

    # top-up an under-covered tool (restrict the pool to its cluster)
    python scripts/generate_picko_data.py --num-samples 200 \
        --output-jsonl data/picko_subset.jsonl --focus pubmed
"""
import argparse
import concurrent.futures as cf
import itertools
import json
import os
import random
import sys
import threading
import time
from collections import Counter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOLS_JSON = os.path.join(HERE, "data", "scientific_pools.json")

import needle.dataset.generate as g


class _RateLimiter:
    """Global min-interval gate so all worker threads together stay under an RPM
    cap (Gemini free tier is 15 RPM). Enforces spacing between request *starts*
    without holding the lock during the sleep, so requests can still overlap."""

    def __init__(self, rpm):
        self.min_interval = 60.0 / max(rpm, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self):
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next)
            self._next = slot + self.min_interval
            wait = slot - now
        if wait > 0:
            time.sleep(wait)


# Default: 13 req/min — safely under the 15 RPM free-tier limit (leaves headroom
# for the occasional retry, which is itself a request).
_LIMITER = _RateLimiter(13)


def set_rate_limit(rpm):
    """Set the global requests-per-minute cap (call before generating)."""
    global _LIMITER
    _LIMITER = _RateLimiter(rpm)
    return _LIMITER.min_interval

# Science-domain personas replacing the on-device assistant contexts.
SCIENCE_CONTEXTS = [
    "The user is a PhD student doing a literature review",
    "The user is a bioinformatician analyzing a new dataset",
    "The user is a machine-learning researcher tracking recent papers",
    "The user is a clinician looking for evidence on a treatment",
    "The user is a research assistant compiling a bibliography",
    "The user is a data scientist searching for a pretrained model",
    "The user is a postdoc preparing a related-work section",
    "The user is a science journalist fact-checking a claim",
    "The user is a grad student new to a research area",
    "The user is a professor preparing lecture material",
    "The user is an undergraduate working on a thesis",
    "The user is an epidemiologist reviewing clinical studies",
]

# Science-task query topics, replacing the on-device assistant SCENARIOS so the
# abstention negatives (none/near_miss) read like real research asks instead of
# "lock the back door".
SCIENCE_SCENARIOS = [
    "finding recent papers on a method", "getting a paper's abstract by its ID",
    "searching PubMed for clinical evidence", "looking up a biomedical article by PMID",
    "downloading a specific journal article", "searching arXiv by category and date",
    "finding pretrained models for a task", "finding ML papers on Hugging Face",
    "searching Wikipedia for a concept", "extracting key facts from a Wikipedia article",
    "comparing literature across databases", "filtering arXiv results by date range",
    "getting a summary of a research topic", "finding datasets for an experiment",
    "tracking new preprints in a subfield", "gathering references for a related-work section",
    "checking citations for a paper", "finding a model by author or organization",
    "looking up an author's publications", "reviewing meta-analyses on a treatment",
    "surveying methods for a benchmark", "finding background on an unfamiliar term",
    "locating the original paper for a model", "sorting models by downloads or likes",
]

# Single-shot heavy mix. PICKO is scoped to one prompt -> one tool call, so we
# drop the multi-call types. We keep some abstention negatives (none/near_miss)
# and light robustness (indirect/disfluent/garbled), which stay 1-call.
_ALL = {t[0]: t for t in g.CALL_TYPES}
PICKO_CALL_TYPES = (
    [_ALL["single"]] * 10
    + [_ALL["none"]] * 2
    + [_ALL["near_miss"]] * 2
    + [_ALL["indirect"]] * 1
    + [_ALL["disfluent"]] * 1
    + [_ALL["garbled"]] * 1
)


def _load_pools(focus=None):
    with open(POOLS_JSON) as f:
        pools = json.load(f)  # {cluster: [tool, ...]}
    if focus:
        wanted = {c.strip() for c in focus.split(",")}
        pools = {k: v for k, v in pools.items() if k in wanted}
        if not pools:
            sys.exit(f"--focus {focus!r} matched no clusters ({list(pools)})")
    return list(pools.values())  # list of pools (each a list of tool dicts)


def _build_overlap_pairs(all_pools):
    """Pairs of semantically-similar tools to co-present (drives D3).

    Pairs every 'search'/'find' tool with every other, so batches routinely
    offer 2+ near-synonymous tools and PICKO must pick the right one.
    """
    flat = [t for pool in all_pools for t in pool]
    searchy = [t for t in flat
               if any(w in t["name"] for w in ("search", "find", "paper"))]
    pairs = list(itertools.combinations(searchy, 2))
    return pairs or list(itertools.combinations(flat[:2], 2))


def _norm(q):
    return " ".join(q.lower().split()).strip(".,!?;:'\"")


def _apply_patches(all_pools):
    """Point needle's generator at our scientific tools (module globals)."""
    g.ALL_POOLS = all_pools
    g._synthesize_tools = lambda *a, **k: None      # never invent novel tools
    g._OVERLAP_PAIRS = _build_overlap_pairs(all_pools)
    g._CONTEXT_SEEDS = SCIENCE_CONTEXTS
    g.SCENARIOS = SCIENCE_SCENARIOS
    g.CALL_TYPES = PICKO_CALL_TYPES
    g.LANGUAGES = ["English"]
    g.MAX_TOOLS = min(10, sum(len(p) for p in all_pools))
    counts = {c: w for c, w in g._TOOL_COUNT_WEIGHTS.items() if c <= g.MAX_TOOLS}
    g._TOOL_COUNT_WEIGHTS = counts or {g.MAX_TOOLS: 1}
    g._TOOL_COUNTS = list(g._TOOL_COUNT_WEIGHTS.keys())
    g._TOOL_WEIGHTS = list(g._TOOL_COUNT_WEIGHTS.values())


class QuotaExhausted(RuntimeError):
    """Raised when the Gemini API reports the quota/rate limit is spent — a fatal
    condition we must NOT retry (retrying just hangs for hours)."""


def _is_quota_error(e):
    s = str(e).lower()
    return "resource_exhausted" in s or "exceeded your current quota" in s or \
           ("429" in s and "quota" in s)


def _batch_with_retry(cp, batch_size, rng, model, tries=3):
    """Run one generate_batch. Fail FAST on quota exhaustion (fatal — abort the
    whole run); retry only briefly on genuinely transient errors."""
    for attempt in range(tries):
        _LIMITER.acquire()          # global RPM gate — every request counts
        try:
            return g.generate_batch(cp, batch_size, rng, model)
        except Exception as e:
            if _is_quota_error(e):
                raise QuotaExhausted(str(e)[:300]) from e
            if attempt == tries - 1:
                return []
            time.sleep(min(2 ** attempt + rng.random(), 15))
    return []


def robust_generate(cp, model, workers, batch_size, max_batches, seen,
                    stop_positive, per_tool, on_batch=None):
    """Submit batches at low concurrency with retry until `stop_positive`
    unique positive examples are collected (or `max_batches` exhausted).

    `seen` is a shared set of normalized queries for cross-run dedup; `per_tool`
    a Counter of positive answers by tool name (both mutated in place).
    Returns the list of new, deduped example dicts.
    """
    out = []
    seed_iter = itertools.count(random.randint(0, 2 ** 31))
    submitted = 0
    n_pos = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def _submit():
            nonlocal submitted
            r = random.Random(next(seed_iter))
            pending.add(pool.submit(_batch_with_retry, cp, batch_size, r, model))
            submitted += 1

        for _ in range(min(workers, max_batches)):
            _submit()

        while pending:
            done, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for f in done:
                try:
                    batch = f.result()
                except QuotaExhausted as e:
                    for p in pending:
                        p.cancel()
                    raise QuotaExhausted(
                        "Gemini quota/rate limit exhausted — stopping. "
                        "Progress so far IS saved. Options: wait for the daily "
                        "free-tier reset, enable billing on the API key, or run "
                        f"fewer tools. (api said: {e})") from None
                for ex in batch:
                    key = _norm(ex["query"])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(ex)
                    try:
                        calls = json.loads(ex["answers"])
                    except (ValueError, TypeError):
                        calls = []
                    if calls:
                        n_pos += 1
                        for c in calls:
                            if isinstance(c, dict) and c.get("name"):
                                per_tool[c["name"]] += 1
                if on_batch:
                    on_batch(len(out), n_pos, per_tool)
                if n_pos < stop_positive and submitted < max_batches:
                    _submit()
    return out


def _append(path, examples):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for ex in examples:
            f.write(json.dumps(
                {"query": ex["query"], "tools": ex["tools"], "answers": ex["answers"]},
                ensure_ascii=False) + "\n")


def _existing_state(path):
    """Load normalized queries + per-tool positive counts already on disk."""
    seen, per_tool = set(), Counter()
    if not os.path.exists(path):
        return seen, per_tool
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                seen.add(_norm(ex["query"]))
                for c in json.loads(ex["answers"]):
                    if isinstance(c, dict) and c.get("name"):
                        per_tool[c["name"]] += 1
            except (ValueError, TypeError, KeyError):
                pass
    return seen, per_tool


def _single_only_call_types():
    singles = [t for t in PICKO_CALL_TYPES if t[0] == "single"]
    return singles or [("single", "exactly 1 tool call")]


def _inject_distractors(examples, target_name, catalog, seed=0, lo=2, hi=6):
    """Rewrite each example's offered `tools` as {target + a few distractors} from
    the catalog, so a forced single-tool example still trains a realistic
    multi-tool context. The answer (which calls `target_name`) is unchanged."""
    by = {t["name"]: t for t in catalog}
    others = [n for n in by if n != target_name]
    rng = random.Random(seed)
    out = []
    for ex in examples:
        k = min(rng.randint(lo, hi), len(others))
        pick = rng.sample(others, k) if k else []
        offered = [by[target_name]] + [by[n] for n in pick]
        rng.shuffle(offered)
        ex = dict(ex)
        ex["tools"] = json.dumps(offered, separators=(",", ":"), ensure_ascii=False)
        out.append(ex)
    return out


def force_tool(cp, tool_dict, catalog, need, seen, per_tool, model,
               workers=2, batch_size=12, max_batches=400, add_distractors=True,
               on_batch=None):
    """Generate `need` examples GUARANTEED to call `tool_dict`.

    Trick: offer ONLY this tool with call_type forced to `single`, so Gemini must
    write a query answerable by it (niche tools no longer get starved). Then
    optionally re-inject distractor tools into the context for realistic training.
    Mutates `seen` (dedup) and `per_tool` (adds this tool's new count). Sequential
    per tool — safe to mutate the generator's module globals here.
    """
    if need <= 0:
        return []
    _apply_patches([[tool_dict]])
    g.CALL_TYPES = _single_only_call_types()
    g._OVERLAP_PAIRS = []          # only the target is offered
    local = Counter()
    got = robust_generate(cp, model, workers, batch_size, max_batches,
                          seen, need, local, on_batch=on_batch)
    keep = []
    for ex in got:                 # safety: keep only true target calls
        try:
            calls = json.loads(ex["answers"])
        except (ValueError, TypeError):
            continue
        if any(isinstance(c, dict) and c.get("name") == tool_dict["name"] for c in calls):
            keep.append(ex)
    if add_distractors:
        keep = _inject_distractors(keep, tool_dict["name"], catalog)
    per_tool[tool_dict["name"]] += len(keep)
    return keep


def balance_dataset(in_path, out_path, n_per_tool=120, max_negatives=None,
                    keep_tools=None, seed=0):
    """Trim a JSONL to EXACTLY `n_per_tool` positive examples per tool (random
    subsample of whatever exists). Optionally cap abstention negatives. Writes
    `out_path` and returns a coverage report (flags any tool still short).

    `keep_tools` (iterable of names) scopes the output to just those tools —
    examples for any other tool are dropped, and `short` is reported only for the
    kept set. Use it to match the balancer to your SELECTION so out-of-scope tools
    in the file don't muddy the report.
    """
    keep = set(keep_tools) if keep_tools is not None else None
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(in_path) if l.strip()]
    by_tool, negatives = {}, []
    for ex in rows:
        try:
            calls = json.loads(ex["answers"])
        except (ValueError, TypeError):
            calls = []
        prim = next((c["name"] for c in calls
                     if isinstance(c, dict) and c.get("name")), None)
        if prim is None:
            negatives.append(ex)
        elif keep is None or prim in keep:
            by_tool.setdefault(prim, []).append(ex)
    out, short = [], {}
    targets = keep if keep is not None else set(by_tool)
    for name in targets:
        exs = by_tool.get(name, [])
        rng.shuffle(exs)
        out.extend(exs[:n_per_tool])
        if len(exs) < n_per_tool:
            short[name] = len(exs)
    if max_negatives is not None:
        rng.shuffle(negatives)
        negatives = negatives[:max_negatives]
    out.extend(negatives)
    rng.shuffle(out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for ex in out:
            f.write(json.dumps({"query": ex["query"], "tools": ex["tools"],
                                "answers": ex["answers"]}, ensure_ascii=False) + "\n")
    return {"per_tool": {k: min(len(v), n_per_tool) for k, v in by_tool.items()},
            "short": short, "negatives": len(negatives), "total": len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-jsonl", type=str, default=os.path.join(HERE, "data", "picko_subset.jsonl"))
    ap.add_argument("--workers", type=int, default=3,
                    help="Concurrency. Keep low (2-4) for a single free-tier key "
                         "to avoid 429 rate-limit failures.")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--model", type=str, default=g.MODEL)
    ap.add_argument("--target-per-tool", type=int, default=140,
                    help="Generate until every tool has >= this many positive "
                         "examples (needle needs >=120).")
    ap.add_argument("--max-batches", type=int, default=400,
                    help="Hard cap on Gemini calls per phase (cost guard).")
    ap.add_argument("--focus", type=str, default=None,
                    help="Comma-separated clusters to restrict the pool "
                         "(pubmed,arxiv,wikipedia,huggingface). Skips auto top-up.")
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey\n"
                 "  export GEMINI_API_KEY=your_key_here")
    if not os.path.exists(POOLS_JSON):
        sys.exit("data/scientific_pools.json missing. Run scripts/make_scientific_pools.py first.")

    full_pools = _load_pools()  # all clusters, name -> cluster map
    cluster_of = {}
    with open(POOLS_JSON) as f:
        for cluster, tools in json.load(f).items():
            for t in tools:
                cluster_of[t["name"]] = cluster
    all_tool_names = list(cluster_of)

    cp = g.ClientPool(g.make_clients())
    seen, per_tool = _existing_state(args.output_jsonl)
    tgt = args.target_per_tool
    print(f"Existing on disk: {sum(per_tool.values())} positives; per-tool {dict(per_tool)}")

    _batches = itertools.count(1)

    def _progress(n_new, n_pos, pt):
        i = next(_batches)
        under = [t for t in all_tool_names if pt[t] < tgt]
        # Newline-terminated + flushed so the log is watchable live.
        print(f"  [batch {i:3}] new={n_new:4} positives={n_pos:4} "
              f"under_{tgt}={len(under)} "
              f"lowest={min((pt[t] for t in all_tool_names), default=0)}",
              flush=True)

    # --- Phase 1: broad pass over all clusters ---
    pools = _load_pools(args.focus)
    _apply_patches(pools)
    print(f"Pools: {[[t['name'] for t in p] for p in pools]}")
    print(f"Model: {args.model}  workers={args.workers}  target/tool={tgt}")
    print("Phase 1 (broad): generating ...")
    # stop when total positives reach ~ n_tools * tgt (rough; top-ups fix skew)
    need = max(0, len(all_tool_names) * tgt - sum(per_tool.values()))
    new = robust_generate(cp, args.model, args.workers, args.batch_size,
                           args.max_batches, seen, need, per_tool, _progress)
    _append(args.output_jsonl, new)
    print(f"\n  phase 1 added {len(new)} examples")

    # --- Phase 2: focused top-ups for any tool still under target ---
    if not args.focus:
        for _ in range(4):  # a few focus rounds
            under = sorted(t for t in all_tool_names if per_tool[t] < tgt)
            if not under:
                break
            clusters = sorted({cluster_of[t] for t in under})
            print(f"\nPhase 2 top-up: under-target {under} -> focus {clusters}")
            _apply_patches(_load_pools(",".join(clusters)))
            need = max(len(under) * tgt - sum(per_tool[t] for t in under), tgt)
            new = robust_generate(cp, args.model, args.workers, args.batch_size,
                                   args.max_batches // 2, seen, need, per_tool, _progress)
            _append(args.output_jsonl, new)
            print(f"\n  top-up added {len(new)} examples")
            if not new:
                break

    print(f"\nFinal per-tool positive counts (on disk):")
    for name in sorted(all_tool_names, key=lambda n: per_tool[n]):
        flag = "  ⚠ <120" if per_tool[name] < 120 else ""
        print(f"  {name:28} {per_tool[name]}{flag}")
    print(f"Total positives: {sum(per_tool.values())}. Output: {args.output_jsonl}")


if __name__ == "__main__":
    main()
