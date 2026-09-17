#!/usr/bin/env python3
"""PICKO research configuration: the 40-tool focus, the Breadth size sweep, the
curated Separation (disambiguation) groups, and the Depth parameter buckets.

Kept separate from the generic `tool_catalog.py` because these are hand-picked
choices for the final study, not catalog facts. Every tool name here is verified
to exist in the 40-tool focus (families arxiv, hf, wikipedia, pubmed).

    from scripts.research_sets import (FOCUS_FAMILIES, focus_names, BREADTH_SIZES,
                                       nested_sets, SIMILAR_GROUPS, param_bucket)
"""
import random

# ---- Focus (the "Original 40") ----
FOCUS_FAMILIES = ["arxiv", "hf", "wikipedia", "pubmed"]


def focus_names(cat):
    """The 40 tool names of the focus families, in catalog order."""
    tools, _ = cat.select_tools(families=FOCUS_FAMILIES)
    return [t["name"] for t in tools]


# ---- Breadth (Tool-Set Size) ----
BREADTH_SIZES = [3, 5, 10, 20, 30]


def nested_sets(names, sizes=BREADTH_SIZES, seed=0):
    """Return {size: [names]} where each smaller set is a strict subset of the
    next (one shuffle, then prefixes). `sizes` are clamped to len(names)."""
    pool = list(names)
    random.Random(seed).shuffle(pool)
    out = {}
    for k in sizes:
        out[k] = pool[:min(k, len(pool))]
    return out


def breadth_pool(cat, focus, seed=0):
    """Ordered tool pool for the Breadth sweep: the `focus` tools (shuffled) first,
    then every other catalog tool (shuffled). Taking prefixes keeps small sizes
    inside the focus while sizes > len(focus) extend to the full 75-tool catalog."""
    rng = random.Random(seed)
    focus = list(focus)
    rng.shuffle(focus)
    others = [t["name"] for t in cat.tools if t["name"] not in set(focus)]
    rng.shuffle(others)
    return focus + others


def size_sets(pool, sizes=BREADTH_SIZES):
    """Nested prefixes of an already-ordered pool: {k: pool[:k]} (no reshuffle)."""
    return {k: pool[:min(k, len(pool))] for k in sizes}


# ---- Separation (Disambiguation) — curated look-alike groups ----
# Each group is a set of tools that do a very similar thing (same action across
# sources, or same source with subtly different actions) and are therefore the
# hardest to tell apart. All names are within the 40-tool focus.
SIMILAR_GROUPS = {
    # same action ("search"), different source
    "cross_source_search": [
        "arxiv_search_papers", "pubmed_search_articles",
        "hf_paper_search", "wikipedia_search_wikipedia",
    ],
    # same source (HuggingFace), search over different resource types
    "hf_search_variants": [
        "hf_hub_repo_search", "hf_model_search", "hf_dataset_search",
        "hf_space_search", "hf_paper_search",
    ],
    # retrieve full paper content (abstract / read / download)
    "get_paper_content": [
        "arxiv_get_abstract", "arxiv_read_paper", "arxiv_download_paper",
        "pubmed_download_article",
    ],
    # arxiv LaTeX sub-family (list vs get vs get-section)
    "arxiv_latex": [
        "arxiv_get_paper_latex", "arxiv_list_paper_latex_sections",
        "arxiv_get_paper_latex_section",
    ],
    # wikipedia: raw retrieval vs summarization
    "wikipedia_retrieve_vs_summarize": [
        "wikipedia_get_article", "wikipedia_get_summary",
        "wikipedia_summarize_article_for_query",
        "wikipedia_summarize_article_section", "wikipedia_extract_key_facts",
    ],
    # short single-item summary across sources
    "single_item_summary": [
        "arxiv_get_abstract", "pubmed_get_article_summaries",
        "wikipedia_get_summary",
    ],
}


# ---- Depth (Parameter Complexity) ----
def param_bucket(total):
    """Bucket a tool by its total parameter count."""
    if total <= 0:
        return "0"
    if total == 1:
        return "1"
    if total <= 3:
        return "2-3"
    return "4+"


PARAM_BUCKET_ORDER = ["0", "1", "2-3", "4+"]


def nargs_bucket(k):
    """Bucket an example by how many arguments its gold answer actually supplies.
    This is what the extraction metrics grade against (unlike the tool's total/
    required schema size), so it is the honest per-example difficulty axis."""
    if k <= 0:
        return "0"
    if k == 1:
        return "1"
    if k == 2:
        return "2"
    return "3+"


NARGS_BUCKET_ORDER = ["0", "1", "2", "3+"]


def gold_n_args(ex):
    """Number of arguments the gold (primary) call in `ex` supplies; 0 if none."""
    import json
    try:
        calls = json.loads(ex.get("answers", "[]"))
    except (ValueError, TypeError):
        return 0
    primary = next((c for c in calls
                    if isinstance(c, dict) and c.get("name")), None)
    if not primary:
        return 0
    args = primary.get("arguments", {})
    return len(args) if isinstance(args, dict) else 0


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.tool_catalog import Catalog
    cat = Catalog()
    fn = focus_names(cat)
    print(f"focus: {len(fn)} tools")
    bad = {n for g in SIMILAR_GROUPS.values() for n in g if n not in cat.by_name}
    print("unknown group tools:", bad or "none")
    outside = {n for g in SIMILAR_GROUPS.values() for n in g if n not in set(fn)}
    print("group tools outside focus:", outside or "none")
    print("nested:", {k: len(v) for k, v in nested_sets(fn).items()})
