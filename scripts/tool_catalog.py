#!/usr/bin/env python3
"""Load and slice the PICKO scientific tool catalog.

Source of truth: `full_tools_53tools_11products.json` (75 tools, 11 families).
Grouping/metadata: `tools_metadata.csv` (family -> category, arg counts).

A "family" (aka product/source) is identified by the tool-name prefix — e.g.
`arxiv_search_papers` -> `arxiv`, `semantic_scholar_search_papers` -> `semantic_scholar`.
Families map up to 4 higher-level categories via the CSV.

Typical use:
    from scripts.tool_catalog import Catalog
    cat = Catalog()
    cat.list_families()                      # {family: {count, category, ...}}
    tools, pools = cat.select_tools(k=11)     # the 11-per-family D3 set
    tools, pools = cat.select_tools(families=["arxiv", "pubmed"])
    tools, pools = cat.select_tools(categories=["Academic & Literature Search"])
`pools` is a list-of-lists (one inner list per family) — the shape
`generate_picko_data._apply_patches` expects.
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER_JSON = os.path.join(HERE, "full_tools_53tools_11products.json")
ELEVEN_JSON = os.path.join(HERE, "11tools_11products.json")
METADATA_CSV = os.path.join(HERE, "tools_metadata.csv")

# Known family prefixes (longest-match first so `semantic_scholar` beats `semantic`).
KNOWN_FAMILIES = [
    "semantic_scholar", "pubmed", "arxiv", "wikipedia", "kaggle", "crossref",
    "openalex", "unpaywall", "google", "github", "hf",
]


def family_of(tool_name):
    """Map a tool name to its family prefix (longest known prefix wins)."""
    for fam in sorted(KNOWN_FAMILIES, key=len, reverse=True):
        if tool_name == fam or tool_name.startswith(fam + "_"):
            return fam
    return tool_name.split("_")[0]  # fallback


def _first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 0


def _compact(tool):
    """A tool dict reduced to {name, description} — no `parameters`. Used to fit
    many tool names in the encoder window when only tool selection is measured."""
    return {"name": tool.get("name", ""),
            "description": tool.get("description", "")}


class Catalog:
    def __init__(self, master=MASTER_JSON, metadata=METADATA_CSV):
        with open(master) as f:
            self.tools = json.load(f)
        self.by_name = {t["name"]: t for t in self.tools}

        # family -> category / source label, from the CSV's representative rows
        self.family_category = {}
        self.family_source = {}
        if os.path.exists(metadata):
            with open(metadata, newline="") as f:
                for row in csv.DictReader(f):
                    # Messy CSV: stray commas make DictReader overflow into a
                    # list under key None — keep only real string columns.
                    row = {(k or "").strip(): (v or "").strip()
                           for k, v in row.items()
                           if isinstance(k, str) and isinstance(v, str)}
                    tname = row.get("tool name", "")
                    if not tname:
                        continue
                    fam = family_of(tname)
                    self.family_category[fam] = row.get("category", "")
                    self.family_source[fam] = row.get("source name", "")

    # ---- introspection ----
    def family_of(self, tool_name):
        return family_of(tool_name)

    def category_of(self, family):
        return self.family_category.get(family, "Uncategorized")

    def params_of(self, tool_name):
        params = self.by_name[tool_name].get("parameters", {}) or {}
        total = len(params)
        required = sum(1 for p in params.values()
                       if isinstance(p, dict) and p.get("required"))
        return required, total

    def list_families(self):
        """{family: {count, category, source, tools:[...]}}, sorted by size desc."""
        fams = {}
        for t in self.tools:
            fam = family_of(t["name"])
            d = fams.setdefault(fam, {"count": 0, "tools": [],
                                      "category": self.category_of(fam),
                                      "source": self.family_source.get(fam, fam)})
            d["count"] += 1
            d["tools"].append(t["name"])
        return dict(sorted(fams.items(), key=lambda kv: -kv[1]["count"]))

    def as_dataframe(self):
        """One row per tool: name, family, category, required/total params."""
        import pandas as pd
        rows = []
        for t in self.tools:
            fam = family_of(t["name"])
            req, tot = self.params_of(t["name"])
            rows.append({"tool": t["name"], "family": fam,
                         "category": self.category_of(fam),
                         "required_params": req, "total_params": tot})
        return pd.DataFrame(rows)

    # ---- selection ----
    def select_tools(self, families=None, categories=None, names=None, k=None,
                     one_per_family=False):
        """Return (flat_tools, pools) for a chosen slice.

        - names:          explicit tool names.
        - families:       whole families by prefix.
        - categories:     whole categories (via CSV mapping).
        - one_per_family: pick the representative tool per family (the 11-set).
        - k:              cap the number of tools (after the above filters).
        `pools` groups the chosen tools by family (list of lists).
        """
        if names:
            chosen = [self.by_name[n] for n in names if n in self.by_name]
        elif one_per_family:
            with open(ELEVEN_JSON) as f:
                reps = json.load(f)
            chosen = [self.by_name.get(t["name"], t) for t in reps]
        else:
            chosen = list(self.tools)
            if families:
                fset = set(families)
                chosen = [t for t in chosen if family_of(t["name"]) in fset]
            if categories:
                cset = set(categories)
                chosen = [t for t in chosen
                          if self.category_of(family_of(t["name"])) in cset]

        if k is not None:
            chosen = chosen[:k]

        # group into pools by family, preserving order
        pools_map = {}
        for t in chosen:
            pools_map.setdefault(family_of(t["name"]), []).append(t)
        return chosen, list(pools_map.values())

    # ---- dataset re-scoping ----
    def restrict_dataset(self, examples, names, offer_all_max=12,
                         keep_negatives=True, neg_offer=8, seed=0,
                         cap_per_tool=None, compact=False,
                         tokenizer=None, max_tokens=900):
        """Re-scope generated examples to a study subset `names`.

        Generated data offers `gold + random distractors from all 75 tools`; to
        study a specific subset we must (a) keep only rows whose PRIMARY gold tool
        is in `names`, and (b) rewrite each row's offered `tools` to contain ONLY
        tools from `names` — otherwise training/eval happen over a tool set that
        isn't the one under study (muddies Breadth/Separation).

        Offered-tools policy:
          - len(names) <= offer_all_max -> offer ALL subset tools (shuffled), so
            every example sees every look-alike (clean Separation);
          - otherwise                   -> gold + sampled distractors from the
            subset, capped at offer_all_max.
        Negatives (empty gold) are kept if `keep_negatives`, offering a random
        `neg_offer`-sized sample of the subset. Answers are untouched.

        Knobs:
          cap_per_tool  subsample positives to at most N per gold tool (faster
                        training). Negatives are capped to the same N.
          compact       offer tool dicts reduced to {name, description} (drop
                        `parameters`) — used for the Breadth study so many tool
                        names fit the encoder window.
          tokenizer     if given, greedily drop distractors (never the gold tool)
            + max_tokens until the offered-tools JSON fits `max_tokens`, so
                        training/eval are never silently truncated by the 1024-tok
                        encoder.

        Returns a new list of {"query","tools","answers"} dicts. Deterministic
        given `seed`.
        """
        import random
        names = list(dict.fromkeys(names))                 # dedupe, keep order
        subset = [n for n in names if n in self.by_name]
        missing = [n for n in names if n not in self.by_name]
        if missing:
            raise KeyError(f"unknown tool names (not in catalog): {missing}")
        subset_set = set(subset)
        offer_all = len(subset) <= offer_all_max
        rng = random.Random(seed)

        def _gold(ex):
            try:
                calls = json.loads(ex.get("answers", "[]"))
            except (ValueError, TypeError):
                return None
            return next((c["name"] for c in calls
                         if isinstance(c, dict) and c.get("name")), None)

        def _dump(offered_names):
            tools = [(_compact(self.by_name[n]) if compact else self.by_name[n])
                     for n in offered_names]
            return json.dumps(tools, separators=(",", ":"), ensure_ascii=False)

        def _tools_field(offered_names, gold):
            offered = list(offered_names)
            rng.shuffle(offered)
            if tokenizer is not None:
                # drop trailing distractors (keep gold) until the JSON fits
                while len(offered) > 1 and \
                        len(tokenizer.encode(_dump(offered))) > max_tokens:
                    drop = next((n for n in reversed(offered) if n != gold), None)
                    if drop is None:
                        break
                    offered.remove(drop)
            return _dump(offered)

        # bucket positives (for cap_per_tool) and collect negatives
        pos_by_tool, negatives = {}, []
        for ex in examples:
            gold = _gold(ex)
            if gold is None:
                negatives.append(ex)
            elif gold in subset_set:
                pos_by_tool.setdefault(gold, []).append(ex)
        for exs in pos_by_tool.values():
            rng.shuffle(exs)
        if cap_per_tool is not None:
            pos_by_tool = {g: exs[:cap_per_tool] for g, exs in pos_by_tool.items()}
            rng.shuffle(negatives)
            negatives = negatives[:cap_per_tool]

        out = []
        for gold, exs in pos_by_tool.items():
            for ex in exs:
                if offer_all:
                    offered = list(subset)
                else:
                    others = [n for n in subset if n != gold]
                    k = min(offer_all_max - 1, len(others))
                    offered = [gold] + (rng.sample(others, k) if k else [])
                out.append({"query": ex["query"],
                            "tools": _tools_field(offered, gold),
                            "answers": ex["answers"]})
        if keep_negatives:
            for ex in negatives:
                k = min(neg_offer, len(subset))
                offered = rng.sample(subset, k) if k else []
                out.append({"query": ex["query"],
                            "tools": _tools_field(offered, None),
                            "answers": ex["answers"]})
        rng.shuffle(out)
        return out


if __name__ == "__main__":
    cat = Catalog()
    print(f"{len(cat.tools)} tools, {len(cat.list_families())} families")
    for fam, d in cat.list_families().items():
        print(f"  {fam:18} n={d['count']:2}  [{d['category']}]")
    flat, pools = cat.select_tools(one_per_family=True)
    print(f"\none-per-family set: {len(flat)} tools -> {[t['name'] for t in flat]}")
