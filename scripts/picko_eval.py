#!/usr/bin/env python3
"""Richer PICKO evaluation: separate TOOL-SELECTION from PARAMETER-EXTRACTION,
broken down per-tool and per-family, plus a D3 confusion matrix and D1
distractor-scaling. Reuses needle's batched greedy decoder.

Metrics returned by `evaluate`:
  selection_acc   fraction of (non-abstention) examples whose predicted primary
                  tool name == the reference primary tool name        (tool function)
  name_f1         set-based tool-name F1                              (tool function)
  args_exact_acc  among selection-correct examples, fraction with the exact
                  argument dict                                       (parameter extraction)
  param_f1        micro per-parameter F1 over reference args          (parameter extraction, finer)
  call_exact      name right AND args exact                           (full call)
  parse_rate      fraction of predictions that were valid JSON
  abstain_acc     among abstention refs ([]), fraction predicted empty
  per_tool        {tool: {n, selection, args_exact, p_tp,p_fp,p_fn}}
  per_family      same, grouped by family

Use:
    from scripts.picko_eval import load_model, predict, evaluate, confusion, base_checkpoint
    model, params, tok = load_model(ckpt)
    preds = predict(model, params, tok, test)          # cached decode
    m = evaluate(test, preds, family_of=family_of)
"""
import json
import os
import re

from needle.model.run import load_checkpoint, generate_batch
from needle.model.architecture import SimpleAttentionNetwork
from needle.dataset.dataset import get_tokenizer


# ---------- model / io ----------
def base_checkpoint():
    from needle.training.finetune import _resolve_checkpoint
    return _resolve_checkpoint(None)  # downloads base if absent


def load_model(ckpt_path):
    params, config = load_checkpoint(ckpt_path)
    return SimpleAttentionNetwork(config), params, get_tokenizer()


def _parse(text):
    try:
        v = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(v, dict):
        v = [v]
    return v if isinstance(v, list) else []


_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _salvage(text):
    """Recover tool-call names from truncated/invalid JSON (regex on `"name":"X"`).
    Lets us cap decode length short — the name appears at the very start of the
    output — and still score tool SELECTION correctly. Arguments are left empty
    (a truncated call has no reliable args), so parameter metrics are unaffected."""
    return [{"name": n, "arguments": {}} for n in _NAME_RE.findall(text or "")]


def _calls(text):
    """Strict JSON parse, falling back to name salvage when it's unparseable."""
    v = _parse(text)
    return v if v is not None else _salvage(text)


def predict(model, params, tok, examples, batch=32, max_gen_len=512, max_enc_len=1024,
            tools_override=None, progress=None):
    """Greedy-decode predictions. `tools_override` (list[str] parallel to examples)
    swaps the offered tools per example (used for D1 distractor scaling)."""
    outs = []
    n = len(examples)
    for i in range(0, n, batch):
        chunk = examples[i:i + batch]
        tools = (tools_override[i:i + batch] if tools_override
                 else [e["tools"] for e in chunk])
        texts = generate_batch(model, params, tok,
                               [e["query"] for e in chunk], tools,
                               max_gen_len=max_gen_len, max_enc_len=max_enc_len,
                               constrained=True)
        outs.extend(texts)
        if progress:
            progress(min(i + batch, n), n)
    return outs


# ---------- metrics ----------
def _primary(calls):
    for c in calls:
        if isinstance(c, dict) and c.get("name"):
            return c
    return None


def _args(c):
    a = c.get("arguments", {}) if isinstance(c, dict) else {}
    return a if isinstance(a, dict) else {}


def evaluate(examples, pred_texts, family_of=None):
    fam = family_of or (lambda n: n.split("_")[0])
    per_tool, per_family = {}, {}

    def slot(d, key):
        return d.setdefault(key, {"n": 0, "selection": 0, "args_exact": 0,
                                  "p_tp": 0, "p_fp": 0, "p_fn": 0})

    n = sel_ok = args_ok_denom = args_ok = parse_ok = 0
    call_exact = 0
    abstain_n = abstain_ok = 0
    name_tp = name_fp = name_fn = 0
    p_tp = p_fp = p_fn = 0

    for ex, ptext in zip(examples, pred_texts):
        ref = _parse(ex.get("answers", "[]")) or []
        pred_strict = _parse(ptext)
        if pred_strict is not None:
            parse_ok += 1
            pred = pred_strict
        else:
            pred = _salvage(ptext)   # recover the name from truncated JSON

        # abstention examples (empty reference)
        if not _primary(ref):
            abstain_n += 1
            if not _primary(pred):
                abstain_ok += 1
            continue

        n += 1
        rp, pp = _primary(ref), _primary(pred)
        rname = rp["name"]
        pname = pp["name"] if pp else None

        # set-based name F1
        rnames = {c["name"] for c in ref if isinstance(c, dict) and c.get("name")}
        pnames = {c["name"] for c in pred if isinstance(c, dict) and c.get("name")}
        name_tp += len(pnames & rnames)
        name_fp += len(pnames - rnames)
        name_fn += len(rnames - pnames)

        tslot = slot(per_tool, rname)
        fslot = slot(per_family, fam(rname))
        tslot["n"] += 1
        fslot["n"] += 1

        selected = (pname == rname)
        if selected:
            sel_ok += 1
            tslot["selection"] += 1
            fslot["selection"] += 1

            # parameter extraction — only meaningful once the tool is right
            rargs, pargs = _args(rp), _args(pp)
            args_ok_denom += 1
            exact = (json.dumps(rargs, sort_keys=True) == json.dumps(pargs, sort_keys=True))
            if exact:
                args_ok += 1
                call_exact += 1
                tslot["args_exact"] += 1
                fslot["args_exact"] += 1
            # per-parameter tp/fp/fn (value must match)
            for k, v in rargs.items():
                if k in pargs and json.dumps(pargs[k], sort_keys=True) == json.dumps(v, sort_keys=True):
                    p_tp += 1; tslot["p_tp"] += 1; fslot["p_tp"] += 1
                else:
                    p_fn += 1; tslot["p_fn"] += 1; fslot["p_fn"] += 1
            for k in pargs:
                if k not in rargs:
                    p_fp += 1; tslot["p_fp"] += 1; fslot["p_fp"] += 1

    def f1(tp, fp, fn):
        return round(2 * tp / max(2 * tp + fp + fn, 1), 4)

    def finalize(d):
        for s in d.values():
            s["selection_acc"] = round(s["selection"] / max(s["n"], 1), 4)
            s["args_exact_acc"] = round(s["args_exact"] / max(s["selection"], 1), 4)
            s["param_f1"] = f1(s["p_tp"], s["p_fp"], s["p_fn"])
        return d

    return {
        "n": n,
        "selection_acc": round(sel_ok / max(n, 1), 4),
        "name_f1": f1(name_tp, name_fp, name_fn),
        "args_exact_acc": round(args_ok / max(args_ok_denom, 1), 4),
        "param_f1": f1(p_tp, p_fp, p_fn),
        "call_exact": round(call_exact / max(n, 1), 4),
        "parse_rate": round(parse_ok / max(len(examples), 1), 4),
        "abstain_acc": round(abstain_ok / max(abstain_n, 1), 4) if abstain_n else None,
        "abstain_n": abstain_n,
        "per_tool": finalize(per_tool),
        "per_family": finalize(per_family),
    }


def evaluate_per_example(examples, pred_texts):
    """One scored row per non-abstention example — the raw material for bucketing
    the test set by any per-example property (e.g. #args). Mirrors `evaluate`'s
    logic but keeps the per-example counts instead of aggregating.

    Row: {ref_tool, selected, args_exact, p_tp, p_fp, p_fn}
      selected    predicted primary name == reference primary name
      args_exact  (only if selected) predicted args dict == reference args dict
      p_tp/fp/fn  (only if selected) per-parameter value-match counts, so a bucket's
                  param_f1 is a micro-average: f1(sum tp, sum fp, sum fn).
    """
    rows = []
    for ex, ptext in zip(examples, pred_texts):
        ref = _parse(ex.get("answers", "[]")) or []
        rp = _primary(ref)
        if not rp:                                   # abstention example — skip
            continue
        strict = _parse(ptext)
        pred = strict if strict is not None else _salvage(ptext)
        pp = _primary(pred)
        selected = bool(pp) and pp["name"] == rp["name"]
        row = {"ref_tool": rp["name"], "selected": int(selected),
               "args_exact": 0, "p_tp": 0, "p_fp": 0, "p_fn": 0}
        if selected:
            rargs, pargs = _args(rp), _args(pp)
            row["args_exact"] = int(
                json.dumps(rargs, sort_keys=True) == json.dumps(pargs, sort_keys=True))
            for k, v in rargs.items():
                if k in pargs and json.dumps(pargs[k], sort_keys=True) == json.dumps(v, sort_keys=True):
                    row["p_tp"] += 1
                else:
                    row["p_fn"] += 1
            row["p_fp"] += sum(1 for k in pargs if k not in rargs)
        rows.append(row)
    return rows


def confusion(examples, pred_texts):
    """{ref_tool: {pred_tool_or_'∅'/'?': count}} over non-abstention examples (D3)."""
    mat = {}
    for ex, ptext in zip(examples, pred_texts):
        ref = _parse(ex.get("answers", "[]")) or []
        rp = _primary(ref)
        if not rp:
            continue
        strict = _parse(ptext)
        pred = strict if strict is not None else _salvage(ptext)
        pp = _primary(pred)
        pname = pp["name"] if pp else ("∅" if strict is not None else "?")
        mat.setdefault(rp["name"], {}).setdefault(pname, 0)
        mat[rp["name"]][pname] += 1
    return mat


def tools_token_len(tools, tokenizer):
    """Token length of a `tools` field (JSON string or list of tool dicts)."""
    if not isinstance(tools, str):
        tools = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    return len(tokenizer.encode(tools))


def n_visible(query, tools, tokenizer, max_enc_len=1024):
    """How many of the offered tools survive the encoder's truncation, i.e. how
    many appear before `[query, <tools>, schemas...]` is cut to `max_enc_len`.
    Mirrors `needle.model.run._build_encoder_input`. `tools` is a list of dicts."""
    q_toks = tokenizer.encode(query)
    max_query = max_enc_len - 2
    q_toks = q_toks[:max_query]
    budget = max_enc_len - len(q_toks) - 1          # tokens left for tool schemas
    visible = 0
    for i in range(len(tools)):
        s = json.dumps(tools[:i + 1], separators=(",", ":"), ensure_ascii=False)
        if len(tokenizer.encode(s)) > budget:
            break
        visible = i + 1
    return visible


def build_tools_override(examples, k, catalog, seed=0):
    """For D1: rebuild each example's offered tools as {gold + (k-1) distractors}
    drawn from the full catalog. `catalog` is a list of tool dicts."""
    import random
    rng = random.Random(seed)
    by_name = {t["name"]: t for t in catalog}
    all_names = list(by_name)
    overrides = []
    for ex in examples:
        ref = _parse(ex.get("answers", "[]")) or []
        gold = [c["name"] for c in ref if isinstance(c, dict) and c.get("name")]
        gold = gold[:1] or []
        pool = [nm for nm in all_names if nm not in gold]
        rng.shuffle(pool)
        chosen = gold + pool[:max(0, k - len(gold))]
        rng.shuffle(chosen)
        overrides.append(json.dumps([by_name[nm] for nm in chosen if nm in by_name],
                                     separators=(",", ":")))
    return overrides
