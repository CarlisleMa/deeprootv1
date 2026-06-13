"""Eval 1 — Lead Hit Identification (recovery-style, head-to-head).

Question: when reading a multi-source passage from the Shen Nong Ben Cao Jing,
does deeproot's KG-grounded pipeline (TaskAValidator + CriticAgent + breadth)
recover more sources and propose better-aligned therapeutic leads than the
same Gemini model given only the raw text?

KG arm pipeline (per region):
  1. Raw Gemini extracts (source, malady) pairs from the passage.
  2. Resolve LLM source names → canonical KG names via name_match.
  3. Per (source, malady) — TaskAValidator.run(): deterministic graph walk
     produces a ClaimVerdict with paths_top (Compound→Target→Disease chains
     with bucket/score/loop_closed) + anchor/corroboration fields.
  4. CriticAgent.run(verdicts=[...]): Pass-2 LLM critic reads the validator's
     enriched ClaimVerdict + auto-pulled enrichment (KNOWN_TREATS, target
     genericity / spectrum, target convergence in source, sibling sources,
     other-maladies-this-source-treats), emits CriticVerdict with
     biological_plausibility, evidence_coherence, key_evidence list
     (compound, target, reached_disease, why_compelling).
  5. Breadth pass — direct Cypher to enumerate compounds extracted from
     each resolved source, ranked by target richness (catches compounds the
     critic didn't flag).
  6. Final lead ranking: critic key_evidence first (score 0.5..1.0 from
     plausibility+coherence), then validator paths_top compounds (0.25..0.5
     from path_score+loop_closed), then breadth (0.0..0.25 from target_count).

Baseline arm: same Gemini model, given only the passage, asked for sources
+ up to 10 compounds per source + per-compound plausibility (no KG hint).

Both arms produce: a source list and a ranked compound list. Compared on:
  - Source recovery: precision / recall / F1.
  - Compound recovery: lenient recall@k (any gold) and strict per-source
    recall@k (compound counted only if attributed to the right source).
  - Plausibility: KG = mean critic biological_plausibility across pairs;
    baseline = mean per-compound plausibility. Plus symmetric Brier.

Single Gemini model for extraction + baseline (sourced from .env via
GEMINI_MODEL). The critic uses GEMINI_MODEL_PRO by default; pass
--critic-model to override. Validator is deterministic — no LLM.

Usage:
    python scripts/eval_lead_recovery.py [--corpus PATH] [--limit N]
    python scripts/eval_lead_recovery.py --plot data/eval/results/eval1_open_*.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._eval_utils import (
    DEFAULT_FLASH_MODEL,
    GeminiCaller,
    TeeStdout,
    aggregate_dicts,
    name_match,
    plot_bar_compare,
    plot_box_compare,
    prf,
    read_jsonl,
    recall_at_k,
    spearman,
    write_jsonl,
)
from src.agents.phase2.task_a_critic import CriticAgent
from src.agents.phase2.task_a_validator import TaskAValidator
from src.graph.client import GraphClient


def _instrument_critic(critic: "CriticAgent", timeout_s: float = 180.0) -> None:
    """Wrap critic._call_gemini for verbose progress + HTTP timeout.

    The new task_a_critic.py has no http_options + no per-call print, so
    a slow / unavailable Pro model produces a silent hang. We wrap the
    method to (a) inject types.HttpOptions(timeout=...), (b) log start +
    duration, (c) re-raise so the agent's existing retry logic runs.
    """
    import json as _json
    import time as _t
    from google.genai import types as _types

    timeout_ms = int(timeout_s * 1000)
    orig_call = critic._call_gemini  # bound method

    def wrapped(pass1, full_paths, enrichment):
        src = pass1.get("source", "?")
        mal = pass1.get("malady", "?")
        print(f"      [critic] start {src!r} -> {mal!r} model={critic._model}",
              flush=True)
        t0 = _t.time()

        # Inline replacement of the original (so we can pass http_options).
        elapsed = _t.time() - critic._last_call
        if elapsed < 1.0:
            _t.sleep(1.0 - elapsed)
        critic._last_call = _t.time()
        prompt = critic._build_prompt(pass1, full_paths, enrichment)

        # Find symbols from the agent module for schema/system prompt.
        from src.agents.phase2 import task_a_critic as _tac
        last_err = None
        for attempt in range(3):
            try:
                resp = critic._gemini.models.generate_content(
                    model=critic._model,
                    contents=prompt,
                    config=_types.GenerateContentConfig(
                        system_instruction=_tac.CRITIC_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=_tac.CRITIC_RESPONSE_SCHEMA,
                        http_options=_types.HttpOptions(timeout=timeout_ms),
                    ),
                )
                out = _json.loads(resp.text)
                print(f"      [critic] ok in {_t.time()-t0:.1f}s", flush=True)
                return out
            except Exception as e:
                last_err = e
                err = str(e)
                transient = any(
                    c in err for c in ("429", "500", "502", "503", "504",
                                       "DEADLINE_EXCEEDED", "UNAVAILABLE")
                )
                if transient and attempt < 2:
                    wait_s = 5 * (attempt + 1)
                    print(f"      [critic] transient {err[:120]}; retry in {wait_s}s",
                          flush=True)
                    _t.sleep(wait_s)
                    continue
                print(f"      [critic] FAIL: {err[:200]}", flush=True)
                raise
        if last_err:
            raise last_err

    critic._call_gemini = wrapped  # type: ignore[assignment]


CORPUS_DEFAULT = Path("data/eval/recovery_corpus.jsonl")
RESULTS_DIR = Path("data/eval/results")


# ---------------------------------------------------------------------------
# Source/Malady extraction (KG arm step 1) — raw Gemini, no KG.
# ---------------------------------------------------------------------------

BATCH_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region_id": {"type": "string"},
                    "pairs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "malady": {"type": "string"},
                            },
                            "required": ["source", "malady"],
                        },
                    },
                },
                "required": ["region_id", "pairs"],
            },
        }
    },
    "required": ["regions"],
}


def _batch_extract_prompt(regions: list[dict]) -> str:
    blocks = []
    for r in regions:
        blocks.append(f"<region id=\"{r['region_id']}\">\n{r['text']}\n</region>")
    joined = "\n\n".join(blocks)
    return (
        "Read ALL passages from a historical Chinese herbal text. For EACH region, "
        "extract every (source, traditional_malady) pair the passage explicitly asserts "
        "— i.e. every claim of the form 'source X treats malady Y'. Use Latin binomial "
        "names for sources where possible. Return one entry per region (matching "
        "region_id), with its own list of pairs. A single source treating multiple "
        "maladies should appear multiple times within that region.\n\n"
        f"{joined}\n"
    )


EXTRACT_PAIRS_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string",
                               "description": "Latin binomial / canonical name of the medicinal source."},
                    "malady": {"type": "string",
                               "description": "The traditional Chinese-medicine ailment / symptom the source is said to treat in the passage."},
                },
                "required": ["source", "malady"],
            },
        }
    },
    "required": ["pairs"],
}


def _extract_pairs_prompt(text: str) -> str:
    return (
        "Read the passage from a historical Chinese herbal text. Extract every "
        "(source, traditional_malady) pair the passage explicitly asserts — i.e. "
        "every claim of the form 'source X treats malady Y'. Return one row per "
        "pair (a single source treating multiple maladies should appear multiple "
        "times). Use Latin binomial names for sources where possible.\n\n"
        f"PASSAGE:\n{text}\n"
    )


# ---------------------------------------------------------------------------
# Baseline (Gemini-only) prompts + schemas
# ---------------------------------------------------------------------------

OPEN_BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "compounds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "plausibility": {"type": "number"},
                                "reasoning": {"type": "string"},
                            },
                            "required": ["name", "plausibility"],
                        },
                    },
                },
                "required": ["source", "compounds"],
            },
        }
    },
    "required": ["sources"],
}


def _open_baseline_prompt(text: str) -> str:
    return (
        "You are a pharmaceutical and natural products expert with broad knowledge of "
        "traditional Chinese medicine, pharmacognosy, and bioactive plant compounds. "
        "You will be given a passage from the Shen Nong Ben Cao Jing and will be asked "
        "to discover and rank plausibility of therapeutic compounds.\n\n"
        "Read this passage from a historical Chinese herbal text. "
        "List each medicinal source discussed; for each, propose up to 10 "
        "therapeutic chemical compounds it likely contains, with a "
        "plausibility score 0.0-1.0 "
        "(0.0 = incoherent or no known mechanism, 1.0 = biologically obvious) "
        "and a one-sentence reasoning grounded in known biochemistry or pharmacology.\n\n"
        f"PASSAGE:\n{text}\n"
    )


# ---------------------------------------------------------------------------
# KG arm
# ---------------------------------------------------------------------------

# For a resolved source, list every Traditional_Malady it claims to treat.
# Used to drive validator.run() with REAL graph maladies — the LLM-extracted
# free-form phrases ("hundreds of diseases of the five viscera") never match
# the KG's canonical malady names, so we use them only for *which sources*
# the passage talks about, not which maladies.
_SOURCE_MALADIES_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE coalesce(s.archived,false)=false
  AND coalesce(m.archived,false)=false
  AND coalesce(r.archived,false)=false
RETURN m.name AS malady
"""


# Breadth: every compound extracted from a resolved source, ranked by how
# many distinct targets that compound has (more targets = more mechanistic
# context). LIMIT applied per-source via the inner subquery.
_BREADTH_QUERY = """
UNWIND $sources AS sname
MATCH (s:Source {name: sname})<-[:IS_EXTRACTED_FROM]-(c:Chemical_Compound)
WHERE coalesce(s.archived,false)=false
  AND coalesce(c.archived,false)=false
OPTIONAL MATCH (c)-[:TARGETS]->(t:Biological_Target)
WHERE coalesce(t.archived,false)=false
WITH sname, c, count(DISTINCT t) AS n_targets
ORDER BY n_targets DESC, c.name
RETURN sname AS source, c.name AS compound, n_targets
LIMIT $limit
"""


def _resolve_to_kg(name: str, kg_sources: list[str]) -> str | None:
    for k in kg_sources:
        if name_match(name, k):
            return k
    return None


def _validator_score(path_score: float, loop_closed: bool) -> float:
    """Map validator path_score (≈0..5) + loop_closed bonus into [0.25, 0.5]."""
    norm = max(0.0, min(1.0, path_score / 5.0)) if path_score else 0.0
    s = 0.25 + 0.20 * norm + (0.05 if loop_closed else 0.0)
    return min(0.5, s)


def _kg_arm_open(
    caller: GeminiCaller,
    validator: TaskAValidator,
    critic: CriticAgent,
    client: GraphClient,
    text: str,
    kg_sources: list[str],
    breadth_limit: int = 60,
    prefetched_pairs: list[dict] | None = None,
    defer_critic: bool = False,
    prefetched_pass2: list[dict] | None = None,
) -> dict:
    # 1. Extract (source, malady) pairs via raw Gemini (skipped if prefetched).
    if prefetched_pairs is not None:
        pairs = [p for p in prefetched_pairs if p.get("source") and p.get("malady")]
    else:
        raw = caller.call(_extract_pairs_prompt(text), schema=EXTRACT_PAIRS_SCHEMA) or {}
        pairs = [p for p in (raw.get("pairs") or []) if p.get("source") and p.get("malady")]
    # Preserve LLM emission order (dedup by first occurrence).
    extracted_raw_srcs: list[str] = []
    _seen_raw: set[str] = set()
    for p in pairs:
        s = p["source"]
        if s not in _seen_raw:
            extracted_raw_srcs.append(s)
            _seen_raw.add(s)

    # 2. Resolve to canonical KG sources (preserve emission order).
    raw_to_kg: dict[str, str] = {}
    unresolved: list[str] = []
    resolved_emission_order: list[str] = []
    _seen_kg: set[str] = set()
    for r in extracted_raw_srcs:
        kg = _resolve_to_kg(r, kg_sources)
        if kg:
            raw_to_kg[r] = kg
            if kg not in _seen_kg:
                resolved_emission_order.append(kg)
                _seen_kg.add(kg)
        else:
            unresolved.append(r)
    resolved_sources = resolved_emission_order  # use emission order as default

    # 3. Pass 1: for each resolved source, fetch the source's real KG
    # maladies, then keep only those that fuzzy-match an LLM-extracted
    # malady for that source. LLM gives us *which maladies the passage
    # discusses*; KG gives us the canonical names + edges. If no LLM
    # malady matches any KG malady for the source, fall back to the full
    # KG malady set for that source (passage clearly mentions the source
    # so any of its claims is in scope).
    pass1_verdicts: list[dict] = []
    pass1_errors: list[dict] = []
    source_to_kg_maladies: dict[str, list[str]] = {}
    source_to_llm_maladies: dict[str, list[str]] = {}
    for p in pairs:
        kg_src = raw_to_kg.get(p["source"])
        if kg_src and p.get("malady"):
            source_to_llm_maladies.setdefault(kg_src, []).append(p["malady"])
    for kg_src in resolved_sources:
        try:
            mrows = client.run(_SOURCE_MALADIES_QUERY, {"source": kg_src}) or []
            kg_maladies = [r["malady"] for r in mrows if r.get("malady")]
        except Exception as e:
            pass1_errors.append({"phase": "source_maladies", "source": kg_src, "error": str(e)})
            kg_maladies = []
        llm_mals = source_to_llm_maladies.get(kg_src, [])
        matched = [m for m in kg_maladies
                   if any(name_match(m, lm) for lm in llm_mals)]
        if not matched and kg_maladies:
            matched = kg_maladies  # fallback: source mentioned, take all KG claims
        source_to_kg_maladies[kg_src] = matched

    total_claims = sum(len(v) for v in source_to_kg_maladies.values())
    n_fb = sum(1 for s in resolved_sources
               if not any(name_match(m, lm)
                          for m in source_to_kg_maladies.get(s, [])
                          for lm in source_to_llm_maladies.get(s, []))
               and source_to_kg_maladies.get(s))
    print(f"    [pass1] {len(resolved_sources)} resolved sources -> {total_claims} (src,malady) claims "
          f"(fallback-all on {n_fb})", flush=True)
    done = 0
    for kg_src, maladies in source_to_kg_maladies.items():
        for malady in maladies:
            done += 1
            try:
                res = validator.run(
                    source_name=kg_src, malady_name=malady,
                    write_graph=False, keep_top_paths=20,
                )
                for v in res.get("verdicts") or []:
                    pass1_verdicts.append(v)
            except Exception as e:
                pass1_errors.append({"source": kg_src, "malady": malady, "error": str(e)})
            if done % 5 == 0 or done == total_claims:
                print(f"    [pass1] {done}/{total_claims}", flush=True)

    # Drop verdicts with no paths (claim_not_found / empty); critic skips them anyway.
    pass1_actionable = [v for v in pass1_verdicts if v.get("path_count", 0) > 0]
    print(f"    [pass1] {len(pass1_actionable)}/{len(pass1_verdicts)} actionable (have paths)",
          flush=True)

    # 4. Pass 2: CriticAgent over the batch.
    pass2_results: list[dict] = []
    if prefetched_pass2 is not None:
        pass2_results = prefetched_pass2
        print(f"    [pass2] using {len(pass2_results)} prefetched (global batch) critic verdicts",
              flush=True)
    elif defer_critic:
        print(f"    [pass2] deferred (will run in global batch)", flush=True)
    elif pass1_actionable:
        print(f"    [pass2] running critic on {len(pass1_actionable)} verdicts (LLM)",
              flush=True)
        try:
            crit_summary = critic.run(verdicts=pass1_actionable, write_graph=False)
            pass2_results = crit_summary.get("results") or []
            print(f"    [pass2] {len(pass2_results)} critic verdicts", flush=True)
        except Exception as e:
            pass1_errors.append({"phase": "critic", "error": str(e),
                                  "tb": traceback.format_exc()[:500]})
            print(f"    [pass2] FAIL: {e}", flush=True)

    ranked, src_score, extracted_sources_kg_ranked = _aggregate_and_rank(
        pass1_actionable, pass2_results, resolved_emission_order,
        client, breadth_limit, pass1_errors,
    )

    return {
        "extracted_pairs": pairs,
        "extracted_sources_raw": extracted_raw_srcs,
        "extracted_sources": extracted_sources_kg_ranked,
        "extracted_sources_emission_order": resolved_emission_order,
        "source_score": src_score,
        "extracted_sources_unresolved": unresolved,
        "source_to_kg_maladies": source_to_kg_maladies,
        "pass1_verdicts": pass1_actionable,
        "pass2_verdicts": pass2_results,
        "errors": pass1_errors,
        "lead_compounds": ranked,
    }


def _aggregate_and_rank(
    pass1_actionable: list[dict],
    pass2_results: list[dict],
    resolved_emission_order: list[str],
    client: GraphClient,
    breadth_limit: int,
    errors_out: list[dict],
) -> tuple[list[dict], dict[str, float], list[str]]:
    leads: dict[str, dict] = {}

    for cv in pass2_results:
        if cv.get("skipped") or cv.get("error"):
            continue
        plaus = float(cv.get("biological_plausibility") or 0.0)
        coh = float(cv.get("evidence_coherence") or 0.0)
        crit_score = 0.5 + 0.5 * ((plaus + coh) / 2.0)  # 0.5..1.0
        for ke in cv.get("key_evidence") or []:
            cname = ke.get("compound")
            if not cname:
                continue
            entry = leads.get(cname)
            if entry is None or crit_score > entry["score"]:
                leads[cname] = {
                    "compound": cname,
                    "source": cv.get("source"),
                    "score": crit_score,
                    "from": "critic",
                    "biological_plausibility": plaus,
                    "evidence_coherence": coh,
                    "target": ke.get("target"),
                    "reached_disease": ke.get("reached_disease"),
                    "why_compelling": ke.get("why_compelling"),
                }

    # 5b. Validator paths_top compounds (next priority).
    for v in pass1_actionable:
        src = v.get("source")
        for path in v.get("paths_top") or []:
            comp = ((path.get("compound") or {}).get("name") or "").strip()
            if not comp:
                continue
            pinfo = path.get("path") or {}
            cand_score = _validator_score(
                float(pinfo.get("score") or 0.0),
                bool(pinfo.get("loop_closed")),
            )
            entry = leads.get(comp)
            if entry is None or cand_score > entry["score"]:
                leads[comp] = {
                    "compound": comp,
                    "source": src,
                    "score": cand_score,
                    "from": "validator_path",
                    "path_score": pinfo.get("score"),
                    "path_bucket": pinfo.get("bucket"),
                    "loop_closed": pinfo.get("loop_closed"),
                    "target": (path.get("target") or {}).get("name"),
                    "reached_disease": (path.get("reached_disease") or {}).get("name"),
                }

    breadth_rows: list[dict] = []
    if resolved_emission_order:
        try:
            breadth_rows = client.run(
                _BREADTH_QUERY,
                {"sources": resolved_emission_order,
                 "limit": breadth_limit * len(resolved_emission_order)},
            ) or []
        except Exception as e:
            errors_out.append({"phase": "breadth", "error": str(e)})
    for r in breadth_rows:
        comp = (r.get("compound") or "").strip()
        if not comp or comp in leads:
            continue
        n_t = int(r.get("n_targets") or 0)
        leads[comp] = {
            "compound": comp,
            "source": r.get("source"),
            "score": min(0.25, 0.05 + 0.01 * n_t),
            "from": "breadth",
            "n_targets": n_t,
        }

    ranked = sorted(leads.values(), key=lambda x: -x["score"])

    src_score: dict[str, float] = {s: 0.0 for s in resolved_emission_order}
    for r in ranked:
        s = r.get("source")
        if s in src_score:
            src_score[s] = max(src_score[s], float(r.get("score") or 0.0))
    emission_index = {s: i for i, s in enumerate(resolved_emission_order)}
    sources_ranked = sorted(
        resolved_emission_order,
        key=lambda s: (-src_score.get(s, 0.0), emission_index.get(s, 1_000_000)),
    )
    return ranked, src_score, sources_ranked


def _baseline_arm_open(caller: GeminiCaller, text: str) -> dict:
    raw = caller.call(_open_baseline_prompt(text), schema=OPEN_BASELINE_SCHEMA)
    if not raw:
        return {"sources": []}
    return raw


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _gold_compound_score(compound: str, source: str,
                         gold_compounds: dict[str, list[str]]) -> int:
    for gold_src, comps in gold_compounds.items():
        if name_match(source, gold_src):
            for c in comps:
                if name_match(compound, c):
                    return 1
    return 0


def _per_source_compound_recall(predictions: list[tuple[str, str]],
                                gold_compounds: dict[str, list[str]],
                                ks: list[int]) -> dict:
    out = {f"per_src_recall@{k}": 0.0 for k in ks}
    if not gold_compounds:
        return out
    per_src_results: dict[int, list[float]] = {k: [] for k in ks}
    for gsrc, gcomps in gold_compounds.items():
        if not gcomps:
            continue
        attributed = [c for (c, s) in predictions if s and name_match(s, gsrc)]
        rk = recall_at_k(attributed, gcomps, ks)
        for k in ks:
            per_src_results[k].append(rk[f"recall@{k}"])
    for k in ks:
        vals = per_src_results[k]
        out[f"per_src_recall@{k}"] = sum(vals) / len(vals) if vals else 0.0
    return out


def _kg_arm_predictions(kg_out: dict) -> list[tuple[str, str]]:
    return [(r["compound"], r.get("source") or "") for r in kg_out["lead_compounds"]]


def _baseline_predictions(base_out: dict) -> list[tuple[str, str, float]]:
    out: list[tuple[str, str, float]] = []
    for s in base_out.get("sources", []):
        sname = s.get("source", "")
        for c in s.get("compounds", []):
            out.append((c.get("name", ""), sname, float(c.get("plausibility", 0.0))))
    return out


def _eval_open_region(region: dict, kg_out: dict, base_out: dict) -> dict:
    gold_sources = region["gold_sources"]
    gold_compounds = region["gold_compounds_per_source"]
    gold_compound_union = sorted({c for cs in gold_compounds.values() for c in cs})
    distractor_compounds = region.get("distractor_compounds_per_source", {})
    all_compound_union = sorted(
        {c for cs in gold_compounds.values() for c in cs}
        | {c for cs in distractor_compounds.values() for c in cs}
    )

    # Source recovery.
    kg_sources = kg_out["extracted_sources"]
    kg_sources_raw = kg_out["extracted_sources_raw"]
    base_sources = [s["source"] for s in base_out.get("sources", [])]
    src_kg = prf(kg_sources, gold_sources)
    src_kg_raw = prf(kg_sources_raw, gold_sources)
    src_base = prf(base_sources, gold_sources)

    # Source recall@k — headline metric. Each region has 3 gold + 7 distractor
    # sources mentioned in the text. recall@k = fraction of gold the arm
    # ranked in its top-k extracted sources. KG arm rank is the order returned
    # by the LLM extraction (preserved through resolution); baseline rank is
    # the order the baseline LLM emitted its `sources` list.
    src_kg_rank = recall_at_k(kg_sources, gold_sources, [3, 5, 10])
    src_base_rank = recall_at_k(base_sources, gold_sources, [3, 5, 10])

    # Compound recovery.
    kg_preds = _kg_arm_predictions(kg_out)
    base_preds_full = _baseline_predictions(base_out)
    kg_compounds_ranked = [c for (c, _) in kg_preds]
    base_compounds_ranked = [c for (c, _, _) in base_preds_full]

    comp_kg = recall_at_k(kg_compounds_ranked, gold_compound_union, [3, 5, 10])
    comp_base = recall_at_k(base_compounds_ranked, gold_compound_union, [3, 5, 10])
    comp_kg_all = recall_at_k(kg_compounds_ranked, all_compound_union, [3, 5, 10])
    comp_base_all = recall_at_k(base_compounds_ranked, all_compound_union, [3, 5, 10])

    per_src_kg = _per_source_compound_recall(kg_preds, gold_compounds, [3, 5, 10])
    per_src_base = _per_source_compound_recall(
        [(c, s) for (c, s, _) in base_preds_full], gold_compounds, [3, 5, 10]
    )

    # Plausibility: KG = mean critic biological_plausibility across pass2 verdicts.
    kg_plaus_vals = [
        float(cv.get("biological_plausibility") or 0.0)
        for cv in kg_out.get("pass2_verdicts") or []
        if not cv.get("skipped") and not cv.get("error")
        and cv.get("biological_plausibility") is not None
    ]
    kg_plausibility = sum(kg_plaus_vals) / len(kg_plaus_vals) if kg_plaus_vals else 0.0

    base_plaus_vals = [p for (_, _, p) in base_preds_full]
    base_plausibility = sum(base_plaus_vals) / len(base_plaus_vals) if base_plaus_vals else 0.0

    # Symmetric Brier: per-compound score vs gold-membership flag.
    kg_brier_diffs = []
    for r in kg_out["lead_compounds"]:
        pseudo_p = max(0.0, min(1.0, float(r.get("score", 0.0))))
        gold_flag = _gold_compound_score(r["compound"], r.get("source") or "", gold_compounds)
        kg_brier_diffs.append((pseudo_p - gold_flag) ** 2)
    kg_brier = sum(kg_brier_diffs) / len(kg_brier_diffs) if kg_brier_diffs else 0.0

    base_brier_diffs = []
    for (cname, sname, plaus) in base_preds_full:
        gold_flag = _gold_compound_score(cname, sname, gold_compounds)
        base_brier_diffs.append((plaus - gold_flag) ** 2)
    base_brier = sum(base_brier_diffs) / len(base_brier_diffs) if base_brier_diffs else 0.0

    # Spearman on shared compounds.
    kg_score = {r["compound"]: float(r.get("score", 0.0)) for r in kg_out["lead_compounds"]}
    base_score = {c: p for (c, _, p) in base_preds_full}
    shared = [(c, kg_score[c], base_score[c]) for c in kg_score if c in base_score]
    rho = spearman([x[1] for x in shared], [x[2] for x in shared]) if len(shared) >= 2 else 0.0

    # Diagnostics.
    leads = kg_out["lead_compounds"]
    n_critic = sum(1 for r in leads if r.get("from") == "critic")
    n_validator = sum(1 for r in leads if r.get("from") == "validator_path")
    n_breadth = sum(1 for r in leads if r.get("from") == "breadth")
    n_pass1 = len(kg_out.get("pass1_verdicts") or [])
    n_pass2 = len(kg_out.get("pass2_verdicts") or [])

    return {
        "region_id": region["region_id"],
        "n_gold_sources": len(gold_sources),
        "n_gold_compounds": len(gold_compound_union),
        "n_all_compounds": len(all_compound_union),
        "n_distractor_compounds": len(all_compound_union) - len(gold_compound_union),

        "src_kg_precision": src_kg["precision"],
        "src_kg_recall": src_kg["recall"],
        "src_kg_f1": src_kg["f1"],
        "src_kg_raw_f1": src_kg_raw["f1"],
        "src_base_precision": src_base["precision"],
        "src_base_recall": src_base["recall"],
        "src_base_f1": src_base["f1"],

        # Source recall@k — headline (3 gold + 7 distractor per region).
        **{f"src_kg_{k}": v for k, v in src_kg_rank.items()},
        **{f"src_base_{k}": v for k, v in src_base_rank.items()},

        # Closed-loop compound recall (primary metric).
        **{f"comp_kg_{k}": v for k, v in comp_kg.items()},
        **{f"comp_base_{k}": v for k, v in comp_base.items()},
        "comp_kg_f1@10": _comp_f1_at_k(comp_kg.get("recall@10", 0.0), len(gold_compound_union), 10),
        "comp_base_f1@10": _comp_f1_at_k(comp_base.get("recall@10", 0.0), len(gold_compound_union), 10),
        # All-sources compound recall (closed-loop + distractor IS_EXTRACTED_FROM).
        **{f"comp_kg_all_{k}": v for k, v in comp_kg_all.items()},
        **{f"comp_base_all_{k}": v for k, v in comp_base_all.items()},

        **{f"strict_kg_{k}": v for k, v in per_src_kg.items()},
        **{f"strict_base_{k}": v for k, v in per_src_base.items()},

        "kg_plausibility": kg_plausibility,
        "base_plausibility": base_plausibility,
        "plausibility_delta": kg_plausibility - base_plausibility,
        "kg_brier": kg_brier,
        "base_brier": base_brier,
        "plausibility_spearman_shared": rho,
        "n_shared_compounds": len(shared),

        "n_extracted_pairs": len(kg_out.get("extracted_pairs", [])),
        "n_extracted_sources_raw": len(kg_sources_raw),
        "n_extracted_sources_resolved": len(kg_sources),
        "n_unresolved_sources": len(kg_out.get("extracted_sources_unresolved", [])),
        "n_pass1_verdicts": n_pass1,
        "n_pass2_verdicts": n_pass2,
        "n_critic_leads": n_critic,
        "n_validator_leads": n_validator,
        "n_breadth_leads": n_breadth,
    }


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_RECALL_PAIRS = [
    ("src_recall@3",    "src_kg_recall@3",      "src_base_recall@3"),
    ("src_recall@5",    "src_kg_recall@5",      "src_base_recall@5"),
    ("src_f1 (set)",    "src_kg_f1",            "src_base_f1"),
    ("comp_recall@3",   "comp_kg_recall@3",     "comp_base_recall@3"),
    ("comp_recall@5",   "comp_kg_recall@5",     "comp_base_recall@5"),
]

_STRICT_PAIRS = [
    ("strict@3",                "strict_kg_per_src_recall@3",   "strict_base_per_src_recall@3"),
    ("strict@5",                "strict_kg_per_src_recall@5",   "strict_base_per_src_recall@5"),
    ("plausibility",            "kg_plausibility",              "base_plausibility"),
    ("brier (lower=better)",    "kg_brier",                     "base_brier"),
]


def _plot_pair(agg: dict, pairs, title: str, out_path: Path) -> None:
    plot_bar_compare(
        [(label, agg.get(kg_key, 0.0), agg.get(base_key, 0.0))
         for (label, kg_key, base_key) in pairs],
        title=title, out_path=out_path,
    )


def _comp_f1_at_k(recall_at_k: float, n_gold: int, k: int) -> float:
    """F1@k from recall@k + n_gold. TP = recall * n_gold; prec = TP/k."""
    if n_gold == 0 or k == 0:
        return 0.0
    tp = recall_at_k * n_gold
    prec = tp / k
    rec = recall_at_k
    denom = prec + rec
    return (2 * prec * rec / denom) if denom > 0 else 0.0


_BOX_TRIPLE = [
    ("src_recall@3",       "src_kg_recall@3",       "src_base_recall@3"),
    ("comp_recall@10",     "comp_kg_recall@10",     "comp_base_recall@10"),
    ("comp_F1@10",         "comp_kg_f1@10",         "comp_base_f1@10"),
    ("plausibility",       "kg_plausibility",       "base_plausibility"),
]


def _plot_boxes(metrics_rows: list[dict], title: str, out_path: Path) -> None:
    """Boxplot (per-region distribution) of headline metrics.
    Derives comp_f1@5 on-the-fly for old saved rows missing the key."""
    for r in metrics_rows:
        if "comp_kg_f1@10" not in r:
            ng = int(r.get("n_gold_compounds", 0))
            r["comp_kg_f1@10"] = _comp_f1_at_k(float(r.get("comp_kg_recall@10", 0.0)), ng, 10)
            r["comp_base_f1@10"] = _comp_f1_at_k(float(r.get("comp_base_recall@10", 0.0)), ng, 10)
    triples = []
    for label, kg_key, base_key in _BOX_TRIPLE:
        kg_vals = [float(r.get(kg_key, 0.0)) for r in metrics_rows]
        base_vals = [float(r.get(base_key, 0.0)) for r in metrics_rows]
        triples.append((label, kg_vals, base_vals))
    plot_box_compare(triples, title=title, out_path=out_path)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_COMPARE_METRICS = [
    ("src_recall@3",   "src_kg_recall@3",       "src_base_recall@3"),
    ("src_recall@5",   "src_kg_recall@5",       "src_base_recall@5"),
    ("comp_recall@5",  "comp_kg_recall@5",      "comp_base_recall@5"),
    ("comp_recall@10", "comp_kg_recall@10",     "comp_base_recall@10"),
    ("strict@5",       "strict_kg_per_src_recall@5", "strict_base_per_src_recall@5"),
    ("plausibility",   "kg_plausibility",       "base_plausibility"),
]


def _metric_rows_from_jsonl(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    out: list[dict] = []
    for r in rows:
        if "metrics" in r:
            out.append(r["metrics"])
    return out


def _plot_compare_three(batch_rows: list[dict], compare_jsonl: Path,
                        out_path: Path, title: str) -> None:
    """3-series bar chart: DeepRoot-batch KG, DeepRoot-discovery KG, baseline.
    Baseline is taken from the discovery JSONL (single source of truth)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped: matplotlib not installed]", flush=True)
        return

    discovery_rows = _metric_rows_from_jsonl(compare_jsonl)
    if not discovery_rows:
        print(f"  [compare] no metrics in {compare_jsonl}; skipping compare plot")
        return

    def mean(rows, key):
        vals = [float(r.get(key, 0.0)) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    names, batch_v, disc_v, base_v = [], [], [], []
    for (label, kg_key, base_key) in _COMPARE_METRICS:
        names.append(label)
        batch_v.append(mean(batch_rows, kg_key))
        disc_v.append(mean(discovery_rows, kg_key))
        base_v.append(mean(discovery_rows, base_key))

    x = list(range(len(names)))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4.2))
    ax.bar([i - w for i in x], batch_v, w, label="DeepRoot (batch)", color="#2e7d32")
    ax.bar(x, disc_v, w, label="DeepRoot (discovery)", color="#3b6ea5")
    ax.bar([i + w for i in x], base_v, w, label="LLM baseline", color="#c45a5a")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    all_vals = batch_v + disc_v + base_v
    ax.set_ylim(0, max(1.0, max(all_vals) * 1.15) if all_vals else 1.0)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [plot saved: {out_path}]", flush=True)


_GLOBAL_CRITIC_SYSTEM_PROMPT = """\
You are a critic for a knowledge-graph-grounded drug-discovery pipeline. You
will be given a JSON LIST of claims. Each claim is a (Source, Traditional
Malady) pair from a historical Chinese herbal text, plus the top mechanistic
paths the deterministic Pass-1 validator found in the KG (Compound -> Target
-> Disease chains).

For EVERY claim, return an object inside `results[]` with EXACTLY the
provided `claim_id` and these fields:
  - verdict: one of [strong_support, moderate_support, partial_support,
    unsupported, mechanistic_only, traditional_only, claim_not_found]
  - biological_plausibility: 0.0-1.0 (does the claim make biological sense
    given known pharmacology of the compounds + their targets?)
  - evidence_coherence: 0.0-1.0 (does the KG evidence converge — do multiple
    paths reach the same disease via mechanistically related targets?)
  - key_evidence: up to 3 compelling (compound, target, reached_disease,
    why_compelling) entries
  - rationale: 1-2 sentences

Score every claim independently — do not let one claim's outcome bias another.
Be rigorous: traditional folk use without mechanistic backing is not strong
support; one weak path is not moderate support.
"""

_GLOBAL_CRITIC_KEY_EVIDENCE_ITEM = {
    "type": "object",
    "properties": {
        "compound": {"type": "string"},
        "target": {"type": "string"},
        "reached_disease": {"type": "string"},
        "why_compelling": {"type": "string"},
    },
    "required": ["compound", "target", "reached_disease", "why_compelling"],
}

_GLOBAL_CRITIC_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "verdict": {"type": "string"},
                    "biological_plausibility": {"type": "number"},
                    "evidence_coherence": {"type": "number"},
                    "key_evidence": {
                        "type": "array",
                        "items": _GLOBAL_CRITIC_KEY_EVIDENCE_ITEM,
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "claim_id", "verdict",
                    "biological_plausibility", "evidence_coherence",
                    "key_evidence", "rationale",
                ],
            },
        }
    },
    "required": ["results"],
}


def _compact_claim(v: dict, max_paths: int = 5) -> dict:
    """Trim a pass-1 verdict to the minimum fields the critic needs.
    Keeps top-N paths_top (already sorted by score)."""
    paths_raw = v.get("paths_top") or []
    paths = []
    for p in paths_raw[:max_paths]:
        comp = (p.get("compound") or {}).get("name")
        targ = (p.get("target") or {}).get("name")
        dis = (p.get("reached_disease") or {}).get("name")
        pi = p.get("path") or {}
        if not (comp and (pi.get("has_target") or targ)):
            continue
        paths.append({
            "compound": comp,
            "target": targ,
            "reached_disease": dis,
            "bucket": pi.get("bucket"),
            "score": pi.get("score"),
            "loop_closed": pi.get("loop_closed"),
        })
    return {
        "claim_id": v["_claim_id"],
        "source": v.get("source"),
        "malady": v.get("malady"),
        "pass1_verdict": v.get("verdict"),
        "top_bucket": v.get("top_bucket"),
        "path_count": v.get("path_count"),
        "paths_loop_closed": v.get("paths_loop_closed"),
        "paths": paths,
    }


def _global_batched_critic(critic: "CriticAgent",
                           tagged_verdicts: list[dict]) -> dict[str, dict]:
    """Single Gemini call scoring ALL verdicts at once. Returns
    {claim_id: critic_dict} shaped like CriticVerdict._cv_to_dict output
    (subset of fields the eval downstream needs)."""
    from google.genai import types as _types
    import time as _t

    claims = [_compact_claim(v) for v in tagged_verdicts]
    prompt = (
        "Score the following claims. Output JSON conforming to the schema. "
        "results[] MUST contain one entry per claim_id below.\n\n"
        + json.dumps({"claims": claims}, ensure_ascii=False)
    )

    print(f"  [global-critic] prompt size: {len(prompt):,} chars; "
          f"{len(claims)} claims", flush=True)
    t0 = _t.time()
    out_raw: dict = {}
    for attempt in range(3):
        try:
            resp = critic._gemini.models.generate_content(
                model=critic._model,
                contents=prompt,
                config=_types.GenerateContentConfig(
                    system_instruction=_GLOBAL_CRITIC_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=_GLOBAL_CRITIC_RESPONSE_SCHEMA,
                    http_options=_types.HttpOptions(timeout=600_000),
                ),
            )
            out_raw = json.loads(resp.text)
            print(f"  [global-critic] ok in {_t.time()-t0:.1f}s", flush=True)
            break
        except Exception as e:
            err = str(e)
            print(f"  [global-critic] attempt {attempt+1} FAIL: {err[:200]}", flush=True)
            if attempt < 2:
                _t.sleep(10 * (attempt + 1))
                continue
            raise

    # Build claim_id -> source/malady map for output enrichment.
    src_map = {v["_claim_id"]: (v.get("source"), v.get("malady")) for v in tagged_verdicts}

    out: dict[str, dict] = {}
    for r in out_raw.get("results") or []:
        cid = r.get("claim_id")
        if not cid or cid not in src_map:
            continue
        src, mal = src_map[cid]
        out[cid] = {
            "source": src,
            "malady": mal,
            "verdict": r.get("verdict") or "",
            "biological_plausibility": float(r.get("biological_plausibility") or 0.0),
            "evidence_coherence": float(r.get("evidence_coherence") or 0.0),
            "key_evidence": r.get("key_evidence") or [],
            "concerns": [],
            "rationale": r.get("rationale") or "",
            "requires_human_review": False,
            "agrees_with_pass1": False,
            "skipped": False,
            "error": None,
        }
    return out


def run(corpus_path: Path, limit: int | None = None,
        critic_model: str | None = None, breadth_limit: int = 60,
        batch_extract: bool = False,
        compare_to: Path | None = None) -> None:
    regions = read_jsonl(corpus_path)
    if limit:
        regions = regions[:limit]
    if not regions:
        print(f"No regions in {corpus_path}")
        return

    print(f"Loaded {len(regions)} regions from {corpus_path}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_tee = TeeStdout(RESULTS_DIR / f"eval1_lead_recovery_{ts}.log")
    print(f"Logging console to: {log_tee.path}")

    with GraphClient() as client:
        validator = TaskAValidator(client)
        # Default critic to the same flash model the rest of the eval uses
        # (single-model fairness vs baseline; Pro defaults are slow and may
        # be unavailable on the user's API key).
        eff_critic_model = critic_model or DEFAULT_FLASH_MODEL
        critic = CriticAgent(
            client,
            gemini_model=eff_critic_model,
            skip_non_actionable=False,            # critique every actionable claim
        )
        _instrument_critic(critic, timeout_s=180.0)
        extract_caller = GeminiCaller(model=DEFAULT_FLASH_MODEL)
        baseline_caller = GeminiCaller(model=DEFAULT_FLASH_MODEL)
        print(f"  Extraction + baseline model: {DEFAULT_FLASH_MODEL}")
        print(f"  Critic model:                {eff_critic_model}")

        kg_rows = client.run(
            "MATCH (s:Source) WHERE coalesce(s.archived,false)=false RETURN s.name AS name"
        )
        kg_sources_list = [r["name"] for r in kg_rows if r.get("name")]
        print(f"  Loaded {len(kg_sources_list)} KG sources for name resolution")

        print("\n=== Eval 1: Lead Hit Identification ===")
        # Batch-extract all (source, malady) pairs up front in one Gemini call.
        prefetched: dict[str, list[dict]] = {}
        if batch_extract:
            print(f"  [batch-extract] one call for {len(regions)} regions")
            raw = extract_caller.call(
                _batch_extract_prompt(regions), schema=BATCH_EXTRACT_SCHEMA
            ) or {}
            for entry in raw.get("regions") or []:
                rid = entry.get("region_id")
                if rid:
                    prefetched[rid] = entry.get("pairs") or []
            print(f"  [batch-extract] received pairs for {len(prefetched)}/{len(regions)} regions")

        open_rows: list[dict] = []
        full_open: list[dict] = []

        if batch_extract:
            # Phase 1: per-region validator + breadth, defer critic.
            cached: list[tuple[dict, dict, dict]] = []  # (region, kg_out, base_out)
            for i, region in enumerate(regions, 1):
                print(f"  [{i}/{len(regions)}] {region['region_id']} (phase 1: validator+breadth)")
                pre = prefetched.get(region["region_id"])
                if pre is None:
                    print(f"    [batch-extract] missing region in batch reply; falling back to per-region extract")
                kg_out = _kg_arm_open(extract_caller, validator, critic, client,
                                      region["text"], kg_sources_list,
                                      breadth_limit=breadth_limit,
                                      prefetched_pairs=pre,
                                      defer_critic=True)
                base_out = _baseline_arm_open(baseline_caller, region["text"])
                cached.append((region, kg_out, base_out))

            # Phase 2: GLOBAL batched critic — one Gemini call for all verdicts.
            tagged_verdicts: list[dict] = []
            for (region, kg_out, _b) in cached:
                rid = region["region_id"]
                for v in kg_out.get("pass1_verdicts") or []:
                    v2 = dict(v)
                    v2["_claim_id"] = f"{rid}||{v.get('source','')}||{v.get('malady','')}"
                    v2["_region_id"] = rid
                    tagged_verdicts.append(v2)
            print(f"\n  [global-critic] running 1 LLM call over {len(tagged_verdicts)} verdicts")
            critic_by_claim = _global_batched_critic(critic, tagged_verdicts) if tagged_verdicts else {}
            print(f"  [global-critic] received {len(critic_by_claim)} critic verdicts\n")

            # Phase 3: assign critic results back to each kg_out, re-aggregate, score.
            for i, (region, kg_out, base_out) in enumerate(cached, 1):
                rid = region["region_id"]
                pass2_for_region: list[dict] = []
                for v in kg_out.get("pass1_verdicts") or []:
                    cid = f"{rid}||{v.get('source','')}||{v.get('malady','')}"
                    cv = critic_by_claim.get(cid)
                    if cv:
                        pass2_for_region.append(cv)
                kg_out["pass2_verdicts"] = pass2_for_region
                # Re-aggregate leads + source ranking with critic data.
                ranked, src_score, sources_ranked = _aggregate_and_rank(
                    kg_out.get("pass1_verdicts") or [],
                    pass2_for_region,
                    kg_out.get("extracted_sources_emission_order") or [],
                    client, breadth_limit,
                    kg_out.setdefault("errors", []),
                )
                kg_out["lead_compounds"] = ranked
                kg_out["source_score"] = src_score
                kg_out["extracted_sources"] = sources_ranked

                metrics = _eval_open_region(region, kg_out, base_out)
                open_rows.append(metrics)
                full_open.append({
                    "region_id": rid,
                    "gold_sources": region["gold_sources"],
                    "gold_compounds_per_source": region["gold_compounds_per_source"],
                    "kg": kg_out,
                    "baseline": base_out,
                    "metrics": metrics,
                })
                print(f"  [{i}/{len(cached)}] {rid} scored")

        else:
            for i, region in enumerate(regions, 1):
                print(f"  [{i}/{len(regions)}] {region['region_id']}")
                kg_out = _kg_arm_open(extract_caller, validator, critic, client,
                                      region["text"], kg_sources_list,
                                      breadth_limit=breadth_limit)
                base_out = _baseline_arm_open(baseline_caller, region["text"])
                metrics = _eval_open_region(region, kg_out, base_out)
                open_rows.append(metrics)
                full_open.append({
                    "region_id": region["region_id"],
                    "gold_sources": region["gold_sources"],
                    "gold_compounds_per_source": region["gold_compounds_per_source"],
                    "kg": kg_out,
                    "baseline": base_out,
                    "metrics": metrics,
                })
                print(
                f"    src@3: kg={metrics['src_kg_recall@3']:.2f} base={metrics['src_base_recall@3']:.2f} | "
                f"src@5: kg={metrics['src_kg_recall@5']:.2f} base={metrics['src_base_recall@5']:.2f} | "
                f"comp@5: kg={metrics['comp_kg_recall@5']:.2f} base={metrics['comp_base_recall@5']:.2f} | "
                f"strict@5: kg={metrics['strict_kg_per_src_recall@5']:.2f} "
                f"base={metrics['strict_base_per_src_recall@5']:.2f} | "
                f"plaus: kg={metrics['kg_plausibility']:.2f} base={metrics['base_plausibility']:.2f} | "
                f"brier: kg={metrics['kg_brier']:.2f} base={metrics['base_brier']:.2f} | "
                f"leads(c/v/b)={metrics['n_critic_leads']}/"
                f"{metrics['n_validator_leads']}/{metrics['n_breadth_leads']}"
            )

        write_jsonl(RESULTS_DIR / f"eval1_open_{ts}.jsonl", full_open)
        agg = aggregate_dicts(open_rows)
        summary = {"eval": "1_lead_recovery", "aggregate": agg}
        (RESULTS_DIR / f"eval1_open_{ts}.summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  Aggregate: {json.dumps(agg, indent=2)}")
        _plot_pair(agg, _RECALL_PAIRS,
                   f"Eval 1 — Source + Lenient Compound Recall (n={agg.get('n', 0)})",
                   RESULTS_DIR / f"eval1_recall_{ts}.png")
        _plot_pair(agg, _STRICT_PAIRS,
                   f"Eval 1 — Strict Recall + Plausibility (n={agg.get('n', 0)})",
                   RESULTS_DIR / f"eval1_strict_plaus_{ts}.png")
        _plot_boxes(open_rows,
                    f"Eval 1 — Headline (n={agg.get('n', 0)})",
                    RESULTS_DIR / f"eval1_box_{ts}.png")

        if compare_to:
            _plot_compare_three(
                batch_rows=open_rows, compare_jsonl=compare_to,
                out_path=RESULTS_DIR / f"eval1_compare_batch_vs_discovery_{ts}.png",
                title=f"Eval 1 — DeepRoot Batch vs Discovery vs Baseline (n={agg.get('n', 0)})",
            )


def replot_from_saved(jsonl_path: Path) -> None:
    rows = read_jsonl(jsonl_path)
    if not rows:
        print(f"No rows in {jsonl_path}")
        return
    stem = jsonl_path.stem
    out_dir = jsonl_path.parent
    # Recompute metrics from raw kg/baseline outputs so we pick up any new
    # @k metrics added since the JSONL was written.
    metrics_rows = []
    for r in rows:
        if "kg" in r and "baseline" in r:
            # synth corpus only stored gold_sources/compounds in the JSONL via
            # the pipeline run; they're inside the per-region `kg` payload's
            # source list — but the original `region` dict isn't preserved.
            # Reconstruct a minimal region-shaped dict from saved fields.
            region = {
                "region_id": r.get("region_id", ""),
                "gold_sources": r.get("gold_sources")
                                or r["kg"].get("gold_sources")
                                or [],
                "gold_compounds_per_source": (
                    r.get("gold_compounds_per_source")
                    or r["kg"].get("gold_compounds_per_source")
                    or {}
                ),
            }
            if not region["gold_sources"] or not region["gold_compounds_per_source"]:
                # Fall back to stored metrics row if gold info missing.
                if "metrics" in r:
                    metrics_rows.append(r["metrics"])
                continue
            metrics_rows.append(_eval_open_region(region, r["kg"], r["baseline"]))
        elif "metrics" in r:
            metrics_rows.append(r["metrics"])
    agg = aggregate_dicts(metrics_rows)
    _plot_pair(agg, _RECALL_PAIRS,
               f"Eval 1 — Source + Lenient Compound Recall (n={agg.get('n', 0)}) [replot]",
               out_dir / f"{stem}_recall.png")
    _plot_pair(agg, _STRICT_PAIRS,
               f"Eval 1 — Strict Recall + Plausibility (n={agg.get('n', 0)}) [replot]",
               out_dir / f"{stem}_strict_plaus.png")
    _plot_boxes(metrics_rows,
                f"Eval 1 — Headline (n={agg.get('n', 0)}) [replot]",
                out_dir / f"{stem}_box.png")
    print(f"Replot done -> {out_dir} ({stem}_recall.png, {stem}_strict_plaus.png, {stem}_box.png)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=CORPUS_DEFAULT)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--critic-model", type=str, default=None,
                   help="Override critic model (default: GEMINI_MODEL_PRO from .env).")
    p.add_argument("--breadth-limit", type=int, default=60,
                   help="Per-region cap on breadth compounds returned from Cypher.")
    p.add_argument("--plot", type=Path, default=None,
                   help="Replot from a saved eval1 JSONL without re-running.")
    p.add_argument("--batch-extract", action="store_true",
                   help="One Gemini call extracts (source,malady) pairs for ALL regions at once.")
    p.add_argument("--compare-to", type=Path, default=None,
                   help="Discovery-mode eval1 JSONL — produces 3-way comparison plot.")
    args = p.parse_args()
    if args.plot:
        replot_from_saved(args.plot)
    else:
        run(args.corpus, limit=args.limit,
            critic_model=args.critic_model, breadth_limit=args.breadth_limit,
            batch_extract=args.batch_extract, compare_to=args.compare_to)
