"""Phase 2 — Task A Baseline C: Graph-only (no LLM).

The "deterministic graph alone" baseline cell for the Task A reasoning
judge eval. Projects each Pass 1 verdict into the same JSON schema as
the LLM critic so the judge can grade all four conditions on identical
fields.

This is NOT a separate computation — it's a structural rendering of
TaskAValidator output into the critic-shaped schema. By construction,
this baseline:

  - has the same `verdict` as Pass 1 (the deterministic ladder)
  - cites real graph edges in `key_evidence` (top paths by tier)
  - cannot have hallucinated evidence (every field is graph-derived)
  - cannot reason about clinical mapping appropriateness, target
    genericity, or pharmacological plausibility — those go to 0.5 / `other`
  - cannot produce a free-form rationale beyond a templated string

The judge will score it low on `reasoning_coherence` and `actionability`.
That's the point — it tests whether the *LLM* layer adds reasoning
value over the *graph* alone, holding the verdict fixed.

No LLM calls, no API costs, no rate limits. Pure projection.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from src.agents.phase2.task_a_critic import _verdict_delta

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier → plausibility / coherence projection tables
# ---------------------------------------------------------------------------

# Map Pass 1's top_bucket onto biological_plausibility ∈ [0,1]. The
# bucket is the WEAKEST link in the strongest path, so a 'gold' bucket
# means at least one fully-gold compound→target→disease chain exists.
_BUCKET_TO_PLAUSIBILITY = {
    "gold":     0.85,
    "silver":   0.65,
    "bronze":   0.45,
    "wood":     0.25,
    "unrated":  0.15,
}


# Mapping-quality penalty for plausibility — a silver/missing mapping
# means the malady→disease link itself is uncertain.
_MAPPING_PENALTY = {
    "gold":    0.0,
    "silver":  -0.10,
    "missing": -0.20,
}


def _plausibility(top_bucket: str | None, mapping_quality: str | None) -> float:
    base = _BUCKET_TO_PLAUSIBILITY.get(top_bucket or "unrated", 0.15)
    base += _MAPPING_PENALTY.get(mapping_quality or "missing", -0.20)
    return max(0.0, min(1.0, base))


def _coherence(path_count: int, paths_loop_closed: int) -> float:
    """Loop-closure ratio bounded to [0, 1]. Zero paths → 0.0; one or
    more loop-closed paths out of N total → ratio."""
    if path_count <= 0:
        return 0.0
    return max(0.0, min(1.0, paths_loop_closed / path_count))


# ---------------------------------------------------------------------------
# Concern derivation (rule-based)
# ---------------------------------------------------------------------------

def _derive_concerns(verdict: dict) -> list[dict]:
    """Rule-based concerns from Pass 1 fields. No LLM, no semantics —
    just structural flags. The judge will rate this LOW on actionability
    relative to the LLM critic.
    """
    concerns: list[dict] = []

    paths_loop_closed = int(verdict.get("paths_loop_closed") or 0)
    path_count = int(verdict.get("path_count") or 0)
    top_bucket = verdict.get("top_bucket") or "unrated"
    mapping_quality = verdict.get("primary_mapping_quality") or "missing"
    paths_with_target = int(verdict.get("paths_with_target") or 0)
    has_compounds = bool(verdict.get("has_compounds"))
    has_mapping = bool(verdict.get("has_mapping"))
    paths_by_bucket = verdict.get("paths_by_bucket") or {}

    if not has_mapping:
        concerns.append({
            "concern_type": "wrong_disease_mapping",
            "explanation": "No primary Modern_Disease mapping exists for this malady.",
        })
    elif mapping_quality == "silver":
        concerns.append({
            "concern_type": "wrong_disease_mapping",
            "explanation": (
                "Mapping quality is silver (mapper_unverified) — the "
                "malady→disease link is approximate."
            ),
        })

    if has_compounds and paths_with_target == 0:
        concerns.append({
            "concern_type": "weak_evidence_only",
            "explanation": (
                "Compounds exist for this Source but no TARGETS edges — "
                "no mechanistic chain can be constructed."
            ),
        })

    if path_count > 0 and paths_loop_closed == 0:
        concerns.append({
            "concern_type": "indirect_mechanism",
            "explanation": (
                f"{path_count} mechanistic paths exist but none loop-close "
                "to the mapped disease (no Compound→Target→Disease chain "
                "reaching this exact disease)."
            ),
        })

    if top_bucket in ("wood", "unrated"):
        concerns.append({
            "concern_type": "weak_evidence_only",
            "explanation": (
                f"Top evidence bucket is '{top_bucket}' — best path uses "
                "weak/phenotypic activities or unrated edges."
            ),
        })

    if (paths_by_bucket.get("gold", 0) > 0
            and paths_loop_closed == 0):
        concerns.append({
            "concern_type": "indirect_mechanism",
            "explanation": (
                f"{paths_by_bucket.get('gold', 0)} gold-tier paths exist but "
                "none reach the mapped disease — global gold-bucket evidence "
                "is not disease-specific."
            ),
        })

    if not concerns:
        concerns.append({
            "concern_type": "other",
            "explanation": "No structural concerns flagged by Pass 1 rules.",
        })
    return concerns


def _derive_key_evidence(verdict: dict, top_n: int = 3) -> list[dict]:
    """Project the top-N paths into the {compound, target, reached_disease,
    why_compelling} schema. why_compelling is a templated tier descriptor.

    paths_top has nested shape from TaskAValidator._path_to_dict:
      {compound: {name, inchikey, ...}, target: {name, gene_symbol, ...},
       reached_disease: {name, icd10_code, ...},
       edges: {ex: {tier, ...}, tgt: {tier, ...}, rel: {tier, ...}},
       path: {bucket, score, loop_closed, component_closed, has_target,
              has_disease}}

    Prefer paths with full chains (has_target AND has_disease) since
    those are what carry meaningful tier information.
    """
    all_paths = verdict.get("paths_top") or []
    # ONLY emit key_evidence for full mechanistic chains. Compound-only
    # paths (no target / no disease) would force placeholders like
    # '<no target>' / '<no disease>' into the evidence struct, which
    # the judge would correctly flag as non-evidence. If no full chains
    # exist, return [] — the rationale + concerns already explain the
    # absence of mechanism.
    chain_paths = [
        p for p in all_paths
        if (p.get("path") or {}).get("has_target")
        and (p.get("path") or {}).get("has_disease")
    ]
    if not chain_paths:
        return []

    out: list[dict] = []
    for p in chain_paths[:top_n]:
        compound_obj = p.get("compound") or {}
        compound = compound_obj.get("name") or compound_obj.get("inchikey")
        if not compound:
            continue  # skip paths even more degenerate than expected

        target_obj = p.get("target") or {}
        target = (
            target_obj.get("gene_symbol")
            or target_obj.get("name")
            or target_obj.get("pref_name")
        )
        if not target:
            continue

        reached_obj = p.get("reached_disease") or {}
        if isinstance(reached_obj, dict):
            reached = reached_obj.get("name")
        else:
            reached = str(reached_obj) if reached_obj else None
        if not reached:
            continue

        edges = p.get("edges") or {}
        ex_tier = (edges.get("ex") or {}).get("tier") or "unrated"
        tgt_tier = (edges.get("tgt") or {}).get("tier") or "unrated"
        rel_tier = (edges.get("rel") or {}).get("tier") or "unrated"

        path_obj = p.get("path") or {}
        bucket = path_obj.get("bucket") or "unrated"
        loop = bool(path_obj.get("loop_closed", False))
        component = bool(path_obj.get("component_closed", False))

        out.append({
            "compound": compound,
            "target": target,
            "reached_disease": reached,
            "why_compelling": (
                f"Path bucket={bucket} (extracted={ex_tier}, "
                f"targets={tgt_tier}, relates={rel_tier}); "
                f"loop_closed={loop}, component_closed={component}."
            ),
        })
    return out


def _build_rationale(verdict: dict) -> str:
    """One templated sentence summarizing Pass 1's deterministic reasoning."""
    return (
        f"Pass 1 deterministic verdict: '{verdict.get('verdict')}'. "
        f"Top bucket = {verdict.get('top_bucket')} "
        f"({verdict.get('top_bucket_path_count')} of "
        f"{verdict.get('path_count')} paths); "
        f"loop-closed = {verdict.get('paths_loop_closed')}; "
        f"unique compounds = {verdict.get('unique_compounds')}; "
        f"primary mapping quality = {verdict.get('primary_mapping_quality')}. "
        f"Pass 1 rationale: {verdict.get('rationale') or '<empty>'}"
    )


def _requires_human_review(verdict: dict) -> bool:
    """Flag claims where the deterministic verdict carries non-trivial
    structural ambiguity. Mirrors what the LLM critic might flag, but
    rule-based.
    """
    paths_by_bucket = verdict.get("paths_by_bucket") or {}
    return bool(
        verdict.get("primary_mapping_quality") == "silver"
        or (paths_by_bucket.get("gold", 0) > 0
            and (verdict.get("paths_loop_closed") or 0) == 0)
        or verdict.get("verdict") == "moderate_support"  # ambiguous middle
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

@dataclass
class BaselineGraphVerdict:
    source: str
    malady: str
    primary_disease: str | None

    verdict: str
    agrees_with_pass1: bool         # always True — verdict IS Pass 1
    verdict_delta: int              # always 0
    biological_plausibility: float
    evidence_coherence: float

    key_evidence: list[dict]
    concerns: list[dict]
    rationale: str
    requires_human_review: bool

    raw_response: str = ""
    error: str | None = None
    skipped: bool = False
    model: str = "graph_only_deterministic"
    duration_s: float = 0.0


def project_pass1_to_critic_schema(verdict: dict) -> BaselineGraphVerdict:
    """Render a single Pass 1 ClaimVerdict (as dict) into the critic's
    JSON schema. Pure function — no I/O, no LLM.
    """
    pass1_verdict = verdict.get("verdict") or "unsupported"
    cv = BaselineGraphVerdict(
        source=verdict.get("source") or "",
        malady=verdict.get("malady") or "",
        primary_disease=verdict.get("primary_disease"),
        verdict=pass1_verdict,
        agrees_with_pass1=True,
        verdict_delta=_verdict_delta(pass1_verdict, pass1_verdict),
        biological_plausibility=_plausibility(
            verdict.get("top_bucket"),
            verdict.get("primary_mapping_quality"),
        ),
        evidence_coherence=_coherence(
            int(verdict.get("path_count") or 0),
            int(verdict.get("paths_loop_closed") or 0),
        ),
        key_evidence=_derive_key_evidence(verdict),
        concerns=_derive_concerns(verdict),
        rationale=_build_rationale(verdict),
        requires_human_review=_requires_human_review(verdict),
        raw_response=json.dumps({"projection": "deterministic_pass1_render"}),
    )
    return cv


def run(
    *,
    pass1_verdicts: list[dict],
    claims: list[dict] | None = None,
) -> dict:
    """Project a list of Pass 1 verdicts into critic-schema verdicts.

    Args:
        pass1_verdicts: list of dicts as produced by
            TaskAValidator.run()['verdicts'].
        claims: optional filter — only project verdicts matching these
            (source, malady) pairs.

    Returns a summary dict in the same shape as TaskAValidator.run().
    """
    if claims is not None:
        wanted = {(c["source"], c["malady"]) for c in claims}
        pass1_verdicts = [
            v for v in pass1_verdicts
            if (v.get("source"), v.get("malady")) in wanted
        ]

    results = [project_pass1_to_critic_schema(v) for v in pass1_verdicts]

    return {
        "agent": "BaselineTaskAGraphOnly",
        "model": "graph_only_deterministic",
        "claims_total": len(results),
        "errors": [],
        "verdicts": [asdict(r) for r in results],
        "completed_at": dt.datetime.utcnow().isoformat(),
    }
