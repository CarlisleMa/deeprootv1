"""Judge-case construction for the Task A reasoning eval.

A "judge case" is everything needed to grade one critic's output on
one claim:

  - the claim metadata (source, malady, traditional evidence, mapped disease)
  - the deterministic Pass 1 signals
  - the EXACT evidence payload the critic could see (reconstructed from
    the same private helpers task_a_critic.py uses internally)
  - the critic's output (verdict, scores, key_evidence, concerns, rationale)
  - results of cheap deterministic precondition checks

The judge LLM grades each case on visible artifacts only.

Two responsibilities:
  1. Stratified case selection from the Pass 1 verdict set.
  2. Per-case prompt-payload reconstruction so the judge can run
     evidence-fidelity checks (citation grounding) deterministically
     before invoking the LLM.

This keeps the judge harness focused on aggregation + LLM calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.phase2.task_a_critic import CriticAgent
from src.agents.phase2.task_a_validator import TaskAValidator
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

DEFAULT_STRATA: dict[str, dict[str, Any]] = {
    "unsupported_actionable": {
        "n": 8,
        "predicate": (
            lambda v: v["verdict"] == "unsupported" and (v.get("path_count") or 0) > 0
        ),
        "description": "unsupported with non-empty path subgraph",
    },
    "strong_support": {
        "n": 8,
        "predicate": lambda v: v["verdict"] == "strong_support",
        "description": "strong_support — gold + loop closure",
    },
    "gold_no_loop_closure": {
        "n": 8,
        "predicate": (
            lambda v: v.get("top_bucket") == "gold"
            and (v.get("paths_loop_closed") or 0) == 0
        ),
        "description": "diagnostic slice: top_bucket=gold but paths_loop_closed=0",
    },
    "mechanistic_only": {
        "n": 4,
        "predicate": lambda v: v["verdict"] == "mechanistic_only",
        "description": "mechanism plausible, traditional support thin",
    },
    "traditional_only": {
        "n": 2,
        "predicate": lambda v: v["verdict"] == "traditional_only",
        "description": "traditional only, no mechanism",
    },
}


def stratified_sample(
    verdicts: list[dict],
    *,
    strata: dict[str, dict[str, Any]] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Pick a stratified sample from Pass 1 verdicts. Each verdict is
    placed in the FIRST stratum whose predicate it matches, so a verdict
    counted in `gold_no_loop_closure` won't double-count in `unsupported`.

    Returns: {stratum_name: [verdict_dict, ...]}.
    """
    strata = strata or DEFAULT_STRATA
    rng = random.Random(seed)

    pools: dict[str, list[dict]] = {name: [] for name in strata}
    used: set[tuple[str, str]] = set()

    for name, spec in strata.items():
        candidates = [
            v for v in verdicts
            if (v["source"], v["malady"]) not in used and spec["predicate"](v)
        ]
        rng.shuffle(candidates)
        chosen = candidates[: spec["n"]]
        pools[name] = chosen
        for v in chosen:
            used.add((v["source"], v["malady"]))

    return pools


def flatten_strata(strata_pools: dict[str, list[dict]]) -> list[dict]:
    """Flatten a stratified sample into a list, annotating each verdict
    with its stratum_name for downstream slicing."""
    out: list[dict] = []
    for name, verdicts in strata_pools.items():
        for v in verdicts:
            v_copy = dict(v)
            v_copy["_stratum"] = name
            out.append(v_copy)
    return out


# ---------------------------------------------------------------------------
# Prompt payload reconstruction
# ---------------------------------------------------------------------------

class PromptPayloadReconstructor:
    """Rebuild the JSON payload the KG critic would have seen for a given
    Pass 1 verdict. Reuses CriticAgent._pull_enrichment + _build_prompt
    so the reconstruction is identical to what the live critic produces.

    Building this requires a populated `pass1_index` (so sibling verdicts
    can be looked up). The full Pass 1 verdict list is fed in once at
    construction; per-claim reconstruction only does the enrichment
    queries (compound profiles, target genericity, etc.).
    """

    def __init__(
        self,
        client: GraphClient,
        all_pass1_verdicts: list[dict],
    ):
        self._critic = CriticAgent(client)
        # Populate the in-memory Pass 1 index used for sibling lookups.
        self._critic._pass1_index = {
            (v.get("source", ""), v.get("malady", "")): v
            for v in all_pass1_verdicts
        }

    def reconstruct(self, pass1_verdict: dict) -> dict:
        """Return the structured payload the critic would have shown the
        LLM (the dict that becomes part of the prompt JSON).

        IMPORTANT: must mirror the live filter in `task_a_critic._critique_one`:
        only paths with both `path.has_target` and `path.has_disease`
        survive into the critic's payload. Compound-only paths are
        excluded so the reconstructor doesn't make the judge see
        evidence the critic could not.
        """
        all_paths = pass1_verdict.get("paths_top") or []
        full_paths = [
            p for p in all_paths
            if (p.get("path") or {}).get("has_target")
            and (p.get("path") or {}).get("has_disease")
        ]
        source = pass1_verdict.get("source") or ""
        malady = pass1_verdict.get("malady") or ""

        enrichment = self._critic._pull_enrichment(source, malady, full_paths)

        ctx = pass1_verdict.get("context") or {}
        payload = {
            "claim": {
                "source": ctx.get("source") or {"name": source},
                "malady": ctx.get("malady") or {"name": malady},
                "treats_edge": ctx.get("treats_edge") or {},
                "primary_disease": ctx.get("primary_disease") or {
                    "name": pass1_verdict.get("primary_disease"),
                },
                "primary_mapping": ctx.get("primary_mapping") or {},
                "component_diseases": pass1_verdict.get("component_diseases") or [],
                "primary_mapping_quality": pass1_verdict.get("primary_mapping_quality"),
            },
            "pass1_signals": {
                "verdict": pass1_verdict.get("verdict"),
                "rationale": pass1_verdict.get("rationale"),
                "path_count": pass1_verdict.get("path_count"),
                "paths_with_target": pass1_verdict.get("paths_with_target"),
                "paths_with_disease": pass1_verdict.get("paths_with_disease"),
                "paths_loop_closed": pass1_verdict.get("paths_loop_closed"),
                "paths_component_closed": pass1_verdict.get("paths_component_closed"),
                "unique_compounds": pass1_verdict.get("unique_compounds"),
                "unique_targets": pass1_verdict.get("unique_targets"),
                "paths_by_bucket": pass1_verdict.get("paths_by_bucket"),
                "top_bucket": pass1_verdict.get("top_bucket"),
                "top_bucket_path_count": pass1_verdict.get("top_bucket_path_count"),
                "top_bucket_max_score": pass1_verdict.get("top_bucket_max_score"),
                "sibling_sources_count": pass1_verdict.get("sibling_sources"),
                "shared_compounds_with_siblings": pass1_verdict.get("shared_compounds"),
            },
            "evidence_paths": full_paths,
            "compound_profiles": enrichment.get("compound_profiles") or {},
            "target_profiles": enrichment.get("target_profiles") or {},
            "target_convergence_in_source":
                enrichment.get("target_convergence_in_source") or [],
            "source_other_maladies":
                enrichment.get("source_other_maladies") or [],
            "sibling_verdicts": enrichment.get("sibling_verdicts") or [],
        }
        return payload


# ---------------------------------------------------------------------------
# Citation set extraction (used for evidence-fidelity checks)
# ---------------------------------------------------------------------------

def collect_tool_call_citations(critic_output: dict) -> dict[str, set[str]]:
    """For the tool-call baseline, the LLM's visible 'payload' is the
    accumulated tool-call results. Walk `tool_call_log[*].result_summary`
    and pull out compound / target / disease names so the citation
    fidelity check can compare against what the LLM actually saw.

    Each tool call's result_summary is shaped like
    {result_key: {count, sample_first}} (see _summarize_result in
    task_a_baseline_tool_call.py). We pull from sample_first plus walk
    the full result if the agent recorded it.
    """
    compounds: set[str] = set()
    targets: set[str] = set()
    diseases: set[str] = set()

    for entry in critic_output.get("tool_call_log") or []:
        summary = entry.get("result_summary") or {}
        for k, v in summary.items():
            if not isinstance(v, dict):
                continue
            # Walk ALL recorded rows (post-fix #4 round 2 keeps full
            # rows; fall back to sample_first for legacy logs).
            rows = v.get("rows")
            if not rows:
                sample = v.get("sample_first")
                rows = [sample] if isinstance(sample, dict) else []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if k == "compounds":
                    if row.get("name"):
                        compounds.add(_norm(row["name"]))
                    if row.get("inchikey"):
                        compounds.add(_norm(row["inchikey"]))
                elif k == "targets":
                    for tk in ("target", "gene_symbol", "uniprot_id",
                               "target_chembl_id", "pref_name"):
                        if row.get(tk):
                            targets.add(_norm(row[tk]))
                elif k == "diseases":
                    for dk in ("disease", "icd10", "mesh"):
                        if row.get(dk):
                            diseases.add(_norm(row[dk]))
                elif k == "mappings":
                    if row.get("disease"):
                        diseases.add(_norm(row["disease"]))
                elif k == "known_treats":
                    if row.get("disease"):
                        diseases.add(_norm(row["disease"]))
                    if row.get("mesh_heading"):
                        diseases.add(_norm(row["mesh_heading"]))

    # Also seed from the critic_output itself: the primary disease and
    # the source name from the prompt are always seen.
    if critic_output.get("primary_disease"):
        diseases.add(_norm(critic_output["primary_disease"]))

    return {"compounds": compounds, "targets": targets, "diseases": diseases}


def collect_payload_citations(payload: dict) -> dict[str, set[str]]:
    """Pull the set of compound names/inchikeys, target names/genes, and
    disease names that appear in the reconstructed payload. Used to check
    whether a critic's `key_evidence` cites things actually present in
    its visible context.
    """
    compounds: set[str] = set()
    targets: set[str] = set()
    diseases: set[str] = set()

    for path in payload.get("evidence_paths") or []:
        c = path.get("compound") or {}
        if c.get("name"):
            compounds.add(_norm(c["name"]))
        if c.get("inchikey"):
            compounds.add(_norm(c["inchikey"]))
        t = path.get("target") or {}
        for k in ("name", "gene_symbol", "uniprot_id", "chembl_id", "pref_name"):
            if t.get(k):
                targets.add(_norm(t[k]))
        # reached_disease is the nested {name, icd10_code, mesh_id, ...}
        # dict — extract .name (and ontology IDs) rather than stringifying
        # the whole dict.
        rd = path.get("reached_disease")
        if isinstance(rd, dict):
            for k in ("name", "icd10_code", "mesh_id", "snomed_id",
                      "efo_id", "mondo_id", "doid_id"):
                if rd.get(k):
                    diseases.add(_norm(rd[k]))
        elif rd:
            diseases.add(_norm(rd))

    cp = payload.get("compound_profiles") or {}
    for ik, prof in cp.items():
        compounds.add(_norm(ik))
        for kt in prof.get("known_treats") or []:
            if kt.get("disease"):
                diseases.add(_norm(kt["disease"]))
        for tg in prof.get("target_spectrum") or []:
            for k in ("target_name", "gene_symbol", "uniprot_id"):
                if tg.get(k):
                    targets.add(_norm(tg[k]))

    tp = payload.get("target_profiles") or {}
    for tid, prof in tp.items():
        targets.add(_norm(tid))
        for d in prof.get("sample_diseases") or []:
            diseases.add(_norm(d))

    for tc in payload.get("target_convergence_in_source") or []:
        if tc.get("target"):
            targets.add(_norm(tc["target"]))
        if tc.get("target_chembl_id"):
            targets.add(_norm(tc["target_chembl_id"]))

    pd = ((payload.get("claim") or {}).get("primary_disease") or {})
    if pd.get("name"):
        diseases.add(_norm(pd["name"]))

    return {
        "compounds": compounds,
        "targets": targets,
        "diseases": diseases,
    }


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


# ---------------------------------------------------------------------------
# Deterministic precondition checks
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class DeterministicChecks:
    schema_ok: CheckResult
    score_ranges_ok: CheckResult
    citation_fidelity: CheckResult
    loop_closure_consistency: CheckResult
    verdict_delta_consistency: CheckResult

    def all_passed(self) -> bool:
        return all(c.passed for c in (
            self.schema_ok, self.score_ranges_ok, self.citation_fidelity,
            self.loop_closure_consistency, self.verdict_delta_consistency,
        ))


def run_deterministic_checks(
    critic_output: dict,
    pass1_verdict: dict,
    payload_citations: dict[str, set[str]] | None,
) -> DeterministicChecks:
    """Cheap pre-LLM checks. None of these depend on the judge LLM —
    they catch obvious failures before we spend tokens on grading.

    payload_citations may be None for conditions that have no
    reconstructable payload (e.g., text-only baseline). In that case
    citation_fidelity returns 'not_applicable' (passed=True with note).
    """
    return DeterministicChecks(
        schema_ok=_check_schema(critic_output),
        score_ranges_ok=_check_score_ranges(critic_output),
        citation_fidelity=_check_citations(critic_output, payload_citations),
        loop_closure_consistency=_check_loop_closure(critic_output, pass1_verdict),
        verdict_delta_consistency=_check_verdict_delta(critic_output, pass1_verdict),
    )


def _check_schema(out: dict) -> CheckResult:
    required = (
        "verdict", "biological_plausibility", "evidence_coherence",
        "key_evidence", "concerns", "rationale", "requires_human_review",
    )
    missing = [k for k in required if k not in out]
    if missing:
        return CheckResult("schema_ok", False, f"missing fields: {missing}")
    if not isinstance(out.get("key_evidence"), list):
        return CheckResult("schema_ok", False, "key_evidence is not a list")
    if not isinstance(out.get("concerns"), list):
        return CheckResult("schema_ok", False, "concerns is not a list")
    return CheckResult("schema_ok", True)


def _check_score_ranges(out: dict) -> CheckResult:
    bp = out.get("biological_plausibility")
    ec = out.get("evidence_coherence")
    fails = []
    for name, v in (("biological_plausibility", bp), ("evidence_coherence", ec)):
        try:
            f = float(v)
            if not (0.0 <= f <= 1.0):
                fails.append(f"{name}={v} out of [0,1]")
        except (TypeError, ValueError):
            fails.append(f"{name}={v} not a number")
    if fails:
        return CheckResult("score_ranges_ok", False, "; ".join(fails))
    return CheckResult("score_ranges_ok", True)


def _check_citations(
    out: dict,
    citations: dict[str, set[str]] | None,
) -> CheckResult:
    if citations is None:
        return CheckResult(
            "citation_fidelity", True,
            "not_applicable (no reconstructable payload for this condition)",
        )
    miss: list[str] = []
    for ke in out.get("key_evidence") or []:
        compound = _norm(ke.get("compound"))
        target = _norm(ke.get("target"))
        disease = _norm(ke.get("reached_disease"))
        if compound and compound not in citations["compounds"]:
            miss.append(f"compound:{ke.get('compound')}")
        if target and target not in citations["targets"]:
            miss.append(f"target:{ke.get('target')}")
        if disease and disease not in citations["diseases"]:
            miss.append(f"disease:{ke.get('reached_disease')}")
    if miss:
        return CheckResult(
            "citation_fidelity", False,
            f"hallucinated citations: {miss[:5]}{'...' if len(miss) > 5 else ''}",
        )
    return CheckResult("citation_fidelity", True)


def _check_loop_closure(
    out: dict,
    pass1: dict,
) -> CheckResult:
    paths_loop_closed = int(pass1.get("paths_loop_closed") or 0)
    rationale = (out.get("rationale") or "").lower()
    claims_loop_closes = (
        "loop closes" in rationale
        or "loop-closure" in rationale
        or "loop closed" in rationale and "no loop" not in rationale
    )
    if claims_loop_closes and paths_loop_closed == 0:
        return CheckResult(
            "loop_closure_consistency", False,
            "rationale claims loop closure but pass1.paths_loop_closed = 0",
        )
    return CheckResult("loop_closure_consistency", True)


def _check_verdict_delta(
    out: dict,
    pass1: dict,
) -> CheckResult:
    """Recompute verdict_delta and agrees_with_pass1 from the verdicts
    themselves and flag mismatches with the values stored in `out`.

    A condition that lacks access to Pass 1 (text_only, tool_call) is
    permitted to leave agrees_with_pass1=False / verdict_delta=0 —
    the harness fills these in post-hoc. We only flag mismatches when
    the stored value disagrees with our recomputation in a NON-trivial
    way (i.e., the critic actually claimed agreement when there is none).
    """
    from src.agents.phase2.task_a_critic import _verdict_delta

    p1 = pass1.get("verdict") or ""
    p2 = out.get("verdict") or ""
    if not p1 or not p2:
        return CheckResult(
            "verdict_delta_consistency", True,
            "skipped: missing verdict on one side",
        )

    expected_delta = _verdict_delta(p1, p2)
    expected_agrees = (expected_delta == 0)

    # Only flag if the critic ACTIVELY mis-stated the relationship.
    # If the field is missing or default-False, that's fine — many
    # baselines don't see Pass 1 and leave it untouched.
    stated_delta = out.get("verdict_delta")
    stated_agrees = out.get("agrees_with_pass1")

    issues: list[str] = []
    if stated_delta is not None and stated_delta != expected_delta:
        # 0 means "no opinion" for baselines that don't see Pass 1
        if stated_delta != 0:
            issues.append(
                f"verdict_delta={stated_delta} but expected {expected_delta} "
                f"(pass1={p1!r} pass2={p2!r})"
            )
    if stated_agrees is not None and stated_agrees != expected_agrees:
        # False is the default for unseen-Pass-1 baselines
        if stated_agrees is True and not expected_agrees:
            issues.append(
                f"agrees_with_pass1={stated_agrees} but verdicts differ "
                f"({p1!r} vs {p2!r})"
            )
    if issues:
        return CheckResult(
            "verdict_delta_consistency", False, "; ".join(issues),
        )
    return CheckResult("verdict_delta_consistency", True)


# ---------------------------------------------------------------------------
# Case object
# ---------------------------------------------------------------------------

@dataclass
class JudgeCase:
    case_id: str
    stratum: str
    source: str
    malady: str
    primary_disease: str | None

    pass1_signals: dict
    payload: dict | None        # None for text-only baseline; full reconstruction otherwise

    critic_output: dict          # the {verdict, ...} dict for the condition under judgment
    condition: str               # "kg_critic" | "text_only" | "graph_only" | "tool_call" | "kg_critic_<model>"

    deterministic_checks: dict   # DeterministicChecks rendered as dict
    payload_hash: str            # sha256 of payload — for resume keys
    output_hash: str             # sha256 of critic output — for resume keys


def make_case_id(condition: str, source: str, malady: str) -> str:
    return f"{condition}::{source}::{malady}"


def hash_dict(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d or {}, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def build_case(
    *,
    condition: str,
    pass1_verdict: dict,
    critic_output: dict,
    payload: dict | None,
    stratum: str = "",
) -> JudgeCase:
    """Build one judge case (deterministic checks included).

    Citation source by condition:
      - kg_critic (any tier) / graph_only: payload-derived citations
        (the live critic's visible evidence).
      - tool_call: derive from `tool_call_log[*].result_summary` —
        what the agent actually saw via its tool calls.
      - text_only: no payload citations (LLM saw the corpus + general
        biomedical knowledge); citation_fidelity stays 'not_applicable'
        with the LLM judge falling back on biomedical plausibility.
    """
    if payload is not None:
        citations = collect_payload_citations(payload)
    elif "tool_call" in condition or "tool" in condition:
        citations = collect_tool_call_citations(critic_output)
    else:
        citations = None
    checks = run_deterministic_checks(critic_output, pass1_verdict, citations)
    return JudgeCase(
        case_id=make_case_id(condition, pass1_verdict["source"], pass1_verdict["malady"]),
        stratum=stratum or pass1_verdict.get("_stratum", ""),
        source=pass1_verdict["source"],
        malady=pass1_verdict["malady"],
        primary_disease=pass1_verdict.get("primary_disease"),
        pass1_signals={
            "verdict": pass1_verdict.get("verdict"),
            "top_bucket": pass1_verdict.get("top_bucket"),
            "paths_loop_closed": pass1_verdict.get("paths_loop_closed"),
            "path_count": pass1_verdict.get("path_count"),
            "paths_by_bucket": pass1_verdict.get("paths_by_bucket"),
            "rationale": pass1_verdict.get("rationale"),
            "primary_mapping_quality": pass1_verdict.get("primary_mapping_quality"),
        },
        payload=payload,
        critic_output=critic_output,
        condition=condition,
        deterministic_checks={
            "schema_ok": _check_to_dict(checks.schema_ok),
            "score_ranges_ok": _check_to_dict(checks.score_ranges_ok),
            "citation_fidelity": _check_to_dict(checks.citation_fidelity),
            "loop_closure_consistency": _check_to_dict(checks.loop_closure_consistency),
            "verdict_delta_consistency": _check_to_dict(checks.verdict_delta_consistency),
            "all_passed": checks.all_passed(),
        },
        payload_hash=hash_dict(payload) if payload else "",
        output_hash=hash_dict(critic_output),
    )


def _check_to_dict(c: CheckResult) -> dict:
    return {"name": c.name, "passed": c.passed, "detail": c.detail}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_pass1(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("verdicts") or []


def save_cases(cases: list[JudgeCase], path: str | Path) -> None:
    out = {"cases": [_case_to_dict(c) for c in cases]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)


def _case_to_dict(c: JudgeCase) -> dict:
    return {
        "case_id": c.case_id,
        "stratum": c.stratum,
        "source": c.source,
        "malady": c.malady,
        "primary_disease": c.primary_disease,
        "pass1_signals": c.pass1_signals,
        "payload": c.payload,
        "critic_output": c.critic_output,
        "condition": c.condition,
        "deterministic_checks": c.deterministic_checks,
        "payload_hash": c.payload_hash,
        "output_hash": c.output_hash,
    }
