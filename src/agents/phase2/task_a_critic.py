"""Phase 2 — Pass 2: Semantic Critic Agent.

Consumes ClaimVerdict objects produced by TaskAValidator (Pass 1) and
applies LLM judgment that Pass 1's deterministic signals can't:

  * Are the molecular targets mechanistically meaningful for the
    reached disease, or just statistically associated?
  * Does the compound's known pharmacology actually support treating
    this condition?
  * Is the malady → modern disease mapping clinically reasonable given
    the symptom description?
  * Are the supporting compounds promiscuous "everything binds them"
    targets (PTPN1-style) or genuinely specific to the pathway?
  * Does the traditional use story make biological sense as a whole?

Pass 1 already did all the math (path counts, tier buckets, loop
closure, scoring). Pass 2 USES those numbers as inputs, doesn't
re-derive them. The LLM is asked for SEMANTIC nuance, not arithmetic.

Output (CriticVerdict):
  verdict                  — same enum as Pass 1 (direct comparison)
  agrees_with_pass1        — bool; explicit disagreement signal
  biological_plausibility  — 0..1
  evidence_coherence       — 0..1
  key_evidence             — list of (compound, target, disease, why)
  concerns                 — list of (concern_type, explanation)
  rationale                — short prose
  requires_human_review    — bool flag for downstream curation

Disagreement between Pass 1 and Pass 2 is itself the signal — the
paper figure is "what fraction of Pass 1 verdicts does the LLM
disagree with, and which direction." Worth tracking explicitly.

With write_graph=True, stamps task_a_critic_* audit properties on
the TREATS_TRADITIONALLY edge (mirroring the Pass 1 task_a_* pattern).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_MODEL_PRO, make_gemini_client
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_RATE_LIMIT_S = 1.0
_GEMINI_MAX_RETRIES = 4

# Verdict enum kept in sync with TaskAValidator so direct comparison is
# meaningful. Don't add new enum values here without mirroring there.
VERDICT_VALUES = [
    "strong_support",
    "moderate_support",
    "partial_support",
    "unsupported",
    "mechanistic_only",
    "traditional_only",
]

# Concern types the LLM is allowed to flag. Constraining the enum keeps
# pass-2 outputs aggregatable across runs (paper-friendly).
CONCERN_TYPES = [
    "generic_target",          # target binds 100s of unrelated diseases
    "weak_evidence_only",      # all paths are wood-tier (phenotypic / weak)
    "indirect_mechanism",      # pathway connection is plausible but distant
    "wrong_disease_mapping",   # MAPS_TO chose a disease that doesn't fit the symptom
    "syndrome_underutilized",  # syndrome components corroborate but Pass 1 didn't use them
    "promiscuous_compound",    # compound is in 50+ sources, low specificity
    "unverified_evidence",     # critical edges are gemini_unverified / llm_verified only
    "other",
]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """\
You are a biomedical reasoning expert evaluating whether a historical \
medicinal Source plausibly treats a modern disease via known mechanisms.

You receive a structured payload with these sections:

  claim — the (Source, Traditional Malady) claim with the actual quoted \
historical text (`treats_edge.evidence_span`), malady description and \
aliases, primary Modern_Disease mapping with full ontology IDs (icd10, \
mesh, snomed, efo, mondo, doid), and the mapper's per-edge rationale + \
alternatives it considered and rejected.

  pass1_signals — the deterministic Pass 1 verdict with path counts, \
evidence-tier bucket distribution, top-bucket score, and sibling source \
count. These are FACTS about the graph, not your judgment material — \
USE them, don't recount them.

  evidence_paths — the top-N FULL mechanistic chains \
(compound→target→reached_disease) with every audit field: per-edge \
evidence_type + tier + confidence; the assay_description / \
mechanism_action / pchembl_score that actually measured the bioactivity; \
Open Targets overall_score / resolved EFO ID / per-source datasource \
score breakdown; the molecule's full SMILES + ChEMBL ID + COCONUT \
np_likeness; the target's gene_symbol + UniProt; the reached disease's \
ontology IDs.

  compound_profiles — per-compound KNOWN_TREATS list (whether this \
compound is already approved / in clinical phase for any indication, \
sourced from ChEMBL drug_indication) + the compound's TOP-5 target \
spectrum (polypharmacology context).

  target_profiles — per-target GENERICITY: how many distinct \
Modern_Diseases this target relates to graph-wide. A target binding 200+ \
diseases is weak evidence even when the path "closes the loop" \
(PTPN1-style universals).

  target_convergence_in_source — targets hit by ≥2 different compounds \
extracted from THIS source. Multi-compound convergence on one target is \
much stronger than a single-compound hit because it's the \
polypharmacological signature TCM remedies actually rely on.

  source_other_maladies — what OTHER maladies this Source claims to treat, \
with cached Pass-1 verdicts. A "kitchen sink" remedy claiming 12 \
unrelated conditions is less specific than one with 2-3 pathologically \
related conditions.

  sibling_verdicts — other Sources treating THIS same malady, with their \
Pass-1 verdicts. Convergence across multiple historical sources is \
collective evidence; "not_yet_validated" means Pass 1 hasn't been run \
on that sibling yet (treat as unknown, not negative).

Your job is SEMANTIC JUDGMENT, not re-counting:

  1. TARGET QUALITY. Are the molecular targets mechanistically meaningful \
for the reached disease, or just statistically associated? Use \
`target_profiles[*].disease_count` — if a target binds 100s of diseases, \
its appearance here is weak evidence regardless of OT score. Use the \
`assay_description` to judge whether the bioactivity was measured \
in a setting relevant to the disease (e.g. radioligand binding != \
in vivo therapeutic effect).

  2. COMPOUND PHARMACOLOGY. Does the compound's known pharmacology actually \
support treating this specific condition? Use `compound_profiles[*].\
known_treats` and `compound_profiles[*].target_spectrum` to anchor \
your reasoning in the compound's real-world profile, not just one path.

  3. DISEASE MAPPING. Is the malady → modern disease mapping clinically \
defensible given the symptom description? Read \
`claim.malady.description` and `claim.malady.evidence_span`, the actual \
TCM symptom text, and judge against `claim.primary_disease.name`. The \
mapper's `mapper_rationale` and `mapping_alternatives` show what was \
considered and rejected — disagree explicitly if you'd have picked an \
alternative.

  4. POLYPHARMACOLOGY / CONVERGENCE. Use \
`target_convergence_in_source` — if 4 different compounds from the \
same source hit COX-2, that's a strong shared mechanism; weight it \
higher than a single-compound hit. Use `sibling_verdicts` — if 5 of 8 \
sibling sources treating the same malady also have strong_support, \
that's collective evidence convergence.

  5. SPECIFICITY. Use `source_other_maladies` — a source claiming to \
treat 12 unrelated conditions is a kitchen-sink remedy and individual \
claims should weight less than a source specialized to 2-3 related \
conditions.

Verdict ladder (you may agree with or override Pass 1):
  strong_support      — mechanistically convincing, multiple gold/silver \
paths, disease mapping is sound, targets are pathway-specific (low \
genericity), compound's KNOWN_TREATS or known pharmacology fits.
  moderate_support    — convincing but less specific or with weaker paths.
  partial_support     — loop closes but evidence is thin, targets are \
generic, OR the dominant evidence is phenotypic / unverified.
  unsupported         — paths exist but don't reach the disease, or \
target genericity makes them uninformative, or the disease mapping is \
clinically wrong.
  mechanistic_only    — has paths but the malady has no Modern_Disease \
anchor; loop is uncheckable in principle.
  traditional_only    — no compounds linked at all.

Set requires_human_review=true when:
  * you disagree with Pass 1 by more than one rung on the verdict ladder
  * the disease mapping looks clinically wrong (cite the alternative)
  * the dominant evidence is unverified (gemini_unverified, llm_verified)
  * top targets are highly generic (disease_count > 50)
  * you can't reach a confident conclusion either way

Be precise, biomedical, and concise. Your rationale should QUOTE specific \
fields from the input (e.g. "SLC6A2 binds 47 diseases per target_profiles" \
or "the assay_description shows radioligand binding, not therapeutic \
in-vivo activity") — never speculate beyond what's provided.

NUMERIC RANGES (strict):
  biological_plausibility ∈ [0.0, 1.0]   0=incoherent, 1=biologically obvious
  evidence_coherence      ∈ [0.0, 1.0]   0=evidence contradicts, 1=internally consistent

Use floats with two decimals (e.g. 0.65, 0.20). Do NOT output 0–10 or 0–100.\
"""


CRITIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICT_VALUES},
        "agrees_with_pass1": {"type": "boolean"},
        "biological_plausibility": {"type": "number"},
        "evidence_coherence": {"type": "number"},
        "key_evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "compound": {"type": "string"},
                    "target": {"type": "string"},
                    "reached_disease": {"type": "string"},
                    "why_compelling": {"type": "string"},
                },
                "required": ["compound", "target", "reached_disease", "why_compelling"],
            },
        },
        "concerns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "concern_type": {"type": "string", "enum": CONCERN_TYPES},
                    "explanation": {"type": "string"},
                },
                "required": ["concern_type", "explanation"],
            },
        },
        "rationale": {"type": "string"},
        "requires_human_review": {"type": "boolean"},
    },
    "required": [
        "verdict",
        "agrees_with_pass1",
        "biological_plausibility",
        "evidence_coherence",
        "key_evidence",
        "concerns",
        "rationale",
        "requires_human_review",
    ],
}


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class CriticVerdict:
    source: str
    malady: str
    pass1_verdict: str
    pass1_top_bucket: str
    pass1_loop_closed: bool

    verdict: str
    agrees_with_pass1: bool
    verdict_delta: int  # ladder rungs Pass2 differs from Pass1 (signed)
    biological_plausibility: float
    evidence_coherence: float

    key_evidence: list[dict]
    concerns: list[dict]
    rationale: str
    requires_human_review: bool

    raw_response: str = ""           # audit trail
    error: str | None = None         # set if LLM call failed
    skipped: bool = False             # True for non-actionable Pass 1 verdicts


# Verdict ladder for delta computation (rung index, lower = weaker).
_VERDICT_LADDER: dict[str, int] = {
    "traditional_only":  0,
    "mechanistic_only":  1,
    "unsupported":       2,
    "partial_support":   3,
    "moderate_support":  4,
    "strong_support":    5,
}


def _verdict_delta(pass1: str, pass2: str) -> int:
    """Pass 2 minus Pass 1 in ladder rungs. Positive = Pass 2 stronger."""
    return _VERDICT_LADDER.get(pass2, -1) - _VERDICT_LADDER.get(pass1, -1)


def _clamp01(x: Any) -> float:
    """Coerce LLM-emitted numbers into [0.0, 1.0]. Defends against models
    that ignore the prompt's range hint and return 0–10 or 0–100."""
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# Cypher (read-only enrichment queries)
# ---------------------------------------------------------------------------

# (Note: we no longer pull malady context separately — Pass 1's enriched
# `context` field carries it.)


# Per-compound KNOWN_TREATS profile. Phase 1's CompoundDiseaseLinker
# populates this from ChEMBL drug_indication. The LLM uses it to check
# whether a compound is already a known drug for ANY indication, which
# changes how to weigh a traditional claim about a related condition.
_COMPOUND_KNOWN_TREATS_QUERY = """
UNWIND $inchikeys AS ik
MATCH (c:Chemical_Compound {inchikey: ik})-[k:KNOWN_TREATS]->(d:Modern_Disease)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
RETURN ik AS inchikey,
       collect({
         disease: d.name,
         clinical_phase: k.clinical_phase,
         evidence_type: k.evidence_type,
         mesh_heading: k.mesh_heading,
         efo_term: k.efo_term
       }) AS known_treats
"""


# Per-target genericity: how many distinct Modern_Diseases this target
# RELATES_TO across the whole graph. PTPN1-style universals (313 diseases)
# are weak evidence even when paths "close the loop". This is the
# CRITIC_TARGET_GENERICITY query from Phase 1's queries.py, batched.
_TARGET_GENERICITY_QUERY = """
UNWIND $target_chembl_ids AS tid
MATCH (t:Biological_Target {target_chembl_id: tid})-[:RELATES_TO]->(d:Modern_Disease)
WHERE (t.archived IS NULL OR t.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
RETURN tid AS target_chembl_id,
       count(DISTINCT d) AS disease_count,
       collect(DISTINCT d.name)[0..10] AS sample_diseases
"""


# Per-compound full target spectrum: all Biological_Targets the compound
# hits, with evidence_type tier, top-5 by mechanism>strong>moderate>weak>
# phenotypic. Reveals polypharmacology context — a compound that hits
# 15 unrelated targets is harder to reason about than a clean one.
_COMPOUND_TARGET_SPECTRUM_QUERY = """
UNWIND $inchikeys AS ik
MATCH (c:Chemical_Compound {inchikey: ik})-[r:TARGETS]->(t:Biological_Target)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (t.archived IS NULL OR t.archived = false)
WITH ik, t, r,
     CASE r.evidence_type
       WHEN 'chembl_mechanism' THEN 0
       WHEN 'chembl_activity_strong' THEN 1
       WHEN 'chembl_activity_moderate' THEN 2
       WHEN 'chembl_activity_weak' THEN 3
       WHEN 'chembl_phenotypic' THEN 4
       ELSE 5
     END AS rank_score
ORDER BY ik, rank_score, -coalesce(r.pchembl_score, 0)
WITH ik, collect({
       target_name: coalesce(t.gene_symbol, t.name),
       gene_symbol: t.gene_symbol,
       uniprot_id: t.uniprot_id,
       target_type: t.target_type,
       evidence_type: r.evidence_type,
       pchembl_score: r.pchembl_score,
       confidence: r.confidence_score
     })[0..5] AS top_targets,
     count(t) AS total_targets
RETURN ik AS inchikey, top_targets, total_targets
"""


# Multi-compound convergence on a single target within ONE source.
# "3 different compounds from this Source all hit COX-2" is much
# stronger than "1 compound hits COX-2 with weak evidence." Direct
# adaptation of CRITIC_TARGET_CONVERGENCE but anchored on the source.
_TARGET_CONVERGENCE_IN_SOURCE_QUERY = """
MATCH (s:Source {name: $source})<-[:IS_EXTRACTED_FROM]-(c:Chemical_Compound)-[:TARGETS]->(t:Biological_Target)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (c.archived IS NULL OR c.archived = false)
  AND (t.archived IS NULL OR t.archived = false)
WITH t, count(DISTINCT c) AS compound_count, collect(DISTINCT c.name) AS compounds
WHERE compound_count >= 2
RETURN t.target_chembl_id AS target_chembl_id,
       coalesce(t.gene_symbol, t.name) AS target,
       t.target_type AS target_type,
       compound_count,
       compounds[0..8] AS sample_compounds
ORDER BY compound_count DESC
LIMIT 10
"""


# Sibling sources: other Sources that also treat THIS Malady. The critic
# uses these names to look up cached Pass-1 verdicts (in-memory) so the
# LLM can see "5 of the 8 sibling sources treating Wind cold also have
# strong_support — collective convergence." If Pass 1 hasn't been run on
# a sibling, the verdict reads "not_yet_validated".
_SIBLING_SOURCES_QUERY = """
MATCH (s1:Source {name: $source})-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
MATCH (s2:Source)-[:TREATS_TRADITIONALLY]->(m)
WHERE s2.name <> $source
  AND (s2.archived IS NULL OR s2.archived = false)
RETURN DISTINCT s2.name AS sibling
ORDER BY s2.name
"""


# Other maladies this Source claims to treat — reveals "kitchen sink"
# remedies (treats 12 unrelated things → low specificity) versus
# "specialty" remedies (treats 2-3 cold-pattern conditions → coherent
# traditional use). Each sibling claim's primary mapped disease is
# included so the LLM can see whether the claims cluster pathologically.
_SOURCE_OTHER_MALADIES_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE m.name <> $exclude_malady
  AND (m.archived IS NULL OR m.archived = false)
OPTIONAL MATCH (m)-[r_map:MAPS_TO]->(d:Modern_Disease)
  WHERE r_map.is_primary = true
    AND (d.archived IS NULL OR d.archived = false)
RETURN m.name AS malady,
       m.mapper_classification AS mapper_classification,
       d.name AS primary_disease,
       r.confidence_score AS treats_conf
ORDER BY m.name
"""


_WRITE_CRITIC_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
SET r += $props
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class CriticAgent(BaseAgent):
    """Pass 2 — LLM semantic critic over Pass 1 verdicts.

    Default model is Gemini Pro (higher reasoning quality). Per-claim
    LLM call with structured JSON output and a 1s rate-limit gate.
    Idempotent — re-running on the same claim overwrites the audit
    properties with the latest LLM output.
    """

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        skip_non_actionable: bool = True,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL_PRO
        # Pass-1 verdicts that don't benefit from semantic critique
        # (no compounds, or no mapping). Default: skip them to save
        # tokens; the run summary still records them as `skipped=True`.
        self._skip_non_actionable = skip_non_actionable
        self._last_call = 0.0
        # In-memory Pass-1 cache for sibling-source lookups. Built lazily
        # from the `verdicts` list passed into run(); never queries
        # disk or the graph. Keyed (source_name, malady_name) → verdict
        # dict. Sources without a cached verdict are reported as
        # "not_yet_validated" rather than re-running Pass 1 mid-flight.
        self._pass1_index: dict[tuple[str, str], dict] = {}

    @property
    def name(self) -> str:
        return "CriticAgent"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        verdicts: list[dict],
        sibling_verdicts: list[dict] | None = None,
        write_graph: bool = False,
        limit: int | None = None,
        **_: Any,
    ) -> dict:
        """Run pass 2 over a list of pass-1 verdict dicts (as produced by
        TaskAValidator.run() → 'verdicts' key, or loaded from --json-out).

        Args:
            verdicts: claims to critique. By default the pass1_index is
                seeded from this list, so sibling lookups within the
                set work.
            sibling_verdicts: optional FULL Pass 1 verdict list used to
                seed the sibling-lookup index. Pass this when `verdicts`
                is a stratified subset and you want sibling verdicts
                outside the subset to resolve correctly. Defaults to
                None → fall back to `verdicts`.
            write_graph / limit: as before.
        """
        if limit is not None:
            verdicts = verdicts[:limit]

        # Two-tier sibling index seeding:
        #   1. Start from `sibling_verdicts` (full Pass 1) if provided.
        #   2. Overlay with `verdicts` so the entries being critiqued
        #      always reflect the latest values supplied by the caller.
        index_source = sibling_verdicts if sibling_verdicts is not None else verdicts
        self._pass1_index = {
            (v.get("source", ""), v.get("malady", "")): v
            for v in index_source
        }
        for v in verdicts:
            self._pass1_index[(v.get("source", ""), v.get("malady", ""))] = v

        self._log_progress(
            f"Critiquing {len(verdicts)} Pass-1 verdict(s) "
            f"(model={self._model}, skip_non_actionable={self._skip_non_actionable}, "
            f"write_graph={write_graph}, pass1_cache={len(self._pass1_index)})"
        )

        results: list[CriticVerdict] = []
        by_pass2_verdict: dict[str, int] = {}
        agree_count = 0
        disagree_strict = 0    # |delta| >= 1
        disagree_major = 0     # |delta| >= 2
        review_count = 0
        skipped_count = 0
        errors: list[str] = []

        for i, v in enumerate(verdicts, 1):
            cv = self._critique_one(v)
            results.append(cv)

            if cv.skipped:
                skipped_count += 1
            elif cv.error:
                errors.append(f"{cv.source} -> {cv.malady}: {cv.error}")
            else:
                by_pass2_verdict[cv.verdict] = by_pass2_verdict.get(cv.verdict, 0) + 1
                if cv.agrees_with_pass1:
                    agree_count += 1
                if abs(cv.verdict_delta) >= 1:
                    disagree_strict += 1
                if abs(cv.verdict_delta) >= 2:
                    disagree_major += 1
                if cv.requires_human_review:
                    review_count += 1

            if write_graph and not cv.skipped and not cv.error:
                self._write_audit_properties(cv)

            if i % 10 == 0 or i == len(verdicts):
                self._log_progress(f"  {i}/{len(verdicts)}")

        critiqued = len(verdicts) - skipped_count - len(errors)
        return {
            "verdicts_in": len(verdicts),
            "critiqued": critiqued,
            "skipped": skipped_count,
            "errors_count": len(errors),
            "by_pass2_verdict": dict(
                sorted(by_pass2_verdict.items(), key=lambda kv: -kv[1])
            ),
            "agreement_rate": (
                round(agree_count / critiqued, 3) if critiqued else 0.0
            ),
            "disagreement_one_rung": disagree_strict,
            "disagreement_two_plus_rungs": disagree_major,
            "requires_human_review": review_count,
            "errors": errors,
            "results": [self._cv_to_dict(cv) for cv in results],
        }

    # ------------------------------------------------------------------
    # Per-verdict critique
    # ------------------------------------------------------------------

    def _critique_one(self, pass1: dict) -> CriticVerdict:
        source = pass1.get("source", "")
        malady = pass1.get("malady", "")
        p1_verdict = pass1.get("verdict", "")
        p1_top_bucket = pass1.get("top_bucket", "unrated")
        p1_loop_closed = bool(pass1.get("paths_loop_closed") or 0)

        # Skip claims where there's no semantic substrate. The summary
        # still records them as `skipped=True` so eval tables can
        # account for them.
        if self._skip_non_actionable and p1_verdict in (
            "claim_not_found", "traditional_only", "mechanistic_only",
        ):
            return CriticVerdict(
                source=source, malady=malady,
                pass1_verdict=p1_verdict,
                pass1_top_bucket=p1_top_bucket,
                pass1_loop_closed=p1_loop_closed,
                verdict=p1_verdict,
                agrees_with_pass1=True,
                verdict_delta=0,
                biological_plausibility=0.0,
                evidence_coherence=0.0,
                key_evidence=[],
                concerns=[],
                rationale="Pass-1 verdict is non-actionable (no compounds or no mapping); skipped.",
                requires_human_review=False,
                skipped=True,
            )

        # Filter compound-only paths out of the LLM payload. They have
        # bucket=ex_tier (often gold) and a single-edge product score,
        # which would crowd out real mechanistic chains. Pass 1's
        # path_count signal already accounts for them at the bucket
        # distribution level.
        full_paths = [
            p for p in (pass1.get("paths_top") or [])
            if (p.get("path") or {}).get("has_target")
            and (p.get("path") or {}).get("has_disease")
        ]

        # Pull cross-cutting enrichment from the graph (read-only).
        enrichment = self._pull_enrichment(source, malady, full_paths)

        try:
            llm_response = self._call_gemini(pass1, full_paths, enrichment)
        except Exception as e:
            logger.warning("Critic Gemini call failed for %r → %r: %s", source, malady, e)
            return CriticVerdict(
                source=source, malady=malady,
                pass1_verdict=p1_verdict,
                pass1_top_bucket=p1_top_bucket,
                pass1_loop_closed=p1_loop_closed,
                verdict=p1_verdict,
                agrees_with_pass1=True,
                verdict_delta=0,
                biological_plausibility=0.0,
                evidence_coherence=0.0,
                key_evidence=[],
                concerns=[],
                rationale="",
                requires_human_review=False,
                error=str(e),
            )

        p2_verdict = llm_response.get("verdict") or p1_verdict
        delta = _verdict_delta(p1_verdict, p2_verdict)
        # Trust the LLM's self-reported agreement flag; cross-check
        # against the computed delta. Sometimes the LLM picks the same
        # verdict but writes agrees=false (or vice versa). The delta is
        # ground-truthier since it's derived.
        agrees = (delta == 0)

        return CriticVerdict(
            source=source, malady=malady,
            pass1_verdict=p1_verdict,
            pass1_top_bucket=p1_top_bucket,
            pass1_loop_closed=p1_loop_closed,
            verdict=p2_verdict,
            agrees_with_pass1=agrees,
            verdict_delta=delta,
            biological_plausibility=_clamp01(
                llm_response.get("biological_plausibility")
            ),
            evidence_coherence=_clamp01(
                llm_response.get("evidence_coherence")
            ),
            key_evidence=llm_response.get("key_evidence") or [],
            concerns=llm_response.get("concerns") or [],
            rationale=str(llm_response.get("rationale") or ""),
            requires_human_review=bool(
                llm_response.get("requires_human_review") or False
            ),
            raw_response=json.dumps(llm_response, ensure_ascii=False),
        )

    # ------------------------------------------------------------------
    # Graph enrichment lookups (read-only)
    # ------------------------------------------------------------------

    def _pull_enrichment(
        self,
        source: str,
        malady: str,
        full_paths: list[dict],
    ) -> dict:
        """Run the four cross-cutting enrichment queries plus the
        sibling-verdict cache lookup. Returns a single dict with sections:

          compound_profiles: {inchikey → {known_treats, target_spectrum, ...}}
          target_profiles:   {target_chembl_id → {disease_count, ...}}
          target_convergence: list of {target, compound_count, sample_compounds}
          source_other_maladies: list of {malady, primary_disease, ...}
          sibling_verdicts: list of {sibling_source, verdict, top_bucket, ...}

        All queries are batched on the inchikeys / target_chembl_ids
        appearing in `full_paths`, so the cost is constant per claim
        regardless of compound/target count beyond top-N.
        """
        inchikeys = sorted({
            (p.get("compound") or {}).get("inchikey")
            for p in full_paths
            if (p.get("compound") or {}).get("inchikey")
        })
        target_ids = sorted({
            (p.get("target") or {}).get("chembl_id")
            for p in full_paths
            if (p.get("target") or {}).get("chembl_id")
        })

        compound_profiles = self._pull_compound_profiles(inchikeys)
        target_profiles = self._pull_target_profiles(target_ids)
        target_convergence = self._pull_target_convergence_in_source(source)
        source_other_maladies = self._pull_source_other_maladies(source, malady)
        sibling_verdicts = self._pull_sibling_verdicts(source, malady)

        return {
            "compound_profiles": compound_profiles,
            "target_profiles": target_profiles,
            "target_convergence_in_source": target_convergence,
            "source_other_maladies": source_other_maladies,
            "sibling_verdicts": sibling_verdicts,
        }

    def _pull_compound_profiles(
        self, inchikeys: list[str],
    ) -> dict[str, dict]:
        """KNOWN_TREATS + target spectrum per compound (batched by inchikey)."""
        if not inchikeys:
            return {}
        kt_rows = self.client.run(
            _COMPOUND_KNOWN_TREATS_QUERY, {"inchikeys": inchikeys},
        )
        spectrum_rows = self.client.run(
            _COMPOUND_TARGET_SPECTRUM_QUERY, {"inchikeys": inchikeys},
        )
        out: dict[str, dict] = {ik: {"known_treats": [], "target_spectrum": []}
                                for ik in inchikeys}
        for r in kt_rows:
            ik = r.get("inchikey")
            if ik in out:
                out[ik]["known_treats"] = r.get("known_treats") or []
        for r in spectrum_rows:
            ik = r.get("inchikey")
            if ik in out:
                out[ik]["target_spectrum"] = r.get("top_targets") or []
                out[ik]["total_targets"] = r.get("total_targets") or 0
        return out

    def _pull_target_profiles(
        self, target_chembl_ids: list[str],
    ) -> dict[str, dict]:
        """Genericity (disease count) per target (batched)."""
        if not target_chembl_ids:
            return {}
        rows = self.client.run(
            _TARGET_GENERICITY_QUERY, {"target_chembl_ids": target_chembl_ids},
        )
        out: dict[str, dict] = {tid: {"disease_count": 0, "sample_diseases": []}
                                for tid in target_chembl_ids}
        for r in rows:
            tid = r.get("target_chembl_id")
            if tid in out:
                out[tid]["disease_count"] = int(r.get("disease_count") or 0)
                out[tid]["sample_diseases"] = r.get("sample_diseases") or []
        return out

    def _pull_target_convergence_in_source(
        self, source: str,
    ) -> list[dict]:
        """Targets hit by ≥2 compounds within this Source (multi-compound
        convergence — strong polypharmacological signal)."""
        rows = self.client.run(
            _TARGET_CONVERGENCE_IN_SOURCE_QUERY, {"source": source},
        )
        return [
            {
                "target": r.get("target"),
                "target_chembl_id": r.get("target_chembl_id"),
                "target_type": r.get("target_type"),
                "compound_count": int(r.get("compound_count") or 0),
                "sample_compounds": r.get("sample_compounds") or [],
            }
            for r in rows
        ]

    def _pull_source_other_maladies(
        self, source: str, exclude_malady: str,
    ) -> list[dict]:
        """Other maladies this Source treats (with their primary mapped
        Modern_Disease + cached Pass-1 verdict if available)."""
        rows = self.client.run(
            _SOURCE_OTHER_MALADIES_QUERY,
            {"source": source, "exclude_malady": exclude_malady},
        )
        out: list[dict] = []
        for r in rows:
            other_malady = r.get("malady") or ""
            cached = self._pass1_index.get((source, other_malady))
            cached_verdict = cached.get("verdict") if cached else "not_yet_validated"
            out.append({
                "malady": other_malady,
                "mapper_classification": r.get("mapper_classification"),
                "primary_disease": r.get("primary_disease"),
                "treats_confidence": r.get("treats_conf"),
                "pass1_verdict": cached_verdict,
            })
        return out

    def _pull_sibling_verdicts(
        self, source: str, malady: str,
    ) -> list[dict]:
        """Other Sources treating THIS malady, with their cached Pass-1
        verdicts. Sources without a cached verdict are reported as
        'not_yet_validated' — the critic should weigh consensus, not
        require it."""
        rows = self.client.run(
            _SIBLING_SOURCES_QUERY, {"source": source, "malady": malady},
        )
        out: list[dict] = []
        for r in rows:
            sibling = r.get("sibling") or ""
            cached = self._pass1_index.get((sibling, malady))
            if cached:
                out.append({
                    "sibling_source": sibling,
                    "pass1_verdict": cached.get("verdict"),
                    "pass1_top_bucket": cached.get("top_bucket"),
                    "pass1_paths_loop_closed": cached.get("paths_loop_closed"),
                    "pass1_top_bucket_max_score": cached.get("top_bucket_max_score"),
                })
            else:
                out.append({
                    "sibling_source": sibling,
                    "pass1_verdict": "not_yet_validated",
                })
        return out

    # ------------------------------------------------------------------
    # Gemini call
    # ------------------------------------------------------------------

    def _call_gemini(
        self, pass1: dict, full_paths: list[dict], enrichment: dict,
    ) -> dict:
        # Polite rate-limit gate (mirrors MaladyDiseaseMapper).
        elapsed = time.time() - self._last_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_call = time.time()

        prompt = self._build_prompt(pass1, full_paths, enrichment)

        last_err: Exception | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                response = self._gemini.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=CRITIC_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=CRITIC_RESPONSE_SCHEMA,
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < _GEMINI_MAX_RETRIES - 1:
                    wait_s = 5 * (2 ** attempt)
                    logger.info(
                        "Critic rate-limited, backoff %ds (attempt %d/%d)",
                        wait_s, attempt + 1, _GEMINI_MAX_RETRIES,
                    )
                    time.sleep(wait_s)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Critic call failed without raising")

    @staticmethod
    def _build_prompt(
        pass1: dict, full_paths: list[dict], enrichment: dict,
    ) -> str:
        """Assemble the full critic payload. Sections:

          claim          — anchor metadata: actual TREATS evidence span,
                           malady description, mapper classification,
                           primary disease ontology IDs, mapping rationale +
                           alternatives. Pulled straight from Pass 1's
                           `context` dict — no re-querying.
          pass1_signals  — verdict + path counts + bucket distribution.
                           The LLM uses these as features, not arithmetic.
          evidence_paths — top-N FULL CHAINS only (compound→target→disease).
                           Each path carries the full edge property bundle
                           (assay_description, OT scores, mapper rationale,
                           ontology IDs).
          compound_profiles — per-compound KNOWN_TREATS (repurposing context)
                           + full target spectrum (polypharmacology).
          target_profiles — per-target genericity (PTPN1-binds-313 problem).
          target_convergence_in_source — multi-compound→target hits within
                           THIS source (strong polypharmacological signal).
          source_other_maladies — Source's other claims with cached
                           Pass-1 verdicts (kitchen-sink-vs-specialty signal).
          sibling_verdicts — other Sources treating THIS malady with their
                           cached Pass-1 verdicts (collective evidence
                           convergence).
        """
        ctx = pass1.get("context") or {}
        payload = {
            "claim": {
                "source": ctx.get("source") or {"name": pass1.get("source")},
                "malady": ctx.get("malady") or {"name": pass1.get("malady")},
                "treats_edge": ctx.get("treats_edge") or {},
                "primary_disease": ctx.get("primary_disease") or {
                    "name": pass1.get("primary_disease"),
                },
                "primary_mapping": ctx.get("primary_mapping") or {},
                "component_diseases": pass1.get("component_diseases") or [],
                "primary_mapping_quality": pass1.get("primary_mapping_quality"),
            },
            "pass1_signals": {
                "verdict": pass1.get("verdict"),
                "rationale": pass1.get("rationale"),
                "path_count": pass1.get("path_count"),
                "paths_with_target": pass1.get("paths_with_target"),
                "paths_with_disease": pass1.get("paths_with_disease"),
                "paths_loop_closed": pass1.get("paths_loop_closed"),
                "paths_component_closed": pass1.get("paths_component_closed"),
                "unique_compounds": pass1.get("unique_compounds"),
                "unique_targets": pass1.get("unique_targets"),
                "paths_by_bucket": pass1.get("paths_by_bucket"),
                "top_bucket": pass1.get("top_bucket"),
                "top_bucket_path_count": pass1.get("top_bucket_path_count"),
                "top_bucket_max_score": pass1.get("top_bucket_max_score"),
                "sibling_sources_count": pass1.get("sibling_sources"),
                "shared_compounds_with_siblings": pass1.get("shared_compounds"),
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

        return (
            "Evaluate the following claim. Output structured JSON per the "
            "schema. Use the cross-cutting signals below (compound profiles, "
            "target genericity, multi-compound convergence, sibling and "
            "self-source patterns) to refine the deterministic Pass-1 "
            "verdict — they capture pharmacological context Pass 1 cannot.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    # ------------------------------------------------------------------
    # Audit writeback
    # ------------------------------------------------------------------

    def _write_audit_properties(self, cv: CriticVerdict) -> None:
        props: dict[str, Any] = {
            "task_a_critic_verdict": cv.verdict,
            "task_a_critic_agrees_with_pass1": cv.agrees_with_pass1,
            "task_a_critic_verdict_delta": cv.verdict_delta,
            "task_a_critic_biological_plausibility": cv.biological_plausibility,
            "task_a_critic_evidence_coherence": cv.evidence_coherence,
            "task_a_critic_rationale": cv.rationale,
            "task_a_critic_requires_human_review": cv.requires_human_review,
            "task_a_critic_concerns": json.dumps(
                cv.concerns, ensure_ascii=False,
            ),
            "task_a_critic_key_evidence": json.dumps(
                cv.key_evidence, ensure_ascii=False,
            ),
            "task_a_critic_attempted_at": dt.datetime.utcnow().isoformat(),
            "task_a_critic_model": self._model,
        }
        self.client.run_write(
            _WRITE_CRITIC_QUERY,
            {"source": cv.source, "malady": cv.malady, "props": props},
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _cv_to_dict(cv: CriticVerdict) -> dict:
        return asdict(cv)
