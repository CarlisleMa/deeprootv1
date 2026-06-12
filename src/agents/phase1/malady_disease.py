"""Phase 1 — Malady → Disease Mapping Agent.

Generate-then-verify, per-malady, with caching. Same architectural shape as
`ExtractionAuditor` and the v2 `SourceCompoundLinker`.

Pipeline (one Gemini call per malady):

  STEP A: GENERATE
    Gemini receives (name, description, evidence_span, aliases) and returns
    structured JSON with one of six `type` values:
      "disease"            — concrete diagnosis, exactly 1 mapping
      "symptom"            — single symptom, exactly 1 mapping
      "syndrome"           — TCM concept spanning multiple modern conditions
                             (1 primary + 0-2 syndrome_components)
      "ambiguous"          — 1 best mapping + alternatives (audit only)
      "tcm_no_equivalent"  — no modern equivalent (0 mappings)
      "error"              — reserved for tool failures, never returned by LLM

    Each mapping carries `name` + `ontology_hint` + `role` + `rationale`.
    The LLM emits NAMES ONLY — no ICD-10/MeSH/SNOMED codes. Codes are
    derived from API lookup in step B.

  STEP B: VERIFY
    For each LLM-suggested name, query OntologyClient (ICD-10 + MeSH +
    SNOMED) in parallel. Prioritise the hinted ontology for exact-match
    short-circuit. Codes come from API responses only.

  STEP C: WRITE
    Properties on the Malady (audit metadata, no edge needed):
      mapper_status            : "linked" | "unverified" | "tcm_no_equivalent" | "error"
      mapper_classification    : the `type` from step A
      mapper_attempted_at      : ISO timestamp
      mapper_raw_response      : Gemini's literal JSON (audit)

    Per-mapping MAPS_TO edges (1 to 3 edges per malady):
      confidence_score, mapping_source, mapping_rank, is_primary,
      mapping_role, mapping_alternatives (only on primary edge for
      type="ambiguous"), mapper_rationale, requires_review

Verifier discipline:
  - The verifier ONLY accepts exact ontology matches. Fuzzy hits never
    promote to a `_Verified` — they're treated as `gemini_unverified`.
  - Strict mode (default): unverified primaries are recorded with
    `mapper_status="unverified"` but NO MAPS_TO / Modern_Disease are
    written. They're picked up again by `--retry-misses`.
  - `allow_unverified=True`: unverified edges ARE written, with
    `requires_review=true` on the edge. Use this when recall matters
    more than precision.

Cap on N (per malady): up to 3 edges total. Rank-2 and rank-3 (always
syndrome_components) are written ONLY when their verifier returned an
exact API match. Anything unverified at rank 2-3 is dropped to the
`mapping_alternatives` JSON on the primary edge.

Idempotent: re-runs skip Maladies whose `mapper_status` is `"linked"`.
`--retry-misses` also re-processes `tcm_no_equivalent`, `unverified`,
and `error`. `--force-remap` ignores status entirely AND deletes the
malady's prior MAPS_TO edges before writing the new plan, so
re-runs never leave stale mappings attached.

Known limitations (deferred — flagged for follow-up, not bugs):
  - Modern_Disease identity is the LLM-emitted disease name string.
    Slight label variants across runs (e.g. "common cold" vs
    "Common cold") can create duplicate disease nodes. A future fix
    would key MERGE on a stable ontology ID (icd10_code if present,
    else mesh_id, else name) — schema migration required.
  - The disease node's `verified_by` property records only the FIRST
    ontology that confirmed the name. Per-edge `mapping_source` is the
    authoritative provenance.
  - Dry-run mode (`--write-graph` not set) still calls Gemini and
    OntologyClient — only the graph writes are skipped. Use `--limit`
    for cheap behavioural previews.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_MODEL, make_gemini_client
from src.data.ontology_client import OntologyClient
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_RATE_LIMIT_S = 1.0      # Polite gap between Gemini calls
_GEMINI_MAX_RETRIES = 4
_MAX_EDGES_PER_MALADY = 3       # 1 primary + at most 2 syndrome_components

# Flat per-evidence-type confidence priors. Keys are
# (role, match_type, source_db). `unverified` collapses across DBs and is
# only used when --allow-unverified is set; in strict mode (default) the
# mapper does NOT write unverified edges at all.
CONFIDENCE: dict[tuple[str, str, str | None], float] = {
    ("primary",            "exact", "icd10"):    0.85,
    ("primary",            "exact", "mesh"):     0.80,
    ("primary",            "exact", "snomed"):   0.80,
    ("primary",            "unverified", None):  0.50,
    # syndrome_components are only ever written when exact (per the cap
    # gate). -0.15 vs the equivalent primary entry.
    ("syndrome_component", "exact", "icd10"):    0.70,
    ("syndrome_component", "exact", "mesh"):     0.65,
    ("syndrome_component", "exact", "snomed"):   0.65,
}

# Symptoms are weaker disease evidence than concrete diagnoses; -0.05.
_SYMPTOM_PENALTY = 0.05


# ---------------------------------------------------------------------------
# Gemini prompt
# ---------------------------------------------------------------------------

MAPPING_SYSTEM_PROMPT = """\
You are a medical terminology expert mapping historical and Traditional \
Chinese Medicine (TCM) terms to modern Western disease classifications.

For each input, return ONE structured response. Choose exactly one `type`:

  type="disease"            — concrete modern diagnosis. Return exactly 1 \
mapping with role="primary".

  type="symptom"            — a single symptom rather than a disease. Return \
exactly 1 mapping with role="primary".

  type="syndrome"           — a TCM concept that legitimately spans multiple \
modern conditions (e.g. "wind heat" → upper respiratory infection + acute \
pharyngitis + fever). Return 1 mapping with role="primary" and 0-2 mappings \
with role="syndrome_component". Use this sparingly — only when the historical \
term genuinely covers a cluster, not when you're just unsure which single \
modern disease fits.

  type="ambiguous"          — the term could plausibly map to one of several \
distinct conditions and you have to pick. Return 1 mapping with \
role="primary" and 1-3 entries in `alternatives` (name + reason_dropped) \
for the audit trail. Alternatives are NOT written as graph edges.

  type="tcm_no_equivalent"  — the term is a TCM-specific concept with no \
clean modern equivalent (e.g. "gu toxin", "evil qi", "five taxations" used \
metaphysically). Return 0 mappings. DO NOT GUESS.

Each mapping must have:
  name           : the canonical English disease/symptom/condition name. Use \
the standard nomenclature (SNOMED CT preferred terms, ICD-10 category names, \
or MeSH headings).
  ontology_hint  : "icd10" | "mesh" | "snomed" — your best guess at which DB \
will recognise this name (used for verifier routing).
  role           : "primary" or "syndrome_component"
  rationale      : one short sentence explaining this specific mapping

DO NOT EMIT ICD-10, MeSH, or SNOMED CODES. Codes will be derived by external \
API lookup of the name. Only emit names.

Use `description` and `evidence_span` as your primary disambiguation \
context — they capture how the term was used in the source text. The bare \
`name` alone is often ambiguous.

Be conservative. If unsure, choose `tcm_no_equivalent` rather than guessing.

Examples follow.
"""

# Few-shot bank — drawn from the curated knowledge in the previous
# implementation's ARCHAIC_SYNONYMS dict. Each example illustrates one of
# the type values and shows the strict no-codes-in-output discipline.
FEW_SHOT_EXAMPLES = """\
EXAMPLES:

Input: name="ague", description="periodic fevers with chills"
Output: {
  "type": "disease",
  "mappings": [{
    "name": "malaria",
    "ontology_hint": "icd10",
    "role": "primary",
    "rationale": "Classical Western archaic term for malaria; periodic chills-and-fever pattern is the definitive sign."
  }],
  "alternatives": [],
  "reasoning": "Standard archaic synonym for malaria with no ambiguity."
}

Input: name="nosebleed", description="recurrent epistaxis"
Output: {
  "type": "symptom",
  "mappings": [{
    "name": "epistaxis",
    "ontology_hint": "icd10",
    "role": "primary",
    "rationale": "Direct symptom mapping; ICD-10 R04.0."
  }],
  "alternatives": [],
  "reasoning": "Single symptom, not a discrete disease entity."
}

Input: name="wind heat", description="acute febrile illness with sore throat, headache, productive cough"
Output: {
  "type": "syndrome",
  "mappings": [
    {
      "name": "upper respiratory infection",
      "ontology_hint": "icd10",
      "role": "primary",
      "rationale": "Best single-condition match for the febrile respiratory symptom cluster."
    },
    {
      "name": "acute pharyngitis",
      "ontology_hint": "icd10",
      "role": "syndrome_component",
      "rationale": "Throat involvement is core to the wind-heat presentation."
    },
    {
      "name": "fever",
      "ontology_hint": "icd10",
      "role": "syndrome_component",
      "rationale": "Heat sign in TCM denotes febrile presentation."
    }
  ],
  "alternatives": [],
  "reasoning": "TCM syndrome that genuinely spans a cluster of modern conditions."
}

Input: name="counterflow", description="upward movement of qi causing nausea and vomiting"
Output: {
  "type": "ambiguous",
  "mappings": [{
    "name": "nausea",
    "ontology_hint": "icd10",
    "role": "primary",
    "rationale": "Best single-symptom match given the description."
  }],
  "alternatives": [
    {"name": "gastroesophageal reflux disease", "reason_dropped": "More mechanistic but not directly named in the text."},
    {"name": "vomiting", "reason_dropped": "Co-occurs but the primary complaint described is the urge, not the act."}
  ],
  "reasoning": "Term has multiple plausible mappings; chose nausea as best primary."
}

Input: name="gu toxin", description="parasitic-magical illness attributed to ingested venoms"
Output: {
  "type": "tcm_no_equivalent",
  "mappings": [],
  "alternatives": [],
  "reasoning": "TCM concept with metaphysical components and no clean modern equivalent. Refusing to map."
}

Input: name="wasting thirst", description="excessive thirst and urination, weight loss"
Output: {
  "type": "disease",
  "mappings": [{
    "name": "diabetes mellitus",
    "ontology_hint": "icd10",
    "role": "primary",
    "rationale": "Polydipsia-polyuria-weight-loss triad is the classical presentation of diabetes mellitus."
  }],
  "alternatives": [],
  "reasoning": "Well-established historical synonym."
}
"""

MAPPING_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                "disease",
                "symptom",
                "syndrome",
                "ambiguous",
                "tcm_no_equivalent",
            ],
        },
        "mappings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ontology_hint": {
                        "type": "string",
                        "enum": ["icd10", "mesh", "snomed"],
                    },
                    "role": {
                        "type": "string",
                        "enum": ["primary", "syndrome_component"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["name", "ontology_hint", "role", "rationale"],
            },
        },
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason_dropped": {"type": "string"},
                },
                "required": ["name", "reason_dropped"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["type", "mappings", "alternatives", "reasoning"],
}


# ---------------------------------------------------------------------------
# Tolerant exact-match normalization
# ---------------------------------------------------------------------------

# Regex for stripping parenthetical / bracketed clauses (e.g.
# "epistaxis (nosebleed)" → "epistaxis"). Run before punctuation collapse.
_PAREN_RE = re.compile(r"\s*[\(\[\{][^\)\]\}]*[\)\]\}]")
# Apostrophes are elision marks, not separators — drop them WITHOUT
# inserting a space. Otherwise "Crohn's disease" → "crohn s disease",
# which would not match "Crohns disease". Includes both ASCII and the
# Unicode right-single-quote that ontologies often emit.
_APOSTROPHE_RE = re.compile(r"['’ʼ]")
# Match any non-word, non-whitespace character — commas, semicolons,
# hyphens, slashes — replace with a space, then collapse.
_PUNCT_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")


def _normalize_for_match(s: str) -> str:
    """Tolerant normalization for verifier exact-match.

    Strips parenthetical clauses, drops apostrophes, replaces remaining
    punctuation with whitespace, collapses whitespace, lowercases. Does
    NOT collapse plurals or strip qualifiers like "acute"/"chronic"/
    "type 2" — those changes can shift meaning and would re-introduce
    the fuzzy-redirect failure.

    Examples (all should normalize equal):
      "Diabetes Mellitus" / "diabetes mellitus"
      "Epistaxis (nosebleed)" / "Epistaxis"
      "Diabetes mellitus, type 2" / "Diabetes mellitus type 2"
      "King's evil" / "kings evil"
      "Crohn's disease" / "Crohns disease"

    Examples that should NOT normalize equal (preserved as distinct):
      "diabetes mellitus" / "diabetes mellitus type 2"
      "pharyngitis" / "acute pharyngitis"
      "scrofula" / "scrofulas"   (plural collapse skipped)
    """
    if not s:
        return ""
    s = _PAREN_RE.sub("", s)
    s = _APOSTROPHE_RE.sub("", s)
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------


def _validate_llm_response(
    classification: str, mappings: list[dict],
) -> str | None:
    """Check shape invariants on the LLM's structured response.

    Schema enforces field PRESENCE; this function enforces cardinality
    and role-per-type rules. Returns None on success, else a string
    describing the violation (used for logging + error status).
    """
    primaries = [m for m in mappings if m.get("role") == "primary"]
    components = [m for m in mappings if m.get("role") == "syndrome_component"]

    if classification == "tcm_no_equivalent":
        if mappings:
            return f"tcm_no_equivalent must have 0 mappings, got {len(mappings)}"
    elif classification in ("disease", "symptom", "ambiguous"):
        if len(primaries) != 1:
            return (
                f"{classification} must have exactly 1 primary mapping, "
                f"got {len(primaries)}"
            )
        if components:
            return (
                f"{classification} cannot have syndrome_component mappings, "
                f"got {len(components)}"
            )
    elif classification == "syndrome":
        if len(primaries) != 1:
            return f"syndrome must have exactly 1 primary mapping, got {len(primaries)}"
        if len(components) > 2:
            return (
                f"syndrome can have at most 2 syndrome_components, "
                f"got {len(components)}"
            )
    else:
        return f"unknown classification type: {classification!r}"

    return None


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _Verified:
    """Verifier output for a single LLM-proposed name."""
    name: str                  # the name actually found in the API (preferred over LLM's casing)
    source_db: str             # "icd10" | "mesh" | "snomed"
    match_type: str            # "exact" | "fuzzy"
    icd10_code: str = ""
    mesh_id: str = ""
    snomed_id: str = ""


@dataclass
class _MappingPlan:
    """One row in the final write plan: an LLM mapping + verifier outcome."""
    llm_name: str              # name as the LLM emitted it
    role: str                  # "primary" | "syndrome_component"
    rationale: str             # the LLM's per-mapping rationale
    ontology_hint: str         # the LLM's ontology hint (audit only)
    verified: _Verified | None # None when no API matched
    rank: int = 0              # filled in by edge selection
    is_primary: bool = False
    confidence: float = 0.0    # filled in by edge selection (flat prior)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class MaladyDiseaseMapper(BaseAgent):
    """Generate-then-verify mapping of Traditional_Malady → Modern_Disease."""

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        no_multi: bool = False,
        allow_unverified: bool = False,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL
        self._ontology = OntologyClient()
        self._no_multi = no_multi  # if True, force N=1 even for type=syndrome
        self._allow_unverified = allow_unverified
        self._last_gemini_call = 0.0
        # Set per run() invocation so _write_outcome can decide whether to
        # delete prior MAPS_TO edges before writing.
        self._force_remap = False

    @property
    def name(self) -> str:
        return "MaladyDiseaseMapper"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        dry_run: bool = True,
        rebuild: bool = False,
        retry_misses: bool = False,
        force_remap: bool = False,
        limit: int | None = None,
        **_: Any,
    ) -> dict:
        if rebuild and not dry_run:
            self._wipe_existing_mappings()

        # Stash for _write_outcome — controls per-malady prior-edge cleanup.
        self._force_remap = force_remap

        cypher = queries.MAPPABLE_MALADIES
        if force_remap:
            cypher = queries.MAPPABLE_MALADIES_FORCE
        elif retry_misses:
            cypher = queries.MAPPABLE_MALADIES_RETRY

        rows = self.client.run(cypher)
        if limit is not None:
            rows = rows[:limit]

        total = len(rows)
        self._log_progress(
            f"Mapping {total} Malady(ies) "
            f"(dry_run={dry_run}, rebuild={rebuild}, "
            f"retry_misses={retry_misses}, force_remap={force_remap}, "
            f"no_multi={self._no_multi}, "
            f"allow_unverified={self._allow_unverified})"
        )

        t_start = time.time()
        by_status: dict[str, int] = {
            "linked": 0,
            "unverified": 0,
            "tcm_no_equivalent": 0,
            "error": 0,
        }
        by_classification: dict[str, int] = {}
        by_mapping_source: dict[str, int] = {}
        edges_primary = 0
        edges_components = 0
        unique_diseases: set[str] = set()
        unverified_examples: list[dict] = []  # for the run summary
        errors: list[str] = []

        for i, row in enumerate(rows, 1):
            try:
                outcome = self._map_one(row)
            except Exception as e:
                msg = f"{row.get('name', '?')}: {e}"
                logger.exception(msg)
                errors.append(msg)
                outcome = {
                    "status": "error",
                    "classification": "error",
                    "raw_response": "",
                    "edges": [],
                }

            status = outcome["status"]
            classification = outcome["classification"]
            by_status[status] = by_status.get(status, 0) + 1
            by_classification[classification] = by_classification.get(classification, 0) + 1

            for plan in outcome["edges"]:
                if plan.is_primary:
                    edges_primary += 1
                else:
                    edges_components += 1
                src_label = self._mapping_source_label(plan.verified)
                by_mapping_source[src_label] = by_mapping_source.get(src_label, 0) + 1
                if plan.verified is None and len(unverified_examples) < 10:
                    unverified_examples.append({
                        "malady": row["name"],
                        "name": plan.llm_name,
                        "ontology_hint": plan.ontology_hint,
                    })
                if plan.verified:
                    unique_diseases.add(plan.verified.name)
                else:
                    unique_diseases.add(plan.llm_name)

            if not dry_run:
                self._write_outcome(row["name"], outcome)

            if i % 25 == 0 or i == total:
                self._log_progress(
                    f"  {i}/{total}  linked={by_status['linked']} "
                    f"tcm_no_equivalent={by_status['tcm_no_equivalent']} "
                    f"errors={by_status['error']}"
                )

        duration_s = round(time.time() - t_start, 2)

        return {
            "dry_run": dry_run,
            "rebuild": rebuild,
            "no_multi": self._no_multi,
            "allow_unverified": self._allow_unverified,
            "maladies_total": total,
            "by_status": by_status,
            "by_classification": dict(
                sorted(by_classification.items(), key=lambda kv: -kv[1])
            ),
            "by_mapping_source": dict(
                sorted(by_mapping_source.items(), key=lambda kv: -kv[1])
            ),
            "edges_primary": edges_primary,
            "edges_syndrome_components": edges_components,
            "unique_modern_diseases": len(unique_diseases),
            "duration_s": duration_s,
            "unverified_examples": unverified_examples,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Per-malady pipeline
    # ------------------------------------------------------------------

    def _map_one(self, row: dict) -> dict:
        """Generate (Gemini) → Verify (APIs) → Plan edges. No graph writes."""
        # Step A: Gemini canonicalize
        try:
            llm_response = self._gemini_map(row)
        except Exception as e:
            logger.warning("Gemini failed for %r: %s", row["name"], e)
            return {
                "status": "error",
                "classification": "error",
                "raw_response": "",
                "edges": [],
                "alternatives_audit": [],
            }

        classification = llm_response.get("type", "error")
        raw_response = json.dumps(llm_response, ensure_ascii=False)
        llm_mappings = llm_response.get("mappings", [])
        llm_alternatives = llm_response.get("alternatives", [])

        # Validate invariants — schema enforces fields exist but not the
        # cardinality / role rules per type. A malformed response is
        # surfaced as `error` so --retry-misses re-processes it.
        violation = _validate_llm_response(classification, llm_mappings)
        if violation:
            logger.warning(
                "Invalid LLM response for %r: %s — marking as error",
                row["name"], violation,
            )
            return {
                "status": "error",
                "classification": "error",
                "raw_response": raw_response,
                "edges": [],
                "alternatives_audit": llm_alternatives,
            }

        # tcm_no_equivalent is the explicit refusal exit
        if classification == "tcm_no_equivalent":
            return {
                "status": "tcm_no_equivalent",
                "classification": "tcm_no_equivalent",
                "raw_response": raw_response,
                "edges": [],
                "alternatives_audit": llm_alternatives,
            }

        # Step B: Verify each LLM-proposed name in parallel
        candidates = [
            (m.get("name", ""), m.get("ontology_hint", "icd10"))
            for m in llm_mappings
        ]
        verified_results = self._verify_parallel(candidates)

        # Step C: Build the write plan with rank/role/confidence
        plans: list[_MappingPlan] = []
        for m, verified in zip(llm_mappings, verified_results):
            plans.append(_MappingPlan(
                llm_name=m.get("name", ""),
                role=m.get("role", "primary"),
                rationale=m.get("rationale", ""),
                ontology_hint=m.get("ontology_hint", "icd10"),
                verified=verified,
            ))

        # Apply the per-malady cap + the strict gate:
        #   - exactly one primary (the first one from the LLM with role=primary)
        #   - at most 2 syndrome_components, and ONLY if exact API match
        #   - in strict mode (default), unverified primary is NOT written
        #   - everything dropped goes into the alternatives_audit list
        edges, dropped = self._select_edges(plans, classification)

        # Combine LLM-emitted alternatives + dropped runner-ups for the
        # primary edge's mapping_alternatives JSON property.
        alternatives_audit: list[dict] = list(llm_alternatives)
        for d in dropped:
            alternatives_audit.append({
                "name": d.llm_name,
                "reason_dropped": (
                    "syndrome_component_unverified"
                    if d.role == "syndrome_component" and d.verified is None
                    else "primary_unverified_strict_mode"
                    if d.role == "primary" and d.verified is None
                    else "exceeded_max_edges_per_malady"
                ),
            })

        # Status: linked if we have edges to write; unverified if we have
        # a primary plan that strict mode dropped; tcm_no_equivalent
        # if there were no plans at all (already short-circuited above).
        if edges:
            status = "linked"
        elif any(p.role == "primary" for p in plans):
            status = "unverified"
        else:
            status = "tcm_no_equivalent"

        return {
            "status": status,
            "classification": classification,
            "raw_response": raw_response,
            "edges": edges,
            "alternatives_audit": alternatives_audit,
        }

    def _select_edges(
        self, plans: list[_MappingPlan], classification: str,
    ) -> tuple[list[_MappingPlan], list[_MappingPlan]]:
        """Apply the cap + strict-exact gate. Returns (kept, dropped)."""
        primaries = [p for p in plans if p.role == "primary"]
        components = [p for p in plans if p.role == "syndrome_component"]

        kept: list[_MappingPlan] = []
        dropped: list[_MappingPlan] = []

        # Exactly one primary. Take the first; demote any extras.
        # In strict mode (default), unverified primaries are dropped — they
        # become alternatives_audit entries and the malady is recorded as
        # `mapper_status="unverified"` rather than written as a graph edge.
        if primaries:
            primary = primaries[0]
            if primary.verified is None and not self._allow_unverified:
                dropped.append(primary)
            else:
                primary.rank = 1
                primary.is_primary = True
                primary.confidence = self._confidence_for(primary, classification)
                kept.append(primary)
            for extra in primaries[1:]:
                dropped.append(extra)

        # Multi-edge gating: only allowed for type=syndrome AND not no_multi
        # AND we actually wrote a primary edge. Each component must have an
        # EXACT verifier match, capped at MAX_EDGES_PER_MALADY total.
        if (
            classification == "syndrome"
            and not self._no_multi
            and kept                          # primary present
            and len(kept) < _MAX_EDGES_PER_MALADY
        ):
            for comp in components:
                if len(kept) >= _MAX_EDGES_PER_MALADY:
                    dropped.append(comp)
                    continue
                if comp.verified is None or comp.verified.match_type != "exact":
                    dropped.append(comp)
                    continue
                comp.rank = len(kept) + 1
                comp.is_primary = False
                comp.confidence = self._confidence_for(comp, classification)
                kept.append(comp)
        else:
            # Components ignored when not type=syndrome or no_multi=True
            for comp in components:
                dropped.append(comp)

        return kept, dropped

    def _confidence_for(self, plan: _MappingPlan, classification: str) -> float:
        if plan.verified is None:
            base = CONFIDENCE[("primary", "unverified", None)]
        else:
            key = (plan.role, plan.verified.match_type, plan.verified.source_db)
            base = CONFIDENCE.get(key, 0.5)
        if classification == "symptom":
            base = max(0.0, base - _SYMPTOM_PENALTY)
        return round(base, 3)

    @staticmethod
    def _mapping_source_label(verified: _Verified | None) -> str:
        # Verifier only produces exact matches now (fuzzy was retired) so
        # the "_exact" suffix is informational. Phase 2 scoring keys on
        # this exact string set — keep it stable.
        if verified is None:
            return "gemini_unverified"
        return f"gemini+{verified.source_db}_exact"

    # ------------------------------------------------------------------
    # Step A — Gemini call (one malady per call)
    # ------------------------------------------------------------------

    def _gemini_map(self, row: dict) -> dict:
        """Single Gemini call; returns the parsed JSON response."""
        # Polite rate limit
        elapsed = time.time() - self._last_gemini_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_gemini_call = time.time()

        aliases = row.get("aliases") or []
        ev = (row.get("evidence_span") or "").strip()
        ev_short = ev[:600] + ("..." if len(ev) > 600 else "")
        desc = (row.get("description") or "").strip()
        desc_short = desc[:300] + ("..." if len(desc) > 300 else "")

        prompt = (
            f"{FEW_SHOT_EXAMPLES}\n\n"
            f"Now map this malady:\n\n"
            f'name="{row["name"]}"\n'
            f'description="{desc_short}"\n'
            f'evidence_span="{ev_short}"\n'
            f"aliases={aliases[:8] if aliases else []}\n"
        )

        last_err: Exception | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                response = self._gemini.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=MAPPING_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=MAPPING_SCHEMA,
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str and attempt < _GEMINI_MAX_RETRIES - 1:
                    wait_s = 5 * (2 ** attempt)
                    logger.info("Rate limited, backoff %ds (attempt %d/%d)",
                                wait_s, attempt + 1, _GEMINI_MAX_RETRIES)
                    time.sleep(wait_s)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Gemini call failed without raising")

    # ------------------------------------------------------------------
    # Step B — Verifier (parallel API lookups)
    # ------------------------------------------------------------------

    def _verify_parallel(
        self, candidates: list[tuple[str, str]],
    ) -> list[_Verified | None]:
        """Verify a batch of (name, ontology_hint) tuples concurrently."""
        if not candidates:
            return []
        with ThreadPoolExecutor(max_workers=8) as pool:
            return list(pool.map(
                lambda c: self._verify_one(c[0], c[1]),
                candidates,
            ))

    def _verify_one(self, name: str, hint: str) -> _Verified | None:
        """Look up a single LLM-proposed name in all 3 ontologies.

        Tolerant exact match: a candidate verifies iff
        `_normalize_for_match(candidate.name) == _normalize_for_match(name)`.
        This is stricter than ontology fuzzy redirects (which can land on
        semantically wrong concepts) but more forgiving than the API's
        byte-equal `exact` flag, which over-rejects on cosmetic differences
        like parentheticals, hyphens, and plurals' adjacent punctuation.

        Priority order for choosing among multiple matches:
          1) match in the LLM's hinted DB (it picked this DB for a reason)
          2) match in any DB, icd10 > mesh > snomed
        """
        if not name or not name.strip():
            return None
        norm_target = _normalize_for_match(name)
        if not norm_target:
            return None

        # Sequential calls — outer ThreadPoolExecutor in _verify_parallel
        # already pipelines candidates across maladies. Adding inner
        # concurrency on the same OntologyClient compounded thread-safety
        # risk for negligible wall-clock benefit at our scale.
        per_db = {
            "icd10":  self._safe_search("icd10", name),
            "mesh":   self._safe_search("mesh", name),
            "snomed": self._safe_search("snomed", name),
        }

        def _matches(c: dict) -> bool:
            return _normalize_for_match(c.get("name", "")) == norm_target

        # 1) Tolerant match in hinted DB
        for c in per_db.get(hint, []):
            if _matches(c):
                return self._to_verified(c)

        # 2) Tolerant match in any DB, in priority order
        for db in ("icd10", "mesh", "snomed"):
            for c in per_db.get(db, []):
                if _matches(c):
                    return self._to_verified(c)

        return None

    def _safe_search(self, db: str, name: str) -> list[dict]:
        """Wrap one OntologyClient call so a single failing API doesn't
        cascade a verifier crash."""
        try:
            if db == "icd10":
                return self._ontology.search_icd10(name, max_results=10)
            if db == "mesh":
                return self._ontology.search_mesh(name, limit=10)
            if db == "snomed":
                return self._ontology.search_snomed(name, rows=10)
        except Exception as e:
            logger.warning("%s search failed for %r: %s", db, name, e)
        return []

    @staticmethod
    def _to_verified(candidate: dict) -> _Verified:
        # Whether the API tagged this as "exact" or "fuzzy" is irrelevant
        # — the caller already accepted it via tolerant normalization,
        # which is our verifier's source of truth. Always store "exact".
        return _Verified(
            name=candidate.get("name", "") or "",
            source_db=candidate.get("source", ""),
            match_type="exact",
            icd10_code=candidate.get("icd10_code", "") or "",
            mesh_id=candidate.get("mesh_id", "") or "",
            snomed_id=candidate.get("snomed_id", "") or "",
        )

    # ------------------------------------------------------------------
    # Step C — Graph writes
    # ------------------------------------------------------------------

    def _write_outcome(self, malady_name: str, outcome: dict) -> None:
        """Write Modern_Disease nodes, MAPS_TO edges, then status props.

        Crash-safety strategy: status props are set LAST. If any earlier
        write fails (network blip, transient Neo4j error), `mapper_status`
        stays NULL so the next default run picks the malady up again. All
        node/edge writes are MERGE-based, so the retry is idempotent — no
        duplicate edges or nodes appear. The opposite ordering (status
        first) would leave a malady marked `linked` with missing edges
        on partial failure, which the resume logic would never retry.
        """
        # When --force-remap re-touches an already-mapped malady, delete
        # its prior MAPS_TO edges first so the new plan replaces them
        # cleanly. Without this, a remap that picks a different disease
        # leaves both the old and new edges attached.
        if self._force_remap:
            self.client.run_write(
                "MATCH (m:Traditional_Malady {name: $name})-[r:MAPS_TO]->() DELETE r",
                {"name": malady_name},
            )

        edges: list[_MappingPlan] = outcome["edges"]

        if edges:
            # Stable disease-name MERGE; codes attached only when API-derived.
            for plan in edges:
                disease_name = (
                    plan.verified.name if plan.verified else plan.llm_name
                )
                if not disease_name:
                    continue
                self._merge_disease_node(disease_name, plan.verified)

            # Write edges. Only the primary edge carries `mapping_alternatives`
            # JSON (it's a per-malady audit blob, not a per-edge property).
            alternatives_json = json.dumps(
                outcome.get("alternatives_audit") or [],
                ensure_ascii=False,
            )
            for plan in edges:
                disease_name = (
                    plan.verified.name if plan.verified else plan.llm_name
                )
                if not disease_name:
                    continue
                self._merge_maps_to_edge(
                    malady_name, disease_name, plan,
                    alternatives_json=alternatives_json if plan.is_primary else None,
                )

        # Status LAST — see docstring for the crash-safety rationale.
        self.client.set_node_properties(
            "Traditional_Malady",
            {"name": malady_name},
            {
                "mapper_status": outcome["status"],
                "mapper_classification": outcome["classification"],
                "mapper_attempted_at": dt.datetime.utcnow().isoformat(),
                "mapper_raw_response": outcome["raw_response"],
            },
        )

    def _merge_disease_node(
        self, disease_name: str, verified: _Verified | None,
    ) -> None:
        """MERGE the disease node. Missing IDs from earlier runs get
        backfilled via `coalesce` (existing values are never overwritten,
        but null/missing fields are populated when a later verifier finds
        them)."""
        # Build the set of fields to backfill. Empty strings are treated
        # as "missing" for the purposes of coalesce.
        backfill: dict[str, str] = {}
        if verified:
            if verified.icd10_code:
                backfill["icd10_code"] = verified.icd10_code
            if verified.mesh_id:
                backfill["mesh_id"] = verified.mesh_id
            if verified.snomed_id:
                backfill["snomed_id"] = verified.snomed_id
            if verified.source_db:
                backfill["verified_by"] = verified.source_db

        # Build coalesce SET clauses. We use `(d.x IS NULL OR d.x = "")`
        # rather than raw coalesce because earlier writes may have stored
        # empty strings rather than nulls; we want both treated as missing.
        coalesce_clauses = ", ".join(
            f"d.{k} = CASE WHEN d.{k} IS NULL OR d.{k} = '' "
            f"THEN $backfill.{k} ELSE d.{k} END"
            for k in backfill
        )
        coalesce_set = f"SET {coalesce_clauses}" if coalesce_clauses else ""

        self.client.run_write(
            f"""
            MERGE (d:Modern_Disease {{name: $name}})
            ON CREATE SET d.created_by = $created_by,
                          d.created_at = $created_at
            {coalesce_set}
            """,
            {
                "name": disease_name,
                "created_by": "malady_disease_mapper",
                "created_at": dt.datetime.utcnow().isoformat(),
                "backfill": backfill,
            },
        )

    def _merge_maps_to_edge(
        self,
        malady_name: str,
        disease_name: str,
        plan: _MappingPlan,
        *,
        alternatives_json: str | None,
    ) -> None:
        rel_props: dict[str, Any] = {
            "confidence_score": plan.confidence,
            "mapping_source": self._mapping_source_label(plan.verified),
            "mapping_rank": plan.rank,
            "is_primary": plan.is_primary,
            "mapping_role": plan.role,
            "mapper_rationale": plan.rationale,
            "ontology_hint": plan.ontology_hint,
            "created_by": "malady_disease_mapper",
            "created_at": dt.datetime.utcnow().isoformat(),
            # Tag the only path that produces unverified edges so the
            # critic / reviewer can filter on `requires_review = true`.
            "requires_review": plan.verified is None,
        }
        if plan.verified:
            if plan.verified.icd10_code:
                rel_props["icd10_code"] = plan.verified.icd10_code
            if plan.verified.mesh_id:
                rel_props["mesh_id"] = plan.verified.mesh_id
            if plan.verified.snomed_id:
                rel_props["snomed_id"] = plan.verified.snomed_id
        if alternatives_json is not None and alternatives_json != "[]":
            rel_props["mapping_alternatives"] = alternatives_json

        self.client.merge_edge(
            "Traditional_Malady", {"name": malady_name},
            "Modern_Disease", {"name": disease_name},
            "MAPS_TO",
            rel_props,
        )

    # ------------------------------------------------------------------
    # Rebuild helper
    # ------------------------------------------------------------------

    def _wipe_existing_mappings(self) -> None:
        """Idempotent clean slate: drop all MAPS_TO edges, all
        Modern_Disease nodes, and clear mapper_* properties on Maladies.

        Safe to run on an already-empty graph."""
        self._log_progress("Rebuild: wiping MAPS_TO + Modern_Disease + mapper_* props")
        self.client.run_write("MATCH ()-[r:MAPS_TO]->() DELETE r")
        self.client.run_write("MATCH (d:Modern_Disease) DETACH DELETE d")
        self.client.run_write("""
            MATCH (m:Traditional_Malady)
            REMOVE m.mapper_status, m.mapper_classification,
                   m.mapper_attempted_at, m.mapper_raw_response
        """)
