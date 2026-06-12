"""Phase 2 — Task A Validator.

Validates (Source, Traditional_Malady) claims using mechanistic evidence
chains pulled from the Phase 1 graph. Reads only Phase 1 outputs; no LLM
in the loop, no new node types, no graph rewrites by default.

Scoring philosophy: TIER-BUCKET PRIMARY, MULTIPLICATIVE TIEBREAK.

  Each path
      Source ←IS_EXTRACTED_FROM← Compound →TARGETS→ Target →RELATES_TO→ Disease
  is bucketed by the WEAKEST evidence_type tier along it (gold > silver >
  bronze > wood). Within a bucket, paths are ordered by the product of
  edge confidences (the strict numeric "score" — used for tiebreak only,
  not for crossing bucket boundaries).

  Tier maps below (EX_TIER, TGT_TIER, REL_TIER) read directly off the
  evidence_type strings written by the Phase 1 linkers — no rederivation.

The verdict per (Source, Malady) is derived deterministically from:
  * whether the Malady has a primary MAPS_TO Modern_Disease (the loop
    target). If not (mapper_status='tcm_no_equivalent' or 'unverified'),
    loop closure is not checkable and the claim drops to mechanistic_only.
  * which paths exist, their bucket, and the count in the top bucket
  * which (if any) close the loop (path's reached Disease == Malady's
    primary mapped Disease)
  * the mapping_source quality on MAPS_TO (gold = ICD-10/MeSH/SNOMED
    verified; silver = gemini_unverified)
  * cross-source corroboration (other Sources treating the same Malady
    that share compounds with this Source)

Verdict ladder (highest to lowest):
  strong_support      loop closed AND ≥2 paths in gold bucket
  moderate_support    loop closed AND top bucket ∈ {gold, silver}
  partial_support     loop closed AND top bucket ∈ {bronze, wood}
  unsupported         has paths AND has mapping AND no loop closes
  mechanistic_only    has paths AND no MAPS_TO mapping (uncheckable loop)
  traditional_only    has TREATS edge AND no compounds linked at all
  claim_not_found     defensive: no active TREATS edge between the pair

With write_graph=True, stamps task_a_* audit properties on the
TREATS_TRADITIONALLY edge so Phase 2 verdicts are queryable. The default
is dry-run; the structured verdict list is always returned via run().
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.base import BaseAgent
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier bucket lookup tables
# ---------------------------------------------------------------------------
# Tier order (numeric weight; larger = better). "wood" is the catch-all
# bottom tier for measurable-but-very-weak evidence (chembl_phenotypic,
# weak activity). "unrated" applies when an edge carries an evidence_type
# string we don't know — we never promote those to silver/gold.

TIER_ORDER: dict[str, int] = {
    "unrated": 0,
    "wood":    1,
    "bronze":  2,
    "silver":  3,
    "gold":    4,
}
_TIER_BY_RANK = {v: k for k, v in TIER_ORDER.items()}


# IS_EXTRACTED_FROM evidence_type → tier (set by SourceCompoundLinker v2).
# Identity-grounded matches (canonical_source = gemini+coconut/pubchem)
# are gold; alias / unverified / formula fallbacks are silver / bronze.
EX_TIER: dict[str, str] = {
    "pubchem_chemical_canonical":   "gold",
    "coconut_organism_canonical":   "gold",
    "pubchem_chemical_unverified":  "silver",
    "coconut_organism_alias":       "silver",
    "pubchem_chemical_formula":     "bronze",
    "coconut_organism_unverified":  "bronze",
}

# TARGETS evidence_type → tier (set by CompoundTargetLinker v2). ChEMBL
# `mechanism` (curated drug-target relationships) is gold; pchembl-tier
# activities split silver/bronze; weak + phenotypic both fall to wood.
TGT_TIER: dict[str, str] = {
    "chembl_mechanism":          "gold",
    "chembl_activity_strong":    "silver",
    "chembl_activity_moderate":  "bronze",
    "chembl_activity_weak":      "wood",
    "chembl_phenotypic":         "wood",
}

# RELATES_TO evidence_type → tier (set by TargetDiseaseLinker). Ontology
# hits (EFO/DOID) and OT-strong are gold; OT-moderate / complex aggregate
# / LLM-verified pathogens are silver; OT-weak is bronze.
REL_TIER: dict[str, str] = {
    "ncbi_pathogen_consensus":          "gold",
    "ncbi_pathogen_efo":                "gold",
    "ncbi_pathogen_doid":               "gold",
    "ot_association_strong":            "gold",
    "ot_association_moderate":          "silver",
    "ot_association_complex_aggregate": "silver",
    "ncbi_pathogen_llm_verified":       "silver",
    "ot_association_weak":              "bronze",
}

# MAPS_TO mapping_source → quality bucket (audit signal, not a path edge).
MAP_QUALITY: dict[str, str] = {
    "gemini+icd10_exact":  "gold",
    "gemini+mesh_exact":   "gold",
    "gemini+snomed_exact": "gold",
    "gemini_unverified":   "silver",
}


def _tier_min(tiers: list[str]) -> str:
    """Path bucket = MIN over present edges (weakest-link).

    Empty input collapses to 'unrated'. Any edge with an unrecognized
    evidence_type drops the whole path to 'unrated' so a stray tag never
    silently inflates the bucket.
    """
    if not tiers:
        return "unrated"
    rank = min(TIER_ORDER.get(t, 0) for t in tiers)
    return _TIER_BY_RANK[rank]


def _tier_rank(t: str) -> int:
    return TIER_ORDER.get(t, 0)


# ---------------------------------------------------------------------------
# Cypher queries
# ---------------------------------------------------------------------------

# Anchor: the TREATS edge plus the Malady's primary MAPS_TO disease and
# any syndrome_components. Components are returned as a list so the
# verdict can flag component-disease hits even though loop closure is
# defined against PRIMARY only (avoids syndrome inflation).
#
# Also pulls every audit field downstream consumers need:
#   - Source: aliases, source_document, evidence_span, canonical_*
#     (lets the critic see the historical text + canonicalisation grade)
#   - Malady: description, evidence_span, aliases, source_document
#     (lets the critic judge the disease mapping against the symptom text)
#   - r_tt: evidence_span (the actual quoted historical text — the most
#     important context for a critic asked to validate a claim)
#   - Primary MAPS_TO: mapper_rationale, mapping_alternatives, mesh_id,
#     icd10_code, snomed_id, requires_review (full mapper audit trail)
#   - Primary disease: ontology IDs (icd10/mesh/snomed/efo/mondo/doid)
_ANCHOR_QUERY = """
MATCH (s:Source {name: $source})-[r_tt:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
OPTIONAL MATCH (m)-[r_map_p:MAPS_TO]->(d_primary:Modern_Disease)
  WHERE r_map_p.is_primary = true
    AND (d_primary.archived IS NULL OR d_primary.archived = false)
OPTIONAL MATCH (m)-[r_map_c:MAPS_TO]->(d_comp:Modern_Disease)
  WHERE coalesce(r_map_c.is_primary, true) = false
    AND (d_comp.archived IS NULL OR d_comp.archived = false)
RETURN
  s.name AS source,
  s.aliases AS source_aliases,
  s.source_document AS source_document,
  s.evidence_span AS source_evidence_span,
  s.canonical_name AS source_canonical_name,
  s.canonical_part AS source_canonical_part,
  s.canonical_type AS source_canonical_type,
  s.canonical_source AS source_canonical_source,
  m.name AS malady,
  m.aliases AS malady_aliases,
  m.description AS malady_description,
  m.evidence_span AS malady_evidence_span,
  m.source_document AS malady_source_document,
  m.mapper_classification AS mapper_classification,
  r_tt.confidence_score AS treats_conf,
  r_tt.evidence_span AS treats_evidence,
  d_primary.name AS primary_disease,
  d_primary.icd10_code AS primary_disease_icd10,
  d_primary.mesh_id AS primary_disease_mesh,
  d_primary.snomed_id AS primary_disease_snomed,
  d_primary.efo_id AS primary_disease_efo,
  d_primary.mondo_id AS primary_disease_mondo,
  d_primary.doid_id AS primary_disease_doid,
  d_primary.verified_by AS primary_disease_verified_by,
  r_map_p.mapping_source AS primary_mapping_source,
  r_map_p.confidence_score AS primary_mapping_conf,
  r_map_p.mapper_rationale AS primary_mapping_rationale,
  r_map_p.mapping_alternatives AS primary_mapping_alternatives,
  r_map_p.requires_review AS primary_mapping_requires_review,
  collect(DISTINCT d_comp.name) AS component_diseases
"""


# All mechanistic paths from Source. OPTIONAL on TARGETS / RELATES_TO so
# partial chains (compound-only, no targets) are visible to the signal
# computer — the verdict needs to distinguish "no compounds" from "has
# compounds but no targets".
#
# Returns the full evidence-level payload per path: every node and edge
# property a downstream critic could plausibly reason over. Cheap because
# it's still one Cypher round-trip; the expensive thing would be running
# this query 100 times instead of once. Compound-only paths are kept in
# the result so Pass 1's bucket distribution is honest about partial
# coverage; the LLM payload filters them out separately.
_PATHS_QUERY = """
MATCH (s:Source {name: $source})
WHERE s.archived IS NULL OR s.archived = false
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)
WHERE c.archived IS NULL OR c.archived = false
OPTIONAL MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)
WHERE t.archived IS NULL OR t.archived = false
OPTIONAL MATCH (t)-[r_rel:RELATES_TO]->(d:Modern_Disease)
WHERE d.archived IS NULL OR d.archived = false
RETURN
  // --- Compound (node) ---
  c.inchikey AS compound_inchikey,
  c.name AS compound,
  c.smiles AS smiles,
  c.molecular_formula AS molecular_formula,
  c.np_likeness AS compound_np_likeness,
  c.annotation_level AS compound_annotation_level,
  c.linker_chembl_id AS compound_chembl_id,
  c.source_db AS compound_source_db,

  // --- IS_EXTRACTED_FROM (edge) ---
  r_ex.evidence_type AS ex_type,
  r_ex.confidence_score AS ex_conf,
  r_ex.confidence_base_prior AS ex_base_prior,
  r_ex.confidence_part_penalty AS ex_part_penalty,
  r_ex.evidence_resolution AS ex_resolution,
  r_ex.species_part_context AS part_context,
  r_ex.part_specific AS part_specific,
  r_ex.np_likeness AS ex_np_likeness,
  r_ex.annotation_level AS ex_annotation_level,
  r_ex.pubchem_formula_ambiguous AS ex_formula_ambiguous,
  r_ex.lookup_query AS ex_lookup_query,

  // --- Target (node) ---
  t.target_chembl_id AS target_chembl_id,
  t.name AS target,
  t.target_pref_name AS target_pref_name,
  t.target_type AS target_type,
  t.gene_symbol AS target_gene_symbol,
  t.uniprot_id AS target_uniprot_id,
  t.ncbi_tax_id AS target_ncbi_tax_id,

  // --- TARGETS (edge) ---
  r_tgt.evidence_type AS tgt_type,
  r_tgt.confidence_score AS tgt_conf,
  r_tgt.pchembl_score AS tgt_pchembl,
  r_tgt.assay_id AS tgt_assay_id,
  r_tgt.assay_type AS tgt_assay_type,
  r_tgt.assay_description AS tgt_assay_description,
  r_tgt.mechanism_action AS tgt_mechanism_action,

  // --- Reached Disease (node) ---
  d.name AS reached_disease,
  d.icd10_code AS reached_icd10,
  d.mesh_id AS reached_mesh,
  d.snomed_id AS reached_snomed,
  d.efo_id AS reached_efo,
  d.mondo_id AS reached_mondo,
  d.doid_id AS reached_doid,

  // --- RELATES_TO (edge) ---
  r_rel.evidence_type AS rel_type,
  r_rel.confidence AS rel_conf,
  r_rel.ot_overall_score AS ot_score,
  r_rel.ot_resolved_id AS ot_resolved_id,
  r_rel.ot_target_ensembl_id AS ot_target_ensembl_id,
  r_rel.ot_top_subunit_uniprot AS ot_top_subunit_uniprot,
  r_rel.ot_datasource_scores AS ot_datasource_scores,
  r_rel.pathogen_lookup_source AS pathogen_lookup_source,
  r_rel.rationale AS rel_rationale,
  r_rel.requires_review AS rel_requires_review
"""


# Cross-source corroboration: count distinct sibling sources (other
# Sources treating the same Malady that share at least one compound
# with this Source) and the number of shared compounds.
_CROSS_CORROB_QUERY = """
MATCH (s1:Source {name: $source})-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s1)
MATCH (c)-[:IS_EXTRACTED_FROM]->(s2:Source)
MATCH (s2)-[:TREATS_TRADITIONALLY]->(m)
WHERE s2.name <> $source
  AND (s2.archived IS NULL OR s2.archived = false)
RETURN count(DISTINCT s2) AS sibling_sources,
       count(DISTINCT c) AS shared_compounds
"""


# All active TREATS claims to evaluate. Default ordering keeps verdict
# JSON deterministic across runs.
_LIST_CLAIMS_QUERY = """
MATCH (s:Source)-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
RETURN s.name AS source, m.name AS malady, r.confidence_score AS treats_conf
ORDER BY s.name, m.name
"""


# Stamp audit properties back onto the TREATS_TRADITIONALLY edge.
_WRITE_VERDICT_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
SET r += $props
"""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------

@dataclass
class _PathFeatures:
    """One Compound→Target→Disease path from a Source. Partial paths
    (no target / no disease) are kept; the bucket reflects whichever
    edges are present.

    The DERIVED fields (ex_tier .. component_closed) drive Pass-1's
    verdict math. The full Cypher row is stashed in `raw_row` so
    serialization can produce a nested compound/target/disease/edges
    payload without bloating the dataclass with 30+ leaf fields.
    """
    compound_inchikey: str
    compound_name: str
    target_chembl_id: str | None
    target_name: str | None
    target_type: str | None
    reached_disease: str | None

    ex_type: str
    ex_conf: float
    tgt_type: str | None
    tgt_conf: float | None
    tgt_pchembl: float | None
    rel_type: str | None
    rel_conf: float | None
    rel_requires_review: bool

    ex_tier: str
    tgt_tier: str | None
    rel_tier: str | None
    path_bucket: str
    path_score: float

    has_target: bool
    has_disease: bool
    loop_closed: bool
    component_closed: bool

    raw_row: dict = field(default_factory=dict)


@dataclass
class ClaimVerdict:
    """Final per-claim verdict + the signals that produced it.

    `context` carries every anchor-level audit field downstream consumers
    (the LLM critic, Phase 2 nominator, paper tables) might want — Source
    canonical info, malady description / aliases / source_document,
    primary disease's ontology IDs, MAPS_TO mapper_rationale and
    mapping_alternatives, the actual TREATS evidence span, etc. Stashing
    them in a single dict keeps ClaimVerdict's flat fields to the
    verdict-relevant signals.
    """
    source: str
    malady: str
    mapper_classification: str | None
    primary_disease: str | None
    component_diseases: list[str]

    treats_confidence: float
    primary_mapping_source: str | None
    primary_mapping_quality: str          # "gold" | "silver" | "missing"

    has_compounds: bool
    has_mechanism: bool
    has_mapping: bool

    # Path-level counts
    path_count: int
    paths_with_target: int
    paths_with_disease: int
    paths_loop_closed: int
    paths_component_closed: int
    unique_compounds: int
    unique_targets: int
    paths_by_bucket: dict[str, int]

    # Top-bucket aggregate
    top_bucket: str
    top_bucket_path_count: int
    top_bucket_max_score: float

    # Cross-source corroboration
    sibling_sources: int
    shared_compounds: int

    verdict: str
    rationale: str

    paths_top: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TaskAValidator(BaseAgent):
    """Deterministic validator for Source→Traditional_Malady claims.

    Read-only by default. With write_graph=True, stamps task_a_* audit
    properties on each TREATS_TRADITIONALLY edge so downstream queries
    can filter on verdict / top_bucket / loop_closed without re-running.
    """

    @property
    def name(self) -> str:
        return "TaskAValidator"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        source_name: str | None = None,
        malady_name: str | None = None,
        limit: int | None = None,
        write_graph: bool = False,
        keep_top_paths: int = 5,
        **_: Any,
    ) -> dict:
        claims = self._list_claims(source_name, malady_name, limit)
        self._log_progress(
            f"Validating {len(claims)} claim(s) "
            f"(write_graph={write_graph}, keep_top_paths={keep_top_paths})"
        )

        verdicts: list[ClaimVerdict] = []
        by_verdict: dict[str, int] = {}
        by_top_bucket: dict[str, int] = {}
        loop_closed_count = 0

        for i, claim in enumerate(claims, 1):
            v = self._evaluate_claim(
                claim["source"], claim["malady"],
                keep_top_paths=keep_top_paths,
            )
            verdicts.append(v)
            by_verdict[v.verdict] = by_verdict.get(v.verdict, 0) + 1
            by_top_bucket[v.top_bucket] = by_top_bucket.get(v.top_bucket, 0) + 1
            if v.paths_loop_closed > 0:
                loop_closed_count += 1

            if write_graph:
                self._write_audit_properties(v)

            if i % 25 == 0 or i == len(claims):
                self._log_progress(f"  {i}/{len(claims)}")

        return {
            "claims_total": len(claims),
            "by_verdict": dict(sorted(by_verdict.items(), key=lambda kv: -kv[1])),
            "by_top_bucket": dict(sorted(by_top_bucket.items(), key=lambda kv: -kv[1])),
            "loop_closure_rate": (
                round(loop_closed_count / len(verdicts), 3) if verdicts else 0.0
            ),
            "verdicts": [self._verdict_to_dict(v) for v in verdicts],
        }

    # ------------------------------------------------------------------
    # Per-claim evaluation
    # ------------------------------------------------------------------

    def _list_claims(
        self,
        source_name: str | None,
        malady_name: str | None,
        limit: int | None,
    ) -> list[dict]:
        if source_name and malady_name:
            return [{"source": source_name, "malady": malady_name}]

        rows = self.client.run(_LIST_CLAIMS_QUERY)
        if source_name:
            rows = [r for r in rows if r["source"] == source_name]
        if malady_name:
            rows = [r for r in rows if r["malady"] == malady_name]
        if limit is not None:
            rows = rows[:limit]
        return rows

    def _evaluate_claim(
        self, source: str, malady: str, *, keep_top_paths: int,
    ) -> ClaimVerdict:
        anchor = self._pull_anchor(source, malady)
        if anchor is None:
            return self._claim_not_found(source, malady)

        primary_disease = anchor.get("primary_disease")
        component_diseases: list[str] = [
            d for d in (anchor.get("component_diseases") or []) if d
        ]
        component_set = set(component_diseases)

        path_rows = self.client.run(_PATHS_QUERY, {"source": source})
        paths: list[_PathFeatures] = [
            self._compute_path_features(r, primary_disease, component_set)
            for r in path_rows
        ]

        corrob = self._pull_corroboration(source, malady)

        return self._build_verdict(
            anchor=anchor,
            paths=paths,
            corrob=corrob,
            component_diseases=component_diseases,
            keep_top_paths=keep_top_paths,
        )

    def _pull_anchor(self, source: str, malady: str) -> dict | None:
        rows = self.client.run(_ANCHOR_QUERY, {"source": source, "malady": malady})
        return rows[0] if rows else None

    def _pull_corroboration(self, source: str, malady: str) -> dict:
        rows = self.client.run(_CROSS_CORROB_QUERY, {"source": source, "malady": malady})
        if not rows:
            return {"sibling_sources": 0, "shared_compounds": 0}
        return {
            "sibling_sources": int(rows[0].get("sibling_sources") or 0),
            "shared_compounds": int(rows[0].get("shared_compounds") or 0),
        }

    @staticmethod
    def _claim_not_found(source: str, malady: str) -> ClaimVerdict:
        return ClaimVerdict(
            source=source, malady=malady,
            mapper_classification=None,
            primary_disease=None, component_diseases=[],
            treats_confidence=0.0,
            primary_mapping_source=None,
            primary_mapping_quality="missing",
            has_compounds=False, has_mechanism=False, has_mapping=False,
            path_count=0, paths_with_target=0, paths_with_disease=0,
            paths_loop_closed=0, paths_component_closed=0,
            unique_compounds=0, unique_targets=0,
            paths_by_bucket={}, top_bucket="unrated",
            top_bucket_path_count=0, top_bucket_max_score=0.0,
            sibling_sources=0, shared_compounds=0,
            verdict="claim_not_found",
            rationale="No active TREATS_TRADITIONALLY edge between source and malady.",
        )

    # ------------------------------------------------------------------
    # Path-level features (tier bucket + multiplicative tiebreak)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_path_features(
        row: dict,
        primary_disease: str | None,
        component_diseases: set[str],
    ) -> _PathFeatures:
        ex_type = row.get("ex_type") or ""
        tgt_type = row.get("tgt_type")
        rel_type = row.get("rel_type")

        ex_tier = EX_TIER.get(ex_type, "unrated")
        tgt_tier = TGT_TIER.get(tgt_type) if tgt_type else None
        rel_tier = REL_TIER.get(rel_type) if rel_type else None

        # Bucket = min over PRESENT edge tiers. compound-only paths bucket
        # by ex_tier alone; that's how "has compounds, no mechanism"
        # surfaces in the distribution.
        present_tiers = [t for t in (ex_tier, tgt_tier, rel_tier) if t]
        path_bucket = _tier_min(present_tiers)

        ex_conf = float(row.get("ex_conf") or 0.0)
        tgt_conf = row.get("tgt_conf")
        rel_conf = row.get("rel_conf")

        # Multiplicative score over PRESENT edge confidences only.
        # Don't penalize a compound-only row for lacking targets — the
        # bucket already captured that.
        score = ex_conf
        if tgt_conf is not None:
            score *= float(tgt_conf)
        if rel_conf is not None:
            score *= float(rel_conf)

        reached = row.get("reached_disease")
        loop_closed = bool(
            reached and primary_disease and reached == primary_disease
        )
        component_closed = bool(
            reached and component_diseases and reached in component_diseases
        )

        return _PathFeatures(
            compound_inchikey=row.get("compound_inchikey") or "",
            compound_name=row.get("compound") or "",
            target_chembl_id=row.get("target_chembl_id"),
            target_name=row.get("target"),
            target_type=row.get("target_type"),
            reached_disease=reached,
            ex_type=ex_type,
            ex_conf=ex_conf,
            tgt_type=tgt_type,
            tgt_conf=float(tgt_conf) if tgt_conf is not None else None,
            tgt_pchembl=(float(row["tgt_pchembl"])
                         if row.get("tgt_pchembl") is not None else None),
            rel_type=rel_type,
            rel_conf=float(rel_conf) if rel_conf is not None else None,
            rel_requires_review=bool(row.get("rel_requires_review")),
            ex_tier=ex_tier,
            tgt_tier=tgt_tier,
            rel_tier=rel_tier,
            path_bucket=path_bucket,
            path_score=round(score, 6),
            has_target=tgt_type is not None,
            has_disease=rel_type is not None,
            loop_closed=loop_closed,
            component_closed=component_closed,
            raw_row=dict(row),
        )

    # ------------------------------------------------------------------
    # Verdict builder
    # ------------------------------------------------------------------

    def _build_verdict(
        self,
        *,
        anchor: dict,
        paths: list[_PathFeatures],
        corrob: dict,
        component_diseases: list[str],
        keep_top_paths: int,
    ) -> ClaimVerdict:
        primary_disease = anchor.get("primary_disease")
        primary_mapping_source = anchor.get("primary_mapping_source")
        treats_conf = float(anchor.get("treats_conf") or 0.0)

        primary_mapping_quality = (
            MAP_QUALITY.get(primary_mapping_source, "silver")
            if primary_mapping_source else "missing"
        )

        has_compounds = bool(paths)
        has_mechanism = any(p.has_target for p in paths)
        has_mapping = primary_disease is not None

        # Aggregate counts
        paths_with_target = sum(1 for p in paths if p.has_target)
        paths_with_disease = sum(1 for p in paths if p.has_disease)
        paths_loop_closed = sum(1 for p in paths if p.loop_closed)
        paths_component_closed = sum(1 for p in paths if p.component_closed)
        unique_compounds = len({p.compound_inchikey for p in paths})
        unique_targets = len({p.target_chembl_id for p in paths if p.target_chembl_id})

        paths_by_bucket: dict[str, int] = {}
        for p in paths:
            paths_by_bucket[p.path_bucket] = paths_by_bucket.get(p.path_bucket, 0) + 1

        # Top bucket = best non-empty bucket. Within that bucket,
        # multiplicative tiebreak gives the max path_score.
        if paths:
            top_bucket = max(
                paths_by_bucket.keys(), key=lambda b: _tier_rank(b),
            )
            top_bucket_paths = [p for p in paths if p.path_bucket == top_bucket]
            top_bucket_path_count = len(top_bucket_paths)
            top_bucket_max_score = max(p.path_score for p in top_bucket_paths)
        else:
            top_bucket = "unrated"
            top_bucket_path_count = 0
            top_bucket_max_score = 0.0

        verdict = self._assign_verdict(
            has_compounds=has_compounds,
            has_mechanism=has_mechanism,
            has_mapping=has_mapping,
            paths_loop_closed=paths_loop_closed,
            top_bucket=top_bucket,
            top_bucket_path_count=top_bucket_path_count,
        )

        rationale = self._build_rationale(
            verdict=verdict,
            primary_disease=primary_disease,
            top_bucket=top_bucket,
            top_bucket_path_count=top_bucket_path_count,
            top_bucket_max_score=top_bucket_max_score,
            paths_loop_closed=paths_loop_closed,
            paths_component_closed=paths_component_closed,
            paths_total=len(paths),
            sibling_sources=int(corrob["sibling_sources"]),
            primary_mapping_quality=primary_mapping_quality,
        )

        # Top-N audit paths: full mechanistic chains (compound→target→
        # disease) FIRST, then closed loops, then bucket rank, then score.
        # The full-chain pre-sort matters because a compound-only path
        # gets bucket=ex_tier (often gold) and a single-edge product score
        # (often ~0.7), which would otherwise outrank a real triple-edge
        # path scoring ~0.18 — and the LLM downstream needs to see the
        # actual mechanistic evidence, not just the COCONUT link.
        sorted_paths = sorted(
            paths,
            key=lambda p: (
                not (p.has_target and p.has_disease),
                not p.loop_closed,
                -_tier_rank(p.path_bucket),
                -p.path_score,
            ),
        )
        paths_top = [self._path_to_dict(p) for p in sorted_paths[:keep_top_paths]]

        return ClaimVerdict(
            source=anchor["source"],
            malady=anchor["malady"],
            mapper_classification=anchor.get("mapper_classification"),
            primary_disease=primary_disease,
            component_diseases=component_diseases,
            treats_confidence=treats_conf,
            primary_mapping_source=primary_mapping_source,
            primary_mapping_quality=primary_mapping_quality,
            has_compounds=has_compounds,
            has_mechanism=has_mechanism,
            has_mapping=has_mapping,
            path_count=len(paths),
            paths_with_target=paths_with_target,
            paths_with_disease=paths_with_disease,
            paths_loop_closed=paths_loop_closed,
            paths_component_closed=paths_component_closed,
            unique_compounds=unique_compounds,
            unique_targets=unique_targets,
            paths_by_bucket=paths_by_bucket,
            top_bucket=top_bucket,
            top_bucket_path_count=top_bucket_path_count,
            top_bucket_max_score=round(top_bucket_max_score, 6),
            sibling_sources=int(corrob["sibling_sources"]),
            shared_compounds=int(corrob["shared_compounds"]),
            verdict=verdict,
            rationale=rationale,
            paths_top=paths_top,
            context=self._build_context(anchor),
        )

    @staticmethod
    def _build_context(anchor: dict) -> dict:
        """Anchor-level audit fields the LLM critic / paper tables consume.

        Captures the actual historical text (treats_evidence_span — the
        most important context for validating a claim), the source's
        canonicalisation grade, the malady's symptom description and
        aliases, the primary disease's full set of ontology IDs, and
        the mapper's per-edge rationale + alternatives. Stored as a flat
        dict because consumers serialize it as JSON anyway.
        """
        return {
            "source": {
                "name": anchor.get("source"),
                "aliases": anchor.get("source_aliases") or [],
                "source_document": anchor.get("source_document"),
                "evidence_span": anchor.get("source_evidence_span"),
                "canonical_name": anchor.get("source_canonical_name"),
                "canonical_part": anchor.get("source_canonical_part"),
                "canonical_type": anchor.get("source_canonical_type"),
                "canonical_source": anchor.get("source_canonical_source"),
            },
            "malady": {
                "name": anchor.get("malady"),
                "aliases": anchor.get("malady_aliases") or [],
                "description": anchor.get("malady_description"),
                "evidence_span": anchor.get("malady_evidence_span"),
                "source_document": anchor.get("malady_source_document"),
                "mapper_classification": anchor.get("mapper_classification"),
            },
            "treats_edge": {
                "evidence_span": anchor.get("treats_evidence"),
                "confidence_score": anchor.get("treats_conf"),
            },
            "primary_disease": {
                "name": anchor.get("primary_disease"),
                "icd10_code": anchor.get("primary_disease_icd10"),
                "mesh_id": anchor.get("primary_disease_mesh"),
                "snomed_id": anchor.get("primary_disease_snomed"),
                "efo_id": anchor.get("primary_disease_efo"),
                "mondo_id": anchor.get("primary_disease_mondo"),
                "doid_id": anchor.get("primary_disease_doid"),
                "verified_by": anchor.get("primary_disease_verified_by"),
            },
            "primary_mapping": {
                "mapping_source": anchor.get("primary_mapping_source"),
                "confidence_score": anchor.get("primary_mapping_conf"),
                "mapper_rationale": anchor.get("primary_mapping_rationale"),
                "mapping_alternatives": anchor.get("primary_mapping_alternatives"),
                "requires_review": anchor.get("primary_mapping_requires_review"),
            },
        }

    @staticmethod
    def _assign_verdict(
        *,
        has_compounds: bool,
        has_mechanism: bool,
        has_mapping: bool,
        paths_loop_closed: int,
        top_bucket: str,
        top_bucket_path_count: int,
    ) -> str:
        if not has_compounds:
            return "traditional_only"
        if not has_mapping:
            # The malady has no Modern_Disease anchor (mapper said
            # tcm_no_equivalent or status='unverified'), so the loop is
            # uncheckable in principle, not just empirically.
            return "mechanistic_only"
        if paths_loop_closed == 0:
            return "unsupported"
        # Loop closes — grade by top bucket
        if top_bucket == "gold" and top_bucket_path_count >= 2:
            return "strong_support"
        if top_bucket in ("gold", "silver"):
            return "moderate_support"
        return "partial_support"

    @staticmethod
    def _build_rationale(
        *,
        verdict: str,
        primary_disease: str | None,
        top_bucket: str,
        top_bucket_path_count: int,
        top_bucket_max_score: float,
        paths_loop_closed: int,
        paths_component_closed: int,
        paths_total: int,
        sibling_sources: int,
        primary_mapping_quality: str,
    ) -> str:
        # Deterministic — same inputs always produce the same string. LLM
        # rationales would belong on a separate Critic agent.
        suffix_parts: list[str] = []
        if sibling_sources:
            suffix_parts.append(
                f"corroborated by {sibling_sources} sibling source(s)"
            )
        if paths_component_closed:
            suffix_parts.append(
                f"{paths_component_closed} path(s) reach a syndrome component"
            )
        if primary_mapping_quality == "silver":
            suffix_parts.append("malady mapping is unverified")
        suffix = " (" + "; ".join(suffix_parts) + ")" if suffix_parts else ""

        if verdict == "strong_support":
            return (
                f"{top_bucket_path_count} {top_bucket}-tier path(s) close the loop "
                f"on '{primary_disease}', top score {top_bucket_max_score:.3f}.{suffix}"
            )
        if verdict == "moderate_support":
            return (
                f"Loop closes on '{primary_disease}' via {top_bucket}-tier "
                f"evidence ({paths_loop_closed} closing path(s)).{suffix}"
            )
        if verdict == "partial_support":
            return (
                f"Loop closes on '{primary_disease}' but only via {top_bucket}-tier "
                f"evidence ({paths_loop_closed} closing path(s)).{suffix}"
            )
        if verdict == "unsupported":
            return (
                f"{paths_total} mechanistic path(s) exist but none reach "
                f"'{primary_disease}'.{suffix}"
            )
        if verdict == "mechanistic_only":
            return (
                f"{paths_total} mechanistic path(s) exist but the malady has no "
                f"Modern_Disease mapping to check against.{suffix}"
            )
        if verdict == "traditional_only":
            return "No compounds linked to this source — claim is purely traditional."
        return ""

    # ------------------------------------------------------------------
    # Audit writeback
    # ------------------------------------------------------------------

    def _write_audit_properties(self, v: ClaimVerdict) -> None:
        """Stamp task_a_* audit properties on the TREATS edge.

        Mirrors the Phase 1 status-flag pattern (linker_status,
        mapper_status, etc.). Idempotent — re-running overwrites with
        the latest verdict and bumps task_a_attempted_at.
        """
        if v.verdict == "claim_not_found":
            return
        props: dict[str, Any] = {
            "task_a_verdict": v.verdict,
            "task_a_top_bucket": v.top_bucket,
            "task_a_top_bucket_path_count": v.top_bucket_path_count,
            "task_a_top_bucket_max_score": v.top_bucket_max_score,
            "task_a_path_count": v.path_count,
            "task_a_paths_loop_closed": v.paths_loop_closed,
            "task_a_paths_component_closed": v.paths_component_closed,
            "task_a_unique_compounds": v.unique_compounds,
            "task_a_unique_targets": v.unique_targets,
            "task_a_sibling_sources": v.sibling_sources,
            "task_a_primary_mapping_quality": v.primary_mapping_quality,
            "task_a_rationale": v.rationale,
            "task_a_attempted_at": dt.datetime.utcnow().isoformat(),
        }
        self.client.run_write(
            _WRITE_VERDICT_QUERY,
            {"source": v.source, "malady": v.malady, "props": props},
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _path_to_dict(p: _PathFeatures) -> dict:
        """Nested per-path payload for the LLM critic.

        Sections:
          compound  — every Chemical_Compound property (smiles, np_likeness, ...)
          target    — every Biological_Target property (gene_symbol, uniprot, ...)
          reached_disease — every Modern_Disease property (icd10, mesh, efo, ...)
          edges.ex / .tgt / .rel — each edge's evidence_type, tier, confidence,
                                   plus all evidence-level audit fields
                                   (assay_description, mechanism_action, OT
                                   datasource scores, mapper rationale, etc.)
          path      — derived signals (bucket, score, loop closure)
        """
        r = p.raw_row or {}
        compound: dict[str, Any] = {
            "name": p.compound_name,
            "inchikey": p.compound_inchikey,
            "smiles": r.get("smiles"),
            "molecular_formula": r.get("molecular_formula"),
            "np_likeness": r.get("compound_np_likeness"),
            "annotation_level": r.get("compound_annotation_level"),
            "chembl_id": r.get("compound_chembl_id"),
            "source_db": r.get("compound_source_db"),
        }
        target: dict[str, Any] | None = None
        if p.has_target:
            target = {
                "name": p.target_name,
                "chembl_id": p.target_chembl_id,
                "pref_name": r.get("target_pref_name"),
                "type": p.target_type,
                "gene_symbol": r.get("target_gene_symbol"),
                "uniprot_id": r.get("target_uniprot_id"),
                "ncbi_tax_id": r.get("target_ncbi_tax_id"),
            }
        reached_disease: dict[str, Any] | None = None
        if p.has_disease:
            reached_disease = {
                "name": p.reached_disease,
                "icd10_code": r.get("reached_icd10"),
                "mesh_id": r.get("reached_mesh"),
                "snomed_id": r.get("reached_snomed"),
                "efo_id": r.get("reached_efo"),
                "mondo_id": r.get("reached_mondo"),
                "doid_id": r.get("reached_doid"),
            }
        edges: dict[str, Any] = {
            "ex": {
                "evidence_type": p.ex_type,
                "tier": p.ex_tier,
                "confidence": p.ex_conf,
                "base_prior": r.get("ex_base_prior"),
                "part_penalty": r.get("ex_part_penalty"),
                "evidence_resolution": r.get("ex_resolution"),
                "species_part_context": r.get("part_context"),
                "part_specific": r.get("part_specific"),
                "np_likeness": r.get("ex_np_likeness"),
                "annotation_level": r.get("ex_annotation_level"),
                "pubchem_formula_ambiguous": r.get("ex_formula_ambiguous"),
                "lookup_query": r.get("ex_lookup_query"),
            },
        }
        if p.has_target:
            edges["tgt"] = {
                "evidence_type": p.tgt_type,
                "tier": p.tgt_tier,
                "confidence": p.tgt_conf,
                "pchembl_score": p.tgt_pchembl,
                "assay_id": r.get("tgt_assay_id"),
                "assay_type": r.get("tgt_assay_type"),
                "assay_description": r.get("tgt_assay_description"),
                "mechanism_action": r.get("tgt_mechanism_action"),
            }
        if p.has_disease:
            edges["rel"] = {
                "evidence_type": p.rel_type,
                "tier": p.rel_tier,
                "confidence": p.rel_conf,
                "ot_overall_score": r.get("ot_score"),
                "ot_resolved_id": r.get("ot_resolved_id"),
                "ot_target_ensembl_id": r.get("ot_target_ensembl_id"),
                "ot_top_subunit_uniprot": r.get("ot_top_subunit_uniprot"),
                "ot_datasource_scores": r.get("ot_datasource_scores"),
                "pathogen_lookup_source": r.get("pathogen_lookup_source"),
                "rationale": r.get("rel_rationale"),
                "requires_review": p.rel_requires_review,
            }
        return {
            "compound": compound,
            "target": target,
            "reached_disease": reached_disease,
            "edges": edges,
            "path": {
                "bucket": p.path_bucket,
                "score": p.path_score,
                "loop_closed": p.loop_closed,
                "component_closed": p.component_closed,
                "has_target": p.has_target,
                "has_disease": p.has_disease,
            },
        }

    @staticmethod
    def _verdict_to_dict(v: ClaimVerdict) -> dict:
        return asdict(v)
