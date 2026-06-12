"""Phase 2 — Task B: Compound Nominator (Pass 3).

Given a Modern_Disease query, nominates novel compound candidates from
the historical knowledge graph that may treat it. Pure deterministic
reads — no LLM in the loop. Drop-in replaceable with future LLM-critic
agents at a later pass.

Walks the graph in BOTH directions:

  Historical chain (backward — the claim provenance):
    Disease ←MAPS_TO← Malady ←TREATS_TRADITIONALLY← Source ←IS_EXTRACTED_FROM← Compound

  Cross-validation chain (forward — modern biomedical loop closure):
    Compound →TARGETS→ Target →RELATES_TO→ Disease

A compound is a "loop-closed" nomination iff at least one of its target
relations RELATES_TO the *same* query disease. Loop-closed candidates
are ranked above non-closed ones.

Scoring (consistent with TaskAValidator):
  - Tier bucket primary, multiplicative tiebreak
  - Forward (cross-validation) bucket = MIN over (TARGETS, RELATES_TO)
    edge tiers. Same EX_TIER/TGT_TIER/REL_TIER lookup tables.
  - Within a bucket, score = product of confidence on the strongest
    forward path.

Composite ranking key per compound (sorted descending):
  1. has_loop_closure             (closed-loop candidates first)
  2. forward_bucket_rank          (gold > silver > bronze > wood)
  3. unique_sources_count         (multi-source corroboration)
  4. unique_maladies_count        (multiple malady → disease routes)
  5. forward_max_score            (multiplicative tiebreak)

Novelty filter:
  - Strict (default): drop compounds with `(c)-[:KNOWN_TREATS]->(d)`
  - In-memory masking: an optional `masked_known_treats` set lets the
    eval harness simulate "this pair was masked" without writing to
    the graph. The set contains (compound_inchikey, disease_name)
    tuples that should be treated as novel for this run.
  - Permissive (`apply_novelty_filter=False`): keep all candidates,
    annotate KNOWN_TREATS status on each card.

Output (per nomination):
  - Compound: name, inchikey, smiles, molecular_formula, np_likeness,
              annotation_level, ChEMBL ID, COCONUT/PubChem flags
  - Top historical paths: source, malady, mapper rationale,
                          treats_evidence_span (the actual TCM text)
  - Top forward paths: target (gene_symbol/uniprot), assay description,
                       OT score, evidence_type
  - Aggregates: unique_sources, unique_maladies, total_paths,
                forward_bucket, forward_max_score, has_loop_closure
  - Provenance: existing KNOWN_TREATS for OTHER diseases (anchor on
                real-world pharmacology)

This nominator is the proposed-system Pass 3 for Task B. The eval
harness uses it directly (with masking) for the drug-repurposing
recovery test (Eval A). Future LLM critic refinement (Pass 4) would
consume these nominations and add semantic judgment.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from src.agents.base import BaseAgent
from src.agents.phase2.task_a_validator import (
    EX_TIER,
    REL_TIER,
    TGT_TIER,
    _tier_min,
    _tier_rank,
)
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher (read-only)
# ---------------------------------------------------------------------------

# Anchor: the disease and any of its mapped maladies (for context)
_ANCHOR_QUERY = """
MATCH (d:Modern_Disease {name: $disease_name})
WHERE d.archived IS NULL OR d.archived = false
RETURN
  d.name AS disease,
  d.icd10_code AS icd10_code,
  d.mesh_id AS mesh_id,
  d.snomed_id AS snomed_id,
  d.efo_id AS efo_id,
  d.mondo_id AS mondo_id,
  d.doid_id AS doid_id
"""


# Pulls every (compound, source, malady) backward route from the disease.
# Returns one row per compound-source-malady triple; aggregation per
# compound happens client-side. Never returns rows for archived nodes.
_BACKWARD_QUERY = """
MATCH (d:Modern_Disease {name: $disease_name})
WHERE d.archived IS NULL OR d.archived = false
MATCH (m:Traditional_Malady)-[r_map:MAPS_TO]->(d)
WHERE r_map.is_primary = true
  AND (m.archived IS NULL OR m.archived = false)
MATCH (s:Source)-[r_tt:TREATS_TRADITIONALLY]->(m)
WHERE s.archived IS NULL OR s.archived = false
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)
WHERE c.archived IS NULL OR c.archived = false
RETURN
  c.inchikey AS compound_inchikey,
  c.name AS compound,
  c.smiles AS smiles,
  c.molecular_formula AS molecular_formula,
  c.np_likeness AS np_likeness,
  c.annotation_level AS annotation_level,
  c.linker_chembl_id AS chembl_id,
  c.source_db AS source_db,
  s.name AS source,
  s.canonical_name AS source_canonical_name,
  s.canonical_part AS source_canonical_part,
  s.canonical_type AS source_canonical_type,
  m.name AS malady,
  m.description AS malady_description,
  m.mapper_classification AS mapper_classification,
  r_ex.evidence_type AS ex_type,
  r_ex.confidence_score AS ex_conf,
  r_ex.species_part_context AS part_context,
  r_tt.confidence_score AS tt_conf,
  r_tt.evidence_span AS treats_evidence_span,
  r_map.confidence_score AS map_conf,
  r_map.mapping_source AS map_source,
  r_map.mapper_rationale AS mapper_rationale
"""


# Pulls every cross-validation forward path: compound → target → disease
# (same disease as query). Used to determine loop closure and forward
# bucket per compound.
_FORWARD_QUERY = """
MATCH (d:Modern_Disease {name: $disease_name})
WHERE d.archived IS NULL OR d.archived = false
MATCH (c:Chemical_Compound)-[r_tgt:TARGETS]->(t:Biological_Target)-[r_rel:RELATES_TO]->(d)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (t.archived IS NULL OR t.archived = false)
RETURN
  c.inchikey AS compound_inchikey,
  t.target_chembl_id AS target_chembl_id,
  t.name AS target,
  t.target_pref_name AS target_pref_name,
  t.gene_symbol AS gene_symbol,
  t.uniprot_id AS uniprot_id,
  t.target_type AS target_type,
  t.ncbi_tax_id AS ncbi_tax_id,
  r_tgt.evidence_type AS tgt_type,
  r_tgt.confidence_score AS tgt_conf,
  r_tgt.pchembl_score AS tgt_pchembl,
  r_tgt.assay_id AS assay_id,
  r_tgt.assay_type AS assay_type,
  r_tgt.assay_description AS assay_description,
  r_tgt.mechanism_action AS mechanism_action,
  r_rel.evidence_type AS rel_type,
  r_rel.confidence AS rel_conf,
  r_rel.ot_overall_score AS ot_score,
  r_rel.ot_resolved_id AS ot_resolved_id,
  r_rel.rationale AS rel_rationale,
  r_rel.requires_review AS rel_requires_review
"""


# Per-compound KNOWN_TREATS for ALL diseases — used to apply the
# novelty filter (strict) AND to surface "this compound is approved for
# X" context in the nomination card.
_KNOWN_TREATS_QUERY = """
UNWIND $inchikeys AS ik
MATCH (c:Chemical_Compound {inchikey: ik})-[k:KNOWN_TREATS]->(d:Modern_Disease)
WHERE (c.archived IS NULL OR c.archived = false)
  AND (d.archived IS NULL OR d.archived = false)
RETURN
  ik AS inchikey,
  collect({
    disease: d.name,
    clinical_phase: k.clinical_phase,
    evidence_type: k.evidence_type,
    mesh_heading: k.mesh_heading,
    efo_term: k.efo_term
  }) AS known_treats
"""


_WRITE_AUDIT_QUERY = """
MATCH (d:Modern_Disease {name: $disease_name})
SET d.task_b_last_run_at = $attempted_at,
    d.task_b_last_run_top_k = $top_k,
    d.task_b_last_run_total_candidates = $total_candidates,
    d.task_b_last_run_loop_closed_count = $loop_closed_count
"""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class _ForwardPath:
    """One cross-validation Compound→Target→Disease path."""
    target_chembl_id: str
    target: str | None
    target_pref_name: str | None
    gene_symbol: str | None
    uniprot_id: str | None
    target_type: str | None
    ncbi_tax_id: str | None
    tgt_type: str | None
    tgt_tier: str
    tgt_conf: float | None
    tgt_pchembl: float | None
    assay_id: str | None
    assay_type: str | None
    assay_description: str | None
    mechanism_action: str | None
    rel_type: str | None
    rel_tier: str
    rel_conf: float | None
    ot_score: float | None
    ot_resolved_id: str | None
    rel_rationale: str | None
    rel_requires_review: bool
    bucket: str
    score: float


@dataclass
class _HistoricalPath:
    """One Compound→Source→Malady→Disease backward route."""
    source: str
    source_canonical_name: str | None
    source_canonical_part: str | None
    source_canonical_type: str | None
    malady: str
    malady_description: str | None
    mapper_classification: str | None
    ex_type: str
    ex_tier: str
    ex_conf: float
    part_context: str | None
    tt_conf: float | None
    treats_evidence_span: str | None
    map_conf: float | None
    map_source: str | None
    mapper_rationale: str | None


@dataclass
class CompoundNomination:
    """One ranked compound candidate with full evidence card."""
    rank: int

    # Compound identity + structural
    compound: str
    compound_inchikey: str
    smiles: str | None
    molecular_formula: str | None
    np_likeness: float | None
    annotation_level: int | None
    chembl_id: str | None
    source_db: str | None

    # Aggregate signals
    unique_sources: int
    unique_maladies: int
    unique_targets: int
    historical_path_count: int
    forward_path_count: int
    has_loop_closure: bool
    forward_bucket: str            # "gold" | "silver" | "bronze" | "wood" | "unrated"
    forward_max_score: float
    forward_bucket_rank: int       # for sort stability

    # Top-N evidence
    top_historical_paths: list[dict]
    top_forward_paths: list[dict]

    # KNOWN_TREATS context
    has_known_treats_for_query: bool
    masked_for_eval: bool          # True if (inchikey, disease) was in mask set
    other_known_treats: list[dict] # KNOWN_TREATS for OTHER diseases (anchor)

    # Composite score (for downstream consumers — not used internally)
    composite_score: float


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TaskBNominator(BaseAgent):
    """Deterministic compound nominator for Task B.

    Read-only by default. With `--write-graph` (the runner) stamps a
    handful of audit properties on the queried Modern_Disease node
    (task_b_last_run_*); never mutates compound or KNOWN_TREATS data.

    Eval-friendly: the `masked_known_treats` argument to run() lets a
    drug-repurposing recovery harness simulate "this (compound, disease)
    KNOWN_TREATS edge was held out" without touching the graph.
    """

    @property
    def name(self) -> str:
        return "TaskBNominator"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        disease_name: str,
        top_k: int = 20,
        apply_novelty_filter: bool = True,
        require_loop_closure: bool = False,
        masked_known_treats: set[tuple[str, str]] | None = None,
        keep_top_paths_per_kind: int = 3,
        write_graph: bool = False,
        **_: Any,
    ) -> dict:
        """Nominate compounds for `disease_name`.

        Args:
          disease_name: exact match on Modern_Disease.name (case-sensitive).
          top_k: max nominations to return.
          apply_novelty_filter: drop compounds with KNOWN_TREATS for the
              query disease (unless that pair is in `masked_known_treats`).
          require_loop_closure: only nominate compounds with at least one
              forward Target→RELATES_TO→disease path. Default False —
              compounds without a forward path are still candidates,
              ranked below loop-closed ones.
          masked_known_treats: set of (compound_inchikey, disease_name)
              pairs to treat as novel for THIS run only. Used by the
              drug-repurposing eval to mask specific edges without
              writing to the graph.
          keep_top_paths_per_kind: top-N historical and top-N forward
              paths to keep per nomination card (default 3 each).
          write_graph: stamp task_b_last_run_* audit properties on the
              Modern_Disease node. Default False (read-only).
        """
        masked = masked_known_treats or set()

        anchor = self._pull_anchor(disease_name)
        if anchor is None:
            return {
                "disease": disease_name,
                "found": False,
                "reason": "no active Modern_Disease with that name",
                "nominations": [],
            }

        backward_rows = self.client.run(_BACKWARD_QUERY, {"disease_name": disease_name})
        forward_rows = self.client.run(_FORWARD_QUERY, {"disease_name": disease_name})

        # Aggregate per compound
        per_compound = self._aggregate_per_compound(
            backward_rows, forward_rows,
        )

        if not per_compound:
            return {
                "disease": disease_name,
                "anchor": anchor,
                "found": True,
                "candidates_total": 0,
                "nominations": [],
            }

        # Pull KNOWN_TREATS for all candidate compounds in one batch
        inchikeys = sorted(per_compound.keys())
        known_treats_by_compound = self._pull_known_treats(inchikeys)

        # Annotate each compound with KNOWN_TREATS info + apply novelty filter
        nominations_raw: list[dict] = []
        for ik, agg in per_compound.items():
            kt_list = known_treats_by_compound.get(ik, [])
            has_kt_for_query = any(
                kt.get("disease") == disease_name for kt in kt_list
            )
            other_kt = [
                kt for kt in kt_list if kt.get("disease") != disease_name
            ]
            masked_for_eval = (ik, disease_name) in masked

            # Novelty filter: skip if has KNOWN_TREATS for query disease,
            # UNLESS that specific pair is masked (eval mode).
            if (
                apply_novelty_filter
                and has_kt_for_query
                and not masked_for_eval
            ):
                continue

            # Loop-closure filter
            if require_loop_closure and not agg["has_loop_closure"]:
                continue

            agg["has_known_treats_for_query"] = has_kt_for_query
            agg["masked_for_eval"] = masked_for_eval
            agg["other_known_treats"] = other_kt
            nominations_raw.append(agg)

        # Sort by composite ranking key (descending). Counts are derived
        # in _aggregate_per_compound — we use the *_count int variants
        # (the *_unique* keys are still sets at this point).
        ranked = sorted(
            nominations_raw,
            key=lambda a: (
                not a["has_loop_closure"],            # closed-loop first
                -_tier_rank(a["forward_bucket"]),     # higher tier first
                -a["unique_sources_count"],
                -a["unique_maladies_count"],
                -a["forward_max_score"],
            ),
        )

        nominations = [
            self._build_nomination(rank, agg, keep_top_paths_per_kind)
            for rank, agg in enumerate(ranked[:top_k], 1)
        ]

        loop_closed_count = sum(1 for n in nominations if n.has_loop_closure)

        if write_graph:
            self._write_audit(
                disease_name=disease_name,
                top_k=top_k,
                total_candidates=len(per_compound),
                loop_closed_count=loop_closed_count,
            )

        self._log_progress(
            f"Disease={disease_name!r}: {len(per_compound)} candidates -> "
            f"{len(ranked)} after filter -> {len(nominations)} nominations "
            f"({loop_closed_count} loop-closed)"
        )

        return {
            "disease": disease_name,
            "anchor": anchor,
            "found": True,
            "candidates_total": len(per_compound),
            "after_filter": len(ranked),
            "nominations_returned": len(nominations),
            "loop_closed_count": loop_closed_count,
            "novelty_filter_applied": apply_novelty_filter,
            "loop_closure_required": require_loop_closure,
            "masked_pairs_count": len(masked),
            "nominations": [self._nomination_to_dict(n) for n in nominations],
        }

    # ------------------------------------------------------------------
    # Anchor + KNOWN_TREATS lookup
    # ------------------------------------------------------------------

    def _pull_anchor(self, disease_name: str) -> dict | None:
        rows = self.client.run(_ANCHOR_QUERY, {"disease_name": disease_name})
        return rows[0] if rows else None

    def _pull_known_treats(
        self, inchikeys: list[str],
    ) -> dict[str, list[dict]]:
        if not inchikeys:
            return {}
        rows = self.client.run(_KNOWN_TREATS_QUERY, {"inchikeys": inchikeys})
        return {r["inchikey"]: (r.get("known_treats") or []) for r in rows}

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_per_compound(
        self,
        backward_rows: list[dict],
        forward_rows: list[dict],
    ) -> dict[str, dict]:
        """Group rows by compound inchikey; compute per-compound aggregate
        signals and keep the raw row lists for top-N path selection."""
        per_compound: dict[str, dict] = {}

        # Backward rows
        for r in backward_rows:
            ik = r.get("compound_inchikey") or ""
            if not ik:
                continue
            agg = per_compound.setdefault(ik, _new_agg(ik, r))
            ex_type = r.get("ex_type") or ""
            ex_tier = EX_TIER.get(ex_type, "unrated")
            ex_conf = float(r.get("ex_conf") or 0.0)
            agg["historical_paths"].append(_HistoricalPath(
                source=r.get("source") or "",
                source_canonical_name=r.get("source_canonical_name"),
                source_canonical_part=r.get("source_canonical_part"),
                source_canonical_type=r.get("source_canonical_type"),
                malady=r.get("malady") or "",
                malady_description=r.get("malady_description"),
                mapper_classification=r.get("mapper_classification"),
                ex_type=ex_type,
                ex_tier=ex_tier,
                ex_conf=ex_conf,
                part_context=r.get("part_context"),
                tt_conf=(float(r["tt_conf"])
                         if r.get("tt_conf") is not None else None),
                treats_evidence_span=r.get("treats_evidence_span"),
                map_conf=(float(r["map_conf"])
                          if r.get("map_conf") is not None else None),
                map_source=r.get("map_source"),
                mapper_rationale=r.get("mapper_rationale"),
            ))
            agg["unique_sources"].add(r.get("source") or "")
            agg["unique_maladies"].add(r.get("malady") or "")

        # Forward rows
        for r in forward_rows:
            ik = r.get("compound_inchikey") or ""
            if not ik:
                continue
            # Forward rows can introduce compounds the backward query
            # didn't (e.g. a compound from another source path that also
            # crosses to this disease). For Task B those are NOT
            # candidates — Task B requires the historical chain. Skip.
            if ik not in per_compound:
                continue
            tgt_type = r.get("tgt_type")
            rel_type = r.get("rel_type")
            tgt_tier = TGT_TIER.get(tgt_type, "unrated") if tgt_type else "unrated"
            rel_tier = REL_TIER.get(rel_type, "unrated") if rel_type else "unrated"
            tgt_conf = r.get("tgt_conf")
            rel_conf = r.get("rel_conf")
            score = 1.0
            if tgt_conf is not None:
                score *= float(tgt_conf)
            if rel_conf is not None:
                score *= float(rel_conf)
            bucket = _tier_min([tgt_tier, rel_tier])

            fp = _ForwardPath(
                target_chembl_id=r.get("target_chembl_id") or "",
                target=r.get("target"),
                target_pref_name=r.get("target_pref_name"),
                gene_symbol=r.get("gene_symbol"),
                uniprot_id=r.get("uniprot_id"),
                target_type=r.get("target_type"),
                ncbi_tax_id=r.get("ncbi_tax_id"),
                tgt_type=tgt_type,
                tgt_tier=tgt_tier,
                tgt_conf=float(tgt_conf) if tgt_conf is not None else None,
                tgt_pchembl=(float(r["tgt_pchembl"])
                             if r.get("tgt_pchembl") is not None else None),
                assay_id=r.get("assay_id"),
                assay_type=r.get("assay_type"),
                assay_description=r.get("assay_description"),
                mechanism_action=r.get("mechanism_action"),
                rel_type=rel_type,
                rel_tier=rel_tier,
                rel_conf=float(rel_conf) if rel_conf is not None else None,
                ot_score=(float(r["ot_score"])
                          if r.get("ot_score") is not None else None),
                ot_resolved_id=r.get("ot_resolved_id"),
                rel_rationale=r.get("rel_rationale"),
                rel_requires_review=bool(r.get("rel_requires_review")),
                bucket=bucket,
                score=round(score, 6),
            )
            agg = per_compound[ik]
            agg["forward_paths"].append(fp)
            agg["unique_targets"].add(r.get("target_chembl_id") or "")

        # Compute per-compound aggregates
        for ik, agg in per_compound.items():
            agg["historical_path_count"] = len(agg["historical_paths"])
            forward_paths: list[_ForwardPath] = agg["forward_paths"]
            agg["forward_path_count"] = len(forward_paths)
            agg["has_loop_closure"] = bool(forward_paths)

            if forward_paths:
                # Best forward path = max bucket rank, then max score
                best = max(
                    forward_paths,
                    key=lambda p: (_tier_rank(p.bucket), p.score),
                )
                agg["forward_bucket"] = best.bucket
                agg["forward_max_score"] = best.score
            else:
                agg["forward_bucket"] = "unrated"
                agg["forward_max_score"] = 0.0
            agg["forward_bucket_rank"] = _tier_rank(agg["forward_bucket"])

            agg["unique_sources_count"] = len(agg["unique_sources"])
            agg["unique_maladies_count"] = len(agg["unique_maladies"])
            agg["unique_targets_count"] = len(agg["unique_targets"])

        return per_compound

    # ------------------------------------------------------------------
    # Nomination card builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nomination(
        rank: int, agg: dict, keep_top: int,
    ) -> CompoundNomination:
        # Top historical paths: rank by ex_tier then by tt_conf*ex_conf
        hist = sorted(
            agg["historical_paths"],
            key=lambda p: (
                -_tier_rank(p.ex_tier),
                -((p.tt_conf or 1.0) * p.ex_conf),
            ),
        )
        # Top forward paths: rank by bucket then score (loop-closed only)
        fwd = sorted(
            agg["forward_paths"],
            key=lambda p: (
                -_tier_rank(p.bucket),
                -p.score,
            ),
        )

        composite = (
            (1 if agg["has_loop_closure"] else 0) * 1000
            + agg["forward_bucket_rank"] * 100
            + agg["unique_sources_count"] * 10
            + agg["forward_max_score"]
        )

        return CompoundNomination(
            rank=rank,
            compound=agg["compound_name"],
            compound_inchikey=agg["compound_inchikey"],
            smiles=agg["smiles"],
            molecular_formula=agg["molecular_formula"],
            np_likeness=agg["np_likeness"],
            annotation_level=agg["annotation_level"],
            chembl_id=agg["chembl_id"],
            source_db=agg["source_db"],
            unique_sources=agg["unique_sources_count"],
            unique_maladies=agg["unique_maladies_count"],
            unique_targets=agg["unique_targets_count"],
            historical_path_count=agg["historical_path_count"],
            forward_path_count=agg["forward_path_count"],
            has_loop_closure=agg["has_loop_closure"],
            forward_bucket=agg["forward_bucket"],
            forward_max_score=agg["forward_max_score"],
            forward_bucket_rank=agg["forward_bucket_rank"],
            top_historical_paths=[asdict(p) for p in hist[:keep_top]],
            top_forward_paths=[asdict(p) for p in fwd[:keep_top]],
            has_known_treats_for_query=agg["has_known_treats_for_query"],
            masked_for_eval=agg["masked_for_eval"],
            other_known_treats=agg["other_known_treats"],
            composite_score=round(composite, 6),
        )

    # ------------------------------------------------------------------
    # Audit writeback
    # ------------------------------------------------------------------

    def _write_audit(
        self,
        *,
        disease_name: str,
        top_k: int,
        total_candidates: int,
        loop_closed_count: int,
    ) -> None:
        self.client.run_write(
            _WRITE_AUDIT_QUERY,
            {
                "disease_name": disease_name,
                "top_k": top_k,
                "total_candidates": total_candidates,
                "loop_closed_count": loop_closed_count,
                "attempted_at": dt.datetime.utcnow().isoformat(),
            },
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _nomination_to_dict(n: CompoundNomination) -> dict:
        return asdict(n)


def _new_agg(inchikey: str, first_row: dict) -> dict:
    """Initial per-compound aggregator with metadata from the first row."""
    return {
        "compound_inchikey": inchikey,
        "compound_name": first_row.get("compound") or "",
        "smiles": first_row.get("smiles"),
        "molecular_formula": first_row.get("molecular_formula"),
        "np_likeness": (float(first_row["np_likeness"])
                        if first_row.get("np_likeness") is not None else None),
        "annotation_level": first_row.get("annotation_level"),
        "chembl_id": first_row.get("chembl_id"),
        "source_db": first_row.get("source_db"),
        "historical_paths": [],
        "forward_paths": [],
        "unique_sources": set(),
        "unique_maladies": set(),
        "unique_targets": set(),
    }
