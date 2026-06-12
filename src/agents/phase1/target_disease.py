"""Phase 1 — Target → Disease RELATES_TO Linker (BATCHED).

Same architectural shape as `CompoundDiseaseLinkerBatched`:

  * Per-Target iteration with bulk-fetch primitives (Open Targets +
    EFO/DOID via OLS + LLM safety net for ORGANISM)
  * Plan-then-apply: ALL matching done in memory; no Neo4j writes
    until the plan is complete
  * UNWIND-batched writes for delete-prior / disease ID backfill /
    edge merge / status update
  * Match-only against existing Modern_Disease nodes (deterministic
    3-tier: efo_id → normalized name → MeSH synonym expansion)
  * Self-healing graph: backfill EFO/MeSH IDs onto matched nodes
  * Per-target `td_linker_status` resume contract
  * Cross-cutting docstring caveats (per-batch atomicity, transient
    cascade protection) inherit the lessons from compound_disease

Four target-type planning branches (all share the same matching +
write infrastructure):

  SINGLE PROTEIN  →  Open Targets associatedDiseases
                     • UniProt → ENSEMBL via OT search
                     • Bulk-fetch all associated diseases per target
                     • Tier-based confidence by OT overall_score:
                         strong (>= 0.7) = 0.85
                         moderate (>= 0.4) = 0.65
                         weak (>= 0.2) = 0.45
                         < 0.2 = dropped via --min-score (default 0.2)
                     • evidence_type: ot_association_{strong,moderate,weak}

  PROTEIN COMPLEX →  Subunit fan-out via ChEMBL target_components
                     • Look up each subunit's UniProt accession
                     • Run the same OT pipeline as SINGLE PROTEIN per subunit
                     • Per-disease MAX-SCORE dedup across subunits
                     • Single RELATES_TO edge per (complex, disease)
                     • evidence_type: ot_association_complex_aggregate
                     • ot_top_subunit_uniprot stored for audit

  PROTEIN FAMILY  →  SKIP — family-level evidence is too coarse.
                     status: skipped_protein_family

  ORGANISM        →  NCBI tax_id → infectious disease via three sources:
                     1. EFO via OLS (caused_by relations)
                     2. DOID via OLS (caused_by relations)
                     3. LLM safety net if both ontologies miss
                        — generate-then-verify pattern: LLM emits
                        candidate disease names + ontology hints,
                        we verify each via NLM ICD-10/MeSH/SNOMED
                     • Confidence: 0.90 deterministic, 0.75 LLM-verified
                     • evidence_type: ncbi_pathogen_efo / _doid /
                       _consensus / _llm_verified

Disease matching (all paths): deterministic three-tier identical to
CompoundDiseaseLinkerBatched —
  Tier 1: efo_id exact (or mesh_id for ORGANISM path which uses MeSH)
  Tier 2: _normalize_for_match(name) exact
  Tier 3: ChEMBL-side MeSH synonym expansion (via OntologyClient)
No matches → dropped, counted as `no_associations` audit signal.

Known limitation: cross-phase atomicity (same as
CompoundDiseaseLinkerBatched) — each of the four write phases runs as
its own batched_unwind_write. Per-batch atomic, not per-target. See
the corresponding section in compound_disease_batched.py for the
recovery contract details.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agents.base import BaseAgent
from src.agents.phase1.malady_disease import _normalize_for_match
from src.config import GEMINI_MODEL, make_gemini_client
from src.data import chembl, open_targets
from src.data.ontology_client import OntologyClient
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class LinkStatus(str, Enum):
    LINKED = "linked"
    NO_ASSOCIATIONS = "no_associations"
    SKIPPED_NO_UNIPROT = "skipped_no_uniprot"
    SKIPPED_NO_SUBUNITS = "skipped_no_subunits"
    SKIPPED_PROTEIN_FAMILY = "skipped_protein_family"
    SKIPPED_NO_TAX_ID = "skipped_no_tax_id"
    SKIPPED_UNKNOWN_TYPE = "skipped_unknown_type"
    ERROR = "error"


# Confidence priors for protein associations (Open Targets-driven)
CONFIDENCE_OT: dict[str, float] = {
    "ot_association_strong":             0.85,   # OT score >= 0.7
    "ot_association_moderate":           0.65,   # OT score >= 0.4
    "ot_association_weak":               0.45,   # OT score >= 0.2
    "ot_association_complex_aggregate":  0.70,   # COMPLEX max-over-subunits
}

# Confidence priors for organism (pathogen → disease)
CONFIDENCE_PATHOGEN: dict[str, float] = {
    "ncbi_pathogen_efo":           0.90,   # EFO ontology authoritative
    "ncbi_pathogen_doid":          0.90,   # DOID ontology authoritative
    "ncbi_pathogen_consensus":     0.92,   # both EFO and DOID agree (small bump)
    "ncbi_pathogen_llm_verified":  0.75,   # LLM-proposed, NLM-verified
}


_DEFAULT_MIN_OT_SCORE = 0.2
_DEFAULT_WRITE_BATCH_SIZE = 500
_SUBUNIT_FAN_OUT_CAP = 10            # don't fan out beyond 10 subunits
_SYNONYM_PREFETCH_WORKERS = 8
_LLM_PREFETCH_WORKERS = 4            # parallel Gemini calls for ORGANISM safety net
_LLM_MAX_RETRIES = 3


def _ot_evidence_type(score: float) -> str | None:
    """OT overall_score → evidence_type tier; None means below threshold."""
    if score >= 0.7:
        return "ot_association_strong"
    if score >= 0.4:
        return "ot_association_moderate"
    if score >= 0.2:
        return "ot_association_weak"
    return None


# ---------------------------------------------------------------------------
# LLM safety net for ORGANISM path
# ---------------------------------------------------------------------------


_PATHOGEN_SYSTEM_PROMPT = """\
You are an infectious disease taxonomist. Given an NCBI taxonomy ID and
the organism name, return the modern Western disease(s) this organism
causes in humans. Use canonical disease nomenclature (MeSH preferred
terms, ICD-10 category names).

If the organism is NOT a human pathogen (e.g. environmental microbe,
plant pathogen, gut commensal, model organism), return an empty list —
DO NOT GUESS at any disease association.

Each mapping must include:
  name           : canonical English disease name
  ontology_hint  : "icd10" | "mesh" | "snomed" — your best guess at which
                   DB will recognize this name (used for verifier routing)
  rationale      : one short sentence explaining the disease causation
"""


_PATHOGEN_SCHEMA = {
    "type": "object",
    "properties": {
        "diseases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "ontology_hint": {
                        "type": "string",
                        "enum": ["icd10", "mesh", "snomed"],
                    },
                    "rationale": {"type": "string"},
                },
                "required": ["name", "ontology_hint", "rationale"],
            },
        },
    },
    "required": ["diseases"],
}


# ---------------------------------------------------------------------------
# Index types
# ---------------------------------------------------------------------------


@dataclass
class _DiseaseIndex:
    """In-memory match index for the existing Modern_Disease universe.

    Built once at run start. Mutated in place as backfills are queued —
    a tier-2/3 match early in the plan upgrades the index so later
    indications hit tier 1 directly.
    """
    by_efo_id: dict[str, str] = field(default_factory=dict)
    by_mesh_id: dict[str, str] = field(default_factory=dict)
    by_mondo_id: dict[str, str] = field(default_factory=dict)
    by_doid_id: dict[str, str] = field(default_factory=dict)
    by_norm_name: dict[str, str] = field(default_factory=dict)
    has_efo: set[str] = field(default_factory=set)
    has_mesh: set[str] = field(default_factory=set)
    has_mondo: set[str] = field(default_factory=set)
    has_doid: set[str] = field(default_factory=set)


@dataclass
class _MatchOutcome:
    disease_name: str | None
    match_tier: str
    backfill_efo: str = ""
    backfill_mesh: str = ""
    backfill_mondo: str = ""
    backfill_doid: str = ""


def _classify_curie(curie: str) -> tuple[str, str]:
    """Return (ontology_kind, normalized_curie).

    ontology_kind ∈ {"efo", "mondo", "doid", "other", ""}.
    normalized_curie uses ":" as the separator (EFO_0001359 → EFO:0001359).
    Empty input returns ("", "").
    """
    if not curie:
        return "", ""
    norm = curie.replace("_", ":") if "_" in curie else curie
    upper = norm.upper()
    if upper.startswith("EFO:"):
        return "efo", norm
    if upper.startswith("MONDO:"):
        return "mondo", norm
    if upper.startswith("DOID:"):
        return "doid", norm
    return "other", norm


# ---------------------------------------------------------------------------
# Cypher write templates (UNWIND-based)
# ---------------------------------------------------------------------------


_DELETE_PRIOR_EDGES_BATCH = """
UNWIND $rows AS r
MATCH (t:Biological_Target {target_chembl_id: r.target_chembl_id})-[k:RELATES_TO]->()
DELETE k
"""


_BACKFILL_DISEASE_BATCH = """
UNWIND $rows AS r
MATCH (d:Modern_Disease {name: r.name})
SET d.efo_id   = CASE WHEN d.efo_id   IS NULL OR d.efo_id   = ''
                 THEN r.efo_id   ELSE d.efo_id   END,
    d.mesh_id  = CASE WHEN d.mesh_id  IS NULL OR d.mesh_id  = ''
                 THEN r.mesh_id  ELSE d.mesh_id  END,
    d.mondo_id = CASE WHEN d.mondo_id IS NULL OR d.mondo_id = ''
                 THEN r.mondo_id ELSE d.mondo_id END,
    d.doid_id  = CASE WHEN d.doid_id  IS NULL OR d.doid_id  = ''
                 THEN r.doid_id  ELSE d.doid_id  END
"""


_MERGE_EDGES_BATCH = """
UNWIND $rows AS r
MATCH (t:Biological_Target {target_chembl_id: r.target_chembl_id})
MATCH (d:Modern_Disease {name: r.disease_name})
MERGE (t)-[k:RELATES_TO]->(d)
SET k += r.props
"""


_UPDATE_STATUS_BATCH = """
UNWIND $rows AS r
MATCH (t:Biological_Target {target_chembl_id: r.target_chembl_id})
SET t.td_linker_status = r.status,
    t.td_linker_attempted_at = r.attempted_at,
    t.td_linker_association_count = r.association_count,
    t.td_linker_dropped_count = r.dropped_count,
    t.td_linker_min_score = r.min_score
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TargetDiseaseLinker(BaseAgent):
    """Batched RELATES_TO linker."""

    def __init__(
        self,
        client: GraphClient,
        *,
        min_score: float = _DEFAULT_MIN_OT_SCORE,
        write_batch_size: int = _DEFAULT_WRITE_BATCH_SIZE,
        ontology_client: OntologyClient | None = None,
        gemini_model: str | None = None,
        subunit_cap: int = _SUBUNIT_FAN_OUT_CAP,
        skip_llm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._min_score = min_score
        self._write_batch_size = write_batch_size
        self._ontology = ontology_client or OntologyClient()
        self._subunit_cap = subunit_cap
        # When True, ORGANISM targets that get no EFO/DOID hits skip the
        # LLM safety net entirely. They land with status=no_associations
        # (which --retry-misses re-processes), letting a subsequent run
        # do the LLM phase alone after parallelization. Two-phase pattern:
        # fast OT-only pass first, slow LLM-only pass second.
        self._skip_llm = skip_llm
        self._gemini = make_gemini_client() if not skip_llm else None
        self._model = gemini_model or GEMINI_MODEL
        self._mesh_synonym_cache: dict[str, list[str]] = {}
        self._pathogen_llm_cache: dict[str, list[dict]] = {}

    @property
    def name(self) -> str:
        return "TargetDiseaseLinker"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        dry_run: bool = True,
        rebuild: bool = False,
        retry_misses: bool = False,
        force_relink: bool = False,
        limit: int | None = None,
        **_: Any,
    ) -> dict:
        if rebuild and not dry_run:
            self._wipe_existing()

        cypher = queries.LINKABLE_TARGETS_FOR_DISEASES
        if force_relink:
            cypher = queries.LINKABLE_TARGETS_FOR_DISEASES_FORCE
        elif retry_misses:
            cypher = queries.LINKABLE_TARGETS_FOR_DISEASES_RETRY

        targets = self.client.run(cypher)
        if limit is not None:
            targets = targets[:limit]

        index = self._load_disease_index()
        total = len(targets)

        # Bucket targets by type for the four planning branches
        single_protein = [t for t in targets if t["target_type"] == "SINGLE PROTEIN"]
        complexes      = [t for t in targets if t["target_type"] == "PROTEIN COMPLEX"]
        families       = [t for t in targets if t["target_type"] == "PROTEIN FAMILY"]
        organisms      = [t for t in targets if t["target_type"] == "ORGANISM"]
        unknown_type   = [t for t in targets if t["target_type"] not in
                          ("SINGLE PROTEIN", "PROTEIN COMPLEX", "PROTEIN FAMILY", "ORGANISM")]

        self._log_progress(
            f"Batched linking {total} target(s) — "
            f"{len(single_protein)} SINGLE PROTEIN, {len(complexes)} PROTEIN COMPLEX, "
            f"{len(families)} PROTEIN FAMILY (skip), {len(organisms)} ORGANISM, "
            f"{len(unknown_type)} unknown type. "
            f"(dry_run={dry_run}, rebuild={rebuild}, "
            f"retry_misses={retry_misses}, force_relink={force_relink}, "
            f"min_score={self._min_score})"
        )
        self._log_progress(
            f"  Modern_Disease universe: {len(index.by_norm_name)} nodes "
            f"({len(index.by_efo_id)} with efo_id, "
            f"{len(index.by_mesh_id)} with mesh_id)"
        )

        t_start = time.time()

        # ----- PHASE 1: bulk OT fetch for SINGLE PROTEIN ---------------
        t_sp = time.time()
        single_protein_results = self._fetch_single_protein_associations(
            single_protein,
        )
        sp_duration = round(time.time() - t_sp, 2)
        self._log_progress(f"  OT bulk SINGLE PROTEIN: {sp_duration}s")

        # ----- PHASE 2: bulk OT fetch for PROTEIN COMPLEX --------------
        t_cx = time.time()
        complex_results = self._fetch_complex_associations(complexes)
        cx_duration = round(time.time() - t_cx, 2)
        self._log_progress(f"  OT bulk PROTEIN COMPLEX: {cx_duration}s")

        # ----- PHASE 3: ORGANISM lookup (EFO + DOID + LLM) -------------
        t_org = time.time()
        organism_results = self._lookup_organisms(organisms)
        org_duration = round(time.time() - t_org, 2)
        self._log_progress(f"  ORGANISM EFO+DOID+LLM: {org_duration}s")

        # ----- PHASE 4: build write plan -------------------------------
        t_plan = time.time()
        plan = self._build_plan(
            single_protein, complexes, families, organisms, unknown_type,
            single_protein_results, complex_results, organism_results,
            index,
        )
        plan_duration = round(time.time() - t_plan, 2)
        self._log_progress(
            f"  Plan: {plan_duration}s — {len(plan['edge_rows'])} edges, "
            f"{len(plan['backfill_rows'])} backfills, "
            f"{len(plan['status_rows'])} status writes"
        )

        # ----- PHASE 5: bulk write -------------------------------------
        t_write = time.time()
        writes_done = 0
        if not dry_run:
            if force_relink and plan["delete_rows"]:
                writes_done += self.client.batched_unwind_write(
                    _DELETE_PRIOR_EDGES_BATCH,
                    plan["delete_rows"],
                    batch_size=self._write_batch_size,
                )
            if plan["backfill_rows"]:
                writes_done += self.client.batched_unwind_write(
                    _BACKFILL_DISEASE_BATCH,
                    plan["backfill_rows"],
                    batch_size=self._write_batch_size,
                )
            if plan["edge_rows"]:
                writes_done += self.client.batched_unwind_write(
                    _MERGE_EDGES_BATCH,
                    plan["edge_rows"],
                    batch_size=self._write_batch_size,
                )
            if plan["status_rows"]:
                writes_done += self.client.batched_unwind_write(
                    _UPDATE_STATUS_BATCH,
                    plan["status_rows"],
                    batch_size=self._write_batch_size,
                )
        write_duration = round(time.time() - t_write, 2)
        self._log_progress(
            f"  Neo4j bulk write: {write_duration}s ({writes_done} rows applied)"
        )

        duration_s = round(time.time() - t_start, 2)

        return {
            "dry_run": dry_run,
            "rebuild": rebuild,
            "min_score": self._min_score,
            "write_batch_size": self._write_batch_size,
            "targets_total": total,
            "modern_disease_universe": len(index.by_norm_name),
            "by_target_type_input": {
                "SINGLE PROTEIN":  len(single_protein),
                "PROTEIN COMPLEX": len(complexes),
                "PROTEIN FAMILY":  len(families),
                "ORGANISM":        len(organisms),
                "UNKNOWN":         len(unknown_type),
            },
            "by_status": plan["by_status"],
            "by_evidence_type": plan["by_evidence_type"],
            "by_match_tier": dict(
                sorted(plan["by_match_tier"].items(), key=lambda kv: -kv[1])
            ),
            "by_pathogen_lookup_source": plan["by_pathogen_lookup_source"],
            "edge_writes": len(plan["edge_rows"]),
            "unique_diseases_linked": len(plan["unique_diseases"]),
            "associations_total": plan["associations_total"],
            "associations_matched": plan["associations_matched"],
            "associations_dropped_no_match": plan["associations_dropped_no_match"],
            "associations_dropped_score": plan["associations_dropped_score"],
            "backfills_efo_id": plan["backfills_efo_id"],
            "backfills_mesh_id": plan["backfills_mesh_id"],
            "backfills_mondo_id": plan["backfills_mondo_id"],
            "backfills_doid_id": plan["backfills_doid_id"],
            "phase_durations_s": {
                "ot_single_protein": sp_duration,
                "ot_complex":        cx_duration,
                "organism_lookup":   org_duration,
                "plan":              plan_duration,
                "neo4j_bulk_write":  write_duration,
            },
            "duration_s": duration_s,
            "errors": plan["errors"],
        }

    # ------------------------------------------------------------------
    # Disease universe loading
    # ------------------------------------------------------------------

    def _load_disease_index(self) -> _DiseaseIndex:
        rows = self.client.run(queries.ALL_MODERN_DISEASES)
        index = _DiseaseIndex()
        for r in rows:
            name = r.get("name") or ""
            if not name:
                continue
            for prop, dest_dict, dest_set in (
                ("efo_id",   index.by_efo_id,   index.has_efo),
                ("mesh_id",  index.by_mesh_id,  index.has_mesh),
                ("mondo_id", index.by_mondo_id, index.has_mondo),
                ("doid_id",  index.by_doid_id,  index.has_doid),
            ):
                value = (r.get(prop) or "").strip()
                if value:
                    # Normalize to canonical CURIE form (EFO:0001359)
                    norm_value = (
                        value.replace("_", ":") if "_" in value else value
                    )
                    dest_dict[norm_value] = name
                    dest_set.add(name)
            norm = _normalize_for_match(name)
            if norm and norm not in index.by_norm_name:
                index.by_norm_name[norm] = name
        return index

    # ------------------------------------------------------------------
    # Per-target-type bulk fetch
    # ------------------------------------------------------------------

    def _fetch_single_protein_associations(
        self, targets: list[dict],
    ) -> dict[str, open_targets.OtTargetResult]:
        """Bulk-fetch OT associations for SINGLE PROTEIN targets that
        carry a uniprot_id."""
        uniprot_to_target_id = {
            t["target_chembl_id"]: t["uniprot_id"]
            for t in targets
            if t.get("uniprot_id")
        }
        if not uniprot_to_target_id:
            return {}
        return open_targets.fetch_target_associations_batch(
            uniprot_to_target_id, min_score=self._min_score,
        )

    def _fetch_complex_associations(
        self, targets: list[dict],
    ) -> dict[str, dict]:
        """For PROTEIN COMPLEX targets, fetch each subunit's UniProt
        from ChEMBL and bulk-resolve associations.

        Returns {complex_target_chembl_id: {
            "components_ok": bool,
            "subunit_results": {subunit_uniprot: OtTargetResult},
        }}.

        `components_ok=False` means ChEMBL's bulk subunit lookup hit a
        transient failure for THIS complex (we don't have a complete
        subunit list, so any "linked" decision would be based on
        partial coverage). The planner treats those complexes as
        `error` rather than `linked` — same blast-radius logic as
        the OT subunit transient case.
        """
        if not targets:
            return {}
        complex_chembl_ids = [t["target_chembl_id"] for t in targets]
        complex_subunits, components_ok = chembl.bulk_fetch_target_components(
            complex_chembl_ids,
        )
        if not components_ok:
            logger.warning(
                "Bulk subunit fetch had transient failures — affected "
                "complexes will be marked error instead of linked"
            )

        # Flatten to a uniprot → "complex_id::uniprot" map for OT batch
        # (we use a synthetic key so we can unmap back per complex)
        per_complex_uniprots: dict[str, list[str]] = {}
        all_unique_uniprots: set[str] = set()
        for cx_id in complex_chembl_ids:
            uniprots = complex_subunits.get(cx_id, [])[: self._subunit_cap]
            per_complex_uniprots[cx_id] = uniprots
            all_unique_uniprots.update(uniprots)

        if not all_unique_uniprots:
            return {
                cx_id: {"components_ok": components_ok, "subunit_results": {}}
                for cx_id in complex_chembl_ids
            }

        # Single OT batch: resolve every unique subunit uniprot, fetch
        # associations once. Then re-fan-out per complex.
        uniprot_to_synthetic = {u: u for u in all_unique_uniprots}
        results = open_targets.fetch_target_associations_batch(
            uniprot_to_synthetic, min_score=self._min_score,
        )

        out: dict[str, dict] = {}
        for cx_id in complex_chembl_ids:
            out[cx_id] = {
                "components_ok": components_ok,
                "subunit_results": {
                    u: results.get(u, open_targets.OtTargetResult(
                        origin_uniprot_id=u, outcome="no_target",
                    ))
                    for u in per_complex_uniprots[cx_id]
                },
            }
        return out

    def _lookup_organisms(
        self, targets: list[dict],
    ) -> dict[str, dict[str, list[dict]]]:
        """For ORGANISM targets, lookup pathogen → disease via EFO,
        DOID, then LLM safety net.

        Returns {target_chembl_id: {"sources": {"efo": [...], "doid": [...],
                                                "llm": [...]},
                                    "tax_id": "5833", "name": "P. falciparum"}}.
        """
        out: dict[str, dict[str, list[dict]]] = {}

        # Phase A: collect unique tax_ids and parallel EFO + DOID lookups
        unique_tax_ids = sorted({
            t.get("ncbi_tax_id") or "" for t in targets if t.get("ncbi_tax_id")
        })
        efo_results: dict[str, list[dict]] = {}
        doid_results: dict[str, list[dict]] = {}

        def _efo_one(tid: str) -> tuple[str, list[dict]]:
            return tid, self._ontology.get_pathogen_diseases_efo(tid)

        def _doid_one(tid: str) -> tuple[str, list[dict]]:
            return tid, self._ontology.get_pathogen_diseases_doid(tid)

        if unique_tax_ids:
            with ThreadPoolExecutor(max_workers=_SYNONYM_PREFETCH_WORKERS) as pool:
                for tid, results in pool.map(_efo_one, unique_tax_ids):
                    efo_results[tid] = results
                for tid, results in pool.map(_doid_one, unique_tax_ids):
                    doid_results[tid] = results

        # Phase B: parallel LLM prefetch for tax_ids needing the safety net.
        # Without this pre-pass, the per-target loop below would serialize
        # ~500 LLM calls — the v1 bottleneck. Now they happen 4-way parallel
        # before the assembly loop, which then sees only cache hits.
        if not self._skip_llm:
            llm_candidates: list[tuple[str, str]] = []
            for t in targets:
                tax_id = t.get("ncbi_tax_id") or ""
                if not tax_id:
                    continue
                if not efo_results.get(tax_id) and not doid_results.get(tax_id):
                    name = t.get("name") or t.get("target_pref_name") or ""
                    llm_candidates.append((tax_id, name))
            self._prefetch_llm_pathogens(llm_candidates)

        # Phase C: per-target assembly (now cache-hit-only for LLM)
        for t in targets:
            tax_id = t.get("ncbi_tax_id") or ""
            tcid = t["target_chembl_id"]
            entry: dict[str, Any] = {
                "tax_id": tax_id,
                "name": t.get("name") or t.get("target_pref_name") or "",
                "sources": {"efo": [], "doid": [], "llm": []},
            }
            if not tax_id:
                out[tcid] = entry
                continue
            entry["sources"]["efo"] = efo_results.get(tax_id, [])
            entry["sources"]["doid"] = doid_results.get(tax_id, [])
            if (
                not self._skip_llm
                and not entry["sources"]["efo"]
                and not entry["sources"]["doid"]
            ):
                # Cache hit (prefetched in Phase B above)
                entry["sources"]["llm"] = self._llm_pathogen_lookup_cached(
                    tax_id, entry["name"],
                )
            out[tcid] = entry
        return out

    # ------------------------------------------------------------------
    # LLM safety net (generate-then-verify)
    # ------------------------------------------------------------------

    def _llm_pathogen_lookup_cached(
        self, tax_id: str, organism_name: str,
    ) -> list[dict]:
        """Cached LLM call for pathogen → disease names + verification.

        After `_prefetch_llm_pathogens` runs, this is a cache hit; the
        underlying LLM call only fires for tax_ids not in the prefetch
        set (e.g., when called outside the batched _lookup_organisms
        flow)."""
        if tax_id in self._pathogen_llm_cache:
            return self._pathogen_llm_cache[tax_id]
        try:
            candidates = self._llm_pathogen_lookup(tax_id, organism_name)
        except Exception as e:
            logger.warning(
                "LLM pathogen lookup failed for tax_id %s (%s): %s",
                tax_id, organism_name, e,
            )
            candidates = []
        self._pathogen_llm_cache[tax_id] = candidates
        return candidates

    def _prefetch_llm_pathogens(
        self, candidates: list[tuple[str, str]],
    ) -> None:
        """Warm the LLM cache in parallel for tax_ids needing the safety net.

        Each (tax_id, organism_name) tuple becomes one parallel Gemini call.
        Cache is populated as results return; subsequent reads via
        `_llm_pathogen_lookup_cached` are O(1) cache hits.

        Without this prefetch, the v1 ORGANISM phase serialized ~500
        unique tax_id LLM calls — the dominant cost (~21 min). Parallel
        prefetch with 4 workers cuts this to ~3 min.
        """
        if not candidates:
            return
        # Dedup by tax_id (keep first organism_name we see) and skip
        # anything already cached (e.g., from a prior --retry-misses run).
        unique: dict[str, str] = {}
        for tid, name in candidates:
            if tid and tid not in self._pathogen_llm_cache and tid not in unique:
                unique[tid] = name or ""
        if not unique:
            return

        def _one(args: tuple[str, str]) -> tuple[str, list[dict]]:
            tid, name = args
            try:
                return tid, self._llm_pathogen_lookup(tid, name)
            except Exception as e:
                logger.warning(
                    "LLM pathogen prefetch failed for tax_id %s (%s): %s",
                    tid, name, e,
                )
                return tid, []

        with ThreadPoolExecutor(max_workers=_LLM_PREFETCH_WORKERS) as pool:
            for tid, result in pool.map(_one, unique.items()):
                self._pathogen_llm_cache[tid] = result

    def _llm_pathogen_lookup(
        self, tax_id: str, organism_name: str,
    ) -> list[dict]:
        """LLM-generate candidate disease names, then verify each via
        OntologyClient (NLM ICD-10/MeSH/SNOMED).

        Thread-safe — multiple calls can run in parallel via the
        prefetch path. The previous artificial 0.5s rate-limit sleep
        was removed because (a) it was the single biggest bottleneck
        in the v1 ORGANISM phase (~21 min for ~500 unique tax_ids),
        and (b) Vertex AI tolerates the throughput we'd hit (4 workers
        × ~1.4s/call = ~3 qps, well below project limits). The 429
        backoff below still handles real rate-limit responses.
        """
        from google.genai import types as genai_types

        prompt = (
            f"NCBI taxonomy ID: {tax_id}\n"
            f"Organism name: {organism_name or '(unknown)'}\n\n"
            "What modern Western disease(s) does this organism cause in humans?"
        )

        last_err: Exception | None = None
        for attempt in range(_LLM_MAX_RETRIES):
            try:
                response = self._gemini.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=_PATHOGEN_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=_PATHOGEN_SCHEMA,
                    ),
                )
                payload = json.loads(response.text)
                proposed = payload.get("diseases") or []
                break
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < _LLM_MAX_RETRIES - 1:
                    time.sleep(5 * (2 ** attempt))
                    continue
                raise
        else:
            if last_err:
                raise last_err
            return []

        # Verify each LLM-emitted name through NLM ontologies
        verified: list[dict] = []
        for cand in proposed:
            name = (cand.get("name") or "").strip()
            hint = cand.get("ontology_hint") or "icd10"
            if not name:
                continue
            v = self._verify_pathogen_disease(name, hint)
            if v is None:
                continue
            verified.append({
                "name": v["name"],
                "mesh_id": v.get("mesh_id", ""),
                "icd10_code": v.get("icd10_code", ""),
                "snomed_id": v.get("snomed_id", ""),
                "rationale": cand.get("rationale", ""),
                "source": "llm",
            })
        return verified

    def _verify_pathogen_disease(
        self, name: str, hint: str,
    ) -> dict | None:
        """Tolerant exact-match verification across NLM ICD-10/MeSH/SNOMED."""
        norm_target = _normalize_for_match(name)
        if not norm_target:
            return None
        per_db = {
            "icd10":  self._safe_search("icd10", name),
            "mesh":   self._safe_search("mesh", name),
            "snomed": self._safe_search("snomed", name),
        }

        def _matches(c: dict) -> bool:
            return _normalize_for_match(c.get("name", "")) == norm_target

        # Hinted DB first
        for c in per_db.get(hint, []):
            if _matches(c):
                return c
        # Fall through priority: icd10 → mesh → snomed
        for db in ("icd10", "mesh", "snomed"):
            for c in per_db.get(db, []):
                if _matches(c):
                    return c
        return None

    def _safe_search(self, db: str, name: str) -> list[dict]:
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

    # ------------------------------------------------------------------
    # Plan builder
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        single_protein: list[dict], complexes: list[dict],
        families: list[dict], organisms: list[dict], unknown: list[dict],
        sp_results: dict[str, open_targets.OtTargetResult],
        cx_results: dict[str, dict[str, open_targets.OtTargetResult]],
        org_results: dict[str, dict[str, list[dict]]],
        index: _DiseaseIndex,
    ) -> dict:
        by_status: dict[str, int] = {s.value: 0 for s in LinkStatus}
        all_evidence_keys = list(CONFIDENCE_OT) + list(CONFIDENCE_PATHOGEN)
        by_evidence_type: dict[str, int] = {k: 0 for k in all_evidence_keys}
        by_match_tier: dict[str, int] = {}
        by_pathogen_source: dict[str, int] = {}
        unique_diseases: set[str] = set()
        associations_total = 0
        associations_matched = 0
        associations_dropped_no_match = 0
        associations_dropped_score = 0
        backfills_efo = 0
        backfills_mesh = 0
        errors: list[str] = []

        delete_rows: list[dict] = []
        backfill_rows: list[dict] = []
        edge_rows: list[dict] = []
        status_rows: list[dict] = []

        backfilled_efos: dict[str, str] = {}
        backfilled_meshes: dict[str, str] = {}
        backfilled_mondos: dict[str, str] = {}
        backfilled_doids: dict[str, str] = {}

        # Mutable accumulators threaded into each planner branch so each
        # branch can record its own contribution to the run-level totals.
        # Replaces the v1 pattern of re-deriving from status_rows[*]
        # .dropped_count, which was always 0 because no branch wrote it.
        accum: dict[str, int] = {
            "associations_total": 0,
            "associations_matched": 0,
            "associations_dropped_score": 0,
            "associations_dropped_no_match": 0,
        }

        attempted_at = dt.datetime.utcnow().isoformat()

        # --- SINGLE PROTEIN branch ---
        for t in single_protein:
            tcid = t["target_chembl_id"]
            uniprot = t.get("uniprot_id") or ""
            if not uniprot:
                by_status[LinkStatus.SKIPPED_NO_UNIPROT.value] += 1
                status_rows.append(self._status_row(
                    tcid, LinkStatus.SKIPPED_NO_UNIPROT, 0, 0, attempted_at,
                ))
                continue
            result = sp_results.get(tcid)
            self._plan_protein_target(
                tcid, result, index,
                edge_rows=edge_rows, status_rows=status_rows,
                delete_rows=delete_rows,
                backfilled_efos=backfilled_efos, backfilled_meshes=backfilled_meshes,
                backfilled_mondos=backfilled_mondos, backfilled_doids=backfilled_doids,
                by_status=by_status, by_evidence_type=by_evidence_type,
                by_match_tier=by_match_tier, unique_diseases=unique_diseases,
                errors=errors, attempted_at=attempted_at,
                accum=accum,
                ot_subunit_uniprot="",
            )

        # --- PROTEIN COMPLEX branch ---
        for t in complexes:
            tcid = t["target_chembl_id"]
            cx_entry = cx_results.get(tcid) or {}
            subunit_results = cx_entry.get("subunit_results", {})
            components_ok = cx_entry.get("components_ok", True)
            # If ChEMBL component fetch failed for this complex, we
            # don't have a complete subunit list — mark as error so
            # --retry-misses re-fetches when ChEMBL is healthy. Without
            # this, partial subunit coverage would be silently treated
            # as "skipped_no_subunits" (a permanent miss).
            if not components_ok:
                by_status[LinkStatus.ERROR.value] += 1
                errors.append(
                    f"{tcid}: transient during ChEMBL component fetch"
                )
                status_rows.append(self._status_row(
                    tcid, LinkStatus.ERROR, 0, 0, attempted_at,
                ))
                continue
            if not subunit_results:
                by_status[LinkStatus.SKIPPED_NO_SUBUNITS.value] += 1
                status_rows.append(self._status_row(
                    tcid, LinkStatus.SKIPPED_NO_SUBUNITS, 0, 0, attempted_at,
                ))
                continue
            self._plan_complex_target(
                tcid, subunit_results, index,
                edge_rows=edge_rows, status_rows=status_rows,
                delete_rows=delete_rows,
                backfilled_efos=backfilled_efos, backfilled_meshes=backfilled_meshes,
                backfilled_mondos=backfilled_mondos, backfilled_doids=backfilled_doids,
                by_status=by_status, by_evidence_type=by_evidence_type,
                by_match_tier=by_match_tier, unique_diseases=unique_diseases,
                errors=errors, attempted_at=attempted_at,
                accum=accum,
            )

        # --- PROTEIN FAMILY branch (skip) ---
        for t in families:
            tcid = t["target_chembl_id"]
            by_status[LinkStatus.SKIPPED_PROTEIN_FAMILY.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.SKIPPED_PROTEIN_FAMILY, 0, 0, attempted_at,
            ))

        # --- ORGANISM branch ---
        for t in organisms:
            tcid = t["target_chembl_id"]
            entry = org_results.get(tcid) or {"sources": {}, "tax_id": "", "name": ""}
            self._plan_organism_target(
                tcid, entry, index,
                edge_rows=edge_rows, status_rows=status_rows,
                delete_rows=delete_rows,
                backfilled_efos=backfilled_efos, backfilled_meshes=backfilled_meshes,
                backfilled_mondos=backfilled_mondos, backfilled_doids=backfilled_doids,
                by_status=by_status, by_evidence_type=by_evidence_type,
                by_match_tier=by_match_tier, by_pathogen_source=by_pathogen_source,
                unique_diseases=unique_diseases,
                attempted_at=attempted_at,
                accum=accum,
            )

        # --- Unknown type ---
        for t in unknown:
            tcid = t["target_chembl_id"]
            by_status[LinkStatus.SKIPPED_UNKNOWN_TYPE.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.SKIPPED_UNKNOWN_TYPE, 0, 0, attempted_at,
            ))

        # Accumulators were updated by each branch as it processed its
        # targets. Authoritative — no re-derivation from status rows.
        associations_total = accum["associations_total"]
        associations_matched = accum["associations_matched"]
        associations_dropped_score = accum["associations_dropped_score"]
        associations_dropped_no_match = accum["associations_dropped_no_match"]

        # Build collapsed backfill rows. One row per disease that needs
        # ANY backfill; the UNWIND template handles all four ID kinds.
        all_backfill_targets = (
            set(backfilled_efos) | set(backfilled_meshes)
            | set(backfilled_mondos) | set(backfilled_doids)
        )
        for disease_name in all_backfill_targets:
            backfill_rows.append({
                "name":     disease_name,
                "efo_id":   backfilled_efos.get(disease_name, ""),
                "mesh_id":  backfilled_meshes.get(disease_name, ""),
                "mondo_id": backfilled_mondos.get(disease_name, ""),
                "doid_id":  backfilled_doids.get(disease_name, ""),
            })
        backfills_efo = len(backfilled_efos)
        backfills_mesh = len(backfilled_meshes)
        backfills_mondo = len(backfilled_mondos)
        backfills_doid = len(backfilled_doids)

        return {
            "by_status": by_status,
            "by_evidence_type": by_evidence_type,
            "by_match_tier": by_match_tier,
            "by_pathogen_lookup_source": by_pathogen_source,
            "unique_diseases": unique_diseases,
            "associations_total": associations_total,
            "associations_matched": associations_matched,
            "associations_dropped_no_match": associations_dropped_no_match,
            "associations_dropped_score": associations_dropped_score,
            "backfills_efo_id": backfills_efo,
            "backfills_mesh_id": backfills_mesh,
            "backfills_mondo_id": backfills_mondo,
            "backfills_doid_id": backfills_doid,
            "errors": errors,
            "delete_rows": delete_rows,
            "backfill_rows": backfill_rows,
            "edge_rows": edge_rows,
            "status_rows": status_rows,
        }

    def _plan_protein_target(
        self, tcid: str, result: open_targets.OtTargetResult | None,
        index: _DiseaseIndex,
        *,
        edge_rows: list[dict], status_rows: list[dict], delete_rows: list[dict],
        backfilled_efos: dict, backfilled_meshes: dict,
        backfilled_mondos: dict, backfilled_doids: dict,
        by_status: dict, by_evidence_type: dict, by_match_tier: dict,
        unique_diseases: set, errors: list, attempted_at: str,
        accum: dict,
        ot_subunit_uniprot: str,
        evidence_type_override: str | None = None,
        ot_score_override: float | None = None,
    ) -> int:
        """Plan a single protein-side target. Returns associations_total
        contribution. Used by both SINGLE PROTEIN and PROTEIN COMPLEX
        (which dispatches per-subunit then aggregates max-score).

        Updates `accum` with score/no-match drop counts in addition to
        the linked-edge contribution."""
        if result is None:
            by_status[LinkStatus.NO_ASSOCIATIONS.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.NO_ASSOCIATIONS, 0, 0, attempted_at,
            ))
            return 0

        if result.outcome == "transient_failure":
            by_status[LinkStatus.ERROR.value] += 1
            if result.error:
                errors.append(f"{tcid}: {result.error}")
            status_rows.append(self._status_row(
                tcid, LinkStatus.ERROR, 0, 0, attempted_at,
            ))
            # Defensive: do NOT add to delete_rows on transient — same
            # protection pattern as compound_disease_batched. A bulk OT
            # outage would otherwise wipe every existing RELATES_TO.
            return 0

        if result.outcome == "no_target" or not result.associations:
            by_status[LinkStatus.NO_ASSOCIATIONS.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.NO_ASSOCIATIONS, 0, 0, attempted_at,
            ))
            delete_rows.append({"target_chembl_id": tcid})
            return 0

        # Per-association: tier + match + queue. Track all four bins.
        score_drops = 0
        no_match_drops = 0
        kept_per_disease: dict[str, tuple[str, open_targets.OtAssociation, _MatchOutcome]] = {}
        for assoc in result.associations:
            accum["associations_total"] += 1
            if assoc.score < self._min_score:
                score_drops += 1
                continue
            et = evidence_type_override or _ot_evidence_type(assoc.score)
            if et is None:
                # Below min-score threshold via the tier mapping path.
                # Counts as a score-drop for telemetry consistency.
                score_drops += 1
                continue
            outcome = self._match_disease(
                assoc.disease_id, assoc.disease_name, index,
            )
            by_match_tier[outcome.match_tier] = (
                by_match_tier.get(outcome.match_tier, 0) + 1
            )
            if outcome.disease_name is None:
                no_match_drops += 1
                continue
            key = outcome.disease_name
            if (
                key not in kept_per_disease
                or assoc.score > kept_per_disease[key][1].score
            ):
                kept_per_disease[key] = (et, assoc, outcome)

        accum["associations_dropped_score"] += score_drops
        accum["associations_dropped_no_match"] += no_match_drops

        if not kept_per_disease:
            by_status[LinkStatus.NO_ASSOCIATIONS.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.NO_ASSOCIATIONS,
                0, score_drops + no_match_drops, attempted_at,
            ))
            delete_rows.append({"target_chembl_id": tcid})
            return 0

        by_status[LinkStatus.LINKED.value] += 1
        delete_rows.append({"target_chembl_id": tcid})
        accum["associations_matched"] += len(kept_per_disease)
        for et, assoc, outcome in kept_per_disease.values():
            by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
            unique_diseases.add(outcome.disease_name or "")
            self._enqueue_backfills(
                outcome, index, backfilled_efos, backfilled_meshes,
                backfilled_mondos, backfilled_doids,
            )
            edge_rows.append({
                "target_chembl_id": tcid,
                "disease_name": outcome.disease_name,
                "props": self._protein_edge_props(
                    et, assoc, ot_subunit_uniprot, attempted_at,
                ),
            })
        status_rows.append(self._status_row(
            tcid, LinkStatus.LINKED,
            len(kept_per_disease), score_drops + no_match_drops, attempted_at,
        ))
        return len(result.associations)

    def _plan_complex_target(
        self, tcid: str,
        subunit_results: dict[str, open_targets.OtTargetResult],
        index: _DiseaseIndex,
        *,
        edge_rows: list[dict], status_rows: list[dict], delete_rows: list[dict],
        backfilled_efos: dict, backfilled_meshes: dict,
        backfilled_mondos: dict, backfilled_doids: dict,
        by_status: dict, by_evidence_type: dict, by_match_tier: dict,
        unique_diseases: set, errors: list, attempted_at: str,
        accum: dict,
    ) -> None:
        """Subunit fan-out with per-disease max-score dedup. Writes ONE
        RELATES_TO per (complex, disease) using the highest-scoring
        subunit's evidence."""
        # Walk subunits; flag transient. ANY transient → mark complex
        # as error (even if other subunits returned ok). The complex's
        # association profile would be incomplete — better to retry
        # next time than to write partial coverage as if it were
        # complete + overwrite prior edges under --force-relink.
        per_disease_best: dict[str, tuple[open_targets.OtAssociation, str]] = {}
        any_transient = False
        score_drops = 0
        no_match_drops = 0
        for uniprot, res in subunit_results.items():
            if res.outcome == "transient_failure":
                any_transient = True
                continue
            if res.outcome != "ok":
                continue
            for assoc in res.associations:
                accum["associations_total"] += 1
                if assoc.score < self._min_score:
                    score_drops += 1
                    continue
                outcome = self._match_disease(
                    assoc.disease_id, assoc.disease_name, index,
                )
                by_match_tier[outcome.match_tier] = (
                    by_match_tier.get(outcome.match_tier, 0) + 1
                )
                if outcome.disease_name is None:
                    no_match_drops += 1
                    continue
                cur = per_disease_best.get(outcome.disease_name)
                if cur is None or assoc.score > cur[0].score:
                    per_disease_best[outcome.disease_name] = (assoc, uniprot)
        accum["associations_dropped_score"] += score_drops
        accum["associations_dropped_no_match"] += no_match_drops

        if any_transient:
            # Whether or not we have partial results, we don't have the
            # full picture. Mark error, skip writes for this target.
            # Defensive: do NOT add to delete_rows (matches the
            # transient-cascade protection in compound_disease_batched).
            by_status[LinkStatus.ERROR.value] += 1
            errors.append(f"{tcid}: at least one subunit transient")
            status_rows.append(self._status_row(
                tcid, LinkStatus.ERROR, 0, 0, attempted_at,
            ))
            return

        if not per_disease_best:
            by_status[LinkStatus.NO_ASSOCIATIONS.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.NO_ASSOCIATIONS, 0, 0, attempted_at,
            ))
            delete_rows.append({"target_chembl_id": tcid})
            return

        by_status[LinkStatus.LINKED.value] += 1
        delete_rows.append({"target_chembl_id": tcid})
        accum["associations_matched"] += len(per_disease_best)
        for disease_name, (assoc, top_subunit) in per_disease_best.items():
            et = "ot_association_complex_aggregate"
            by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
            unique_diseases.add(disease_name)
            # Re-run match to get the outcome (it succeeded above)
            outcome = self._match_disease(
                assoc.disease_id, assoc.disease_name, index,
            )
            self._enqueue_backfills(
                outcome, index, backfilled_efos, backfilled_meshes,
                backfilled_mondos, backfilled_doids,
            )
            edge_rows.append({
                "target_chembl_id": tcid,
                "disease_name": disease_name,
                "props": self._protein_edge_props(
                    et, assoc, top_subunit, attempted_at,
                ),
            })
        status_rows.append(self._status_row(
            tcid, LinkStatus.LINKED, len(per_disease_best),
            score_drops + no_match_drops, attempted_at,
        ))

    def _plan_organism_target(
        self, tcid: str, entry: dict, index: _DiseaseIndex,
        *,
        edge_rows: list[dict], status_rows: list[dict], delete_rows: list[dict],
        backfilled_efos: dict, backfilled_meshes: dict,
        backfilled_mondos: dict, backfilled_doids: dict,
        by_status: dict, by_evidence_type: dict, by_match_tier: dict,
        by_pathogen_source: dict,
        unique_diseases: set, attempted_at: str,
        accum: dict,
    ) -> None:
        tax_id = entry.get("tax_id") or ""
        if not tax_id:
            by_status[LinkStatus.SKIPPED_NO_TAX_ID.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.SKIPPED_NO_TAX_ID, 0, 0, attempted_at,
            ))
            return

        sources = entry.get("sources") or {}
        efo_hits = sources.get("efo") or []
        doid_hits = sources.get("doid") or []
        llm_hits = sources.get("llm") or []

        # Aggregate: (disease_name, source_set, identifying_id, candidate)
        # Use the matching tier to land on existing Modern_Disease nodes.
        per_disease_best: dict[str, dict] = {}
        no_match_drops = 0

        def _consider(cand: dict, source: str) -> None:
            nonlocal no_match_drops
            accum["associations_total"] += 1
            disease_id = cand.get("id") or ""
            outcome = self._match_disease(
                disease_id_curie=disease_id,
                disease_name=cand.get("name", ""),
                index=index,
                pathogen_lookup=True,
                pathogen_extra_ids=cand,
            )
            by_match_tier[outcome.match_tier] = (
                by_match_tier.get(outcome.match_tier, 0) + 1
            )
            if outcome.disease_name is None:
                no_match_drops += 1
                return
            cur = per_disease_best.get(outcome.disease_name)
            if cur is None:
                per_disease_best[outcome.disease_name] = {
                    "outcome": outcome,
                    "sources": {source},
                    "candidate": cand,
                }
            else:
                cur["sources"].add(source)

        for c in efo_hits:
            _consider(c, "efo")
        for c in doid_hits:
            _consider(c, "doid")
        for c in llm_hits:
            _consider(c, "llm")

        accum["associations_dropped_no_match"] += no_match_drops

        if not per_disease_best:
            by_status[LinkStatus.NO_ASSOCIATIONS.value] += 1
            status_rows.append(self._status_row(
                tcid, LinkStatus.NO_ASSOCIATIONS, 0, no_match_drops, attempted_at,
            ))
            delete_rows.append({"target_chembl_id": tcid})
            return

        by_status[LinkStatus.LINKED.value] += 1
        delete_rows.append({"target_chembl_id": tcid})
        accum["associations_matched"] += len(per_disease_best)
        for disease_name, info in per_disease_best.items():
            sources_set = info["sources"]
            if "efo" in sources_set and "doid" in sources_set:
                et = "ncbi_pathogen_consensus"
                source_label = "efo+doid"
            elif "llm" in sources_set:
                et = "ncbi_pathogen_llm_verified"
                source_label = "llm_verified"
            elif "efo" in sources_set:
                et = "ncbi_pathogen_efo"
                source_label = "efo"
            else:
                et = "ncbi_pathogen_doid"
                source_label = "doid"
            by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
            by_pathogen_source[source_label] = (
                by_pathogen_source.get(source_label, 0) + 1
            )
            unique_diseases.add(disease_name)
            self._enqueue_backfills(
                info["outcome"], index, backfilled_efos, backfilled_meshes,
                backfilled_mondos, backfilled_doids,
            )
            edge_rows.append({
                "target_chembl_id": tcid,
                "disease_name": disease_name,
                "props": self._organism_edge_props(
                    et, source_label, tax_id, info["candidate"], attempted_at,
                ),
            })
        status_rows.append(self._status_row(
            tcid, LinkStatus.LINKED,
            len(per_disease_best), no_match_drops, attempted_at,
        ))

    # ------------------------------------------------------------------
    # Disease matching (3-tier)
    # ------------------------------------------------------------------

    def _match_disease(
        self,
        disease_id_curie: str,
        disease_name: str,
        index: _DiseaseIndex,
        *,
        pathogen_lookup: bool = False,
        pathogen_extra_ids: dict | None = None,
    ) -> _MatchOutcome:
        """Tier 1/2/3 match a candidate against existing Modern_Disease.

        Critical: each ID type routes to its own index (efo_id, mondo_id,
        doid_id, mesh_id) and is backfilled to its OWN property on the
        matched node — we never stamp a MONDO ID into the efo_id field
        or vice versa.
        """
        kind, normalized_curie = _classify_curie(disease_id_curie)

        # Helper: build a backfill outcome with all relevant ID kinds set
        # from the inputs, gated on whether the matched node already has
        # them. Used by tier 2 + tier 3 paths.
        def _outcome(disease: str, match_tier: str) -> _MatchOutcome:
            backfill_efo = ""
            backfill_mondo = ""
            backfill_doid = ""
            backfill_mesh = ""
            if kind == "efo" and disease not in index.has_efo:
                backfill_efo = normalized_curie
            elif kind == "mondo" and disease not in index.has_mondo:
                backfill_mondo = normalized_curie
            elif kind == "doid" and disease not in index.has_doid:
                backfill_doid = normalized_curie
            if pathogen_extra_ids:
                mid = (pathogen_extra_ids.get("mesh_id") or "").strip()
                if mid and disease not in index.has_mesh:
                    backfill_mesh = mid
            return _MatchOutcome(
                disease_name=disease,
                match_tier=match_tier,
                backfill_efo=backfill_efo,
                backfill_mesh=backfill_mesh,
                backfill_mondo=backfill_mondo,
                backfill_doid=backfill_doid,
            )

        # Tier 1: exact ID match
        # Pathogen path also checks mesh_id from the extra ids dict
        if pathogen_lookup and pathogen_extra_ids:
            mesh_id = (pathogen_extra_ids.get("mesh_id") or "").strip()
            if mesh_id and mesh_id in index.by_mesh_id:
                return _MatchOutcome(
                    disease_name=index.by_mesh_id[mesh_id],
                    match_tier="mesh_id",
                )
        if normalized_curie:
            if kind == "efo" and normalized_curie in index.by_efo_id:
                return _MatchOutcome(
                    disease_name=index.by_efo_id[normalized_curie],
                    match_tier="efo_id",
                )
            if kind == "mondo" and normalized_curie in index.by_mondo_id:
                return _MatchOutcome(
                    disease_name=index.by_mondo_id[normalized_curie],
                    match_tier="mondo_id",
                )
            if kind == "doid" and normalized_curie in index.by_doid_id:
                return _MatchOutcome(
                    disease_name=index.by_doid_id[normalized_curie],
                    match_tier="doid_id",
                )

        # Tier 2: normalized name
        if disease_name:
            norm = _normalize_for_match(disease_name)
            if norm and norm in index.by_norm_name:
                return _outcome(index.by_norm_name[norm], "norm_name")

        # Tier 3: MeSH synonym expansion (only useful when we have a
        # MeSH ID — most pathogen-disease + LLM-verified candidates do)
        if pathogen_extra_ids:
            mesh_id = (pathogen_extra_ids.get("mesh_id") or "").strip()
            if mesh_id:
                for syn in self._get_mesh_synonyms_cached(mesh_id):
                    norm = _normalize_for_match(syn)
                    if norm and norm in index.by_norm_name:
                        return _outcome(
                            index.by_norm_name[norm], "mesh_synonym",
                        )

        return _MatchOutcome(disease_name=None, match_tier="no_match")

    def _get_mesh_synonyms_cached(self, mesh_id: str) -> list[str]:
        if mesh_id not in self._mesh_synonym_cache:
            self._mesh_synonym_cache[mesh_id] = (
                self._ontology.get_mesh_synonyms(mesh_id)
            )
        return self._mesh_synonym_cache[mesh_id]

    def _enqueue_backfills(
        self,
        outcome: _MatchOutcome,
        index: _DiseaseIndex,
        backfilled_efos: dict, backfilled_meshes: dict,
        backfilled_mondos: dict, backfilled_doids: dict,
    ) -> None:
        """Queue backfills onto the matched Modern_Disease + upgrade the
        in-memory index so later candidates in the SAME run can hit
        tier 1 directly. Each ID type routes to its own field."""
        name = outcome.disease_name or ""
        if outcome.backfill_efo:
            existing = backfilled_efos.get(name)
            if existing != outcome.backfill_efo:
                backfilled_efos[name] = outcome.backfill_efo
                index.by_efo_id[outcome.backfill_efo] = name
                index.has_efo.add(name)
        if outcome.backfill_mesh:
            existing = backfilled_meshes.get(name)
            if existing != outcome.backfill_mesh:
                backfilled_meshes[name] = outcome.backfill_mesh
                index.by_mesh_id[outcome.backfill_mesh] = name
                index.has_mesh.add(name)
        if outcome.backfill_mondo:
            existing = backfilled_mondos.get(name)
            if existing != outcome.backfill_mondo:
                backfilled_mondos[name] = outcome.backfill_mondo
                index.by_mondo_id[outcome.backfill_mondo] = name
                index.has_mondo.add(name)
        if outcome.backfill_doid:
            existing = backfilled_doids.get(name)
            if existing != outcome.backfill_doid:
                backfilled_doids[name] = outcome.backfill_doid
                index.by_doid_id[outcome.backfill_doid] = name
                index.has_doid.add(name)

    # ------------------------------------------------------------------
    # Edge prop builders
    # ------------------------------------------------------------------

    def _protein_edge_props(
        self,
        evidence_type: str,
        assoc: open_targets.OtAssociation,
        top_subunit_uniprot: str,
        attempted_at: str,
    ) -> dict:
        confidence = CONFIDENCE_OT.get(evidence_type, 0.5)
        rationale = self._template_rationale_protein(evidence_type, assoc)
        props: dict[str, Any] = {
            "confidence": confidence,
            "evidence_type": evidence_type,
            "source_db": "OpenTargets",
            "ot_overall_score": assoc.score,
            "ot_resolved_id": assoc.disease_id,
            "ot_target_ensembl_id": assoc.target_ensembl_id,
            "ot_datasource_scores": json.dumps(assoc.datasource_scores or {}),
            "rationale": rationale,
            "requires_review": False,
            "created_by": "target_disease_linker",
            "created_at": attempted_at,
        }
        if top_subunit_uniprot:
            props["ot_top_subunit_uniprot"] = top_subunit_uniprot
        return props

    def _organism_edge_props(
        self,
        evidence_type: str,
        source_label: str,
        tax_id: str,
        candidate: dict,
        attempted_at: str,
    ) -> dict:
        confidence = CONFIDENCE_PATHOGEN.get(evidence_type, 0.5)
        rationale = self._template_rationale_organism(
            evidence_type, source_label, tax_id, candidate,
        )
        is_llm = "llm" in source_label
        props: dict[str, Any] = {
            "confidence": confidence,
            "evidence_type": evidence_type,
            "source_db": "EFO+DOID" if "+" in source_label else (
                "EFO" if source_label == "efo" else
                "DOID" if source_label == "doid" else "Gemini+NLM"
            ),
            "ncbi_tax_id": tax_id,
            "pathogen_lookup_source": source_label,
            "rationale": rationale,
            "requires_review": is_llm,
            "created_by": "target_disease_linker",
            "created_at": attempted_at,
        }
        # Carry the ID we used for the match, plus any candidate IDs
        if candidate.get("id"):
            props["ot_resolved_id"] = candidate["id"]
        if candidate.get("mesh_id"):
            props["mesh_id"] = candidate["mesh_id"]
        if candidate.get("icd10_code"):
            props["icd10_code"] = candidate["icd10_code"]
        return props

    @staticmethod
    def _template_rationale_protein(
        evidence_type: str, assoc: open_targets.OtAssociation,
    ) -> str:
        ds = assoc.datasource_scores or {}
        top_sources = sorted(ds.items(), key=lambda kv: -(kv[1] or 0))[:3]
        bits = ", ".join(f"{k}={v:.2f}" for k, v in top_sources if v)
        tier = evidence_type.replace("ot_association_", "")
        if bits:
            return (
                f"OT {tier} association (overall_score={assoc.score:.2f}); "
                f"top sources: {bits}"
            )
        return f"OT {tier} association (overall_score={assoc.score:.2f})"

    @staticmethod
    def _template_rationale_organism(
        evidence_type: str, source_label: str, tax_id: str, candidate: dict,
    ) -> str:
        cand_name = candidate.get("name", "")
        if "consensus" in evidence_type:
            return (
                f"Pathogen-disease link (NCBI tax_id {tax_id} → {cand_name}) "
                f"confirmed by both EFO and DOID."
            )
        if "llm_verified" in evidence_type:
            llm_rationale = candidate.get("rationale", "").strip()
            base = (
                f"Pathogen-disease link (NCBI tax_id {tax_id} → {cand_name}) "
                f"proposed by LLM, verified via NLM ontology lookup."
            )
            return f"{base} {llm_rationale}" if llm_rationale else base
        return (
            f"Pathogen-disease link (NCBI tax_id {tax_id} → {cand_name}) "
            f"from {source_label.upper()}."
        )

    def _status_row(
        self, target_chembl_id: str, status: LinkStatus,
        association_count: int, dropped: int, attempted_at: str,
    ) -> dict:
        return {
            "target_chembl_id": target_chembl_id,
            "status": status.value,
            "attempted_at": attempted_at,
            "association_count": association_count,
            "dropped_count": dropped,
            "min_score": self._min_score,
        }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _wipe_existing(self) -> None:
        self._log_progress(
            "Rebuild: wiping RELATES_TO + td_linker_* props"
        )
        self.client.run_write("MATCH ()-[r:RELATES_TO]->() DELETE r")
        self.client.run_write("""
            MATCH (t:Biological_Target)
            REMOVE t.td_linker_status,
                   t.td_linker_attempted_at,
                   t.td_linker_association_count,
                   t.td_linker_dropped_count,
                   t.td_linker_min_score
        """)
