"""Phase 1 — Compound → Disease KNOWN_TREATS Linking Agent.

Pure routing layer over ChEMBL `drug_indication`. Mirrors the v2
CompoundTargetLinker shape: no LLM in the loop, idempotent via
per-Compound `kt_linker_status`, flat per-evidence-type confidence
priors, crash-safe write order.

The KNOWN_TREATS edge is the gold-standard "this compound is clinically
known to treat this disease" signal. Phase 2 uses it two ways:

  * Task B's novelty filter — exclude compounds already approved for the
    queried disease (so we don't recommend an existing drug as "novel").
  * Phase 2 scoring — a positive prior on (compound, disease) pairs that
    have a clinical indication.

Pipeline (per Chemical_Compound):

  STEP A: LOOKUP  (ChEMBL drug_indication, hierarchy-expanded)
    - Walk parent + salts + tautomers from c.linker_chembl_id
    - Fetch /drug_indication?molecule_chembl_id__in=...
    - Per (compound, mesh_id) pair, keep max-phase row only

  STEP B: MATCH   (three-tier, MATCH-ONLY — no new disease nodes)
    Tier 1: graph node mesh_id == ChEMBL's mesh_id (deterministic)
    Tier 2: _normalize_for_match(name) exact match (cosmetic differences)
    Tier 3: ChEMBL-side MeSH synonym expansion — look up ChEMBL's mesh_id
            via OntologyClient.get_mesh_synonyms(), normalize each, and
            check against any Modern_Disease name. Critically, the
            synonym source is ChEMBL's mesh_id NOT the graph node's, so
            this works even when the graph node has no MeSH code at all.
    No match → drop the indication and count it (audit signal).

  STEP C: WRITE  (MERGE-only on existing node; status set LAST)
    1. (force_relink only) DELETE prior KNOWN_TREATS edges for this
       compound
    2. Coalesce-backfill mesh_id and efo_id onto the matched
       Modern_Disease node (when ChEMBL has IDs the graph lacks).
       Updates the in-memory index so subsequent matches in the SAME
       run hit tier 1 directly.
    3. MERGE KNOWN_TREATS edge with confidence + clinical_phase +
       evidence_type + mesh_id + efo_id + audit fields
    4. SET kt_linker_status / kt_linker_attempted_at /
       kt_linker_indication_count / kt_linker_dropped_count on the
       Compound (LAST)

Confidence priors (by max_phase_for_ind):
  4 (approved)  → chembl_indication_approved   0.95
  3            → chembl_indication_phase3     0.85
  2            → chembl_indication_phase2     0.65
  1            → chembl_indication_phase1     0.45
  0 (preclinical) → DROPPED ENTIRELY (relation is "known to treat")

Failure handling:
  - HTTP 404 / no record → outcome=not_in_chembl → status=no_indications
  - Network error / 5xx  → outcome=transient_failure → status=error
                           (re-tried via --retry-misses)
  - Per-compound exception in pipeline → status=error
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.agents.base import BaseAgent
from src.agents.phase1.malady_disease import _normalize_for_match
from src.data import chembl
from src.data.ontology_client import OntologyClient
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class LinkStatus(str, Enum):
    LINKED = "linked"
    NO_INDICATIONS = "no_indications"
    ERROR = "error"


# Flat per-evidence-type confidence priors. Keyed on the `evidence_type`
# string written to each KNOWN_TREATS edge. Phase 0 (preclinical) is
# absent intentionally — the relation name is "known to treat" and
# preclinical indications don't qualify.
CONFIDENCE: dict[str, float] = {
    "chembl_indication_approved": 0.95,   # max_phase 4 — FDA-approved
    "chembl_indication_phase3":   0.85,
    "chembl_indication_phase2":   0.65,
    "chembl_indication_phase1":   0.45,
}


_DEFAULT_WORKERS = 8
_DEFAULT_MIN_PHASE = 1   # drop preclinical (0) by default


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence_type_for_phase(phase: int) -> str | None:
    """Map ChEMBL max_phase_for_ind → evidence_type label.

    Returns None for phase 0 (preclinical) so the caller drops the row.
    Phase >= 4 collapses to `approved` (rare cases of phase 4.5 etc.).
    """
    if phase >= 4:
        return "chembl_indication_approved"
    if phase == 3:
        return "chembl_indication_phase3"
    if phase == 2:
        return "chembl_indication_phase2"
    if phase == 1:
        return "chembl_indication_phase1"
    return None


@dataclass
class _DiseaseIndex:
    """In-memory match index for the existing Modern_Disease universe.

    Built once at run start from ALL_MODERN_DISEASES. Mutated in place
    as we backfill mesh_id/efo_id onto matched nodes — so a tier-2 or
    tier-3 match earlier in the run upgrades the same disease to a
    tier-1 hit later in the same run.
    """
    by_mesh_id: dict[str, str] = field(default_factory=dict)
    by_norm_name: dict[str, str] = field(default_factory=dict)
    has_mesh: set[str] = field(default_factory=set)   # disease names that have a mesh_id
    has_efo: set[str] = field(default_factory=set)    # disease names that have an efo_id


@dataclass
class _MatchOutcome:
    """Result of matching one ChEMBL indication against the disease index."""
    disease_name: str | None       # None = no match (dropped)
    match_tier: str                # "mesh_id" | "norm_name" | "mesh_synonym" | "no_match"
    backfill_mesh: str = ""        # non-empty → backfill onto matched node
    backfill_efo: str = ""         # non-empty → backfill onto matched node


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CompoundDiseaseLinker(BaseAgent):
    """Route Chemical_Compound nodes to ChEMBL drug_indication and write
    KNOWN_TREATS edges to existing Modern_Disease nodes (match-only)."""

    def __init__(
        self,
        client: GraphClient,
        *,
        workers: int = _DEFAULT_WORKERS,
        min_phase: int = _DEFAULT_MIN_PHASE,
        ontology_client: OntologyClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._workers = workers
        self._min_phase = min_phase
        self._ontology = ontology_client or OntologyClient()
        self._mesh_synonym_cache: dict[str, list[str]] = {}
        self._force_relink = False  # set per-run via run()

    @property
    def name(self) -> str:
        return "CompoundDiseaseLinker"

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

        self._force_relink = force_relink

        cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS
        if force_relink:
            cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS_FORCE
        elif retry_misses:
            cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS_RETRY

        compounds = self.client.run(cypher)
        if limit is not None:
            compounds = compounds[:limit]

        # Build the in-memory disease universe ONCE per run. Match-only
        # mode means we never grow this set from ChEMBL data — anything
        # that doesn't match here is dropped and counted.
        index = self._load_disease_index()

        total = len(compounds)
        self._log_progress(
            f"Linking {total} compound(s) to {len(index.by_norm_name)} "
            f"existing Modern_Disease node(s) "
            f"(dry_run={dry_run}, rebuild={rebuild}, "
            f"retry_misses={retry_misses}, force_relink={force_relink}, "
            f"min_phase={self._min_phase}, workers={self._workers})"
        )

        t_start = time.time()
        by_status: dict[str, int] = {s.value: 0 for s in LinkStatus}
        by_evidence_type: dict[str, int] = {k: 0 for k in CONFIDENCE}
        by_match_tier: dict[str, int] = {}
        edge_writes = 0
        unique_diseases: set[str] = set()
        indications_total = 0
        indications_matched = 0
        indications_dropped_no_match = 0
        indications_dropped_phase = 0
        backfills_mesh = 0
        backfills_efo = 0
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futs = {
                pool.submit(
                    chembl.fetch_compound_indications,
                    c["chembl_id"], inchikey=c["inchikey"],
                ): c for c in compounds
            }
            done = 0
            for fut in as_completed(futs):
                compound_row = futs[fut]
                done += 1
                try:
                    result = fut.result()
                except Exception as e:
                    msg = f"{compound_row.get('name', '?')}: {e}"
                    logger.exception(msg)
                    errors.append(msg)
                    by_status[LinkStatus.ERROR.value] += 1
                    if not dry_run:
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.ERROR,
                            indication_count=0, dropped=0,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                if result.outcome == "transient_failure":
                    by_status[LinkStatus.ERROR.value] += 1
                    if result.error:
                        errors.append(
                            f"{compound_row.get('name', '?')}: {result.error}"
                        )
                    if not dry_run:
                        # Force-relink wipes prior edges even on error so
                        # stale edges don't leak through.
                        self._maybe_delete_prior_edges(compound_row["inchikey"])
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.ERROR,
                            indication_count=0, dropped=0,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                if result.outcome == "not_in_chembl" or not result.indications:
                    by_status[LinkStatus.NO_INDICATIONS.value] += 1
                    if not dry_run:
                        self._maybe_delete_prior_edges(compound_row["inchikey"])
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.NO_INDICATIONS,
                            indication_count=0, dropped=0,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                # Match each indication against the in-graph disease universe
                indications_total += len(result.indications)
                edges_written_this_compound: list[tuple[str, chembl.ChemblIndication, _MatchOutcome]] = []
                phase_dropped_this = 0
                no_match_this = 0
                for ind in result.indications:
                    # Enforce --min-phase BEFORE evidence-type mapping.
                    # _evidence_type_for_phase only drops phase 0; the
                    # CLI knob requires an explicit gate here.
                    if ind.max_phase_for_ind < self._min_phase:
                        phase_dropped_this += 1
                        continue
                    et = _evidence_type_for_phase(ind.max_phase_for_ind)
                    if et is None:
                        phase_dropped_this += 1
                        continue
                    outcome = self._match_indication(ind, index)
                    by_match_tier[outcome.match_tier] = (
                        by_match_tier.get(outcome.match_tier, 0) + 1
                    )
                    if outcome.disease_name is None:
                        no_match_this += 1
                        continue
                    edges_written_this_compound.append((et, ind, outcome))

                indications_dropped_phase += phase_dropped_this
                indications_dropped_no_match += no_match_this

                # Per-disease dedup at max-phase. Two distinct mesh_ids
                # can land on the same Modern_Disease node via tier 3
                # — keep the highest-phase row only.
                best_per_disease: dict[str, tuple[str, chembl.ChemblIndication, _MatchOutcome]] = {}
                for et, ind, outcome in edges_written_this_compound:
                    key = outcome.disease_name or ""
                    if (
                        key not in best_per_disease
                        or ind.max_phase_for_ind
                            > best_per_disease[key][1].max_phase_for_ind
                    ):
                        best_per_disease[key] = (et, ind, outcome)
                kept = list(best_per_disease.values())

                if not kept:
                    by_status[LinkStatus.NO_INDICATIONS.value] += 1
                    if not dry_run:
                        self._maybe_delete_prior_edges(compound_row["inchikey"])
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.NO_INDICATIONS,
                            indication_count=0,
                            dropped=phase_dropped_this + no_match_this,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                # Write edges + update in-memory index for in-run upgrades
                by_status[LinkStatus.LINKED.value] += 1
                indications_matched += len(kept)
                for et, ind, outcome in kept:
                    by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
                    unique_diseases.add(outcome.disease_name or "")
                edge_writes += len(kept)

                if not dry_run:
                    self._maybe_delete_prior_edges(compound_row["inchikey"])
                    for et, ind, outcome in kept:
                        if outcome.backfill_mesh or outcome.backfill_efo:
                            self._backfill_disease_ids(
                                outcome.disease_name or "", outcome,
                            )
                            if outcome.backfill_mesh:
                                backfills_mesh += 1
                                # Upgrade the in-memory index so later
                                # indications in the same run hit tier 1.
                                index.by_mesh_id[outcome.backfill_mesh] = outcome.disease_name or ""
                                index.has_mesh.add(outcome.disease_name or "")
                            if outcome.backfill_efo:
                                backfills_efo += 1
                                index.has_efo.add(outcome.disease_name or "")
                        self._merge_known_treats_edge(
                            compound_row["inchikey"], outcome.disease_name or "",
                            ind, et,
                        )
                    self._persist_status(
                        compound_row["inchikey"],
                        LinkStatus.LINKED,
                        indication_count=len(kept),
                        dropped=phase_dropped_this + no_match_this,
                    )

                self._maybe_log_progress(done, total, by_status)

        duration_s = round(time.time() - t_start, 2)

        return {
            "dry_run": dry_run,
            "rebuild": rebuild,
            "min_phase": self._min_phase,
            "compounds_total": total,
            "modern_disease_universe": len(index.by_norm_name),
            "by_status": by_status,
            "by_evidence_type": by_evidence_type,
            "by_match_tier": dict(
                sorted(by_match_tier.items(), key=lambda kv: -kv[1])
            ),
            "edge_writes": edge_writes,
            "unique_diseases_linked": len(unique_diseases),
            "indications_total": indications_total,
            "indications_matched": indications_matched,
            "indications_dropped_no_match": indications_dropped_no_match,
            "indications_dropped_phase": indications_dropped_phase,
            "backfills_mesh_id": backfills_mesh,
            "backfills_efo_id": backfills_efo,
            "duration_s": duration_s,
            "errors": errors,
        }

    def _maybe_log_progress(
        self, done: int, total: int, by_status: dict[str, int],
    ) -> None:
        if done % 50 == 0 or done == total:
            self._log_progress(
                f"  {done}/{total}  linked={by_status['linked']} "
                f"no_indications={by_status['no_indications']} "
                f"errors={by_status['error']}"
            )

    # ------------------------------------------------------------------
    # Step A — load existing disease universe
    # ------------------------------------------------------------------

    def _load_disease_index(self) -> _DiseaseIndex:
        rows = self.client.run(queries.ALL_MODERN_DISEASES)
        index = _DiseaseIndex()
        for r in rows:
            name = r.get("name") or ""
            if not name:
                continue
            mesh_id = (r.get("mesh_id") or "").strip()
            efo_id = (r.get("efo_id") or "").strip()
            if mesh_id:
                index.by_mesh_id[mesh_id] = name
                index.has_mesh.add(name)
            if efo_id:
                index.has_efo.add(name)
            norm = _normalize_for_match(name)
            if norm and norm not in index.by_norm_name:
                index.by_norm_name[norm] = name
        return index

    # ------------------------------------------------------------------
    # Step B — three-tier match
    # ------------------------------------------------------------------

    def _match_indication(
        self, ind: chembl.ChemblIndication, index: _DiseaseIndex,
    ) -> _MatchOutcome:
        # Tier 1: ChEMBL mesh_id == graph node mesh_id (deterministic).
        if ind.mesh_id and ind.mesh_id in index.by_mesh_id:
            disease = index.by_mesh_id[ind.mesh_id]
            return _MatchOutcome(
                disease_name=disease,
                match_tier="mesh_id",
                backfill_mesh="",  # already present (that's how we matched)
                backfill_efo=(
                    ind.efo_id if ind.efo_id and disease not in index.has_efo
                    else ""
                ),
            )

        # Tier 2: normalized name match against either mesh_heading or
        # efo_term. Mesh_heading is preferred (more curated) but efo_term
        # is the fallback when ChEMBL only carried EFO.
        for candidate in (ind.mesh_heading, ind.efo_term):
            if not candidate:
                continue
            norm = _normalize_for_match(candidate)
            if norm and norm in index.by_norm_name:
                disease = index.by_norm_name[norm]
                return _MatchOutcome(
                    disease_name=disease,
                    match_tier="norm_name",
                    backfill_mesh=(
                        ind.mesh_id
                        if ind.mesh_id and disease not in index.has_mesh
                        else ""
                    ),
                    backfill_efo=(
                        ind.efo_id
                        if ind.efo_id and disease not in index.has_efo
                        else ""
                    ),
                )

        # Tier 3: ChEMBL-side MeSH synonym expansion. Source the synonyms
        # from ChEMBL's mesh_id (NOT the graph's), so this works even
        # when the graph node has no mesh_id at all. Cached per mesh_id
        # to avoid hammering NLM on hot indications.
        if ind.mesh_id:
            for syn in self._get_mesh_synonyms_cached(ind.mesh_id):
                norm = _normalize_for_match(syn)
                if norm and norm in index.by_norm_name:
                    disease = index.by_norm_name[norm]
                    return _MatchOutcome(
                        disease_name=disease,
                        match_tier="mesh_synonym",
                        backfill_mesh=(
                            ind.mesh_id
                            if disease not in index.has_mesh
                            else ""
                        ),
                        backfill_efo=(
                            ind.efo_id
                            if ind.efo_id and disease not in index.has_efo
                            else ""
                        ),
                    )

        return _MatchOutcome(disease_name=None, match_tier="no_match")

    def _get_mesh_synonyms_cached(self, mesh_id: str) -> list[str]:
        if mesh_id not in self._mesh_synonym_cache:
            self._mesh_synonym_cache[mesh_id] = (
                self._ontology.get_mesh_synonyms(mesh_id)
            )
        return self._mesh_synonym_cache[mesh_id]

    # ------------------------------------------------------------------
    # Step C — graph writes
    # ------------------------------------------------------------------

    def _maybe_delete_prior_edges(self, compound_inchikey: str) -> None:
        """When --force-relink re-touches a compound, delete its prior
        KNOWN_TREATS edges before writing the new plan. Called from
        EVERY terminal-status branch so a relink that now resolves to
        zero matches still clears the stale edges."""
        if not self._force_relink:
            return
        self.client.run_write(
            "MATCH (c:Chemical_Compound {inchikey: $inchikey})"
            "-[r:KNOWN_TREATS]->() DELETE r",
            {"inchikey": compound_inchikey},
        )

    def _backfill_disease_ids(
        self, disease_name: str, outcome: _MatchOutcome,
    ) -> None:
        """Coalesce-backfill mesh_id and/or efo_id onto an existing
        Modern_Disease node. Same coalesce pattern MaladyDiseaseMapper
        uses — never overwrite existing values, only fill missing ones.
        """
        backfill: dict[str, str] = {}
        if outcome.backfill_mesh:
            backfill["mesh_id"] = outcome.backfill_mesh
        if outcome.backfill_efo:
            backfill["efo_id"] = outcome.backfill_efo
        if not backfill:
            return
        coalesce_clauses = ", ".join(
            f"d.{k} = CASE WHEN d.{k} IS NULL OR d.{k} = '' "
            f"THEN $backfill.{k} ELSE d.{k} END"
            for k in backfill
        )
        self.client.run_write(
            f"""
            MATCH (d:Modern_Disease {{name: $name}})
            SET {coalesce_clauses}
            """,
            {"name": disease_name, "backfill": backfill},
        )

    def _merge_known_treats_edge(
        self,
        compound_inchikey: str,
        disease_name: str,
        ind: chembl.ChemblIndication,
        evidence_type: str,
    ) -> None:
        confidence = CONFIDENCE.get(evidence_type, 0.5)
        rel_props: dict[str, Any] = {
            "confidence_score": confidence,
            "evidence_type": evidence_type,
            "source_db": "ChEMBL",
            "clinical_phase": ind.max_phase_for_ind,
            "drug_indication_id": ind.drug_indication_id,
            "molecule_chembl_id": ind.molecule_chembl_id,
            "created_by": "compound_disease_linker",
            "created_at": dt.datetime.utcnow().isoformat(),
        }
        if ind.mesh_id:
            rel_props["mesh_id"] = ind.mesh_id
        if ind.mesh_heading:
            rel_props["mesh_heading"] = ind.mesh_heading
        if ind.efo_id:
            rel_props["efo_id"] = ind.efo_id
        if ind.efo_term:
            rel_props["efo_term"] = ind.efo_term

        self.client.merge_edge(
            "Chemical_Compound", {"inchikey": compound_inchikey},
            "Modern_Disease", {"name": disease_name},
            "KNOWN_TREATS",
            rel_props,
            from_key="inchikey",
        )

    def _persist_status(
        self,
        compound_inchikey: str,
        status: LinkStatus,
        *,
        indication_count: int,
        dropped: int,
    ) -> None:
        """Set kt_linker_* props on the Compound. Always called LAST in
        the per-compound write — partial failures upstream leave the
        status NULL so the next default run picks the compound up."""
        props: dict[str, Any] = {
            "kt_linker_status": status.value,
            "kt_linker_attempted_at": dt.datetime.utcnow().isoformat(),
            "kt_linker_indication_count": indication_count,
            "kt_linker_dropped_count": dropped,
            "kt_linker_min_phase": self._min_phase,
        }
        self.client.run_write(
            "MATCH (c:Chemical_Compound {inchikey: $inchikey}) SET c += $props",
            {"inchikey": compound_inchikey, "props": props},
        )

    # ------------------------------------------------------------------
    # Rebuild helper
    # ------------------------------------------------------------------

    def _wipe_existing(self) -> None:
        """Idempotent clean slate: drop all KNOWN_TREATS edges and
        clear kt_linker_* properties on Compounds. Does NOT delete
        Modern_Disease nodes — those belong to MaladyDiseaseMapper."""
        self._log_progress(
            "Rebuild: wiping KNOWN_TREATS + kt_linker_* props"
        )
        self.client.run_write("MATCH ()-[r:KNOWN_TREATS]->() DELETE r")
        self.client.run_write("""
            MATCH (c:Chemical_Compound)
            REMOVE c.kt_linker_status,
                   c.kt_linker_attempted_at,
                   c.kt_linker_indication_count,
                   c.kt_linker_dropped_count,
                   c.kt_linker_min_phase
        """)
