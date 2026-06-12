"""Phase 1 — Compound → Disease KNOWN_TREATS Linker (BATCHED).

Throughput-optimized variant of `CompoundDiseaseLinker`. Same matching
semantics, same disease-universe scoping, same confidence priors —
only the I/O strategy differs.

What changes vs the per-compound version:

  * ChEMBL fetches use `__in=` filters (~100 IDs per URL) instead of
    one URL per compound. Three bulk phases — molecule resolve,
    hierarchy expand, drug_indication fetch — each parallelized over
    chunks. ~50× fewer HTTP round trips at our scale.

  * Neo4j writes use UNWIND-batched transactions instead of per-compound
    MERGE. Replaces ~5,500 round trips with ~12 — one chunk-tx per
    `batch_size` rows for each of the four logical write phases
    (delete-prior, disease-id backfill, edge merge, compound status).
    Each batch is one Neo4j transaction (so per-batch atomicity), but
    cross-phase atomicity is NOT guaranteed: a failure between phases
    can leave partial state (e.g. force-delete succeeded, edge merge
    failed → orphaned compound with cleared edges and stale status).
    See "Known limitation: cross-phase atomicity" below.

  * "Plan-then-apply" structure: ALL matching happens in memory before
    any write touches Neo4j. The summary now includes `planned_*`
    counts that you can inspect via `--dry-run` to validate the
    write set before applying.

Same as the per-compound version:

  * Match-only mode — never creates new Modern_Disease nodes
  * Three-tier matching: mesh_id → normalized name → MeSH synonym
    expansion (sourced from ChEMBL's mesh_id, not the graph node's)
  * Per-clinical-phase confidence priors; phase 0 dropped
  * Self-healing graph via coalesce backfill of mesh_id / efo_id
  * `kt_linker_status` resume contract (NULL / linked / no_indications
    / error); same three Cypher variants for default / retry-misses /
    force-relink
  * Returns the same summary dict shape — drop-in replaceable

Trade-offs of the batched approach:

  * Coarser failure attribution. A bulk ChEMBL transient failure marks
    EVERY compound in the batch as `error`. The next `--retry-misses`
    run re-processes them. For ChEMBL's reliability at our scale this
    rarely matters; per-compound granularity remains available via
    the original linker.
  * Resumability is per-batch instead of per-compound. A crash
    mid-write loses the in-flight UNWIND chunk (default 500 compounds)
    rather than the in-flight compound. Smaller `--write-batch-size`
    values reduce the blast radius if you care.

Known limitation: cross-phase atomicity
  Each of the four write phases (delete / backfill / merge edges /
  status) runs as its own batched_unwind_write call. Within ONE
  Neo4j transaction (one UNWIND of up to `batch_size` rows), all rows
  succeed or all fail. ACROSS phases, a partial-state window exists:
  if phase 1 (delete) succeeds and phase 3 (edge merge) fails, the
  compound's prior edges are gone and `kt_linker_status` still reads
  whatever it was before the run. The next default rerun would skip
  it. `--retry-misses` recovers it because the status check tolerates
  no_indications and error rows.

  Cross-phase atomicity could be fixed by combining all four phase
  operations into a single Cypher transaction per chunk (messy but
  possible) or using Neo4j 5's `CALL { ... } IN TRANSACTIONS OF N
  ROWS`. Deferred — same scope as the cross-cutting refactor that
  would make all four Phase 1 linkers atomically-batched.

Transient-failure handling: bulk ChEMBL transients flip EVERY compound
in the batch to `transient_failure` (coarse failure semantics from
fetch_compound_indications_batch / _all_transient). Compounds in this
state are NEVER added to the delete_rows set — even with --force-relink.
If we did, a single ChEMBL outage during a --force-relink run would
wipe ALL existing KNOWN_TREATS at once. Prior edges survive the blip;
status moves to `error`; --retry-misses picks the compound up next run.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agents.base import BaseAgent
from src.agents.phase1.compound_disease import (
    CONFIDENCE,
    LinkStatus,
    _DiseaseIndex,
    _MatchOutcome,
    _evidence_type_for_phase,
)
from src.agents.phase1.malady_disease import _normalize_for_match
from src.data import chembl
from src.data.ontology_client import OntologyClient
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


_DEFAULT_MIN_PHASE = 1
_DEFAULT_WRITE_BATCH_SIZE = 500
_SYNONYM_PREFETCH_WORKERS = 8


# ---------------------------------------------------------------------------
# Cypher write templates (UNWIND-based)
# ---------------------------------------------------------------------------


_DELETE_PRIOR_EDGES_BATCH = """
UNWIND $rows AS r
MATCH (c:Chemical_Compound {inchikey: r.inchikey})-[k:KNOWN_TREATS]->()
DELETE k
"""


_BACKFILL_DISEASE_BATCH = """
UNWIND $rows AS r
MATCH (d:Modern_Disease {name: r.name})
SET d.mesh_id = CASE WHEN d.mesh_id IS NULL OR d.mesh_id = ''
                THEN r.mesh_id ELSE d.mesh_id END,
    d.efo_id  = CASE WHEN d.efo_id IS NULL OR d.efo_id = ''
                THEN r.efo_id  ELSE d.efo_id  END
"""


_MERGE_EDGES_BATCH = """
UNWIND $rows AS r
MATCH (c:Chemical_Compound {inchikey: r.inchikey})
MATCH (d:Modern_Disease {name: r.disease_name})
MERGE (c)-[k:KNOWN_TREATS]->(d)
SET k += r.props
"""


_UPDATE_STATUS_BATCH = """
UNWIND $rows AS r
MATCH (c:Chemical_Compound {inchikey: r.inchikey})
SET c.kt_linker_status = r.status,
    c.kt_linker_attempted_at = r.attempted_at,
    c.kt_linker_indication_count = r.indication_count,
    c.kt_linker_dropped_count = r.dropped_count,
    c.kt_linker_min_phase = r.min_phase
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CompoundDiseaseLinkerBatched(BaseAgent):
    """Throughput-optimized KNOWN_TREATS linker.

    See module docstring for the architectural diff vs CompoundDiseaseLinker.
    """

    def __init__(
        self,
        client: GraphClient,
        *,
        min_phase: int = _DEFAULT_MIN_PHASE,
        write_batch_size: int = _DEFAULT_WRITE_BATCH_SIZE,
        ontology_client: OntologyClient | None = None,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._min_phase = min_phase
        self._write_batch_size = write_batch_size
        self._ontology = ontology_client or OntologyClient()
        self._mesh_synonym_cache: dict[str, list[str]] = {}

    @property
    def name(self) -> str:
        return "CompoundDiseaseLinkerBatched"

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

        cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS
        if force_relink:
            cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS_FORCE
        elif retry_misses:
            cypher = queries.LINKABLE_COMPOUNDS_FOR_INDICATIONS_RETRY

        compounds = self.client.run(cypher)
        if limit is not None:
            compounds = compounds[:limit]

        index = self._load_disease_index()

        total = len(compounds)
        self._log_progress(
            f"Batched linking {total} compound(s) to {len(index.by_norm_name)} "
            f"existing Modern_Disease node(s) "
            f"(dry_run={dry_run}, rebuild={rebuild}, "
            f"retry_misses={retry_misses}, force_relink={force_relink}, "
            f"min_phase={self._min_phase}, "
            f"write_batch_size={self._write_batch_size})"
        )

        t_start = time.time()

        # ----- PHASE 1: bulk-fetch from ChEMBL --------------------------
        t_fetch = time.time()
        inchikey_to_chembl_id = {c["inchikey"]: c["chembl_id"] for c in compounds}
        results = chembl.fetch_compound_indications_batch(inchikey_to_chembl_id)
        fetch_duration = round(time.time() - t_fetch, 2)
        self._log_progress(f"  ChEMBL bulk fetch: {fetch_duration}s")

        # ----- PHASE 1.5: pre-fetch MeSH synonyms in parallel -----------
        # Without this, the per-indication tier-3 lookups in _build_plan
        # are sequential against the rate-limited OntologyClient — the
        # main bottleneck of the batched run. We KNOW which mesh_ids might
        # need synonym lookup (every unique mesh_id ChEMBL returned),
        # so warming the cache in parallel cuts seconds-per-lookup ×
        # hundreds-of-lookups down to one parallel batch.
        t_syn = time.time()
        unique_mesh_ids = {
            ind.mesh_id
            for r in results.values()
            for ind in r.indications
            if ind.mesh_id and ind.mesh_id not in index.by_mesh_id
        }
        self._prefetch_synonyms(unique_mesh_ids)
        synonym_duration = round(time.time() - t_syn, 2)
        self._log_progress(
            f"  MeSH synonym prefetch: {synonym_duration}s "
            f"({len(unique_mesh_ids)} unique mesh_ids)"
        )

        # ----- PHASE 2: match + plan ------------------------------------
        t_plan = time.time()
        plan = self._build_plan(compounds, results, index)
        plan_duration = round(time.time() - t_plan, 2)
        self._log_progress(
            f"  Plan: {plan_duration}s — "
            f"{len(plan['edge_rows'])} edges, "
            f"{len(plan['backfill_rows'])} backfills, "
            f"{len(plan['status_rows'])} status writes"
        )

        # ----- PHASE 3: bulk write --------------------------------------
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
            f"  Neo4j bulk write: {write_duration}s "
            f"({writes_done} rows applied)"
        )

        duration_s = round(time.time() - t_start, 2)

        return {
            "dry_run": dry_run,
            "rebuild": rebuild,
            "min_phase": self._min_phase,
            "write_batch_size": self._write_batch_size,
            "compounds_total": total,
            "modern_disease_universe": len(index.by_norm_name),
            "by_status": plan["by_status"],
            "by_evidence_type": plan["by_evidence_type"],
            "by_match_tier": dict(
                sorted(plan["by_match_tier"].items(), key=lambda kv: -kv[1])
            ),
            "edge_writes": len(plan["edge_rows"]),
            "unique_diseases_linked": len(plan["unique_diseases"]),
            "indications_total": plan["indications_total"],
            "indications_matched": plan["indications_matched"],
            "indications_dropped_no_match": plan["indications_dropped_no_match"],
            "indications_dropped_phase": plan["indications_dropped_phase"],
            "backfills_mesh_id": plan["backfills_mesh_id"],
            "backfills_efo_id": plan["backfills_efo_id"],
            "phase_durations_s": {
                "chembl_bulk_fetch": fetch_duration,
                "synonym_prefetch": synonym_duration,
                "plan": plan_duration,
                "neo4j_bulk_write": write_duration,
            },
            "duration_s": duration_s,
            "errors": plan["errors"],
        }

    # ------------------------------------------------------------------
    # Disease universe loading (mirrors per-compound version)
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
    # Match + plan (in-memory; no Neo4j writes)
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        compounds: list[dict],
        results: dict[str, chembl.IndicationResult],
        index: _DiseaseIndex,
    ) -> dict:
        """Build the full write plan in memory. Mutates `index` in place
        for in-run mesh_id/efo_id upgrades (so a tier-3 match early in the
        plan upgrades the index for a tier-1 hit later).
        """
        by_status: dict[str, int] = {s.value: 0 for s in LinkStatus}
        by_evidence_type: dict[str, int] = {k: 0 for k in CONFIDENCE}
        by_match_tier: dict[str, int] = {}
        unique_diseases: set[str] = set()
        indications_total = 0
        indications_matched = 0
        indications_dropped_no_match = 0
        indications_dropped_phase = 0
        backfills_mesh = 0
        backfills_efo = 0
        errors: list[str] = []

        delete_rows: list[dict] = []   # for force_relink
        backfill_rows: list[dict] = []
        edge_rows: list[dict] = []
        status_rows: list[dict] = []

        # Track which (disease_name, backfill) we've already enqueued so
        # we don't backfill the same node twice in one run.
        backfilled_meshes: dict[str, str] = {}    # disease_name → mesh_id
        backfilled_efos: dict[str, str] = {}      # disease_name → efo_id

        attempted_at = dt.datetime.utcnow().isoformat()

        for compound_row in compounds:
            inchikey = compound_row["inchikey"]
            result = results.get(inchikey)
            if result is None:
                # Shouldn't happen — every input has a result. Defensive.
                # Same delete-cascade logic as transient_failure: don't
                # add to delete_rows. Status set to error; --retry-misses
                # recovers. Prior edges remain valid.
                by_status[LinkStatus.ERROR.value] += 1
                errors.append(f"{compound_row.get('name', '?')}: no result returned")
                status_rows.append(self._status_row(
                    inchikey, LinkStatus.ERROR, 0, 0, attempted_at,
                ))
                continue

            if result.outcome == "transient_failure":
                by_status[LinkStatus.ERROR.value] += 1
                if result.error:
                    errors.append(
                        f"{compound_row.get('name', '?')}: {result.error}"
                    )
                status_rows.append(self._status_row(
                    inchikey, LinkStatus.ERROR, 0, 0, attempted_at,
                ))
                # IMPORTANT: do NOT add to delete_rows on transient failure.
                # Bulk transients in fetch_compound_indications_batch flip
                # EVERY compound in the batch to `transient_failure` via
                # _all_transient() (coarse failure semantics). If we
                # included transient compounds in the force-relink delete
                # set, a single ChEMBL outage during a --force-relink run
                # would wipe ALL existing KNOWN_TREATS at once. Better
                # behavior: prior edges survive the blip; status is set to
                # `error` so --retry-misses picks the compound up next run.
                continue

            if result.outcome == "not_in_chembl" or not result.indications:
                by_status[LinkStatus.NO_INDICATIONS.value] += 1
                status_rows.append(self._status_row(
                    inchikey, LinkStatus.NO_INDICATIONS, 0, 0, attempted_at,
                ))
                delete_rows.append({"inchikey": inchikey})
                continue

            # Per-indication: gate on phase, match against disease index
            indications_total += len(result.indications)
            phase_dropped_this = 0
            no_match_this = 0
            kept_per_disease: dict[
                str, tuple[str, chembl.ChemblIndication, _MatchOutcome]
            ] = {}

            for ind in result.indications:
                # min_phase gate
                if ind.max_phase_for_ind < self._min_phase:
                    phase_dropped_this += 1
                    continue
                evidence_type = _evidence_type_for_phase(ind.max_phase_for_ind)
                if evidence_type is None:
                    phase_dropped_this += 1
                    continue
                outcome = self._match_indication(ind, index)
                by_match_tier[outcome.match_tier] = (
                    by_match_tier.get(outcome.match_tier, 0) + 1
                )
                if outcome.disease_name is None:
                    no_match_this += 1
                    continue
                # Per-disease max-phase dedup
                key = outcome.disease_name
                if (
                    key not in kept_per_disease
                    or ind.max_phase_for_ind > kept_per_disease[key][1].max_phase_for_ind
                ):
                    kept_per_disease[key] = (evidence_type, ind, outcome)

            indications_dropped_phase += phase_dropped_this
            indications_dropped_no_match += no_match_this

            if not kept_per_disease:
                by_status[LinkStatus.NO_INDICATIONS.value] += 1
                status_rows.append(self._status_row(
                    inchikey, LinkStatus.NO_INDICATIONS,
                    0, phase_dropped_this + no_match_this, attempted_at,
                ))
                delete_rows.append({"inchikey": inchikey})
                continue

            # Linked path: enqueue backfills, edges, status
            by_status[LinkStatus.LINKED.value] += 1
            indications_matched += len(kept_per_disease)
            delete_rows.append({"inchikey": inchikey})
            for et, ind, outcome in kept_per_disease.values():
                by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
                unique_diseases.add(outcome.disease_name or "")

                # Enqueue backfill (dedup by disease_name + ID type)
                if outcome.backfill_mesh:
                    if backfilled_meshes.get(outcome.disease_name) != outcome.backfill_mesh:
                        backfilled_meshes[outcome.disease_name] = outcome.backfill_mesh
                        # Update in-memory index for in-run tier-1 upgrades
                        index.by_mesh_id[outcome.backfill_mesh] = outcome.disease_name
                        index.has_mesh.add(outcome.disease_name)
                        backfills_mesh += 1
                if outcome.backfill_efo:
                    if backfilled_efos.get(outcome.disease_name) != outcome.backfill_efo:
                        backfilled_efos[outcome.disease_name] = outcome.backfill_efo
                        index.has_efo.add(outcome.disease_name)
                        backfills_efo += 1

                edge_rows.append({
                    "inchikey": inchikey,
                    "disease_name": outcome.disease_name,
                    "props": self._edge_props(ind, et, attempted_at),
                })

            status_rows.append(self._status_row(
                inchikey, LinkStatus.LINKED,
                len(kept_per_disease),
                phase_dropped_this + no_match_this,
                attempted_at,
            ))

        # Collapse backfilled_meshes + backfilled_efos into the row list
        # (one row per disease that needs ANY backfill)
        all_backfill_targets = set(backfilled_meshes) | set(backfilled_efos)
        for disease_name in all_backfill_targets:
            backfill_rows.append({
                "name": disease_name,
                "mesh_id": backfilled_meshes.get(disease_name, ""),
                "efo_id": backfilled_efos.get(disease_name, ""),
            })

        return {
            "by_status": by_status,
            "by_evidence_type": by_evidence_type,
            "by_match_tier": by_match_tier,
            "unique_diseases": unique_diseases,
            "indications_total": indications_total,
            "indications_matched": indications_matched,
            "indications_dropped_no_match": indications_dropped_no_match,
            "indications_dropped_phase": indications_dropped_phase,
            "backfills_mesh_id": backfills_mesh,
            "backfills_efo_id": backfills_efo,
            "errors": errors,
            "delete_rows": delete_rows,
            "backfill_rows": backfill_rows,
            "edge_rows": edge_rows,
            "status_rows": status_rows,
        }

    def _match_indication(
        self, ind: chembl.ChemblIndication, index: _DiseaseIndex,
    ) -> _MatchOutcome:
        # Tier 1: mesh_id exact
        if ind.mesh_id and ind.mesh_id in index.by_mesh_id:
            disease = index.by_mesh_id[ind.mesh_id]
            return _MatchOutcome(
                disease_name=disease,
                match_tier="mesh_id",
                backfill_mesh="",
                backfill_efo=(
                    ind.efo_id if ind.efo_id and disease not in index.has_efo
                    else ""
                ),
            )

        # Tier 2: normalized name on mesh_heading or efo_term
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

        # Tier 3: ChEMBL-side MeSH synonym expansion
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

    def _prefetch_synonyms(self, mesh_ids: set[str]) -> None:
        """Warm the synonym cache in parallel before planning.

        OntologyClient is thread-safe (its internal lock guards both
        the rate-limit timestamp and the requests.Session). Workers
        will serialize on the rate-limit gate but not on the actual
        HTTP wait — net wall-clock drops linearly with worker count
        until the rate limiter saturates.
        """
        if not mesh_ids:
            return
        with ThreadPoolExecutor(
            max_workers=_SYNONYM_PREFETCH_WORKERS,
        ) as pool:
            for mid, synonyms in zip(
                mesh_ids,
                pool.map(self._ontology.get_mesh_synonyms, mesh_ids),
            ):
                self._mesh_synonym_cache[mid] = synonyms

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _edge_props(
        self,
        ind: chembl.ChemblIndication,
        evidence_type: str,
        attempted_at: str,
    ) -> dict:
        confidence = CONFIDENCE.get(evidence_type, 0.5)
        props: dict[str, Any] = {
            "confidence_score": confidence,
            "evidence_type": evidence_type,
            "source_db": "ChEMBL",
            "clinical_phase": ind.max_phase_for_ind,
            "drug_indication_id": ind.drug_indication_id,
            "molecule_chembl_id": ind.molecule_chembl_id,
            "created_by": "compound_disease_linker_batched",
            "created_at": attempted_at,
        }
        if ind.mesh_id:
            props["mesh_id"] = ind.mesh_id
        if ind.mesh_heading:
            props["mesh_heading"] = ind.mesh_heading
        if ind.efo_id:
            props["efo_id"] = ind.efo_id
        if ind.efo_term:
            props["efo_term"] = ind.efo_term
        return props

    def _status_row(
        self,
        inchikey: str,
        status: LinkStatus,
        indication_count: int,
        dropped: int,
        attempted_at: str,
    ) -> dict:
        return {
            "inchikey": inchikey,
            "status": status.value,
            "attempted_at": attempted_at,
            "indication_count": indication_count,
            "dropped_count": dropped,
            "min_phase": self._min_phase,
        }

    # ------------------------------------------------------------------
    # Rebuild
    # ------------------------------------------------------------------

    def _wipe_existing(self) -> None:
        """Same scope as the per-compound version: KNOWN_TREATS edges +
        kt_linker_* props on Compounds. Modern_Disease nodes are NOT
        touched."""
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
