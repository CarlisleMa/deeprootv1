"""Phase 1 — Compound → Target Linking Agent (v2).

Pure routing layer over ChEMBL. Mirrors the v2 SourceCompoundLinker
shape: no LLM in the loop, idempotent via per-node `target_linker_status`,
flat per-evidence-type confidence priors, crash-safe write order.

Pipeline (per Chemical_Compound):

  STEP A: LOOKUP  (ChEMBL — InChIKey → SMILES → name priority, paginated)
    - mechanisms (curated clinical drug-target relationships, any allowed type)
    - activities (filtered to allowed_assay_types + standard_relation in
      {"=", "~"} + data_validity_comment empty)
    - target_types accepted: SINGLE PROTEIN, PROTEIN COMPLEX,
      PROTEIN FAMILY, ORGANISM (organism is critical for parasite /
      antimicrobial evidence — many TCM remedies treat infections)

  STEP B: NORMALIZE
    - Biological_Target identity = target_chembl_id (always present, works
      for protein and non-protein targets equally)
    - Per-tier confidence prior:
        chembl_mechanism             0.95
        chembl_activity_strong  >=7  0.75
        chembl_activity_moderate >=6 0.60
        chembl_activity_weak    >=5  0.40
    - Cap at MAX_TARGETS_PER_COMPOUND (default 20), top by pchembl

  STEP C: WRITE  (MERGE-only; status set LAST for crash safety)
    1. (force_relink only) DELETE prior TARGETS edges for this compound
    2. MERGE Biological_Target by target_chembl_id with coalesce-style
       backfill on uniprot_id / gene_symbol / ncbi_tax_id / target_type / name
    3. MERGE TARGETS edge with confidence + evidence_type + pchembl +
       assay_id + assay_type + assay_description
    4. SET target_linker_status / linker_attempted_at / linker_target_count
       / linker_dropped_count / linker_chembl_id / linker_lookup_method /
       linker_pchembl_floor / linker_max_targets on the Compound (LAST)

Failure handling:
  - HTTP 404 / "molecule not in ChEMBL"  → status="no_targets_found"
  - Network error / 5xx mid-flight       → status="error" (re-tried via
                                            --retry-misses)
  - Per-compound exception in pipeline   → status="error"
  This avoids the v1 bug of recording transient failures as permanent
  no_targets_found, which silently mis-labelled compounds during outages.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from typing import Any

from src.agents.base import BaseAgent
from src.data import chembl
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class LinkStatus(str, Enum):
    LINKED = "linked"
    NO_TARGETS_FOUND = "no_targets_found"
    ERROR = "error"


# Flat per-evidence-type confidence priors. Keyed on the `evidence_type`
# string written to each TARGETS edge so Phase 2 scoring can look up the
# prior directly.
CONFIDENCE: dict[str, float] = {
    "chembl_mechanism":          0.95,
    "chembl_activity_strong":    0.75,
    "chembl_activity_moderate":  0.60,
    "chembl_activity_weak":      0.40,
    # Phenotypic = qualitative bioactivity in functional / organism /
    # whole-cell assays where ChEMBL didn't compute a potency.
    # Same tier as activity_weak (0.40) — lower than quantitative
    # weak binding because we have no IC50 to anchor the strength,
    # but still grounded in real measured biology (parasite killing,
    # antimicrobial activity, etc.). Distinct evidence_type tag lets
    # downstream consumers filter independently.
    "chembl_phenotypic":         0.40,
}

_PCHEMBL_STRONG = 7.0      # <= 100 nM — drug-like potency
_PCHEMBL_MODERATE = 6.0    # <= 1 μM   — meaningful binding
_PCHEMBL_WEAK = 5.0        # <= 10 μM  — written; downstream filter via prior

_DEFAULT_PCHEMBL_FLOOR = 5.0
_DEFAULT_MAX_TARGETS = 20
_DEFAULT_WORKERS = 8


def _evidence_type_for(target: chembl.ChemblTarget) -> str:
    """Map a ChemblTarget to the evidence_type label used for confidence
    lookup + downstream scoring."""
    if target.evidence_type == "mechanism":
        return "chembl_mechanism"
    # `pchembl_score is None` is the explicit signal that this came from
    # the phenotypic-activity path (qualitative — no quantitative potency
    # was computed). 0.0 vs None matters: a quantitative measurement of
    # exactly 0.0 is impossible, so None unambiguously means phenotypic.
    if target.pchembl_score is None:
        return "chembl_phenotypic"
    p = target.pchembl_score
    if p >= _PCHEMBL_STRONG:
        return "chembl_activity_strong"
    if p >= _PCHEMBL_MODERATE:
        return "chembl_activity_moderate"
    return "chembl_activity_weak"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class CompoundTargetLinker(BaseAgent):
    """Route Chemical_Compound nodes to ChEMBL → write Biological_Target +
    TARGETS edges."""

    def __init__(
        self,
        client: GraphClient,
        *,
        pchembl_floor: float = _DEFAULT_PCHEMBL_FLOOR,
        max_targets_per_compound: int = _DEFAULT_MAX_TARGETS,
        workers: int = _DEFAULT_WORKERS,
        allowed_target_types: tuple[str, ...] = chembl.DEFAULT_ALLOWED_TARGET_TYPES,
        allowed_assay_types: tuple[str, ...] = chembl.DEFAULT_ALLOWED_ASSAY_TYPES,
        include_phenotypic: bool = False,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._pchembl_floor = pchembl_floor
        self._max_targets = max_targets_per_compound
        self._workers = workers
        self._allowed_target_types = allowed_target_types
        self._allowed_assay_types = allowed_assay_types
        self._include_phenotypic = include_phenotypic
        self._force_relink = False  # set per-run via run()

    @property
    def name(self) -> str:
        return "CompoundTargetLinker"

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

        cypher = queries.LINKABLE_COMPOUNDS_FOR_TARGETS
        if force_relink:
            cypher = queries.LINKABLE_COMPOUNDS_FORCE
        elif retry_misses:
            cypher = queries.LINKABLE_COMPOUNDS_RETRY

        compounds = self.client.run(cypher)
        if limit is not None:
            compounds = compounds[:limit]

        total = len(compounds)
        self._log_progress(
            f"Linking {total} compound(s) "
            f"(dry_run={dry_run}, rebuild={rebuild}, "
            f"retry_misses={retry_misses}, force_relink={force_relink}, "
            f"pchembl_floor={self._pchembl_floor}, "
            f"max_targets={self._max_targets}, workers={self._workers}, "
            f"target_types={list(self._allowed_target_types)}, "
            f"assay_types={list(self._allowed_assay_types)}, "
            f"include_phenotypic={self._include_phenotypic})"
        )

        t_start = time.time()
        by_status: dict[str, int] = {s.value: 0 for s in LinkStatus}
        by_evidence_type: dict[str, int] = {k: 0 for k in CONFIDENCE}
        by_target_type: dict[str, int] = {}
        by_lookup_method: dict[str, int] = {}
        by_chembl_outcome: dict[str, int] = {}
        edge_writes = 0
        unique_targets: set[str] = set()
        targets_per_compound: list[int] = []
        targets_dropped_total = 0
        errors: list[str] = []

        # Submit all ChEMBL lookups concurrently; iterate as they complete.
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futs = {
                pool.submit(
                    chembl.fetch_compound_data,
                    {
                        "inchikey": c["inchikey"],
                        "name": c["name"] or "",
                        "smiles": c["smiles"] or "",
                    },
                    pchembl_floor=self._pchembl_floor,
                    allowed_target_types=self._allowed_target_types,
                    allowed_assay_types=self._allowed_assay_types,
                    include_phenotypic=self._include_phenotypic,
                ): c
                for c in compounds
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
                            target_count=0, dropped=0,
                            chembl_id="", lookup_method="exception",
                        )
                    continue

                by_lookup_method[result.lookup_method] = (
                    by_lookup_method.get(result.lookup_method, 0) + 1
                )
                by_chembl_outcome[result.outcome] = (
                    by_chembl_outcome.get(result.outcome, 0) + 1
                )

                # Discriminated outcome: transient failure ≠ permanent miss
                if result.outcome == "transient_failure":
                    by_status[LinkStatus.ERROR.value] += 1
                    if result.error:
                        errors.append(
                            f"{compound_row.get('name', '?')}: {result.error}"
                        )
                    if not dry_run:
                        # Force-relink must wipe stale targets EVEN for
                        # error / not_in_chembl outcomes — otherwise the
                        # old edges leak through unchanged.
                        self._maybe_delete_prior_targets(compound_row["inchikey"])
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.ERROR,
                            target_count=0, dropped=0,
                            chembl_id=result.chembl_id,
                            lookup_method=result.lookup_method,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                if result.outcome == "not_in_chembl" or not result.targets:
                    by_status[LinkStatus.NO_TARGETS_FOUND.value] += 1
                    targets_per_compound.append(0)
                    if not dry_run:
                        self._maybe_delete_prior_targets(compound_row["inchikey"])
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.NO_TARGETS_FOUND,
                            target_count=0, dropped=0,
                            chembl_id=result.chembl_id,
                            lookup_method=result.lookup_method,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                # Apply the cap. Sort: mechanisms first (strongest), then
                # activities by pchembl desc.
                sorted_targets = sorted(
                    result.targets,
                    key=lambda t: (
                        0 if t.evidence_type == "mechanism" else 1,
                        -(t.pchembl_score or 0.0),
                    ),
                )
                kept = sorted_targets[: self._max_targets]
                dropped = max(0, len(sorted_targets) - self._max_targets)
                targets_dropped_total += dropped

                if not kept:
                    by_status[LinkStatus.NO_TARGETS_FOUND.value] += 1
                    targets_per_compound.append(0)
                    if not dry_run:
                        self._persist_status(
                            compound_row["inchikey"],
                            LinkStatus.NO_TARGETS_FOUND,
                            target_count=0, dropped=dropped,
                            chembl_id=result.chembl_id,
                            lookup_method=result.lookup_method,
                        )
                    self._maybe_log_progress(done, total, by_status)
                    continue

                # Tally
                by_status[LinkStatus.LINKED.value] += 1
                targets_per_compound.append(len(kept))
                for t in kept:
                    et = _evidence_type_for(t)
                    by_evidence_type[et] = by_evidence_type.get(et, 0) + 1
                    by_target_type[t.target_type] = (
                        by_target_type.get(t.target_type, 0) + 1
                    )
                    unique_targets.add(t.target_chembl_id)
                edge_writes += len(kept)

                if not dry_run:
                    self._write_compound_targets(
                        compound_row["inchikey"], kept,
                    )
                    self._persist_status(
                        compound_row["inchikey"],
                        LinkStatus.LINKED,
                        target_count=len(kept),
                        dropped=dropped,
                        chembl_id=result.chembl_id,
                        lookup_method=result.lookup_method,
                    )

                self._maybe_log_progress(done, total, by_status)

        duration_s = round(time.time() - t_start, 2)

        return {
            "dry_run": dry_run,
            "rebuild": rebuild,
            "pchembl_floor": self._pchembl_floor,
            "max_targets_per_compound": self._max_targets,
            "allowed_target_types": list(self._allowed_target_types),
            "allowed_assay_types": list(self._allowed_assay_types),
            "include_phenotypic": self._include_phenotypic,
            "compounds_total": total,
            "by_status": by_status,
            "by_evidence_type": by_evidence_type,
            "by_target_type": dict(
                sorted(by_target_type.items(), key=lambda kv: -kv[1])
            ),
            "by_lookup_method": dict(
                sorted(by_lookup_method.items(), key=lambda kv: -kv[1])
            ),
            "by_chembl_outcome": dict(
                sorted(by_chembl_outcome.items(), key=lambda kv: -kv[1])
            ),
            "edge_writes": edge_writes,
            "unique_target_chembl_ids": len(unique_targets),
            "targets_dropped_to_cap": targets_dropped_total,
            "targets_per_linked_compound": {
                "p50": _percentile(targets_per_compound, 50),
                "p75": _percentile(targets_per_compound, 75),
                "p95": _percentile(targets_per_compound, 95),
                "max": max(targets_per_compound) if targets_per_compound else 0,
            },
            "duration_s": duration_s,
            "errors": errors,
        }

    def _maybe_log_progress(
        self, done: int, total: int, by_status: dict[str, int],
    ) -> None:
        if done % 100 == 0 or done == total:
            self._log_progress(
                f"  {done}/{total}  linked={by_status['linked']} "
                f"no_targets={by_status['no_targets_found']} "
                f"errors={by_status['error']}"
            )

    # ------------------------------------------------------------------
    # Step C — graph writes
    # ------------------------------------------------------------------

    def _write_compound_targets(
        self, compound_inchikey: str, targets: list[chembl.ChemblTarget],
    ) -> None:
        """Write all kept targets as a unit. Order: per-compound DELETE
        (force_relink only) → MERGE Biological_Target → MERGE TARGETS edge.
        Status update is _persist_status, called separately AFTER this so
        partial failures are self-healing."""
        self._maybe_delete_prior_targets(compound_inchikey)
        for t in targets:
            self._merge_target_node(t)
            self._merge_targets_edge(compound_inchikey, t)

    def _maybe_delete_prior_targets(self, compound_inchikey: str) -> None:
        """When --force-relink re-touches a compound, delete its prior
        TARGETS edges before writing the new plan. Called from EVERY
        terminal-status branch (linked / no_targets_found / error) — not
        just the linked path — so a relink that now resolves to zero
        targets correctly clears the stale edges, instead of leaving
        them attached. (v2 bug fix.)"""
        if not self._force_relink:
            return
        self.client.run_write(
            "MATCH (c:Chemical_Compound {inchikey: $inchikey})"
            "-[r:TARGETS]->() DELETE r",
            {"inchikey": compound_inchikey},
        )

    def _merge_target_node(self, t: chembl.ChemblTarget) -> None:
        """MERGE Biological_Target by target_chembl_id, backfilling
        missing display fields without overwriting existing ones."""
        # Display name: gene symbol > pref_name > target_chembl_id
        display_name = t.gene_symbol or t.target_name or t.target_chembl_id

        backfill: dict[str, str] = {
            "name": display_name,
            "target_type": t.target_type or "",
            "target_pref_name": t.target_name or "",
            "uniprot_id": t.uniprot_id or "",
            "gene_symbol": t.gene_symbol or "",
            "ncbi_tax_id": t.ncbi_tax_id or "",
        }
        backfill = {k: v for k, v in backfill.items() if v}

        coalesce_clauses = ", ".join(
            f"d.{k} = CASE WHEN d.{k} IS NULL OR d.{k} = '' "
            f"THEN $backfill.{k} ELSE d.{k} END"
            for k in backfill
        )
        coalesce_set = f"SET {coalesce_clauses}" if coalesce_clauses else ""

        self.client.run_write(
            f"""
            MERGE (d:Biological_Target {{target_chembl_id: $target_chembl_id}})
            ON CREATE SET d.created_by = $created_by,
                          d.created_at = $created_at
            {coalesce_set}
            """,
            {
                "target_chembl_id": t.target_chembl_id,
                "created_by": "compound_target_linker",
                "created_at": dt.datetime.utcnow().isoformat(),
                "backfill": backfill,
            },
        )

    def _merge_targets_edge(
        self, compound_inchikey: str, t: chembl.ChemblTarget,
    ) -> None:
        evidence_type = _evidence_type_for(t)
        confidence = CONFIDENCE.get(evidence_type, 0.5)
        rel_props: dict[str, Any] = {
            "confidence_score": confidence,
            "evidence_type": evidence_type,
            "source_db": "ChEMBL",
            "assay_id": t.assay_id or "",
            "target_chembl_id": t.target_chembl_id or "",
            "target_type": t.target_type or "",
            "created_by": "compound_target_linker",
            "created_at": dt.datetime.utcnow().isoformat(),
        }
        if t.pchembl_score is not None:
            rel_props["pchembl_score"] = t.pchembl_score
        if t.assay_type:
            rel_props["assay_type"] = t.assay_type
        if t.assay_description:
            rel_props["assay_description"] = t.assay_description
        if t.mechanism_action:
            rel_props["mechanism_action"] = t.mechanism_action

        self.client.merge_edge(
            "Chemical_Compound", {"inchikey": compound_inchikey},
            "Biological_Target", {"target_chembl_id": t.target_chembl_id},
            "TARGETS",
            rel_props,
            from_key="inchikey",
            to_key="target_chembl_id",
        )

    def _persist_status(
        self,
        compound_inchikey: str,
        status: LinkStatus,
        *,
        target_count: int,
        dropped: int,
        chembl_id: str,
        lookup_method: str,
    ) -> None:
        """Set status + audit-trail props on the Compound. Always called
        LAST in write_outcome — if any node/edge MERGE failed earlier,
        status stays NULL and the next default run picks up the compound.

        Audit props include the run-config (pchembl floor, max targets)
        AND per-compound resolution metadata (chembl_id, lookup_method)
        so a future debugging session can reconstruct what happened
        without re-running ChEMBL.
        """
        props: dict[str, Any] = {
            "target_linker_status": status.value,
            "linker_attempted_at": dt.datetime.utcnow().isoformat(),
            "linker_target_count": target_count,
            # Always set so it doesn't go stale across runs (was a v1 bug:
            # only set when truthy, leaving stale counts after a re-run
            # that dropped zero).
            "linker_dropped_count": dropped,
            "linker_pchembl_floor": self._pchembl_floor,
            "linker_max_targets": self._max_targets,
            "linker_lookup_method": lookup_method or "",
            "linker_chembl_id": chembl_id or "",
        }
        self.client.run_write(
            "MATCH (c:Chemical_Compound {inchikey: $inchikey}) SET c += $props",
            {"inchikey": compound_inchikey, "props": props},
        )

    # ------------------------------------------------------------------
    # Rebuild helper
    # ------------------------------------------------------------------

    def _wipe_existing(self) -> None:
        """Idempotent clean slate: drop all TARGETS edges, all
        Biological_Target nodes, and clear linker_* properties on
        Compounds. Safe to run on an empty graph."""
        self._log_progress(
            "Rebuild: wiping TARGETS + Biological_Target + linker_* props"
        )
        self.client.run_write("MATCH ()-[r:TARGETS]->() DELETE r")
        self.client.run_write("MATCH (t:Biological_Target) DETACH DELETE t")
        self.client.run_write("""
            MATCH (c:Chemical_Compound)
            REMOVE c.target_linker_status,
                   c.linker_attempted_at,
                   c.linker_target_count,
                   c.linker_dropped_count,
                   c.linker_pchembl_floor,
                   c.linker_max_targets,
                   c.linker_lookup_method,
                   c.linker_chembl_id
        """)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return s[idx]
