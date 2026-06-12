"""Phase 1 — Reviewer Agent: post-pipeline graph quality pass.

The reviewer runs *after* every per-row Phase 1 agent (extraction →
auditor → linkers → mapper) and is the only agent that looks at the
whole graph rather than walking individual rows. Its job is to enforce
graph-level invariants the per-row agents can't see, using the status
flags those agents already wrote.

Pure deterministic — no LLM, no hand-curated word lists. Every decision
is grounded in graph properties produced by upstream agents.

Phases (each is independently runnable; default runs all):

  1. orphan_malady_dedup
     A "Malady orphan" is a Traditional_Malady with mapper_status="linked"
     (the mapper ran and wrote a MAPS_TO edge) AND no incoming
     TREATS_TRADITIONALLY edge (no Source claims to treat it). These
     exist because the extraction LLM emitted the malady in `maladies[]`
     but referenced a slightly different name (case/qualifier variant)
     in `relationships[]` for the same passage — one passage produces
     two Malady nodes; only one gets a TREATS edge.

     Dedup key: the *primary* MAPS_TO Modern_Disease. Two maladies that
     were independently mapped onto the same Modern_Disease are the same
     concept (the malady_disease mapper LLM agreed on the mapping twice
     — strong cross-validation). Action: archive the orphan as
     `merged_into:<keeper>` and cascade-archive its outgoing MAPS_TO
     edges. Same shape as the auditor's Source-merge logic in
     extraction_auditor.py, just keyed on Modern_Disease equality
     instead of canonical_name+canonical_part.

     Orphans with NO claimed sibling for their Modern_Disease are left
     alone for now — they'll be picked up by a future "dead-end
     detection" phase.

Idempotency: every phase guards on `archived IS NULL OR archived = false`
so re-running on a partially-applied graph is a no-op for already-handled
nodes. No status-flag-on-Reviewer for now — phases are cheap enough to
re-walk every run.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base import BaseAgent
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase 1: orphan-Malady dedup
# ---------------------------------------------------------------------------

# Find every orphan + its primary Modern_Disease + every claimed sibling
# sharing that disease, in one round-trip. Per-orphan keeper selection is
# done client-side so tiebreakers can be applied without nested Cypher.
ORPHAN_MALADY_PLAN_QUERY = """
MATCH (orphan:Traditional_Malady)-[r_o:MAPS_TO]->(d:Modern_Disease)
WHERE (orphan.archived IS NULL OR orphan.archived = false)
  AND orphan.mapper_status = 'linked'
  AND (r_o.is_primary = true OR r_o.is_primary IS NULL)
  AND (d.archived IS NULL OR d.archived = false)
  AND NOT EXISTS { (:Source)-[:TREATS_TRADITIONALLY]->(orphan) }
OPTIONAL MATCH (claimed:Traditional_Malady)-[r_c:MAPS_TO]->(d)
  WHERE claimed.name <> orphan.name
    AND (claimed.archived IS NULL OR claimed.archived = false)
    AND (r_c.is_primary = true OR r_c.is_primary IS NULL)
    AND EXISTS { (:Source)-[:TREATS_TRADITIONALLY]->(claimed) }
WITH orphan, d, claimed
OPTIONAL MATCH (:Source)-[r_tt:TREATS_TRADITIONALLY]->(claimed)
WITH orphan, d, claimed, count(r_tt) AS keeper_in_degree
RETURN orphan.name AS orphan_name,
       d.name AS modern_disease,
       collect({name: claimed.name, in_degree: keeper_in_degree}) AS candidates
ORDER BY orphan.name
"""

ARCHIVE_ORPHAN_NODE = """
MATCH (m:Traditional_Malady {name: $orphan})
WHERE m.archived IS NULL OR m.archived = false
SET m.archived = true,
    m.archive_reason = 'merged_into:' + $keeper,
    m.merged_canonical_target = $modern_disease,
    m.reviewed_by = 'reviewer_agent'
"""

ARCHIVE_ORPHAN_EDGES = """
MATCH (m:Traditional_Malady {name: $orphan})-[r:MAPS_TO]->(:Modern_Disease)
WHERE r.archived IS NULL OR r.archived = false
SET r.archived = true,
    r.archive_reason = 'source_node_archived'
RETURN count(r) AS archived_edges
"""


def _pick_orphan_malady_keeper(candidates: list[dict]) -> str | None:
    """Highest TREATS_TRADITIONALLY in-degree wins; alphabetical tiebreak.

    The keeper choice mostly affects display — every TREATS edge already
    points to the keeper (orphans have none by definition), and Phase 2
    traverses MAPS_TO to Modern_Disease. So degree-max anchors the
    surviving node where the most evidence already lives.
    """
    real = [c for c in candidates if c.get("name")]
    if not real:
        return None
    real.sort(key=lambda c: (-(c.get("in_degree") or 0), c["name"]))
    return real[0]["name"]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReviewerAgent(BaseAgent):
    """Post-pipeline graph quality pass. Deterministic, multi-phase."""

    def __init__(self, client: GraphClient, **kwargs: Any):
        super().__init__(client, **kwargs)

    @property
    def name(self) -> str:
        return "ReviewerAgent"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        write_graph: bool = False,
        limit: int = 0,
        phases: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Execute review phases.

        Args:
            write_graph: If False (default), build the plan but don't
                write — every phase respects this. If True, apply.
            limit: Per-phase cap on the number of actionable items
                (0 = no cap). Useful for spot-checking.
            phases: Subset of phases to run. None or empty → all.
                Currently only "orphan_malady_dedup" is implemented;
                future phases will slot in here.

        Returns:
            {phase_name: phase_summary_dict, ...}
        """
        all_phases = ["orphan_malady_dedup"]
        run_phases = phases or all_phases

        results: dict[str, Any] = {"mode": "WRITE" if write_graph else "DRY-RUN"}

        if "orphan_malady_dedup" in run_phases:
            self._log_progress("Phase: orphan_malady_dedup")
            results["orphan_malady_dedup"] = self._phase_orphan_malady_dedup(
                write_graph=write_graph, limit=limit,
            )

        return results

    # ------------------------------------------------------------------
    # Phase 1: orphan-Malady dedup
    # ------------------------------------------------------------------

    def _phase_orphan_malady_dedup(
        self, *, write_graph: bool, limit: int,
    ) -> dict:
        plan = self._build_orphan_malady_plan()
        actionable = [p for p in plan if p["keeper"]]
        skipped = [p for p in plan if not p["keeper"]]

        if limit > 0:
            actionable = actionable[:limit]

        applied: list[dict] = []
        if write_graph and actionable:
            for item in actionable:
                edges = self._apply_orphan_malady_dedup(item)
                applied.append({**item, "archived_edges": edges})
            self._log_progress(
                f"Archived {len(applied)} orphan Malady node(s); "
                f"{sum(a['archived_edges'] for a in applied)} MAPS_TO edge(s) cascaded"
            )

        return self._summarize_orphan_malady(plan, actionable, applied, skipped)

    def _build_orphan_malady_plan(self) -> list[dict]:
        rows = self.client.run(ORPHAN_MALADY_PLAN_QUERY)
        plan: list[dict] = []
        for row in rows:
            keeper = _pick_orphan_malady_keeper(row.get("candidates") or [])
            plan.append({
                "orphan": row["orphan_name"],
                "modern_disease": row["modern_disease"],
                "keeper": keeper,
                "candidates": [
                    c["name"]
                    for c in (row.get("candidates") or [])
                    if c.get("name")
                ],
            })
        return plan

    def _apply_orphan_malady_dedup(self, item: dict) -> int:
        """Archive the orphan + cascade-archive its outgoing MAPS_TO edges.

        Returns the number of MAPS_TO edges archived. Idempotent — both
        writes guard on `archived IS NULL OR archived = false`.
        """
        params = {
            "orphan": item["orphan"],
            "keeper": item["keeper"],
            "modern_disease": item["modern_disease"],
        }
        self.client.run_write(ARCHIVE_ORPHAN_NODE, params)
        rows = self.client.run_write(ARCHIVE_ORPHAN_EDGES, params)
        return rows[0]["archived_edges"] if rows else 0

    @staticmethod
    def _summarize_orphan_malady(
        plan: list[dict],
        actionable: list[dict],
        applied: list[dict],
        skipped: list[dict],
    ) -> dict:
        by_disease: dict[str, int] = {}
        for p in actionable:
            d = p["modern_disease"]
            by_disease[d] = by_disease.get(d, 0) + 1

        return {
            "orphans_total": len(plan),
            "actionable": len(actionable),
            "applied": len(applied),
            "no_keeper": len(skipped),
            "edges_cascaded": sum(a.get("archived_edges", 0) for a in applied),
            "by_disease": dict(sorted(by_disease.items(), key=lambda kv: -kv[1])),
            "no_keeper_orphans": [p["orphan"] for p in skipped],
            "plan": actionable,
        }
