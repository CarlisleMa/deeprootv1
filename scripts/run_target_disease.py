#!/usr/bin/env python3
"""Target → Disease RELATES_TO Linker (BATCHED) CLI runner.

Same architectural pattern as run_compound_disease_batched.py — bulk
external-DB fetches + UNWIND-batched Neo4j writes + match-only against
existing Modern_Disease nodes.

Four target-type planning branches (auto-dispatched inside the agent):
  SINGLE PROTEIN  → Open Targets associatedDiseases
  PROTEIN COMPLEX → Subunit fan-out via ChEMBL → OT, max-score per disease
  PROTEIN FAMILY  → SKIP (status: skipped_protein_family)
  ORGANISM        → EFO/DOID via OLS, LLM safety net via Gemini (verified
                    via NLM ICD-10/MeSH/SNOMED)

Usage:
    # Dry-run preview (calls OT + OLS + Gemini, no Neo4j writes)
    python scripts/run_target_disease.py --limit 50

    # Apply on the live graph
    python scripts/run_target_disease.py --write-graph

    # Wipe RELATES_TO + td_linker_* props before linking
    python scripts/run_target_disease.py --write-graph --rebuild

    # Re-process targets with status=no_associations or error
    python scripts/run_target_disease.py --write-graph --retry-misses

    # Ignore td_linker_status entirely
    python scripts/run_target_disease.py --write-graph --force-relink

    # Tighter OT filtering (default 0.2; set 0.4 for moderate+ only)
    python scripts/run_target_disease.py --write-graph --min-score 0.4

    # Smaller Neo4j batches (less blast radius if AuraDB throttles)
    python scripts/run_target_disease.py --write-graph --write-batch-size 100

    # Limit subunit fan-out for PROTEIN COMPLEX (default 10)
    python scripts/run_target_disease.py --write-graph --subunit-cap 5

    # Raw JSON
    python scripts/run_target_disease.py --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase1.target_disease import TargetDiseaseLinker
from src.graph.client import GraphClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _print_summary(summary: dict) -> None:
    mode = "DRY RUN" if summary["dry_run"] else "WRITE MODE"
    rebuild = " + REBUILD" if summary.get("rebuild") else ""
    print(f"\n=== Target→Disease RELATES_TO Linker (BATCHED, {mode}{rebuild}) ===")
    print(f"  Targets processed:        {summary['targets_total']}")
    print(f"  Modern_Disease universe:  {summary['modern_disease_universe']}")
    print(f"  Wall clock total:         {summary['duration_s']}s")
    print(f"  min_score (OT):           {summary['min_score']}")
    print(f"  write_batch_size:         {summary['write_batch_size']}")

    pd = summary.get("phase_durations_s", {})
    if pd:
        print(f"\n  --- phase wall clock ---")
        print(f"    OT bulk SINGLE PROTEIN:   {pd.get('ot_single_protein', 0)}s")
        print(f"    OT bulk PROTEIN COMPLEX:  {pd.get('ot_complex', 0)}s")
        print(f"    ORGANISM (EFO+DOID+LLM):  {pd.get('organism_lookup', 0)}s")
        print(f"    Plan (in-memory):         {pd.get('plan', 0)}s")
        print(f"    Neo4j bulk write:         {pd.get('neo4j_bulk_write', 0)}s")

    print("\n  --- by target_type (input) ---")
    for tt, count in summary.get("by_target_type_input", {}).items():
        if count:
            print(f"    {tt:18s} {count}")

    print("\n  --- by td_linker_status ---")
    for status, count in summary["by_status"].items():
        if count:
            print(f"    {status:30s} {count}")

    print("\n  --- by evidence_type (per edge) ---")
    nonzero = {k: v for k, v in summary["by_evidence_type"].items() if v}
    if not nonzero:
        print("    (none)")
    else:
        for et, count in sorted(nonzero.items(), key=lambda kv: -kv[1]):
            print(f"    {et:38s} {count}")

    print("\n  --- by match_tier (per association) ---")
    for tier, count in summary.get("by_match_tier", {}).items():
        print(f"    {tier:18s} {count}")

    bps = summary.get("by_pathogen_lookup_source", {})
    if bps:
        print("\n  --- by pathogen lookup source (ORGANISM) ---")
        for src, count in bps.items():
            print(f"    {src:18s} {count}")

    print("\n  --- associations ---")
    print(f"    Associations fetched:                 {summary['associations_total']}")
    print(f"    Matched to Modern_Disease:            {summary['associations_matched']}")
    print(f"    Dropped (no in-graph disease):        {summary['associations_dropped_no_match']}")
    print(f"    Dropped (below min_score):            {summary['associations_dropped_score']}")

    print("\n  --- writes ---")
    print(f"    RELATES_TO edge writes:               {summary['edge_writes']}")
    print(f"    Unique Modern_Disease linked:         {summary['unique_diseases_linked']}")
    print(f"    Backfills: efo_id={summary['backfills_efo_id']}  "
          f"mesh_id={summary['backfills_mesh_id']}")

    if summary["errors"]:
        print(f"\n  --- {len(summary['errors'])} error(s) ---")
        for err in summary["errors"][:10]:
            print(f"    {err}")
        if len(summary["errors"]) > 10:
            print(f"    ... and {len(summary['errors']) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BATCHED Target→Disease RELATES_TO linker"
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help="Apply UNWIND-batched RELATES_TO writes to Neo4j (default: "
             "dry-run). NOTE: dry-run STILL calls OT/OLS/Gemini — only "
             "graph writes are skipped. Use --limit when previewing.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe RELATES_TO edges + td_linker_* props before linking. "
             "Idempotent. Does NOT delete Modern_Disease or Biological_Target "
             "nodes. Requires --write-graph.",
    )
    parser.add_argument(
        "--retry-misses", action="store_true",
        help="Also re-process targets with td_linker_status='no_associations' "
             "or 'error'.",
    )
    parser.add_argument(
        "--force-relink", action="store_true",
        help="Ignore td_linker_status entirely; re-process every target. "
             "Combine with --rebuild for a clean slate.",
    )
    parser.add_argument(
        "--min-score", type=float, default=0.2,
        help="Minimum OT overall_score to retain (default 0.2). Set higher "
             "(e.g. 0.4) for stricter associations.",
    )
    parser.add_argument(
        "--write-batch-size", type=int, default=500,
        help="Rows per UNWIND transaction. Default 500.",
    )
    parser.add_argument(
        "--subunit-cap", type=int, default=10,
        help="Max subunits to fan out per PROTEIN COMPLEX target. Default 10.",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Skip the LLM safety net for ORGANISM targets. Targets that "
             "don't get EFO/DOID hits land as no_associations (rather "
             "than burning per-target sequential Gemini calls). Use this "
             "for a fast OT-only pass; follow with --retry-misses after "
             "the LLM phase is parallelized.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N targets (debugging)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON summary instead of pretty output",
    )
    args = parser.parse_args()

    if args.retry_misses and args.force_relink:
        parser.error("--retry-misses and --force-relink are mutually exclusive")
    if args.rebuild and not args.write_graph:
        parser.error(
            "--rebuild requires --write-graph (would have no effect in dry-run)"
        )

    with GraphClient() as client:
        linker = TargetDiseaseLinker(
            client,
            min_score=args.min_score,
            write_batch_size=args.write_batch_size,
            subunit_cap=args.subunit_cap,
            skip_llm=args.skip_llm,
        )
        summary = linker.run(
            dry_run=not args.write_graph,
            rebuild=args.rebuild,
            retry_misses=args.retry_misses,
            force_relink=args.force_relink,
            limit=args.limit,
        )

    if args.json:
        print(json.dumps(summary, indent=2, default=str, ensure_ascii=False))
        return

    _print_summary(summary)
    print("\nDone.")


if __name__ == "__main__":
    main()
