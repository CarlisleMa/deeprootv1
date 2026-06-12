#!/usr/bin/env python3
"""Compound → Disease KNOWN_TREATS Linker — BATCHED runner.

Throughput-optimized variant of run_compound_disease.py. Same semantics
(match-only, three-tier disease matching, per-clinical-phase priors,
self-healing backfill, kt_linker_status resume contract) but bulk
ChEMBL `__in=` fetches + UNWIND-batched Neo4j writes. Targets
~25-50× speedup on full-corpus runs vs the per-compound version.

Usage:
    # Dry-run preview (calls ChEMBL bulk endpoints, no Neo4j writes)
    python scripts/run_compound_disease_batched.py --limit 200

    # Apply on the live graph
    python scripts/run_compound_disease_batched.py --write-graph

    # Wipe KNOWN_TREATS + kt_linker_* props before linking
    python scripts/run_compound_disease_batched.py --write-graph --rebuild

    # Re-process compounds with status=no_indications or error
    python scripts/run_compound_disease_batched.py --write-graph --retry-misses

    # Ignore kt_linker_status entirely
    python scripts/run_compound_disease_batched.py --write-graph --force-relink

    # Tighter write batches (smaller blast radius on crash; more roundtrips)
    python scripts/run_compound_disease_batched.py --write-graph --write-batch-size 100

    # Only FDA-approved indications
    python scripts/run_compound_disease_batched.py --write-graph --min-phase 4

    # Raw JSON
    python scripts/run_compound_disease_batched.py --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase1.compound_disease_batched import CompoundDiseaseLinkerBatched
from src.graph.client import GraphClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _print_summary(summary: dict) -> None:
    mode = "DRY RUN" if summary["dry_run"] else "WRITE MODE"
    rebuild = " + REBUILD" if summary.get("rebuild") else ""
    print(f"\n=== Compound→Disease KNOWN_TREATS Linker (BATCHED, {mode}{rebuild}) ===")
    print(f"  Compounds processed:        {summary['compounds_total']}")
    print(f"  Modern_Disease universe:    {summary['modern_disease_universe']}")
    print(f"  Wall clock total:           {summary['duration_s']}s")
    print(f"  min_phase:                  {summary['min_phase']}")
    print(f"  write_batch_size:           {summary['write_batch_size']}")

    pd = summary.get("phase_durations_s", {})
    if pd:
        print(f"\n  --- phase wall clock ---")
        print(f"    ChEMBL bulk fetch:    {pd.get('chembl_bulk_fetch', 0)}s")
        print(f"    Synonym prefetch:     {pd.get('synonym_prefetch', 0)}s")
        print(f"    Plan (in-memory):     {pd.get('plan', 0)}s")
        print(f"    Neo4j bulk write:     {pd.get('neo4j_bulk_write', 0)}s")

    print("\n  --- by kt_linker_status ---")
    for status, count in summary["by_status"].items():
        if count:
            print(f"    {status:25s} {count}")

    print("\n  --- by evidence_type (per edge) ---")
    nonzero = {k: v for k, v in summary["by_evidence_type"].items() if v}
    if not nonzero:
        print("    (none)")
    else:
        for et, count in sorted(nonzero.items(), key=lambda kv: -kv[1]):
            print(f"    {et:32s} {count}")

    print("\n  --- by match_tier (per indication) ---")
    for tier, count in summary.get("by_match_tier", {}).items():
        print(f"    {tier:18s} {count}")

    print("\n  --- indications ---")
    print(f"    Indications fetched:                 {summary['indications_total']}")
    print(f"    Matched to Modern_Disease:           {summary['indications_matched']}")
    print(f"    Dropped (no matching disease):       {summary['indications_dropped_no_match']}")
    print(f"    Dropped (below min_phase):           {summary['indications_dropped_phase']}")

    print("\n  --- writes ---")
    print(f"    KNOWN_TREATS edge writes:            {summary['edge_writes']}")
    print(f"    Unique Modern_Disease linked:        {summary['unique_diseases_linked']}")
    print(f"    Backfills: mesh_id={summary['backfills_mesh_id']}  "
          f"efo_id={summary['backfills_efo_id']}")

    if summary["errors"]:
        print(f"\n  --- {len(summary['errors'])} error(s) ---")
        for err in summary["errors"][:10]:
            print(f"    {err}")
        if len(summary["errors"]) > 10:
            print(f"    ... and {len(summary['errors']) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the BATCHED Compound→Disease KNOWN_TREATS linker"
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help="Apply UNWIND-batched KNOWN_TREATS writes to Neo4j (default: "
             "dry-run). NOTE: dry-run STILL calls the ChEMBL bulk endpoints "
             "— only graph writes are skipped. Use --limit when previewing.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe KNOWN_TREATS edges + kt_linker_* props before linking. "
             "Idempotent — safe on an empty graph. Does NOT delete "
             "Modern_Disease nodes (those belong to MaladyDiseaseMapper). "
             "Requires --write-graph.",
    )
    parser.add_argument(
        "--retry-misses", action="store_true",
        help="Also re-process compounds with kt_linker_status="
             "'no_indications' or 'error'.",
    )
    parser.add_argument(
        "--force-relink", action="store_true",
        help="Ignore kt_linker_status entirely; re-process every compound. "
             "Combine with --rebuild for a clean slate.",
    )
    parser.add_argument(
        "--min-phase", type=int, default=1,
        help="Drop indications below this clinical phase. Default 1 "
             "(phase 0 / preclinical dropped).",
    )
    parser.add_argument(
        "--write-batch-size", type=int, default=500,
        help="Rows per UNWIND transaction. Default 500. Smaller values "
             "reduce blast radius on crash; larger values reduce Neo4j "
             "round-trip count.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N compounds (debugging)",
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
        linker = CompoundDiseaseLinkerBatched(
            client,
            min_phase=args.min_phase,
            write_batch_size=args.write_batch_size,
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
