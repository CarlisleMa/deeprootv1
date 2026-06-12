#!/usr/bin/env python3
"""Compound → Target linker (v2) CLI runner.

Pure routing layer over ChEMBL. Reads each Chemical_Compound's InChIKey
(SMILES + name as fallbacks), looks up qualifying biological targets
(mechanisms + activities above the pchembl floor), and writes
Biological_Target nodes (keyed by ChEMBL target_chembl_id, with uniprot_id
/ ncbi_tax_id as informational properties) plus TARGETS edges with flat
per-evidence-type confidence priors.

Accepts four target types: SINGLE PROTEIN, PROTEIN COMPLEX, PROTEIN
FAMILY, ORGANISM. ORGANISM is included specifically to capture
parasite/microbe evidence (e.g. antimalarial assays against Plasmodium)
that's load-bearing for TCM corpora.

Usage:
    # Dry-run (default — no graph writes; STILL calls ChEMBL)
    python scripts/run_compound_target.py --limit 5

    # Apply on the live graph
    python scripts/run_compound_target.py --write-graph

    # Wipe TARGETS + Biological_Target + linker_* props before linking
    # (combine with --write-graph for a clean rebuild)
    python scripts/run_compound_target.py --write-graph --rebuild

    # Re-process compounds that previously found no targets or errored
    python scripts/run_compound_target.py --write-graph --retry-misses

    # Ignore target_linker_status entirely
    python scripts/run_compound_target.py --write-graph --force-relink

    # Tuning knobs
    python scripts/run_compound_target.py --pchembl-floor 7.0    # stricter
    python scripts/run_compound_target.py --include-weak         # equiv to floor=5.0
    python scripts/run_compound_target.py --max-targets 10       # tighter cap
    python scripts/run_compound_target.py --workers 12           # default 8

    # Raw JSON
    python scripts/run_compound_target.py --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase1.compound_target import CompoundTargetLinker
from src.graph.client import GraphClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _print_summary(summary: dict) -> None:
    mode = "DRY RUN" if summary["dry_run"] else "WRITE MODE"
    rebuild = " + REBUILD" if summary.get("rebuild") else ""
    print(f"\n=== Compound→Target Linker v2 ({mode}{rebuild}) ===")
    print(f"  Compounds processed:  {summary['compounds_total']}")
    print(f"  Wall clock:           {summary['duration_s']}s")
    print(f"  pchembl floor:        {summary['pchembl_floor']}")
    print(f"  max targets/compound: {summary['max_targets_per_compound']}")
    print(f"  target types:         {summary['allowed_target_types']}")
    print(f"  assay types:          {summary['allowed_assay_types']}")
    print(f"  include phenotypic:   {summary.get('include_phenotypic', False)}")

    print("\n  --- by target_linker_status ---")
    for status, count in summary["by_status"].items():
        if count:
            print(f"    {status:25s} {count}")

    print("\n  --- by evidence_type (per edge) ---")
    nonzero = {k: v for k, v in summary["by_evidence_type"].items() if v}
    if not nonzero:
        print("    (none)")
    else:
        for et, count in sorted(nonzero.items(), key=lambda kv: -kv[1]):
            print(f"    {et:30s} {count}")

    print("\n  --- by target_type (per edge) ---")
    for tt, count in summary.get("by_target_type", {}).items():
        print(f"    {tt:18s} {count}")

    print("\n  --- by ChEMBL lookup_method ---")
    for method, count in summary["by_lookup_method"].items():
        print(f"    {method:15s} {count}")

    print("\n  --- by ChEMBL outcome ---")
    for outcome, count in summary.get("by_chembl_outcome", {}).items():
        print(f"    {outcome:25s} {count}")

    print("\n  --- targets ---")
    print(f"    TARGETS edge writes:                 {summary['edge_writes']}")
    print(f"    Unique Biological_Target (ChEMBL):   {summary['unique_target_chembl_ids']}")
    print(f"    Targets dropped to cap:              {summary['targets_dropped_to_cap']}")
    dist = summary["targets_per_linked_compound"]
    print(
        f"    Per-linked-compound: p50={dist['p50']}  p75={dist['p75']}  "
        f"p95={dist['p95']}  max={dist['max']}"
    )

    if summary["errors"]:
        print(f"\n  --- {len(summary['errors'])} error(s) ---")
        for err in summary["errors"][:10]:
            print(f"    {err}")
        if len(summary["errors"]) > 10:
            print(f"    ... and {len(summary['errors']) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v2 Compound→Target linker"
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help="Apply Biological_Target + TARGETS writes to Neo4j (default: "
             "dry-run). NOTE: dry-run STILL calls ChEMBL — only graph "
             "writes are skipped. Use --limit when previewing.",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="Wipe TARGETS edges + Biological_Target nodes + linker_* "
             "props before linking. Idempotent — safe on an empty graph. "
             "Requires --write-graph.",
    )
    parser.add_argument(
        "--retry-misses", action="store_true",
        help="Also re-process compounds with target_linker_status="
             "'no_targets_found' or 'error' (default: skip them)",
    )
    parser.add_argument(
        "--force-relink", action="store_true",
        help="Ignore target_linker_status entirely; re-process every compound",
    )
    parser.add_argument(
        "--pchembl-floor", type=float, default=5.0,
        help="Minimum pchembl_value for activity-tier edges. Default 5.0 "
             "(<= 10 μM, all tiers). The per-edge `evidence_type` "
             "(weak/moderate/strong) preserves the strength so downstream "
             "queries can filter via confidence_score >= 0.5 to drop the "
             "weak tier without losing the data. Use 7.0 to skip everything "
             "below drug-like potency at write time.",
    )
    parser.add_argument(
        "--max-targets", type=int, default=20,
        help="Cap on TARGETS edges per compound (top by pchembl, mechanisms "
             "first). Default 20.",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="ChEMBL parallel-lookup worker count. Default 8.",
    )
    parser.add_argument(
        "--protein-only", action="store_true",
        help="Restrict allowed_target_types to {SINGLE PROTEIN}. Default: "
             "{SINGLE PROTEIN, PROTEIN COMPLEX, PROTEIN FAMILY, ORGANISM} — "
             "ORGANISM is critical for parasite/antimicrobial evidence.",
    )
    parser.add_argument(
        "--binding-only", action="store_true",
        help="Restrict allowed_assay_types to {B}. Default: {B, F} (Binding + "
             "Functional) — Functional assays are needed for ORGANISM and "
             "whole-cell evidence.",
    )
    parser.add_argument(
        "--include-phenotypic", action="store_true",
        help="Add a second activity-fetch pass for rows where ChEMBL has no "
             "pchembl_value (qualitative bioactivity in functional / organism "
             "assays). Recovers compounds like terpenes / sterols / sugars "
             "that have rich antiparasitic / antimicrobial data but no "
             "quantitative potency. Tagged as evidence_type=chembl_phenotypic "
             "at confidence 0.40. Off by default for backward-compatibility.",
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

    from src.data.chembl import (
        DEFAULT_ALLOWED_ASSAY_TYPES, DEFAULT_ALLOWED_TARGET_TYPES,
    )
    target_types = (
        ("SINGLE PROTEIN",) if args.protein_only
        else DEFAULT_ALLOWED_TARGET_TYPES
    )
    assay_types = (
        ("B",) if args.binding_only
        else DEFAULT_ALLOWED_ASSAY_TYPES
    )

    with GraphClient() as client:
        linker = CompoundTargetLinker(
            client,
            pchembl_floor=args.pchembl_floor,
            max_targets_per_compound=args.max_targets,
            workers=args.workers,
            allowed_target_types=target_types,
            allowed_assay_types=assay_types,
            include_phenotypic=args.include_phenotypic,
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
