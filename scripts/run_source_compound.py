#!/usr/bin/env python3
"""Source → Compound linker (v2) CLI runner.

The v2 linker requires the ExtractionAuditor to have written
`canonical_source` on every Source node first. It then:

  - reads canonical_name + canonical_type from each Source
  - routes organisms to COCONUT (local parquet) and chemicals to
    PubChem (HTTP, cached)
  - MERGEs Chemical_Compound by InChIKey (RDKit-computed)
  - writes IS_EXTRACTED_FROM with flat per-evidence_type confidence

Usage:
    # Dry-run (default — no graph writes, prints projected counts)
    python scripts/run_source_compound.py --dry-run

    # Apply on the live graph
    python scripts/run_source_compound.py

    # Resume / retry control
    python scripts/run_source_compound.py --retry-misses
    python scripts/run_source_compound.py --force-relink
    python scripts/run_source_compound.py --limit 10

    # Raw JSON output
    python scripts/run_source_compound.py --json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase1.source_compound import SourceCompoundLinker
from src.graph.client import GraphClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def _print_summary(summary: dict) -> None:
    mode = "DRY RUN" if summary["dry_run"] else "WRITE MODE"
    print(f"\n=== Source→Compound Linker v2 ({mode}) ===")
    print(f"  Sources processed: {summary['sources_total']}")
    print(f"  Wall clock: {summary['duration_s']}s")

    print("\n  --- by linker_status ---")
    for status, count in summary["by_status"].items():
        if count == 0:
            continue
        print(f"    {status:30s} {count}")

    print("\n  --- by evidence_type ---")
    nonzero_evidence = {k: v for k, v in summary["by_evidence_type"].items() if v}
    if not nonzero_evidence:
        print("    (none)")
    else:
        for et, count in sorted(nonzero_evidence.items(), key=lambda kv: -kv[1]):
            print(f"    {et:35s} {count}")

    print("\n  --- compounds ---")
    print(f"    Unique compounds (by inchikey): {summary['compounds_unique']}")
    print(f"    IS_EXTRACTED_FROM edge writes:  {summary['edge_writes']}")
    dist = summary["compound_count_distribution"]
    print(
        f"    Per-source distribution: p50={dist['p50']}  p75={dist['p75']}  "
        f"p95={dist['p95']}  max={dist['max']}"
    )

    print("\n  --- part-context split (organism evidence is species-level) ---")
    total_edges = summary.get("edge_writes") or 1
    ps = summary.get("edges_part_specific", 0)
    wc = summary.get("edges_whole_or_compound_level", 0)
    print(
        f"    Part-specific (penalty applied): {ps}  ({100 * ps / total_edges:.1f}%)"
    )
    print(
        f"    Whole / compound-level:          {wc}  ({100 * wc / total_edges:.1f}%)"
    )
    by_part = summary.get("by_part_context", {})
    if by_part:
        # Show top 10 parts to keep the report scannable
        for part, count in list(by_part.items())[:10]:
            print(f"      {part:20s} {count}")
        if len(by_part) > 10:
            tail = sum(c for _, c in list(by_part.items())[10:])
            print(f"      {'(other)':20s} {tail}")

    print("\n  --- external calls ---")
    ec = summary["external_calls"]
    print(f"    COCONUT lookup calls:         {ec['coconut_lookup_calls']}")
    print(f"      hits:                       {ec['coconut_lookup_hits']}")
    print(f"      misses:                     {ec['coconut_lookup_misses']}")
    print(f"      alias-fallback hits:        {ec['coconut_alias_fallback_hits']}")
    print(f"    PubChem name calls:           {ec['pubchem_name_calls']}")
    print(f"    PubChem formula calls:        {ec['pubchem_formula_calls']}")
    print(f"    PubChem cache hits:           {ec['pubchem_cache_hits']}")
    print(f"    PubChem errors:               {ec['pubchem_errors']}")

    if summary["errors"]:
        print(f"\n  --- {len(summary['errors'])} error(s) ---")
        for err in summary["errors"][:10]:
            print(f"    {err}")
        if len(summary["errors"]) > 10:
            print(f"    ... and {len(summary['errors']) - 10} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2 Source→Compound linker")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't write to the graph; print the projected plan only",
    )
    parser.add_argument(
        "--retry-misses", action="store_true",
        help="Also retry Sources with linker_status='no_compounds_found' "
             "(use after a PubChem cache or COCONUT snapshot update)",
    )
    parser.add_argument(
        "--force-relink", action="store_true",
        help="Ignore linker_status entirely and re-process every Source",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N Sources (debugging)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON summary instead of the pretty report",
    )
    args = parser.parse_args()

    if args.retry_misses and args.force_relink:
        parser.error("--retry-misses and --force-relink are mutually exclusive")

    with GraphClient() as client:
        linker = SourceCompoundLinker(client)
        summary = linker.run(
            dry_run=args.dry_run,
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
