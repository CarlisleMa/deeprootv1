#!/usr/bin/env python3
"""Phase 2 — Task A Validator CLI runner.

Validates Source → Traditional_Malady claims using mechanistic evidence
chains from the Phase 1 graph. Tier-bucket primary, multiplicative
tiebreak.

Usage:
    # Validate every active claim, dry-run, print summary
    python scripts/run_task_a_validator.py

    # Validate a specific (Source, Malady) pair
    python scripts/run_task_a_validator.py --source "Cortex Cinnamomi Cassiae" \\
        --malady "Abdominal cold pain"

    # Validate first 20 claims (spot-check)
    python scripts/run_task_a_validator.py --limit 20

    # Apply audit properties to TREATS edges on the live graph
    python scripts/run_task_a_validator.py --write-graph

    # Emit raw verdict JSON to a file (for paper tables)
    python scripts/run_task_a_validator.py --json-out task_a_results.json

    # Limit the number of evidence paths kept per verdict (default 5)
    python scripts/run_task_a_validator.py --keep-top-paths 10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase2.task_a_validator import TaskAValidator
from src.graph.client import GraphClient


def _print_human_summary(summary: dict) -> None:
    print(f"\n=== Task A — {summary['claims_total']} claim(s) ===\n")
    print("by_verdict:")
    for k, n in summary["by_verdict"].items():
        print(f"  {k:22s} {n}")
    print("\nby_top_bucket:")
    for k, n in summary["by_top_bucket"].items():
        print(f"  {k:10s} {n}")
    print(f"\nloop_closure_rate: {summary['loop_closure_rate']:.3f}")


def _print_per_claim_table(verdicts: list[dict], max_rows: int = 30) -> None:
    if not verdicts:
        return
    print(f"\n--- Per-claim verdicts (first {min(max_rows, len(verdicts))}) ---")
    for v in verdicts[:max_rows]:
        loop_tag = "LOOP" if v["paths_loop_closed"] > 0 else "    "
        bucket_tag = v["top_bucket"]
        path_tag = f"paths={v['path_count']:3d}"
        score_tag = f"score={v['top_bucket_max_score']:.3f}"
        rationale = v["rationale"]
        print(
            f"  [{loop_tag}] {bucket_tag:8s} {path_tag} {score_tag}"
            f"  {v['verdict']:18s} {v['source'][:32]!r} -> {v['malady'][:30]!r}"
        )
        if rationale:
            print(f"           ↳ {rationale}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Task A claim validator (Phase 2)."
    )
    parser.add_argument(
        "--source",
        help=(
            "Run only on this Source name. Combine with --malady to "
            "score one claim, or with no --malady to score every claim "
            "from this source."
        ),
    )
    parser.add_argument(
        "--malady",
        help=(
            "Run only on this Traditional_Malady name. Combine with "
            "--source to target one specific claim."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of claims to evaluate (after filtering).",
    )
    parser.add_argument(
        "--keep-top-paths", type=int, default=5,
        help="Number of evidence paths to retain per verdict (default 5).",
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help=(
            "Stamp task_a_* audit properties on each TREATS_TRADITIONALLY "
            "edge (default: dry-run, read-only)."
        ),
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Optional path to dump the full verdict JSON.",
    )
    parser.add_argument(
        "--show", type=int, default=30,
        help="Max rows in the per-claim table printed to stdout (default 30).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with GraphClient() as client:
        agent = TaskAValidator(client)
        summary = agent.run(
            source_name=args.source,
            malady_name=args.malady,
            limit=args.limit,
            write_graph=args.write_graph,
            keep_top_paths=args.keep_top_paths,
        )

    _print_human_summary(summary)
    _print_per_claim_table(summary.get("verdicts", []), max_rows=args.show)

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        print(f"\nVerdict JSON written to: {out_path}")


if __name__ == "__main__":
    main()
