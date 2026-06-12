#!/usr/bin/env python3
"""Reviewer Agent CLI runner.

Post-pipeline graph quality pass. Deterministic, multi-phase. Currently
implements orphan-Malady dedup; future phases (dead-end detection,
retry-flagging, generic-category cleanup) slot into the same agent.

Usage:
    # Dry-run all phases (default — print plan, no writes)
    python scripts/run_reviewer.py

    # Apply on the live graph
    python scripts/run_reviewer.py --write-graph

    # Limit per-phase actionable items (spot-check)
    python scripts/run_reviewer.py --limit 5

    # Run a specific phase only
    python scripts/run_reviewer.py --phase orphan_malady_dedup

    # Raw JSON
    python scripts/run_reviewer.py --json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase1.reviewer import ReviewerAgent
from src.graph.client import GraphClient


def _print_phase_orphan_malady(phase_summary: dict) -> None:
    plan = phase_summary.get("plan") or []
    print("=== Phase: orphan_malady_dedup ===")
    print(f"  orphans_total:    {phase_summary['orphans_total']}")
    print(f"  actionable:       {phase_summary['actionable']}")
    print(f"  applied:          {phase_summary['applied']}")
    print(f"  no_keeper:        {phase_summary['no_keeper']}")
    print(f"  edges_cascaded:   {phase_summary['edges_cascaded']}")
    print()

    if plan:
        print("  Plan:")
        for p in plan:
            cand_str = (
                f" (chose from {p['candidates']})"
                if len(p["candidates"]) > 1 else ""
            )
            print(
                f"    {p['orphan']!r}  →  {p['keeper']!r}  "
                f"[via Modern_Disease: {p['modern_disease']}]{cand_str}"
            )
        print()

    if phase_summary.get("by_disease"):
        print("  By Modern_Disease (top 10):")
        for d, n in list(phase_summary["by_disease"].items())[:10]:
            print(f"    {d}: {n}")
        print()

    no_keeper = phase_summary.get("no_keeper_orphans") or []
    if no_keeper:
        head = no_keeper[:10]
        suffix = "..." if len(no_keeper) > 10 else ""
        print(
            f"  no_keeper_orphans (left for future dead-end phase): "
            f"{head}{suffix}"
        )


_PHASE_PRINTERS = {
    "orphan_malady_dedup": _print_phase_orphan_malady,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Reviewer Agent (post-pipeline graph quality pass)."
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help="Apply on the live graph (default: dry-run)",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Per-phase cap on actionable items (0 = no cap)",
    )
    parser.add_argument(
        "--phase", action="append", dest="phases", default=None,
        help="Run only this phase (repeatable). Default: all phases.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit raw JSON in addition to the human-readable summary",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    with GraphClient() as client:
        agent = ReviewerAgent(client)
        results = agent.run(
            write_graph=args.write_graph,
            limit=args.limit,
            phases=args.phases,
        )

        print(f"=== Reviewer Agent ({results.get('mode', 'DRY-RUN')}) ===\n")
        for phase, summary in results.items():
            if phase == "mode":
                continue
            printer = _PHASE_PRINTERS.get(phase)
            if printer:
                printer(summary)
            else:
                print(f"=== Phase: {phase} ===")
                print(json.dumps(summary, indent=2, default=str))

        if args.json:
            print("\n--- Raw JSON ---")
            print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
