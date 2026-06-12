#!/usr/bin/env python3
"""Phase 2 — Pass 2 Critic CLI runner.

Consumes Pass 1 (TaskAValidator) verdicts and runs the LLM critic over
each one. Two input modes:

  --pass1-json <path>   load verdicts from a previously-saved Pass 1 JSON
  (default)             run TaskAValidator inline (live graph)

Usage:
    # Run pass 1 inline on first 10 claims, then critique them
    python scripts/run_task_a_critic.py --limit 10

    # Critique a previously-saved pass-1 JSON dump
    python scripts/run_task_a_critic.py --pass1-json task_a_results.json

    # Filter to one specific claim (works in inline mode)
    python scripts/run_task_a_critic.py --source "Aconitum carmichaelii" --malady "Wind cold"

    # Apply audit properties on the live graph
    python scripts/run_task_a_critic.py --limit 5 --write-graph

    # Skip the "non-actionable" filter — critique every claim including
    # traditional_only / mechanistic_only ones (more expensive)
    python scripts/run_task_a_critic.py --include-non-actionable

    # Save pass-2 results to JSON for later analysis / paper tables
    python scripts/run_task_a_critic.py --json-out critic_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.phase2.task_a_critic import CriticAgent
from src.agents.phase2.task_a_validator import TaskAValidator
from src.graph.client import GraphClient


def _print_summary(summary: dict) -> None:
    print(f"\n=== Critic Pass 2 — {summary['verdicts_in']} input(s) ===")
    print(f"  critiqued:                    {summary['critiqued']}")
    print(f"  skipped (non-actionable):     {summary['skipped']}")
    print(f"  errors:                       {summary['errors_count']}")
    print(f"  agreement_rate:               {summary['agreement_rate']:.3f}")
    print(f"  disagreement (>=1 rung):      {summary['disagreement_one_rung']}")
    print(f"  disagreement (>=2 rungs):     {summary['disagreement_two_plus_rungs']}")
    print(f"  flagged for human review:     {summary['requires_human_review']}")

    if summary.get("by_pass2_verdict"):
        print("\nby_pass2_verdict:")
        for k, n in summary["by_pass2_verdict"].items():
            print(f"  {k:22s} {n}")


def _print_per_claim(results: list[dict], max_rows: int = 30) -> None:
    if not results:
        return
    actionable = [r for r in results if not r.get("skipped") and not r.get("error")]
    if not actionable:
        return
    print(f"\n--- Per-claim critic verdicts (first {min(max_rows, len(actionable))}) ---")
    for r in actionable[:max_rows]:
        delta = r["verdict_delta"]
        delta_tag = (
            "AGREE" if delta == 0 else
            f"+{delta}" if delta > 0 else f"{delta}"
        )
        review_tag = " [REVIEW]" if r["requires_human_review"] else ""
        print(
            f"  [{delta_tag:>5}]{review_tag} "
            f"{r['source'][:30]!r} -> {r['malady'][:28]!r}  "
            f"p1={r['pass1_verdict']:18s} p2={r['verdict']:18s}  "
            f"plaus={r['biological_plausibility']:.2f} coh={r['evidence_coherence']:.2f}"
        )
        if r.get("rationale"):
            print(f"           ↳ {r['rationale'][:200]}")
        for c in r.get("concerns", [])[:3]:
            print(f"             concern[{c.get('concern_type')}]: {c.get('explanation', '')[:160]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Pass 2 LLM critic over Pass 1 verdicts."
    )
    parser.add_argument(
        "--pass1-json", type=str, default=None,
        help="Load verdicts from a saved TaskAValidator JSON dump.",
    )
    parser.add_argument(
        "--source", help="Inline mode: filter Pass 1 to this Source.",
    )
    parser.add_argument(
        "--malady", help="Inline mode: filter Pass 1 to this Malady.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on the number of claims to run through Pass 2.",
    )
    parser.add_argument(
        "--keep-top-paths", type=int, default=5,
        help="Top-N evidence paths fed to the LLM per claim (default 5).",
    )
    parser.add_argument(
        "--write-graph", action="store_true",
        help="Stamp task_a_critic_* audit properties on TREATS edges.",
    )
    parser.add_argument(
        "--include-non-actionable", action="store_true",
        help=(
            "Don't skip Pass 1 traditional_only / mechanistic_only verdicts. "
            "Costs more LLM tokens but covers every claim."
        ),
    )
    parser.add_argument(
        "--gemini-model", type=str, default=None,
        help="Override Gemini model (default: GEMINI_MODEL_PRO).",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Path to dump full Pass 2 result JSON.",
    )
    parser.add_argument(
        "--show", type=int, default=30,
        help="Max rows in per-claim table (default 30).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # -- Load pass-1 verdicts ---------------------------------------------
    with GraphClient() as client:
        if args.pass1_json:
            print(f"Loading Pass 1 verdicts from {args.pass1_json} ...")
            with open(args.pass1_json, "r", encoding="utf-8") as f:
                pass1_data = json.load(f)
            verdicts = pass1_data.get("verdicts") or []
            print(f"  Loaded {len(verdicts)} Pass-1 verdicts.")
        else:
            print("Running Pass 1 (TaskAValidator) inline ...")
            task_a = TaskAValidator(client)
            pass1_summary = task_a.run(
                source_name=args.source,
                malady_name=args.malady,
                limit=args.limit,
                keep_top_paths=args.keep_top_paths,
            )
            verdicts = pass1_summary.get("verdicts") or []
            print(f"  Pass 1 produced {len(verdicts)} verdicts.")

        # -- Run pass 2 ---------------------------------------------------
        critic = CriticAgent(
            client,
            gemini_model=args.gemini_model,
            skip_non_actionable=not args.include_non_actionable,
        )
        summary = critic.run(
            verdicts=verdicts,
            write_graph=args.write_graph,
            limit=args.limit if args.pass1_json else None,
        )

    _print_summary(summary)
    _print_per_claim(summary.get("results", []), max_rows=args.show)

    if summary.get("errors"):
        print(f"\n--- Errors ({len(summary['errors'])}) ---")
        for e in summary["errors"][:10]:
            print(f"  {e}")

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str, ensure_ascii=False)
        print(f"\nPass 2 JSON written to: {out_path}")


if __name__ == "__main__":
    main()
