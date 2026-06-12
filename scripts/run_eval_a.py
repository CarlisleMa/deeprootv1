#!/usr/bin/env python3
"""Eval A — Drug Repurposing Recovery (held-out KNOWN_TREATS) CLI runner.

The marquee quantitative eval for the Phase-2 Task B pipeline. For each
held-out (compound, disease) pair, masks all stereoisomers of the
compound, runs TaskBNominator on the disease, and checks whether the
masked compound is recovered in the top-K.

Usage:
    # Full corpus run (default top_k=20)
    python scripts/run_eval_a.py

    # Quick spot-check on 10 trials
    python scripts/run_eval_a.py --limit 10

    # Stricter eval — only count loop-closed candidates
    python scripts/run_eval_a.py --require-loop-closure

    # Larger top-K
    python scripts/run_eval_a.py --top-k 50

    # Dump full per-trial JSON (for paper tables / debugging)
    python scripts/run_eval_a.py --json-out results/eval_a.json

    # Run on a specific subset (one disease)
    python scripts/run_eval_a.py --filter-disease "Common Cold"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.eval_drug_repurposing import (
    DrugRepurposingEval,
    summary_to_dict,
)
from src.graph.client import GraphClient


def _print_summary(summary, show_per_disease: int = 10) -> None:
    print(f"\n{'='*60}")
    print(f"  Eval A — Drug Repurposing Recovery")
    print(f"{'='*60}")
    print(f"  config_mode:        {summary.config_mode}")
    print(f"  reachability_mode:  {summary.reachability_mode}")
    print(f"  test_set_size:      {summary.test_set_size}")
    print(f"  top_k:              {summary.top_k}")
    if summary.use_critic:
        print(f"  critic_input_top_n: {summary.critic_input_top_n}")
    print(f"  duration:           {summary.duration_s}s")
    cd = summary.coverage_diagnostic or {}
    if cd:
        print(f"\n  Coverage diagnostic (corpus reach into KNOWN_TREATS):")
        broad = cd.get("broad", 0)
        for mode in ("broad", "historical", "strict"):
            n = cd.get(mode, 0)
            pct = (n / broad * 100) if broad else 0.0
            print(f"    {mode:11s}: {n:4d}  ({pct:5.1f}% of broad)")
    print()
    print(f"  recall@1:           {summary.recall_at_1:.3f}")
    print(f"  recall@5:           {summary.recall_at_5:.3f}")
    print(f"  recall@10:          {summary.recall_at_10:.3f}")
    print(f"  recall@{summary.top_k:<3d}         {summary.recall_at_top_k:.3f}")
    print(f"  MRR:                {summary.mrr:.3f}")
    print()
    print(f"  found:              {summary.found_count}")
    print(f"  not_found:          {summary.not_found_count}")

    if summary.rank_histogram:
        print(f"\n  rank histogram (top 15 ranks):")
        for r, cnt in list(summary.rank_histogram.items())[:15]:
            bar = "#" * min(40, cnt)
            print(f"    rank {r:2d}: {cnt:3d}  {bar}")

    if summary.per_disease_recall_at_top_k and show_per_disease > 0:
        print(f"\n  per-disease recall@{summary.top_k} (top {show_per_disease}):")
        for d, r in list(summary.per_disease_recall_at_top_k.items())[:show_per_disease]:
            print(f"    {r:.2f}  {d}")


def _print_failures(summary, max_failures: int = 15) -> None:
    fails = [t for t in summary.trials if not t.found]
    if not fails:
        return
    print(f"\n  Sample failures (first {min(max_failures, len(fails))}):")
    for t in fails[:max_failures]:
        print(
            f"    {t.test_compound[:30]!r:<32s} -> "
            f"{t.test_disease[:30]!r:<32s}  "
            f"(masked {t.masked_inchikey_count} stereo(s); "
            f"{t.candidates_total} candidates total)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Eval A — drug repurposing recovery on held-out KNOWN_TREATS.",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Distinct compounds to evaluate per trial (default 20).",
    )
    parser.add_argument(
        "--reachability", choices=["broad", "historical", "strict"],
        default="historical",
        help=(
            "Test-set reachability mode (default 'historical'). "
            "'broad' = ~301 pairs (most unreachable); "
            "'historical' = ~21 pairs with full backward chain; "
            "'strict' = ~13 pairs with backward chain AND forward closure."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the test set size (useful for quick spot-checks).",
    )
    parser.add_argument(
        "--filter-disease", type=str, default=None,
        help="Restrict trials to test pairs for this disease.",
    )
    parser.add_argument(
        "--require-loop-closure", action="store_true",
        help="Stricter eval: only count loop-closed candidates as nominations.",
    )
    parser.add_argument(
        "--mode", choices=["pass3", "pass3_critic", "llm_passages"],
        default="pass3",
        help=(
            "Configuration to evaluate (default: pass3). "
            "pass3 = deterministic Task B nominator only. "
            "pass3_critic = Pass 3 + Pass 4 LLM critic re-rank. "
            "llm_passages = single LLM call with raw corpus, no graph "
            "(Eval-B baseline cell)."
        ),
    )
    parser.add_argument(
        "--use-critic", action="store_true",
        help="Shortcut for --mode pass3_critic (kept for back-compat).",
    )
    parser.add_argument(
        "--baseline-corpus-path", type=str, default=None,
        help="Override corpus path for the llm_passages baseline.",
    )
    parser.add_argument(
        "--critic-input-top-n", type=int, default=50,
        help="Number of Pass-3 candidates fed to the critic per trial "
             "(default 50). Only relevant with --use-critic.",
    )
    parser.add_argument(
        "--gemini-model", type=str, default=None,
        help="Override Gemini model for the critic.",
    )
    parser.add_argument(
        "--progress-every", type=int, default=25,
        help="Log a heartbeat every N trials (default 25).",
    )
    parser.add_argument(
        "--json-out", type=str, default=None,
        help="Path to dump the full eval result JSON (per-trial detail).",
    )
    parser.add_argument(
        "--show-per-disease", type=int, default=10,
        help="Top-N diseases to print per-disease recall for (default 10).",
    )
    parser.add_argument(
        "--show-failures", type=int, default=15,
        help="Sample failure trials to print (default 15).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Resolve effective mode: --mode takes precedence; legacy --use-critic
    # only applies when --mode wasn't explicitly switched.
    effective_mode = args.mode
    if args.use_critic and effective_mode == "pass3":
        effective_mode = "pass3_critic"

    with GraphClient() as client:
        evaluator = DrugRepurposingEval(
            client,
            mode=effective_mode,
            use_critic=args.use_critic,
            critic_input_top_n=args.critic_input_top_n,
            critic_gemini_model=args.gemini_model,
            baseline_corpus_path=args.baseline_corpus_path,
        )

        # Optional disease filter — applied before the eval driver
        # iterates, so progress and metrics reflect the filtered set.
        if args.filter_disease:
            full = evaluator.pull_test_set(args.reachability)
            filtered = [
                p for p in full if p["test_disease"] == args.filter_disease
            ]
            if not filtered:
                print(f"No test pairs for disease {args.filter_disease!r}.")
                return
            # Override pull_test_set for any subsequent calls in this run
            evaluator.pull_test_set = lambda mode=args.reachability: filtered

        summary = evaluator.run(
            top_k=args.top_k,
            reachability_mode=args.reachability,
            require_loop_closure=args.require_loop_closure,
            limit=args.limit,
            progress_every=args.progress_every,
        )

    _print_summary(summary, show_per_disease=args.show_per_disease)
    _print_failures(summary, max_failures=args.show_failures)

    if args.json_out:
        out = Path(args.json_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(summary_to_dict(summary), f, indent=2,
                     default=str, ensure_ascii=False)
        print(f"\n  Eval A JSON written to: {out}")


if __name__ == "__main__":
    main()
