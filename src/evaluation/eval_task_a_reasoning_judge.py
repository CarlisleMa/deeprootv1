"""Eval — Task A reasoning judge (LLM-as-judge over critic outputs).

Grades each critic output across the 4 conditions (KG critic at one or
more model tiers, text-only baseline, graph-only baseline, tool-call
baseline) on the 6-dimension rubric in judge_prompts.py.

Pipeline:

  1. Load Pass 1 verdicts.
  2. Stratify claims (judge_cases.DEFAULT_STRATA).
  3. For each (condition, claim) load the precomputed critic output.
  4. Reconstruct the visible payload for conditions that had one
     (KG critic). Text-only and tool-call baselines have no
     reconstructable graph payload — citation_fidelity falls back to
     biomedical-plausibility judgement only.
  5. Run deterministic precondition checks.
  6. Build the JudgeCase struct.
  7. Run the LLM judge (Claude Opus 4.7, cross-family) — POINTWISE.
     Pairwise is scaffolded but NOT run by default.
  8. Aggregate: per-condition dimension means, flag rates, slice cuts,
     lowest-scoring exemplars.

Read-only — no graph writes. Resume support keyed by
case_id + output_hash + judge_model.

CLI usage:
  python -m src.evaluation.eval_task_a_reasoning_judge \\
      --pass1-json results/task_a_pass1.json \\
      --critic-jsons results/task_a_pass2_kg.json results/task_a_pass2_baseline_text.json \\
      --conditions kg_critic_pro text_only \\
      --out src/evaluation/results/task_a_judge_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import ANTHROPIC_API_KEY
from src.evaluation.judge_cases import (
    DEFAULT_STRATA,
    JudgeCase,
    PromptPayloadReconstructor,
    build_case,
    flatten_strata,
    hash_dict,
    load_pass1,
    save_cases,
    stratified_sample,
)
from src.evaluation.judge_prompts import (
    DIMENSIONS,
    FLAGS,
    PAIRWISE_RESPONSE_SCHEMA,
    PAIRWISE_SYSTEM,
    POINTWISE_RESPONSE_SCHEMA,
    POINTWISE_SYSTEM,
    render_pairwise_user,
    render_pointwise_user,
)
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"   # cross-family vs Gemini critic
RATE_LIMIT_S = 1.2
MAX_TOKENS = 2048
MAX_RETRIES = 3

# Conditions whose outputs we know how to reconstruct payloads for.
# text_only and tool_call had no pre-computed graph payload, so the
# citation-fidelity check is "not applicable" for them.
PAYLOAD_RECONSTRUCTABLE = {"kg_critic", "graph_only"}
PAYLOAD_RECONSTRUCTABLE_PREFIX = ("kg_critic",)  # supports kg_critic_<model>


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

def _make_anthropic_client():
    import anthropic
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Add it to .env: ANTHROPIC_API_KEY=sk-ant-..."
        )
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def _call_claude_json(
    client,
    system: str,
    user: str,
    *,
    model: str,
    schema_hint: dict[str, Any],
    last_call: list[float],
) -> dict | None:
    """One Claude call returning parsed JSON. The prompt includes a JSON
    schema hint in the system text; we still defensively parse and
    schema-check on return.
    """
    user_with_schema = (
        user
        + "\n\nReturn ONLY a JSON object matching this schema:\n"
        + json.dumps(schema_hint, indent=2)
    )
    for attempt in range(MAX_RETRIES):
        elapsed = time.time() - last_call[0]
        if elapsed < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - elapsed)
        try:
            # Claude Opus 4.7 deprecates `temperature` — only set it for
            # older Sonnet / Haiku families that still accept it.
            create_kwargs: dict[str, Any] = dict(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user_with_schema}],
            )
            if not model.startswith("claude-opus-4-7"):
                create_kwargs["temperature"] = 0.0
            resp = client.messages.create(**create_kwargs)
            last_call[0] = time.time()
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                # Strip code fences
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("Judge JSON parse failed (attempt %d): %s", attempt + 1, e)
            last_call[0] = time.time()
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except Exception as e:
            err = str(e)
            last_call[0] = time.time()
            if "429" in err and attempt < MAX_RETRIES - 1:
                time.sleep(10 * (attempt + 1))
                continue
            logger.warning("Judge call failed: %s", e)
            return None
    return None


# ---------------------------------------------------------------------------
# Pointwise judging
# ---------------------------------------------------------------------------

@dataclass
class PointwiseResult:
    case_id: str
    condition: str
    source: str
    malady: str
    stratum: str
    overall_score: int
    dimension_scores: dict           # {dim: {score, justification}}
    flags: dict                       # {flag: bool}
    recommended_status: str
    summary: str
    deterministic_checks: dict
    pass1_signals: dict
    judge_model: str
    output_hash: str
    error: str | None = None


def _judge_one_case(
    case: JudgeCase,
    *,
    client: Any,
    judge_model: str,
) -> PointwiseResult:
    """Worker function for one case. Each thread holds its own
    last_call gate (the SDK's 429-retry handles cross-thread rate
    pressure, so per-thread gating is sufficient).
    """
    user = render_pointwise_user(
        condition=case.condition,
        source=case.source,
        malady=case.malady,
        primary_disease=case.primary_disease,
        pass1_signals=case.pass1_signals,
        payload=case.payload,
        critic_output=case.critic_output,
        deterministic_checks=case.deterministic_checks,
    )
    last_call = [0.0]
    judge_resp = _call_claude_json(
        client,
        POINTWISE_SYSTEM,
        user,
        model=judge_model,
        schema_hint=POINTWISE_RESPONSE_SCHEMA,
        last_call=last_call,
    )

    if judge_resp is None:
        return PointwiseResult(
            case_id=case.case_id,
            condition=case.condition,
            source=case.source,
            malady=case.malady,
            stratum=case.stratum,
            overall_score=0,
            dimension_scores={},
            flags={f: False for f in FLAGS},
            recommended_status="human_review",
            summary="judge call failed",
            deterministic_checks=case.deterministic_checks,
            pass1_signals=case.pass1_signals,
            judge_model=judge_model,
            output_hash=case.output_hash,
            error="judge_failed",
        )

    return PointwiseResult(
        case_id=case.case_id,
        condition=case.condition,
        source=case.source,
        malady=case.malady,
        stratum=case.stratum,
        overall_score=int(judge_resp.get("overall_score") or 0),
        dimension_scores=judge_resp.get("dimension_scores") or {},
        flags=judge_resp.get("flags") or {f: False for f in FLAGS},
        recommended_status=judge_resp.get("recommended_status") or "human_review",
        summary=str(judge_resp.get("summary") or ""),
        deterministic_checks=case.deterministic_checks,
        pass1_signals=case.pass1_signals,
        judge_model=judge_model,
        output_hash=case.output_hash,
    )


def _checkpoint_partial(
    results: list["PointwiseResult"],
    checkpoint_path: "Path",
    judge_model: str,
) -> None:
    """Atomically write partial pointwise results so a kill-then-restart
    picks up where we left off via the (case_id, output_hash, model)
    resume key in _load_existing()."""
    tmp = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "judge_model": judge_model,
        "pointwise": [asdict(r) for r in results],
        "pairwise": [],
        "summary": {"checkpoint": True, "n_done": len(results)},
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, default=str)
    tmp.replace(checkpoint_path)


def run_pointwise(
    cases: list[JudgeCase],
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    existing: dict[str, dict] | None = None,
    max_workers: int = 8,
    checkpoint_every: int = 20,
    checkpoint_path: "Path | None" = None,
) -> list[PointwiseResult]:
    """Run pointwise judging over a list of cases. Parallel via
    ThreadPoolExecutor (default 8 workers). Resumable: cases whose
    (case_id, output_hash, judge_model) is already in `existing` are
    skipped. Periodically checkpoints to `checkpoint_path` so a kill
    mid-run can be resumed.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = _make_anthropic_client()
    existing = existing or {}
    results: list[PointwiseResult] = []
    pending: list[JudgeCase] = []

    for case in cases:
        resume_key = f"{case.case_id}|{case.output_hash}|{judge_model}"
        if resume_key in existing:
            results.append(_dict_to_pointwise(existing[resume_key]))
        else:
            pending.append(case)

    total = len(cases)
    if results:
        logger.info("Resume: %d/%d already judged; %d pending",
                    len(results), total, len(pending))
    if not pending:
        return results

    logger.info(
        "Pointwise judge: %d cases pending, %d workers, model=%s",
        len(pending), max_workers, judge_model,
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_judge_one_case, c, client=client, judge_model=judge_model): c
            for c in pending
        }
        for future in as_completed(futures):
            case = futures[future]
            try:
                res = future.result()
            except Exception as e:
                logger.warning("[worker error] %s: %s", case.case_id, e)
                res = PointwiseResult(
                    case_id=case.case_id,
                    condition=case.condition,
                    source=case.source,
                    malady=case.malady,
                    stratum=case.stratum,
                    overall_score=0,
                    dimension_scores={},
                    flags={f: False for f in FLAGS},
                    recommended_status="human_review",
                    summary=f"worker exception: {e!r}",
                    deterministic_checks=case.deterministic_checks,
                    pass1_signals=case.pass1_signals,
                    judge_model=judge_model,
                    output_hash=case.output_hash,
                    error=str(e),
                )
            results.append(res)
            completed += 1
            if completed % checkpoint_every == 0 or completed == len(pending):
                logger.info(
                    "[%d/%d] pointwise judge progress (%d cumulative results)",
                    completed, len(pending), len(results),
                )
                if checkpoint_path is not None:
                    _checkpoint_partial(results, checkpoint_path, judge_model)

    return results


def _dict_to_pointwise(d: dict) -> PointwiseResult:
    return PointwiseResult(
        case_id=d.get("case_id", ""),
        condition=d.get("condition", ""),
        source=d.get("source", ""),
        malady=d.get("malady", ""),
        stratum=d.get("stratum", ""),
        overall_score=int(d.get("overall_score", 0)),
        dimension_scores=d.get("dimension_scores") or {},
        flags=d.get("flags") or {f: False for f in FLAGS},
        recommended_status=d.get("recommended_status", "human_review"),
        summary=d.get("summary", ""),
        deterministic_checks=d.get("deterministic_checks") or {},
        pass1_signals=d.get("pass1_signals") or {},
        judge_model=d.get("judge_model", ""),
        output_hash=d.get("output_hash", ""),
        error=d.get("error"),
    )


# ---------------------------------------------------------------------------
# Pairwise judging — scaffold only, NOT run by default
# ---------------------------------------------------------------------------

@dataclass
class PairwiseResult:
    claim_id: str       # source::malady
    source: str
    malady: str
    condition_a: str
    condition_b: str
    a_was_first: bool   # whether A's output was shown first (for position-bias correction)
    dimension_preferences: dict
    overall: dict
    judge_model: str
    error: str | None = None


def run_pairwise(
    cases_by_condition: dict[str, list[JudgeCase]],
    *,
    pair: tuple[str, str],
    judge_model: str = DEFAULT_JUDGE_MODEL,
    seed: int = 42,
) -> list[PairwiseResult]:
    """Compare TWO conditions side-by-side on the SAME claims.

    Args:
        cases_by_condition: {condition_name -> list of JudgeCase}, must
            contain the two conditions in `pair` with overlapping claims.
        pair: (condition_a, condition_b) names.
        judge_model: Anthropic model.
        seed: RNG seed for A/B order randomization.

    Pairs claims by (source, malady). Skips claims missing in either
    condition. Randomizes which output is shown as A vs B and unmaps
    after the call.
    """
    import random
    cond_a, cond_b = pair
    by_claim_a = {(c.source, c.malady): c for c in cases_by_condition.get(cond_a, [])}
    by_claim_b = {(c.source, c.malady): c for c in cases_by_condition.get(cond_b, [])}
    common_claims = sorted(set(by_claim_a) & set(by_claim_b))

    rng = random.Random(seed)
    client = _make_anthropic_client()
    last_call = [0.0]
    results: list[PairwiseResult] = []

    for i, (source, malady) in enumerate(common_claims, 1):
        case_a = by_claim_a[(source, malady)]
        case_b = by_claim_b[(source, malady)]
        a_first = rng.random() < 0.5
        if a_first:
            cond_x, out_x = cond_a, case_a.critic_output
            cond_y, out_y = cond_b, case_b.critic_output
        else:
            cond_x, out_x = cond_b, case_b.critic_output
            cond_y, out_y = cond_a, case_a.critic_output

        # Use case_a's payload (both conditions share the same claim, but
        # for fairness use the KG-side payload if available).
        payload = case_a.payload if case_a.payload else case_b.payload

        user = render_pairwise_user(
            source=source,
            malady=malady,
            primary_disease=case_a.primary_disease,
            pass1_signals=case_a.pass1_signals,
            payload=payload,
            condition_a=cond_x,
            condition_b=cond_y,
            output_a=out_x,
            output_b=out_y,
        )

        logger.info("[%d/%d] pairwise %s ↔ %s for %s",
                    i, len(common_claims), cond_a, cond_b, source)

        judge_resp = _call_claude_json(
            client,
            PAIRWISE_SYSTEM,
            user,
            model=judge_model,
            schema_hint=PAIRWISE_RESPONSE_SCHEMA,
            last_call=last_call,
        )
        if judge_resp is None:
            results.append(PairwiseResult(
                claim_id=f"{source}::{malady}",
                source=source, malady=malady,
                condition_a=cond_a, condition_b=cond_b,
                a_was_first=a_first,
                dimension_preferences={},
                overall={},
                judge_model=judge_model,
                error="judge_failed",
            ))
            continue

        # Unmap A/B back to original conditions if randomization swapped.
        prefs = judge_resp.get("dimension_preferences") or {}
        overall = judge_resp.get("overall") or {}
        if not a_first:
            prefs = {d: _swap_pref(p) for d, p in prefs.items()}
            overall = _swap_pref(overall) if overall else overall

        results.append(PairwiseResult(
            claim_id=f"{source}::{malady}",
            source=source, malady=malady,
            condition_a=cond_a, condition_b=cond_b,
            a_was_first=a_first,
            dimension_preferences=prefs,
            overall=overall,
            judge_model=judge_model,
        ))

    return results


def _swap_pref(p: dict) -> dict:
    """Swap A↔B in a single preference dict (keeps justification unchanged)."""
    out = dict(p)
    pref = out.get("preferred")
    if pref == "A":
        out["preferred"] = "B"
    elif pref == "B":
        out["preferred"] = "A"
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_pointwise(results: list[PointwiseResult]) -> dict:
    """Compute summary stats per condition + slice cuts."""
    import statistics
    by_condition: dict[str, list[PointwiseResult]] = {}
    for r in results:
        by_condition.setdefault(r.condition, []).append(r)

    summary: dict[str, Any] = {
        "total_cases": len(results),
        "by_condition": {},
        "slice_cuts": {},
    }

    for cond, rs in by_condition.items():
        graded = [r for r in rs if r.overall_score > 0]
        n = len(graded)
        if n == 0:
            continue

        dim_means: dict[str, float] = {}
        for d in DIMENSIONS:
            scores = [
                int(r.dimension_scores.get(d, {}).get("score") or 0)
                for r in graded
                if r.dimension_scores.get(d)
            ]
            dim_means[d] = round(statistics.mean(scores), 3) if scores else 0.0

        flag_rates: dict[str, float] = {}
        for f in FLAGS:
            tripped = sum(1 for r in graded if r.flags.get(f))
            flag_rates[f] = round(tripped / n, 3)

        status_dist: dict[str, int] = {}
        for r in graded:
            status_dist[r.recommended_status] = status_dist.get(r.recommended_status, 0) + 1

        # Lowest-scoring 5 exemplars
        lowest = sorted(graded, key=lambda r: r.overall_score)[:5]

        summary["by_condition"][cond] = {
            "n": n,
            "overall_mean": round(statistics.mean(r.overall_score for r in graded), 3),
            "overall_median": statistics.median(r.overall_score for r in graded),
            "dimension_means": dim_means,
            "flag_rates": flag_rates,
            "status_distribution": status_dist,
            "lowest_scoring": [
                {
                    "case_id": r.case_id,
                    "overall_score": r.overall_score,
                    "summary": r.summary,
                }
                for r in lowest
            ],
        }

    # Slice cuts (across conditions, combined)
    summary["slice_cuts"] = {
        "by_stratum": _slice_by(results, lambda r: r.stratum),
        "by_pass1_top_bucket": _slice_by(
            results, lambda r: r.pass1_signals.get("top_bucket") or "unrated"
        ),
        "gold_no_loop_closure": _slice_by(
            results,
            lambda r: (
                "gold_no_loop"
                if r.pass1_signals.get("top_bucket") == "gold"
                and (r.pass1_signals.get("paths_loop_closed") or 0) == 0
                else "other"
            ),
        ),
    }
    return summary


def _slice_by(results: list[PointwiseResult], key_fn) -> dict:
    import statistics
    from collections import defaultdict
    buckets: dict[str, dict[str, list[PointwiseResult]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in results:
        if r.overall_score <= 0:
            continue
        buckets[key_fn(r)][r.condition].append(r)
    out: dict[str, dict[str, dict]] = {}
    for slice_name, by_cond in buckets.items():
        out[slice_name] = {}
        for cond, rs in by_cond.items():
            out[slice_name][cond] = {
                "n": len(rs),
                "overall_mean": round(
                    statistics.mean(r.overall_score for r in rs), 3
                ) if rs else 0,
            }
    return out


# ---------------------------------------------------------------------------
# Case construction (orchestrator)
# ---------------------------------------------------------------------------

def build_cases_for_condition(
    *,
    condition: str,
    pass1_verdicts_lookup: dict[tuple[str, str], dict],
    critic_verdicts: list[dict],
    payload_reconstructor: PromptPayloadReconstructor | None,
    sampled_claims: set[tuple[str, str]],
    stratum_lookup: dict[tuple[str, str], str],
) -> list[JudgeCase]:
    """Build JudgeCase objects for one condition.

    Args:
        condition: condition name (e.g. "kg_critic_pro", "text_only").
        pass1_verdicts_lookup: {(source, malady) -> Pass 1 verdict dict}.
        critic_verdicts: list of critic outputs from this condition.
        payload_reconstructor: reuses CriticAgent helpers; pass None for
            text_only / tool_call (no reconstructable payload).
        sampled_claims: only build cases for these (source, malady) pairs.
        stratum_lookup: {(source, malady) -> stratum_name}.
    """
    cases: list[JudgeCase] = []
    needs_payload = (
        condition in PAYLOAD_RECONSTRUCTABLE
        or condition.startswith(PAYLOAD_RECONSTRUCTABLE_PREFIX)
    )

    for cv in critic_verdicts:
        key = (cv.get("source"), cv.get("malady"))
        if key not in sampled_claims:
            continue
        pass1 = pass1_verdicts_lookup.get(key)
        if pass1 is None:
            continue

        if needs_payload and payload_reconstructor is not None:
            try:
                payload = payload_reconstructor.reconstruct(pass1)
            except Exception as e:
                logger.warning(
                    "Payload reconstruction failed for %s::%s: %s",
                    cv.get("source"), cv.get("malady"), e,
                )
                payload = None
        else:
            payload = None

        cases.append(build_case(
            condition=condition,
            pass1_verdict=pass1,
            critic_output=cv,
            payload=payload,
            stratum=stratum_lookup.get(key, ""),
        ))

    return cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open() as f:
        d = json.load(f)
    out: dict[str, dict] = {}
    for r in d.get("pointwise") or []:
        key = f"{r.get('case_id')}|{r.get('output_hash')}|{r.get('judge_model')}"
        out[key] = r
    return out


def _save_results(
    *,
    out_path: Path,
    pointwise: list[PointwiseResult],
    pairwise: list[PairwiseResult],
    summary: dict,
    judge_model: str,
    config: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "judge_model": judge_model,
        "config": config,
        "pointwise": [asdict(r) for r in pointwise],
        "pairwise": [asdict(r) for r in pairwise],
        "summary": summary,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task A reasoning-quality judge (LLM-as-judge over critic outputs)."
    )
    parser.add_argument(
        "--pass1-json", required=True,
        help="Path to Pass 1 verdicts (results/task_a_pass1.json).",
    )
    parser.add_argument(
        "--critic-jsons", nargs="+", required=True,
        help="One critic-output JSON per condition (parallel order with --conditions).",
    )
    parser.add_argument(
        "--conditions", nargs="+", required=True,
        help=(
            "Condition names matching --critic-jsons. Conditions starting "
            "with 'kg_critic' get full payload reconstruction; others are "
            "graded without citation-fidelity (no_payload mode)."
        ),
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON path for judge results.",
    )
    parser.add_argument(
        "--cases-out", default=None,
        help="Optional: dump reconstructed JudgeCase records here for audit.",
    )
    parser.add_argument(
        "--judge-model", default=DEFAULT_JUDGE_MODEL,
        help=f"Anthropic model for the judge (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument(
        "--stratify", action="store_true", default=True,
        help="Use stratified sampling (default: on). Currently the only mode.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for stratified sampling.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total claims for smoke tests.",
    )
    parser.add_argument(
        "--skip-llm", action="store_true",
        help="Run deterministic checks + case construction only; skip LLM judge.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Ignore cached results and rejudge all cases.",
    )
    parser.add_argument(
        "--pairwise", nargs=2, default=None,
        metavar=("CONDITION_A", "CONDITION_B"),
        help="Run pairwise judge for these two conditions (deferred for v1).",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel judge workers (default 8). Anthropic SDK is "
             "thread-safe; per-call 429s are retried by the worker.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=20,
        help="Atomically write partial results to --out every N "
             "completed cases (default 20). Resume picks up from there.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if len(args.critic_jsons) != len(args.conditions):
        parser.error("--critic-jsons and --conditions must have the same length")

    # 1. Load Pass 1 + stratify
    pass1_verdicts = load_pass1(args.pass1_json)
    pass1_lookup = {(v["source"], v["malady"]): v for v in pass1_verdicts}
    pools = stratified_sample(pass1_verdicts, seed=args.seed)
    sampled = flatten_strata(pools)
    if args.limit is not None:
        sampled = sampled[: args.limit]
    sampled_keys = {(v["source"], v["malady"]) for v in sampled}
    stratum_lookup = {(v["source"], v["malady"]): v["_stratum"] for v in sampled}

    print(f"[judge] {len(sampled_keys)} stratified claims selected from "
          f"{len(pass1_verdicts)} Pass 1 verdicts")

    # 2. Build payload reconstructor (one shared instance, reuses pass1 cache)
    reconstructor: PromptPayloadReconstructor | None = None
    needs_reconstruction = any(
        c in PAYLOAD_RECONSTRUCTABLE or c.startswith(PAYLOAD_RECONSTRUCTABLE_PREFIX)
        for c in args.conditions
    )
    client_cm: GraphClient | None = None
    if needs_reconstruction:
        client_cm = GraphClient()
        client_cm.__enter__()
        reconstructor = PromptPayloadReconstructor(client_cm, pass1_verdicts)

    try:
        # 3. Build JudgeCases per condition
        all_cases: list[JudgeCase] = []
        cases_by_cond: dict[str, list[JudgeCase]] = {}
        for critic_path, condition in zip(args.critic_jsons, args.conditions):
            with open(critic_path, encoding="utf-8") as f:
                critic_data = json.load(f)
            # Support both schemas: the new baselines emit 'verdicts',
            # the underlying CriticAgent.run() emits 'results'. Try both.
            critic_verdicts = (
                critic_data.get("verdicts")
                or critic_data.get("results")
                or []
            )
            built = build_cases_for_condition(
                condition=condition,
                pass1_verdicts_lookup=pass1_lookup,
                critic_verdicts=critic_verdicts,
                payload_reconstructor=reconstructor,
                sampled_claims=sampled_keys,
                stratum_lookup=stratum_lookup,
            )
            print(f"[judge] {condition}: {len(built)} cases built (from {critic_path})")
            cases_by_cond[condition] = built
            all_cases.extend(built)

        if args.cases_out:
            save_cases(all_cases, args.cases_out)
            print(f"[judge] Cases written to {args.cases_out}")

        # 4. Run pointwise judge (unless --skip-llm)
        out_path = Path(args.out)
        existing = {} if args.force else _load_existing(out_path)
        pointwise: list[PointwiseResult] = []
        if not args.skip_llm:
            pointwise = run_pointwise(
                all_cases,
                judge_model=args.judge_model,
                existing=existing,
                max_workers=args.workers,
                checkpoint_every=args.checkpoint_every,
                checkpoint_path=out_path,
            )
        else:
            print("[judge] --skip-llm: deterministic checks only, no LLM grading")

        # 5. Pairwise (deferred for v1; only runs if --pairwise specified)
        pairwise: list[PairwiseResult] = []
        if args.pairwise:
            cond_a, cond_b = args.pairwise
            pairwise = run_pairwise(
                cases_by_cond,
                pair=(cond_a, cond_b),
                judge_model=args.judge_model,
                seed=args.seed,
            )
            print(f"[judge] pairwise {cond_a} ↔ {cond_b}: {len(pairwise)} comparisons")

        # 6. Aggregate + write
        summary = aggregate_pointwise(pointwise)
        config = {
            "pass1_json": args.pass1_json,
            "critic_jsons": args.critic_jsons,
            "conditions": args.conditions,
            "stratify": args.stratify,
            "seed": args.seed,
            "limit": args.limit,
            "skip_llm": args.skip_llm,
            "pairwise_pair": args.pairwise,
        }
        _save_results(
            out_path=out_path,
            pointwise=pointwise,
            pairwise=pairwise,
            summary=summary,
            judge_model=args.judge_model,
            config=config,
        )

        print(f"\n[judge] Wrote results to {out_path}")
        print(f"[judge] Pointwise cases judged: {sum(1 for r in pointwise if r.overall_score > 0)}/{len(pointwise)}")
        if summary.get("by_condition"):
            print("\n[judge] Per-condition overall mean:")
            for cond, stats in summary["by_condition"].items():
                print(f"  {cond:30s}  n={stats['n']:3d}  mean={stats['overall_mean']:.2f}")

    finally:
        if client_cm is not None:
            client_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
