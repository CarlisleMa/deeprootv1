"""Prompts and response schemas for the Task A reasoning judge.

Two modes:

  POINTWISE — grade one critic's output for one claim on six dimensions
              (1-5 scale + per-dimension justification + binary flags).

  PAIRWISE  — compare two critics' outputs for the SAME claim across the
              same six dimensions, returning {A | B | TIE} per dimension
              plus an overall preference with strength.

Both modes are designed for Claude (cross-family from the Gemini critic
under judgement). Temperature 0, structured JSON output.

The dimensions are tuned to the current task_a_critic.py output schema
(verdict / biological_plausibility / evidence_coherence / key_evidence /
concerns / rationale / requires_human_review) and to the design questions
TASK_A_LLM_JUDGE.md identifies. An earlier 5-dimension rubric was the
inspiration, but the dimensions and schema differ to match the new
critic shape.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

DIMENSIONS = [
    "evidence_fidelity",
    "verdict_alignment",
    "reasoning_coherence",
    "clinical_mapping_awareness",
    "uncertainty_calibration",
    "actionability",
]

DIMENSION_DESCRIPTIONS: dict[str, str] = {
    "evidence_fidelity": (
        "Does the critic cite ONLY evidence present in the visible payload? "
        "Every compound, target, disease, count, and score in the rationale "
        "or key_evidence should be traceable to a payload field. "
        "  5 = every cited fact is grounded in the payload\n"
        "  3 = mostly grounded, with loose paraphrases or minor unsupported details\n"
        "  1 = cites fabricated / payload-invisible evidence that affects the verdict"
    ),
    "verdict_alignment": (
        "Does the verdict follow from the visible evidence (paths, loop closure "
        "status, mapping quality, enrichment context)? Verdicts should reflect "
        "the actual signal, not surface heuristics like global top_bucket. "
        "  5 = verdict is well supported by loop closure status, path quality, "
        "      mapping quality, and enrichment\n"
        "  3 = verdict plausible but incompletely justified\n"
        "  1 = verdict contradicts the visible evidence or ignores decisive signals"
    ),
    "reasoning_coherence": (
        "Does the rationale actually explain THIS claim's chain (traditional "
        "text → modern disease → compound → target → reached disease)? Or is "
        "it boilerplate that could apply to any herb? "
        "  5 = clear claim-specific chain across all links\n"
        "  3 = understandable but generic or with key links missing\n"
        "  1 = incoherent, boilerplate, or internally contradictory"
    ),
    "clinical_mapping_awareness": (
        "Does the critic handle the Traditional_Malady → Modern_Disease mapping "
        "responsibly? Flag weak mappings, point out alternatives, avoid clinical "
        "category errors. "
        "  5 = explicitly checks mapping fit; identifies plausible alternatives\n"
        "  3 = accepts mapping without scrutiny but commits no error\n"
        "  1 = makes a clinical category error or ignores an obviously weak mapping"
    ),
    "uncertainty_calibration": (
        "Are biological_plausibility, evidence_coherence, concerns, and "
        "requires_human_review calibrated to the actual uncertainty? "
        "  5 = scores, concerns, and review flag reflect real uncertainty\n"
        "  3 = limitations mentioned but confidence slightly off\n"
        "  1 = overconfident, underconfident, or flags review inconsistently"
    ),
    "actionability": (
        "Would a curator reading this output know what to inspect next? "
        "Specific weakest links (missing loop closure, generic target, weak "
        "mapping, off-disease evidence) are useful; vague concerns are not. "
        "  5 = identifies specific weakest link a curator can act on\n"
        "  3 = useful but broad concern\n"
        "  1 = no actionable explanation"
    ),
}

FLAGS = [
    "hallucinated_evidence",        # citations not in payload
    "unsupported_verdict_jump",     # verdict materially stronger than evidence supports
    "ignored_loop_closure_status",  # critic ignored paths_loop_closed signal
    "overclaims_strength",          # claims gold-tier support when bucket=gold but loop=0
    "contradictory_scores",         # bp / ec contradict the verdict
    "needs_human_review",           # judge thinks this case is too ambiguous to grade fully
]


# ---------------------------------------------------------------------------
# Pointwise prompts
# ---------------------------------------------------------------------------

POINTWISE_SYSTEM = """\
You are an independent biomedical evaluation judge. You are grading the \
quality of an automated critic's VISIBLE reasoning, not deciding whether \
the traditional therapeutic claim is absolutely true.

Rules of the game:
  - Grade ONLY the critic's visible artifacts (verdict, scores, key_evidence, \
concerns, rationale, requires_human_review).
  - Use ONLY the supplied evidence payload for evidence_fidelity. You may \
draw on general biomedical knowledge ONLY for clinical_mapping_awareness \
and broad biological_plausibility.
  - Treat Pass 1 as an INPUT signal, not ground truth. A critic can be good \
even when it agrees with an imperfect Pass 1 verdict, if it explains the \
limitation correctly.
  - Penalize rationales that confuse global top_bucket evidence with \
loop-closing disease support. (A claim where the source has 76 gold-bucket \
paths but ZERO of those paths reach the mapped disease is NOT well-supported.)
  - Penalize hallucinated citations: every compound / target / disease named \
by the critic must trace to a payload field, unless explicitly tagged as \
external context.

Citation-checking sources by condition:
  - kg_critic / graph_only:  the reconstructed evidence payload below.
  - tool_call:                the agent's tool_call_log inside the critic
                              output records every result row the agent
                              actually saw (compounds, targets, diseases
                              returned by graph-query tools). Citations
                              must trace to those rows.
  - text_only:                the historical passage(s) the LLM was
                              shown plus widely-known biomedical facts.
                              Score evidence_fidelity on plausibility:
                              fabricated compound→target pairings should
                              be penalized; commonly accepted active
                              ingredients of named herbs are acceptable.

Return ONLY valid JSON matching the requested schema.\
"""


POINTWISE_USER_TEMPLATE = """\
EVALUATE THIS CRITIC OUTPUT.

Condition under judgement: {condition}

Claim:
  Source:                {source}
  Traditional Malady:    {malady}
  Mapped Modern Disease: {primary_disease}

Pass 1 (deterministic) signals:
{pass1_signals}

Visible payload (the structured evidence the critic could see — empty {{}} if \
the condition has no graph payload):
{payload}

Critic output (under judgement):
{critic_output}

Deterministic preconditions already run (informational — these are facts, \
not for you to repeat):
{deterministic_checks}

Grade on the following six dimensions, 1-5 scale, with a 1-2 sentence \
justification per dimension. Then set the binary flags. Then give an \
overall_score (1-5) and recommended_status.

DIMENSION DESCRIPTIONS:
{dimension_descriptions}

FLAG MEANINGS:
{flag_descriptions}

Recommended_status:
  pass         — overall_score >= 4 AND no severe flags
  weak_pass    — overall_score >= 3 AND only minor flags
  fail         — hallucinated_evidence true, contradictory_scores true, or \
overall_score < 3
  human_review — clinically sensitive or ambiguous; you can't grade fully

Output JSON:
"""


POINTWISE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {"type": "integer"},
        "dimension_scores": {
            "type": "object",
            "properties": {
                d: {
                    "type": "object",
                    "properties": {
                        "score": {"type": "integer"},
                        "justification": {"type": "string"},
                    },
                    "required": ["score", "justification"],
                }
                for d in DIMENSIONS
            },
            "required": DIMENSIONS,
        },
        "flags": {
            "type": "object",
            "properties": {f: {"type": "boolean"} for f in FLAGS},
            "required": FLAGS,
        },
        "recommended_status": {
            "type": "string",
            "enum": ["pass", "weak_pass", "fail", "human_review"],
        },
        "summary": {"type": "string"},
    },
    "required": [
        "overall_score", "dimension_scores", "flags",
        "recommended_status", "summary",
    ],
}


# ---------------------------------------------------------------------------
# Pairwise prompts (scaffolded, not run in v1; harness exposes them so we
# can pick the comparand pair after seeing the full pointwise pipeline)
# ---------------------------------------------------------------------------

PAIRWISE_SYSTEM = """\
You are an independent biomedical evaluation judge comparing two critic \
outputs for the SAME therapeutic claim, produced by two different systems. \
Each system was evaluated on the dimensions: evidence_fidelity, \
verdict_alignment, reasoning_coherence, clinical_mapping_awareness, \
uncertainty_calibration, actionability.

Determine which output is BETTER per dimension and overall. A well-reasoned \
weaker verdict (e.g. unsupported with clear rationale about missing loop \
closure) can be better than a poorly-reasoned stronger verdict.

The position (A vs B) is randomized — do not let position influence you.

Return ONLY valid JSON matching the requested schema.\
"""


PAIRWISE_USER_TEMPLATE = """\
COMPARE TWO CRITIC OUTPUTS for the same claim.

Claim:
  Source:                {source}
  Traditional Malady:    {malady}
  Mapped Modern Disease: {primary_disease}

Pass 1 signals:
{pass1_signals}

Visible payload (may be empty {{}} for some conditions):
{payload}

=== EVALUATOR A ===
Condition: {condition_a}
Output:
{output_a}

=== EVALUATOR B ===
Condition: {condition_b}
Output:
{output_b}

For each dimension and overall, indicate which evaluator is better \
(A | B | TIE). For the overall preference, also indicate the strength \
(strong | moderate | slight).

DIMENSION DESCRIPTIONS:
{dimension_descriptions}
"""


PAIRWISE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dimension_preferences": {
            "type": "object",
            "properties": {
                d: {
                    "type": "object",
                    "properties": {
                        "preferred": {"type": "string", "enum": ["A", "B", "TIE"]},
                        "justification": {"type": "string"},
                    },
                    "required": ["preferred", "justification"],
                }
                for d in DIMENSIONS
            },
            "required": DIMENSIONS,
        },
        "overall": {
            "type": "object",
            "properties": {
                "preferred": {"type": "string", "enum": ["A", "B", "TIE"]},
                "strength": {"type": "string", "enum": ["strong", "moderate", "slight"]},
                "justification": {"type": "string"},
            },
            "required": ["preferred", "strength", "justification"],
        },
    },
    "required": ["dimension_preferences", "overall"],
}


# ---------------------------------------------------------------------------
# Helpers — render templates
# ---------------------------------------------------------------------------

def _flag_descriptions() -> str:
    return "\n".join(f"  {f}" for f in FLAGS)


def _dimension_descriptions() -> str:
    return "\n\n".join(
        f"{d.upper()}\n{DIMENSION_DESCRIPTIONS[d]}" for d in DIMENSIONS
    )


def render_pointwise_user(
    *,
    condition: str,
    source: str,
    malady: str,
    primary_disease: str | None,
    pass1_signals: dict,
    payload: dict | None,
    critic_output: dict,
    deterministic_checks: dict,
) -> str:
    import json
    return POINTWISE_USER_TEMPLATE.format(
        condition=condition,
        source=source,
        malady=malady,
        primary_disease=primary_disease or "<not mapped>",
        pass1_signals=json.dumps(pass1_signals, indent=2, ensure_ascii=False),
        payload=json.dumps(payload or {}, indent=2, ensure_ascii=False)[:30000],
        critic_output=json.dumps(critic_output, indent=2, ensure_ascii=False),
        deterministic_checks=json.dumps(deterministic_checks, indent=2, ensure_ascii=False),
        dimension_descriptions=_dimension_descriptions(),
        flag_descriptions=_flag_descriptions(),
    )


def render_pairwise_user(
    *,
    source: str,
    malady: str,
    primary_disease: str | None,
    pass1_signals: dict,
    payload: dict | None,
    condition_a: str,
    condition_b: str,
    output_a: dict,
    output_b: dict,
) -> str:
    import json
    return PAIRWISE_USER_TEMPLATE.format(
        source=source,
        malady=malady,
        primary_disease=primary_disease or "<not mapped>",
        pass1_signals=json.dumps(pass1_signals, indent=2, ensure_ascii=False),
        payload=json.dumps(payload or {}, indent=2, ensure_ascii=False)[:25000],
        condition_a=condition_a,
        condition_b=condition_b,
        output_a=json.dumps(output_a, indent=2, ensure_ascii=False),
        output_b=json.dumps(output_b, indent=2, ensure_ascii=False),
        dimension_descriptions=_dimension_descriptions(),
    )
