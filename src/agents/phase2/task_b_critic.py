"""Phase 2 — Task B Critic (Pass 4): LLM re-ranker over deterministic nominations.

Pass 3 (`TaskBNominator`) produces ranked compound nominations using
tier-bucket scoring + multiplicative tiebreak — fast, deterministic,
auditable. Pass 4 (this agent) takes the top-N nominations and asks
Gemini Pro to re-rank them based on biomedical plausibility, target
specificity, and pharmacological coherence.

Architecture: ONE LLM call per disease (not per candidate). The full
top-N list is serialized into a compact payload — compound metadata,
top forward path, top historical path, KNOWN_TREATS for other
diseases, aggregate signals — and the LLM returns a re-ordered top-K
with per-candidate plausibility, concerns, and rationale.

This mirrors Task A's two-pass design:
  Task A: TaskAValidator (Pass 1) → CriticAgent (Pass 2)
  Task B: TaskBNominator (Pass 3) → TaskBCritic (Pass 4)

Default behaviour:
  - Input: top-50 nominations from Pass 3
  - Output: re-ranked top-20 with structured per-candidate notes
  - Read-only — no graph writes
  - Skips diseases with empty / very short candidate lists

The eval harness uses Pass 4 by passing `--use-critic`. Pass 4's
output is dedup'd by planar key the same way Pass 3's is, so
recall@K comparisons across configurations are apples-to-apples.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_MODEL_PRO, make_gemini_client
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GEMINI_RATE_LIMIT_S = 1.0
_GEMINI_MAX_RETRIES = 4

# Concern enum mirrors Task A's critic so paper aggregations work
# across both tasks.
CONCERN_TYPES = [
    "generic_target",
    "weak_evidence_only",
    "indirect_mechanism",
    "wrong_disease_mapping",
    "syndrome_underutilized",
    "promiscuous_compound",
    "unverified_evidence",
    "mismatched_pharmacology",   # NEW for Task B: candidate's known activity doesn't fit the disease class
    "non_drug_substance",        # NEW for Task B: candidate is a nutrient/mineral, not a drug-shaped molecule
    "other",
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = """\
You are a biomedical drug-discovery expert evaluating compound \
candidates for a query disease. The candidates are nominated by a \
deterministic graph search over historical Chinese medicine records \
combined with modern target/disease databases. Each candidate has:

  - Historical evidence: which TCM source(s) contain the compound and \
which traditional malady the source treats (with evidence span from \
the historical text).
  - Modern mechanistic evidence: which biological target(s) the compound \
binds, and whether those targets associate with the query disease in \
Open Targets or other ontologies.
  - Polypharmacology context: KNOWN_TREATS edges to OTHER diseases \
(this is real-world clinical reality, not the held-out signal).
  - Quantitative signals: source corroboration count, target convergence \
count, evidence_type tier, multiplicative score.

Your job is to RE-RANK these candidates by biomedical plausibility for \
the query disease. Apply criteria the deterministic system can't:

  1. TARGET SPECIFICITY. Does the candidate hit a target mechanistically \
relevant to the disease, or is it a generic target that binds 100s of \
diseases? Mention it explicitly when you down-rank for genericity.
  2. PHARMACOLOGICAL COHERENCE. Does the compound's known pharmacology \
(KNOWN_TREATS for OTHER diseases, target spectrum) actually fit a story \
where it could treat the query disease? E.g. an anti-inflammatory \
compound for an inflammatory disease is coherent; a vitamin / mineral \
without specific target binding is incoherent.
  3. HISTORICAL CONSISTENCY. Does the malady the compound's source \
treats look semantically related to the modern disease? "Wind cold" → \
common cold is reasonable; "wind cold" → epilepsy is suspicious.
  4. EVIDENCE QUALITY. Phenotypic radioligand binding ≠ functional \
therapeutic efficacy. Wood-tier paths are real biology but weak \
evidence; gold-tier mechanism / strong activity / OT-strong are much \
stronger.

Output a re-ordered top-K list with:
  - rank (1 to K, 1 = most plausible)
  - compound_inchikey (must match an input candidate exactly)
  - plausibility ∈ [0.0, 1.0]
  - concerns: list of {concern_type (from enum), explanation}
  - rationale: 1-2 sentences citing specific input fields

Constraints:
  - Output exactly K candidates, ordered by your judgment.
  - Every output compound_inchikey MUST appear in the input list.
  - Cite specific input fields (target name, source name, KNOWN_TREATS \
disease, etc.) — never speculate beyond what's provided.
  - plausibility is a 2-decimal float in [0, 1]; don't output 0-10 or \
0-100.
"""


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


def _make_schema(top_k: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "disease": {"type": "string"},
            "rerank": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "compound_inchikey": {"type": "string"},
                        "compound_name": {"type": "string"},
                        "plausibility": {"type": "number"},
                        "concerns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "concern_type": {
                                        "type": "string",
                                        "enum": CONCERN_TYPES,
                                    },
                                    "explanation": {"type": "string"},
                                },
                                "required": ["concern_type", "explanation"],
                            },
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "rank",
                        "compound_inchikey",
                        "compound_name",
                        "plausibility",
                        "concerns",
                        "rationale",
                    ],
                },
            },
            "overall_assessment": {"type": "string"},
        },
        "required": ["disease", "rerank", "overall_assessment"],
    }


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class CriticReranked:
    """One re-ranked compound nomination after the LLM critic pass."""
    rank: int
    compound: str
    compound_inchikey: str

    # LLM judgment
    plausibility: float
    concerns: list[dict]
    rationale: str

    # Original Pass-3 fields (carried forward for evidence card display)
    pass3_rank: int | None
    pass3_has_loop_closure: bool | None
    pass3_forward_bucket: str | None
    pass3_forward_max_score: float | None
    pass3_unique_sources: int | None
    pass3_unique_targets: int | None

    # Audit
    pass3_card: dict | None = None      # full original nomination dict (optional)


@dataclass
class TaskBCriticResult:
    disease: str
    input_count: int
    reranked: list[CriticReranked]
    overall_assessment: str
    skipped: bool = False
    error: str | None = None
    duration_s: float = 0.0
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp01(x: Any) -> float:
    if x is None:
        return 0.0
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if v != v:
        return 0.0
    return max(0.0, min(1.0, v))


def _compact_candidate(n: dict) -> dict:
    """Compact representation of one Pass-3 nomination for the LLM payload.

    Trades verbosity for token budget. Each card is ~150-250 tokens.
    Top forward + top historical paths only; aggregate signals; KNOWN_TREATS
    for other diseases.
    """
    top_fwd = (n.get("top_forward_paths") or [None])[0]
    top_hist = (n.get("top_historical_paths") or [None])[0]

    fwd_compact = None
    if top_fwd:
        fwd_compact = {
            "target": top_fwd.get("gene_symbol") or top_fwd.get("target"),
            "uniprot_id": top_fwd.get("uniprot_id"),
            "target_type": top_fwd.get("target_type"),
            "evidence_type": top_fwd.get("tgt_type"),
            "tier": top_fwd.get("tgt_tier"),
            "pchembl": top_fwd.get("tgt_pchembl"),
            "assay_description": (top_fwd.get("assay_description") or "")[:240],
            "mechanism_action": top_fwd.get("mechanism_action"),
            "rel_evidence_type": top_fwd.get("rel_type"),
            "rel_tier": top_fwd.get("rel_tier"),
            "ot_overall_score": top_fwd.get("ot_score"),
            "rel_rationale": (top_fwd.get("rel_rationale") or "")[:160],
        }
    hist_compact = None
    if top_hist:
        hist_compact = {
            "source": top_hist.get("source"),
            "malady": top_hist.get("malady"),
            "malady_description": (top_hist.get("malady_description") or "")[:200],
            "treats_evidence_span": (top_hist.get("treats_evidence_span") or "")[:240],
            "ex_evidence_type": top_hist.get("ex_type"),
            "ex_tier": top_hist.get("ex_tier"),
            "mapper_rationale": (top_hist.get("mapper_rationale") or "")[:200],
        }

    other_kt = (n.get("other_known_treats") or [])[:5]
    other_kt_compact = [
        {
            "disease": k.get("disease"),
            "clinical_phase": k.get("clinical_phase"),
        }
        for k in other_kt
    ]

    return {
        "pass3_rank": n.get("rank"),
        "compound_inchikey": n.get("compound_inchikey"),
        "compound": n.get("compound"),
        "smiles": n.get("smiles"),
        "molecular_formula": n.get("molecular_formula"),
        "np_likeness": n.get("np_likeness"),
        "chembl_id": n.get("chembl_id"),
        "aggregate_signals": {
            "unique_sources": n.get("unique_sources"),
            "unique_maladies": n.get("unique_maladies"),
            "unique_targets": n.get("unique_targets"),
            "historical_path_count": n.get("historical_path_count"),
            "forward_path_count": n.get("forward_path_count"),
            "has_loop_closure": n.get("has_loop_closure"),
            "forward_bucket": n.get("forward_bucket"),
            "forward_max_score": n.get("forward_max_score"),
            "composite_score": n.get("composite_score"),
        },
        "top_forward_path": fwd_compact,
        "top_historical_path": hist_compact,
        "other_known_treats": other_kt_compact,
    }


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TaskBCritic(BaseAgent):
    """Pass-4 LLM re-ranker. Read-only, one Gemini Pro call per disease."""

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL_PRO
        self._last_call = 0.0

    @property
    def name(self) -> str:
        return "TaskBCritic"

    # ------------------------------------------------------------------

    def run(
        self,
        *,
        disease_name: str,
        nominations: list[dict],
        input_top_n: int = 50,
        output_top_k: int = 20,
        include_pass3_cards: bool = False,
        **_: Any,
    ) -> TaskBCriticResult:
        """Re-rank Pass-3 nominations.

        Args:
          disease_name: query disease.
          nominations: list of Pass-3 nomination dicts (from
              TaskBNominator.run()['nominations']).
          input_top_n: take the top N Pass-3 candidates as LLM input
              (default 50).
          output_top_k: ask the LLM to return the top K of those
              (default 20).
          include_pass3_cards: when True, attach the full original
              Pass-3 nomination dict to each CriticReranked entry.
        """
        t_start = time.time()

        if not nominations:
            return TaskBCriticResult(
                disease=disease_name, input_count=0,
                reranked=[], overall_assessment="(no candidates to rank)",
                skipped=True, duration_s=0.0,
            )

        # Take top-N from Pass 3 as input
        candidates = nominations[:input_top_n]
        # Effective output K can't exceed the input list size
        k = min(output_top_k, len(candidates))

        # Index by inchikey for post-LLM lookup
        by_ik: dict[str, dict] = {
            n.get("compound_inchikey", ""): n for n in candidates
        }

        try:
            llm_response = self._call_gemini(disease_name, candidates, k)
        except Exception as e:
            logger.warning("TaskBCritic Gemini call failed for %r: %s",
                           disease_name, e)
            return TaskBCriticResult(
                disease=disease_name,
                input_count=len(candidates),
                reranked=[], overall_assessment="",
                error=str(e),
                duration_s=round(time.time() - t_start, 2),
            )

        # Map LLM output back to full nomination cards. Trust the LLM's
        # ordering; coerce its inchikey to one in our input set; drop
        # fabricated entries.
        rerank_in = llm_response.get("rerank") or []
        reranked: list[CriticReranked] = []
        seen: set[str] = set()
        for entry in rerank_in:
            ik = entry.get("compound_inchikey", "")
            if ik not in by_ik or ik in seen:
                continue
            seen.add(ik)
            orig = by_ik[ik]
            reranked.append(CriticReranked(
                rank=len(reranked) + 1,
                compound=orig.get("compound", entry.get("compound_name", "")),
                compound_inchikey=ik,
                plausibility=_clamp01(entry.get("plausibility")),
                concerns=entry.get("concerns") or [],
                rationale=str(entry.get("rationale") or ""),
                pass3_rank=orig.get("rank"),
                pass3_has_loop_closure=orig.get("has_loop_closure"),
                pass3_forward_bucket=orig.get("forward_bucket"),
                pass3_forward_max_score=orig.get("forward_max_score"),
                pass3_unique_sources=orig.get("unique_sources"),
                pass3_unique_targets=orig.get("unique_targets"),
                pass3_card=orig if include_pass3_cards else None,
            ))
            if len(reranked) >= k:
                break

        return TaskBCriticResult(
            disease=disease_name,
            input_count=len(candidates),
            reranked=reranked,
            overall_assessment=str(llm_response.get("overall_assessment") or ""),
            duration_s=round(time.time() - t_start, 2),
            raw_response=json.dumps(llm_response, ensure_ascii=False),
        )

    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        disease_name: str,
        candidates: list[dict],
        top_k: int,
    ) -> dict:
        # Polite rate-limit (mirrors Task A critic)
        elapsed = time.time() - self._last_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_call = time.time()

        prompt = self._build_prompt(disease_name, candidates, top_k)
        schema = _make_schema(top_k)

        last_err: Exception | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                response = self._gemini.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=CRITIC_SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < _GEMINI_MAX_RETRIES - 1:
                    wait_s = 5 * (2 ** attempt)
                    logger.info(
                        "TaskBCritic rate-limited, backoff %ds (attempt %d/%d)",
                        wait_s, attempt + 1, _GEMINI_MAX_RETRIES,
                    )
                    time.sleep(wait_s)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("TaskBCritic call failed without raising")

    @staticmethod
    def _build_prompt(
        disease_name: str,
        candidates: list[dict],
        top_k: int,
    ) -> str:
        compact = [_compact_candidate(n) for n in candidates]
        payload = {
            "query_disease": disease_name,
            "task": (
                f"Re-rank the following {len(candidates)} compound "
                f"candidates by biomedical plausibility for treating the "
                f"query disease. Return exactly {top_k} candidates in "
                "ranked order."
            ),
            "output_top_k": top_k,
            "candidates": compact,
        }
        return (
            "Here are the candidate nominations to re-rank. Output "
            "structured JSON per the schema.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )


def reranked_to_dict(r: CriticReranked) -> dict:
    return asdict(r)


def result_to_dict(r: TaskBCriticResult) -> dict:
    out = asdict(r)
    out["reranked"] = [reranked_to_dict(x) for x in r.reranked]
    return out
