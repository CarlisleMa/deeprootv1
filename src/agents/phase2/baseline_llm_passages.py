"""Phase 2 — Baseline: LLM with raw corpus passages, no graph.

Eval-B baseline cell. Tests: *does the structured graph and the
multi-agent pipeline add value over a single Gemini Pro call that
sees the same raw historical text?*

Architecture (one LLM call per disease query):

  Inputs:
    - disease name (the query)
    - the FULL Shen Nong Ben Cao Jing text (~130k tokens, fits Pro)
    - a list of compound names to EXCLUDE (the masked stereo-siblings
      of the held-out test compound, so the eval is fair)

  Output:
    - structured JSON: top-K ranked compound nominations
      {compound_name, source_herb, rationale, plausibility ∈ [0,1]}

The LLM has to:
  (1) read the corpus and find herbs claimed to treat the disease
  (2) bridge herb name → known active compound (drawing on its
      training data on TCM pharmacology)
  (3) rank by likelihood of treating the query disease

This is a pure single-agent baseline — no Phase 1 extraction, no
Phase 2 deterministic ranking, no critic. The LLM does everything in
one shot from raw text. It mirrors what a domain expert would do if
handed the corpus and asked for candidate compounds.

Caveat: The LLM was trained on biomedical literature including
ChEMBL drug_indication content. Some leakage of the held-out
KNOWN_TREATS signal is possible — we explicitly mask the test
compound's planar siblings in the prompt to mitigate this, but a
well-trained model may still recall published associations. Report
numbers honestly with this disclosure.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_MODEL_PRO, make_gemini_client
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CORPUS_PATH = "data/historical_corpus/shen_nong_ben_cao_jing_full.txt"
_GEMINI_RATE_LIMIT_S = 1.0
_GEMINI_MAX_RETRIES = 4


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a biomedical expert specializing in classical Chinese medicine
and natural-product drug discovery. You will receive:

  - A query: a modern Western disease name to find treatments for
  - The full text of a classical Chinese medical text (Shen Nong Ben
    Cao Jing, "The Divine Farmer's Materia Medica")
  - An EXCLUDE list: chemical compound names that should NOT be in
    your output (these are masked from the evaluation)

Your task:

  1. Identify herbs in the text that are claimed (or that you know
     from training data on TCM pharmacology) to treat the query
     disease or related conditions.
  2. For each such herb, name its primary active chemical compound(s)
     (using your knowledge of natural-product chemistry).
  3. Filter out any compound on the EXCLUDE list.
  4. Return a ranked list of top-K candidate compounds, ordered from
     most to least likely treatment.

For each candidate, provide:
  - compound_name: the molecular compound (e.g., "ephedrine",
    "berberine", "artemisinin"), NOT the herb name
  - source_herb: the herb mentioned in the text where this compound
    is found
  - rationale: 1-2 sentences citing relevant text or pharmacology
  - plausibility: float in [0.0, 1.0] reflecting your confidence

Constraints:
  - Use the most common English or Latin chemical name
  - DO NOT include compounds in the exclude list
  - Provide novel candidates, not already-known treatments — this is
    a drug-repurposing task
  - If you don't have strong evidence for a candidate, give it a
    plausibility < 0.5 rather than excluding it (we want broad
    coverage in the top-K)
"""


def _make_response_schema(top_k: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "disease": {"type": "string"},
            "nominations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer"},
                        "compound_name": {"type": "string"},
                        "source_herb": {"type": "string"},
                        "rationale": {"type": "string"},
                        "plausibility": {"type": "number"},
                    },
                    "required": [
                        "rank",
                        "compound_name",
                        "source_herb",
                        "rationale",
                        "plausibility",
                    ],
                },
            },
            "overall_strategy": {"type": "string"},
        },
        "required": ["disease", "nominations", "overall_strategy"],
    }


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass
class LLMNomination:
    rank: int
    compound_name: str
    source_herb: str
    rationale: str
    plausibility: float

    # Resolved against our graph (post-LLM)
    matched_inchikey: str | None = None
    matched_planar_key: str | None = None
    matched_compound_name: str | None = None  # canonical name in graph


@dataclass
class BaselineLLMPassagesResult:
    disease: str
    masked_compound_count: int
    input_tokens_estimate: int
    nominations: list[LLMNomination]
    overall_strategy: str
    duration_s: float
    error: str | None = None
    raw_response: str = ""


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class BaselineLLMPassages(BaseAgent):
    """Single LLM call, raw corpus, no graph. Eval-B baseline cell."""

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        corpus_path: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL_PRO
        self._last_call = 0.0
        path = Path(corpus_path or _DEFAULT_CORPUS_PATH)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")
        self._corpus_text = path.read_text(encoding="utf-8")
        self._corpus_path = str(path)

    @property
    def name(self) -> str:
        return "BaselineLLMPassages"

    # ------------------------------------------------------------------

    def run(
        self,
        *,
        disease_name: str,
        excluded_compound_names: list[str] | None = None,
        top_k: int = 20,
        resolve_in_graph: bool = True,
        **_: Any,
    ) -> BaselineLLMPassagesResult:
        """Run the baseline for one disease.

        Args:
          disease_name: the query disease (e.g. "Stroke", "Common Cold").
          excluded_compound_names: compound names to instruct the LLM to
              exclude. The eval harness passes the masked planar
              siblings of the held-out test compound here.
          top_k: number of nominations to return.
          resolve_in_graph: if True, attempt to map each LLM-output
              compound name to a Chemical_Compound node in our graph,
              attaching the matched inchikey + planar key for downstream
              recall computation. Set False for inspection-only runs.
        """
        t_start = time.time()
        excluded = excluded_compound_names or []

        try:
            llm_response = self._call_gemini(
                disease_name=disease_name,
                excluded=excluded,
                top_k=top_k,
            )
        except Exception as e:
            logger.warning(
                "BaselineLLMPassages failed for %r: %s",
                disease_name, e,
            )
            return BaselineLLMPassagesResult(
                disease=disease_name,
                masked_compound_count=len(excluded),
                input_tokens_estimate=len(self._corpus_text) // 4,
                nominations=[],
                overall_strategy="",
                duration_s=round(time.time() - t_start, 2),
                error=str(e),
            )

        # Parse + clamp
        raw_noms = llm_response.get("nominations") or []
        nominations: list[LLMNomination] = []
        for i, n in enumerate(raw_noms[:top_k]):
            nominations.append(LLMNomination(
                rank=int(n.get("rank") or i + 1),
                compound_name=str(n.get("compound_name") or ""),
                source_herb=str(n.get("source_herb") or ""),
                rationale=str(n.get("rationale") or ""),
                plausibility=_clamp01(n.get("plausibility")),
            ))

        # Resolve compound names → graph nodes for recall computation
        if resolve_in_graph and nominations:
            self._resolve_in_graph(nominations)

        return BaselineLLMPassagesResult(
            disease=disease_name,
            masked_compound_count=len(excluded),
            input_tokens_estimate=len(self._corpus_text) // 4,
            nominations=nominations,
            overall_strategy=str(llm_response.get("overall_strategy") or ""),
            duration_s=round(time.time() - t_start, 2),
            raw_response=json.dumps(llm_response, ensure_ascii=False),
        )

    # ------------------------------------------------------------------

    def _call_gemini(
        self,
        disease_name: str,
        excluded: list[str],
        top_k: int,
    ) -> dict:
        elapsed = time.time() - self._last_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_call = time.time()

        excluded_str = (
            ", ".join(repr(n) for n in excluded)
            if excluded else "(none)"
        )
        prompt = (
            f"QUERY DISEASE: {disease_name}\n\n"
            f"OUTPUT TOP-K: {top_k}\n\n"
            f"EXCLUDE THESE COMPOUNDS (masked from evaluation): "
            f"{excluded_str}\n\n"
            f"--- BEGIN CLASSICAL CORPUS (Shen Nong Ben Cao Jing) ---\n\n"
            f"{self._corpus_text}\n\n"
            f"--- END CORPUS ---\n\n"
            f"Now produce the ranked top-{top_k} compound nominations "
            f"for {disease_name!r}. Output JSON per the schema."
        )

        schema = _make_response_schema(top_k)
        last_err: Exception | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                response = self._gemini.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
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
                        "BaselineLLMPassages rate-limited, backoff %ds (%d/%d)",
                        wait_s, attempt + 1, _GEMINI_MAX_RETRIES,
                    )
                    time.sleep(wait_s)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("BaselineLLMPassages call failed without raising")

    # ------------------------------------------------------------------
    # Graph resolution
    # ------------------------------------------------------------------

    def _resolve_in_graph(self, nominations: list[LLMNomination]) -> None:
        """Map LLM-output compound names to Chemical_Compound nodes via
        case-insensitive name match. Attach inchikey + planar key for
        downstream recall computation. Leaves matched_* as None when no
        match is found."""
        names = [n.compound_name for n in nominations if n.compound_name]
        if not names:
            return

        # Case-insensitive batched lookup. Returns multiple inchikeys per
        # name (stereo siblings); we take the first.
        rows = self.client.run("""
            UNWIND $names AS qname
            MATCH (c:Chemical_Compound)
            WHERE toLower(c.name) = toLower(qname)
              AND (c.archived IS NULL OR c.archived = false)
            RETURN qname, c.inchikey AS inchikey, c.name AS canonical
            LIMIT 200
        """, {"names": names})

        # Group by query name (lowercased), keep first match
        by_name: dict[str, dict] = {}
        for r in rows:
            qname = (r.get("qname") or "").lower()
            if qname not in by_name:
                by_name[qname] = {
                    "inchikey": r.get("inchikey"),
                    "canonical": r.get("canonical"),
                }

        for n in nominations:
            key = n.compound_name.lower()
            if key in by_name:
                ik = by_name[key]["inchikey"]
                n.matched_inchikey = ik
                n.matched_planar_key = (ik or "")[:14] or None
                n.matched_compound_name = by_name[key]["canonical"]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def nomination_to_dict(n: LLMNomination) -> dict:
    return asdict(n)


def result_to_dict(r: BaselineLLMPassagesResult) -> dict:
    out = asdict(r)
    out["nominations"] = [nomination_to_dict(x) for x in r.nominations]
    out["corpus_path"] = getattr(r, "_corpus_path", None)
    return out
