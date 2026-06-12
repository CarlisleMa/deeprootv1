"""Phase 2 — Task A Baseline B: Text + LLM (no graph, no Pass 1).

The "relevant historical passage + LLM" baseline cell for the Task A
reasoning judge eval. Tests whether the structured graph + multi-agent
pipeline adds value over a single LLM call that sees the historical
text snippets RELEVANT to this Source plus the modern-disease mapping.

By default, this baseline retrieves a per-source passage from the
Shen Nong Ben Cao Jing — ALL mention-windows (±15 lines around each
appearance of the source name or any of its aliases), merged and
deduplicated. This mirrors what an LLM-with-RAG would see if asked
to evaluate the claim. The full-corpus mode (--full-corpus) is
preserved as an alternative for parity with Task B's BaselineLLMPassages.

Architecture (one LLM call per claim):

  Inputs:
    - source name + aliases + traditional_evidence_span
    - malady name + description
    - primary_disease (the modern mapping)
    - relevant historical passage(s) for this source

  Output:
    - SAME JSON schema as task_a_critic.py:
      {verdict, agrees_with_pass1, biological_plausibility, evidence_coherence,
       key_evidence:[{compound, target, reached_disease, why_compelling}],
       concerns:[{concern_type, explanation}], rationale, requires_human_review}

The LLM must:
  (1) Read the historical passage(s) for this source.
  (2) Use parametric knowledge to name compounds, targets, diseases.
  (3) Reason about whether modern science supports the traditional claim.
  (4) Self-assess plausibility, coherence, concerns.

It has NO access to:
  - the structured KG (no compound/target/disease nodes by ID)
  - Pass 1's path counts, tier buckets, loop-closure flags
  - any external lookup or tool beyond the LLM's parametric knowledge

`agrees_with_pass1` is computed by the harness after the call, not
asked of the model (since it never sees Pass 1).

Default model: Gemini 3.1 Flash Lite (cheapest cross-baseline tier).
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
from src.agents.phase2.task_a_critic import (
    CONCERN_TYPES,
    VERDICT_VALUES,
    CRITIC_RESPONSE_SCHEMA,
    _clamp01,
    _verdict_delta,
)
from src.config import GEMINI_MODEL, make_gemini_client
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CORPUS_PATH = "data/historical_corpus/shen_nong_ben_cao_jing_full.txt"
_GEMINI_RATE_LIMIT_S = 1.0
_GEMINI_MAX_RETRIES = 4

# Per-source passage retrieval: take ±N lines around each mention.
_PASSAGE_WINDOW_LINES = 15
# Cap on total passage length per claim. Beyond this, the retrieval
# truncates rather than ballooning the prompt for sources mentioned
# 50+ times across the corpus.
_PASSAGE_CHAR_CAP = 60_000


SYSTEM_PROMPT = """\
You are a biomedical reviewer evaluating whether modern pharmacology supports \
a therapeutic claim from classical Chinese medicine. You have access to:

  - The historical passage(s) about THIS specific Source from the
    Shen Nong Ben Cao Jing (mention-windows merged into one excerpt)
  - The traditional claim itself (a Source treats a Traditional_Malady)
  - The modern Western disease the malady has been mapped to

You DO NOT have access to:
  - any structured database, knowledge graph, or external tool
  - any pre-computed analysis, path scoring, or compound/target lookup

Use only the corpus text and your own training knowledge of natural-product \
pharmacology, target biology, and clinical medicine.

Your task is to:
  1. Find the relevant passage about this Source in the corpus.
  2. Reason about candidate active compounds (from your training knowledge).
  3. Reason about plausible molecular targets and disease mechanisms.
  4. Decide whether the traditional claim is supported by modern evidence.
  5. Self-assess plausibility, coherence, concerns, and review-worthiness.

Output a verdict in EXACTLY the structured JSON schema requested. Be honest \
about uncertainty — if you can't name a specific compound for a target, say \
the link is speculative rather than fabricating one.

NUMERIC RANGES (strict):
  biological_plausibility ∈ [0.0, 1.0]   0=incoherent, 1=biologically obvious
  evidence_coherence      ∈ [0.0, 1.0]   0=evidence contradicts, 1=internally consistent

VERDICT enum (pick exactly one):
  traditional_only    — only traditional support, no modern mechanism known
  mechanistic_only    — modern mechanism plausible but traditional support thin
  unsupported         — no support either way
  partial_support     — some bridge between traditional and modern
  moderate_support    — clear modern mechanism that fits the traditional use
  strong_support      — well-characterized compound, target, and disease link

CONCERN concern_type enum: generic_target, weak_evidence_only, indirect_mechanism, \
wrong_disease_mapping, syndrome_underutilized, promiscuous_compound, \
unverified_evidence, other.

Always set agrees_with_pass1=false (you don't see Pass 1; the harness recomputes it).
"""


CLAIM_PROMPT_TEMPLATE = """\
EVALUATE THIS CLAIM:

  Source:                {source}
  Aliases:               {aliases}
  Traditional Malady:    {malady}
  Mapped Modern Disease: {primary_disease}

  Traditional evidence span (from TREATS edge):
    "{treats_evidence}"

  Malady description:
    "{malady_description}"

HISTORICAL PASSAGE (mention-windows around this Source from the
Shen Nong Ben Cao Jing — {n_windows} window(s), {passage_chars} chars):
<<<
{passage}
>>>

Output JSON matching the schema. The 'agrees_with_pass1' field should be \
set to false; the harness recomputes it after the call.
"""


# ---------------------------------------------------------------------------
# Cypher (read-only — fetch claim metadata only, no path/enrichment)
# ---------------------------------------------------------------------------

_CLAIM_ANCHOR_QUERY = """
MATCH (s:Source {name: $source})-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady})
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
OPTIONAL MATCH (m)-[r_map:MAPS_TO]->(d:Modern_Disease)
  WHERE r_map.is_primary = true
    AND (d.archived IS NULL OR d.archived = false)
RETURN
  s.name AS source,
  s.aliases AS source_aliases,
  s.evidence_span AS source_evidence_span,
  m.name AS malady,
  m.description AS malady_description,
  r.evidence_span AS treats_evidence,
  r.confidence_score AS treats_confidence,
  d.name AS primary_disease
"""


# ---------------------------------------------------------------------------
# Per-source passage retrieval
# ---------------------------------------------------------------------------

def extract_relevant_passage(
    corpus: str,
    source_name: str,
    aliases: list[str] | None = None,
    *,
    window_lines: int = _PASSAGE_WINDOW_LINES,
    char_cap: int = _PASSAGE_CHAR_CAP,
) -> tuple[str, int]:
    """Find every line containing `source_name` or one of `aliases`,
    expand to a ±window_lines context, merge overlapping windows, and
    return (passage_text, n_merged_windows).

    Match is case-insensitive, whole-line substring (not strict word
    boundary — the corpus reformats names with extra spaces, e.g.
    "Ban  Mao  (Mylabris)" so any substring hit on the alias counts).
    """
    aliases = aliases or []
    needles_raw = [source_name] + list(aliases)
    needles = [
        n.lower().strip()
        for n in needles_raw
        if n and len(n.strip()) >= 3  # avoid pathological 1-2 char hits
    ]
    if not needles:
        return ("", 0)

    lines = corpus.splitlines()
    n = len(lines)
    hit_idx: list[int] = []
    for i, line in enumerate(lines):
        low = line.lower()
        if any(needle in low for needle in needles):
            hit_idx.append(i)

    if not hit_idx:
        return ("", 0)

    # Build (start, end) ranges with the window, then merge overlaps.
    ranges: list[tuple[int, int]] = []
    for i in hit_idx:
        start = max(0, i - window_lines)
        end = min(n, i + window_lines + 1)
        ranges.append((start, end))
    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Concatenate with separators, capped at char_cap.
    parts: list[str] = []
    total_chars = 0
    for start, end in merged:
        chunk = "\n".join(lines[start:end])
        header = f"\n--- corpus lines {start + 1}-{end} ---\n"
        block = header + chunk
        if total_chars + len(block) > char_cap:
            remaining = char_cap - total_chars
            if remaining > 200:
                parts.append(block[:remaining] + "\n... [truncated] ...")
            break
        parts.append(block)
        total_chars += len(block)

    return ("\n".join(parts).strip(), len(merged))


_ALL_CLAIMS_QUERY = """
MATCH (s:Source)-[r:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
RETURN s.name AS source, m.name AS malady
ORDER BY source, malady
"""


# ---------------------------------------------------------------------------
# Data shape — mirrors task_a_critic.CriticVerdict so the judge can grade
# all conditions on the same struct
# ---------------------------------------------------------------------------

@dataclass
class BaselineTextVerdict:
    source: str
    malady: str
    primary_disease: str | None
    treats_evidence: str | None

    verdict: str
    agrees_with_pass1: bool             # set by harness, not by LLM
    verdict_delta: int                  # set by harness
    biological_plausibility: float
    evidence_coherence: float

    key_evidence: list[dict]
    concerns: list[dict]
    rationale: str
    requires_human_review: bool

    raw_response: str = ""
    error: str | None = None
    skipped: bool = False
    model: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BaselineTaskATextOnly(BaseAgent):
    """Baseline B: single LLM call with raw corpus, no graph, no Pass 1.

    The output schema mirrors task_a_critic.CriticVerdict so the judge
    can grade all four conditions (KG critic, text-only, graph-only,
    tool-call) on the same fields.
    """

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        corpus_path: str | None = None,
        full_corpus: bool = False,
        passage_window_lines: int = _PASSAGE_WINDOW_LINES,
        passage_char_cap: int = _PASSAGE_CHAR_CAP,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL
        self._corpus_path = Path(corpus_path or _DEFAULT_CORPUS_PATH)
        self._corpus_text: str | None = None
        self._full_corpus = full_corpus
        self._window_lines = passage_window_lines
        self._char_cap = passage_char_cap
        self._last_call = 0.0

    @property
    def name(self) -> str:
        return "BaselineTaskATextOnly"

    def _load_corpus(self) -> str:
        if self._corpus_text is None:
            self._corpus_text = self._corpus_path.read_text(encoding="utf-8")
            self._log_progress(
                f"Loaded corpus: {len(self._corpus_text)} chars from {self._corpus_path}"
            )
        return self._corpus_text

    def run(
        self,
        *,
        claims: list[dict] | None = None,
        source_name: str | None = None,
        malady_name: str | None = None,
        limit: int | None = None,
        pass1_lookup: dict[tuple[str, str], str] | None = None,
        **_: Any,
    ) -> dict:
        """Critique a list of (source, malady) claims with the text-only baseline.

        Args:
            claims: list of {"source": ..., "malady": ...} dicts. If None,
                    all claims are pulled from the graph (or filtered by
                    source_name / malady_name).
            source_name / malady_name: optional single-claim mode.
            limit: cap claims after filtering.
            pass1_lookup: optional {(source, malady) -> pass1_verdict} dict.
                          If provided, the harness fills in verdict_delta /
                          agrees_with_pass1 against Pass 1 verdicts.

        Returns a summary dict in the same shape as TaskAValidator.run().
        """
        target_claims = self._resolve_claims(claims, source_name, malady_name)
        if limit is not None:
            target_claims = target_claims[:limit]

        corpus = self._load_corpus()

        self._log_progress(
            f"Critiquing {len(target_claims)} claim(s) with text-only baseline "
            f"(model={self._model}, corpus_chars={len(corpus)})"
        )

        results: list[BaselineTextVerdict] = []
        errors: list[str] = []

        for i, c in enumerate(target_claims, 1):
            anchor = self._fetch_anchor(c["source"], c["malady"])
            if anchor is None:
                self._log_progress(
                    f"[{i}/{len(target_claims)}] SKIP {c['source']} → {c['malady']} (anchor not found)"
                )
                continue

            self._log_progress(
                f"[{i}/{len(target_claims)}] {anchor['source']} → {anchor['malady']}"
            )
            t0 = time.time()
            try:
                response = self._call_gemini(anchor, corpus)
            except Exception as e:
                errors.append(f"{anchor['source']} -> {anchor['malady']}: {e}")
                results.append(BaselineTextVerdict(
                    source=anchor["source"],
                    malady=anchor["malady"],
                    primary_disease=anchor.get("primary_disease"),
                    treats_evidence=anchor.get("treats_evidence"),
                    verdict="unsupported",
                    agrees_with_pass1=False,
                    verdict_delta=0,
                    biological_plausibility=0.0,
                    evidence_coherence=0.0,
                    key_evidence=[],
                    concerns=[],
                    rationale="",
                    requires_human_review=True,
                    raw_response="",
                    error=str(e),
                    model=self._model,
                    duration_s=time.time() - t0,
                ))
                continue

            verdict = response.get("verdict", "unsupported")
            if verdict not in VERDICT_VALUES:
                verdict = "unsupported"

            cv = BaselineTextVerdict(
                source=anchor["source"],
                malady=anchor["malady"],
                primary_disease=anchor.get("primary_disease"),
                treats_evidence=anchor.get("treats_evidence"),
                verdict=verdict,
                agrees_with_pass1=False,        # harness will recompute
                verdict_delta=0,
                biological_plausibility=_clamp01(response.get("biological_plausibility")),
                evidence_coherence=_clamp01(response.get("evidence_coherence")),
                key_evidence=list(response.get("key_evidence") or []),
                concerns=list(response.get("concerns") or []),
                rationale=str(response.get("rationale") or ""),
                requires_human_review=bool(response.get("requires_human_review")),
                raw_response=json.dumps(response, ensure_ascii=False),
                model=self._model,
                duration_s=time.time() - t0,
            )

            if pass1_lookup is not None:
                pass1_v = pass1_lookup.get((anchor["source"], anchor["malady"]))
                if pass1_v:
                    cv.verdict_delta = _verdict_delta(pass1_v, cv.verdict)
                    cv.agrees_with_pass1 = (cv.verdict_delta == 0)

            results.append(cv)

        return {
            "agent": self.name,
            "model": self._model,
            "claims_total": len(results),
            "errors": errors,
            "verdicts": [asdict(r) for r in results],
            "completed_at": dt.datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_claims(
        self,
        claims: list[dict] | None,
        source: str | None,
        malady: str | None,
    ) -> list[dict]:
        if claims is not None:
            return list(claims)
        if source and malady:
            return [{"source": source, "malady": malady}]
        if source:
            return [
                r for r in self.client.run(_ALL_CLAIMS_QUERY)
                if r["source"] == source
            ]
        return list(self.client.run(_ALL_CLAIMS_QUERY))

    def _fetch_anchor(self, source: str, malady: str) -> dict | None:
        rows = self.client.run(
            _CLAIM_ANCHOR_QUERY, {"source": source, "malady": malady},
        )
        return rows[0] if rows else None

    def _build_passage(self, anchor: dict, corpus: str) -> tuple[str, int]:
        """Pick the historical passage to send to the LLM. Default:
        per-source mention windows. With --full-corpus: the whole
        Shen Nong text.
        """
        if self._full_corpus:
            return (corpus, 1)
        aliases = anchor.get("source_aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        passage, n_windows = extract_relevant_passage(
            corpus,
            anchor["source"],
            aliases,
            window_lines=self._window_lines,
            char_cap=self._char_cap,
        )
        # Always seed with the source's curated evidence_span (if any)
        # so the LLM at least sees what Phase 1 used to anchor the node,
        # in case the corpus retrieval missed.
        seed = anchor.get("source_evidence_span") or ""
        if seed and seed.strip():
            passage = (
                f"--- source.evidence_span (Phase 1 anchor) ---\n{seed.strip()}\n\n"
                + passage
            )
        if not passage.strip():
            # Fall back to just the seed + treats_evidence so the LLM has
            # something to work with even when the source name doesn't
            # appear in the corpus the way we expect.
            passage = (
                f"--- source.evidence_span ---\n{seed or '<no anchor>'}\n"
                f"--- treats edge evidence_span ---\n"
                f"{anchor.get('treats_evidence') or '<no treats span>'}"
            )
            n_windows = 0
        return passage, n_windows

    def _call_gemini(self, anchor: dict, corpus: str) -> dict:
        elapsed = time.time() - self._last_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_call = time.time()

        passage, n_windows = self._build_passage(anchor, corpus)
        aliases = anchor.get("source_aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        prompt = CLAIM_PROMPT_TEMPLATE.format(
            source=anchor["source"],
            aliases=", ".join(aliases) if aliases else "<none>",
            malady=anchor["malady"],
            primary_disease=anchor.get("primary_disease") or "<not mapped>",
            treats_evidence=anchor.get("treats_evidence") or "<no evidence span>",
            malady_description=anchor.get("malady_description") or "<no description>",
            n_windows=n_windows,
            passage_chars=len(passage),
            passage=passage,
        )

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
                        response_schema=CRITIC_RESPONSE_SCHEMA,
                    ),
                )
                return json.loads(response.text)
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < _GEMINI_MAX_RETRIES - 1:
                    wait_s = 5 * (2 ** attempt)
                    logger.info(
                        "Baseline rate-limited, backoff %ds (attempt %d/%d)",
                        wait_s, attempt + 1, _GEMINI_MAX_RETRIES,
                    )
                    time.sleep(wait_s)
                    continue
                raise
        if last_err:
            raise last_err
        raise RuntimeError("Baseline call failed without raising")
