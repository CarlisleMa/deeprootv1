"""Phase 2 — Task A Baseline D: LLM with EXTERNAL biomedical APIs.

The "agentic AI" baseline cell: an LLM with the SAME external biomedical
APIs we used to build the Phase-1 KG. Tests the ablation question:

  "Can an agentic LLM with real biomedical APIs match what the
   pre-built KG gives the proposed-system critic?"

That's a stronger ablation than wrapping our own graph queries (which
answered a softer "scripted vs agentic on the same KG" question).

Tools available to the agent (4 client-executed function tools, all
free public APIs):

    1. lookup_chembl_compound(compound_name): ChEMBL REST API.
       Returns mechanisms, activities (with pChEMBL), and clinical
       indications for a compound. Wraps src/data/chembl.fetch_compound_data
       and fetch_compound_indications.

    2. query_opentargets_target_diseases(uniprot_id):
       Open Targets GraphQL. Returns top diseases the target is
       associated with, with overall + per-datasource scores. Wraps
       src/data/open_targets.fetch_target_associations_batch.

    3. pubmed_search(query, max_results=5): NCBI E-utils esearch +
       esummary. Returns top PMIDs with titles for evidence anchoring.

    4. lookup_mesh_term(term): NLM MeSH lookup. Resolves a free-text
       symptom or disease to MeSH descriptors with IDs.

google_search was experimentally bundled in but Gemini Flash Lite
picked it 27/30 times and never invoked the biomedical APIs (see
results/task_a_pass2_baseline_tools_SEARCH_DOMINANT.json). For the
clean ablation against the KG critic we drop it; the LLM's parametric
knowledge already covers the "open web search" baseline.

Architecture:
  - One Gemini call per round. Up to MAX_TOOL_ROUNDS=10 rounds.
  - Final round forces structured JSON in the same critic schema.
  - Each tool result is recorded in tool_call_log[*].result_summary.rows
    so the judge can citation-fidelity-check what the agent actually saw.

Default model: Gemini 3.1 Flash Lite (cheapest cross-baseline tier).
The agent has NO access to the proposed-system KG — its only path to
graph-side data is via the same external APIs that originally populated
the KG.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import quote

import requests
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
from src.data import chembl, open_targets
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TOOL_ROUNDS = 10
_GEMINI_RATE_LIMIT_S = 1.0
_GEMINI_MAX_RETRIES = 4

_NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_MESH_LOOKUP = "https://id.nlm.nih.gov/mesh/lookup/descriptor"
_HTTP_TIMEOUT_S = 15
_TOOL_RESULT_ROW_CAP = 20  # cap rows returned per tool call (cost + audit)


# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------

_LOOKUP_CHEMBL_DECL = types.FunctionDeclaration(
    name="lookup_chembl_compound",
    description=(
        "Look up a natural-product or drug compound by name in the ChEMBL "
        "database. Returns curated mechanisms of action, bioactivity assays "
        "with pChEMBL binding scores, target IDs (gene_symbol, uniprot_id, "
        "chembl_id), and clinical indications (max-phase, MeSH disease)."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "compound_name": types.Schema(
                type="STRING",
                description=(
                    "The canonical compound name (e.g. 'cantharidin', "
                    "'amygdalin', 'baicalein'). The tool tries InChIKey/name "
                    "fuzzy matching."
                ),
            ),
        },
        required=["compound_name"],
    ),
)

_QUERY_OT_DECL = types.FunctionDeclaration(
    name="query_opentargets_target_diseases",
    description=(
        "Look up the top diseases associated with a biological target via "
        "the Open Targets Platform. Returns disease names, EFO IDs, and "
        "overall association scores (0-1) drawn from genetics, drugs, "
        "literature, RNA expression, and animal models."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "target_id": types.Schema(
                type="STRING",
                description=(
                    "UniProt ID (e.g. 'P23219' for COX-1) OR ChEMBL target "
                    "ID (e.g. 'CHEMBL230'). UniProt is preferred."
                ),
            ),
        },
        required=["target_id"],
    ),
)

_PUBMED_DECL = types.FunctionDeclaration(
    name="pubmed_search",
    description=(
        "Search PubMed for biomedical literature on a query (compound name, "
        "target, disease, or mechanism). Returns top PMIDs with titles and "
        "publication years for evidence anchoring."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "query": types.Schema(
                type="STRING",
                description=(
                    "PubMed-style query, e.g. 'cantharidin urolithiasis', "
                    "'amygdalin antitussive', or 'baicalein COX-2'."
                ),
            ),
            "max_results": types.Schema(
                type="INTEGER",
                description="Number of results to return (1-20, default 5).",
            ),
        },
        required=["query"],
    ),
)

_MESH_DECL = types.FunctionDeclaration(
    name="lookup_mesh_term",
    description=(
        "Resolve a free-text symptom or disease term to formal MeSH "
        "descriptors with their MeSH IDs. Useful for grounding the "
        "traditional malady → modern disease mapping in standard medical "
        "ontology (e.g. 'urinary obstruction' → MeSH descriptors)."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "term": types.Schema(
                type="STRING",
                description="Free-text disease or symptom name to resolve.",
            ),
        },
        required=["term"],
    ),
)


_FUNCTION_TOOL = types.Tool(function_declarations=[
    _LOOKUP_CHEMBL_DECL,
    _QUERY_OT_DECL,
    _PUBMED_DECL,
    _MESH_DECL,
])
# google_search was tested and dropped: with mixed tools, Gemini Flash
# Lite picks the cheap one-shot search 27/30 times (see
# task_a_pass2_baseline_tools_SEARCH_DOMINANT.json for that experiment).
# The valid ablation question — does a pre-built KG add value over an
# LLM agent that can hit the SAME external sources we used to build it
# — needs the agent to actually exercise ChEMBL / Open Targets / PubMed
# / MeSH, so we expose only those.
_TOOLS = [_FUNCTION_TOOL]
_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(mode="AUTO"),
)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a biomedical reviewer evaluating whether modern pharmacology supports \
a therapeutic claim from classical Chinese medicine. You have access to four \
biomedical-database tools and you must use them to answer:

  • lookup_chembl_compound: ChEMBL — compound mechanisms, activities, indications
  • query_opentargets_target_diseases: Open Targets — target → disease scores
  • pubmed_search: PubMed — biomedical literature (titles + PMIDs)
  • lookup_mesh_term: MeSH — formal disease/term ontology

You DO NOT have access to a pre-built knowledge graph of TCM compounds, \
targets, or diseases. Build your case using the tools as a real expert \
would investigating a herb monograph: pick candidate active compounds, \
look up their targets in ChEMBL, see which diseases those targets relate to \
in Open Targets, scan PubMed for mechanism papers, and verify the malady → \
modern disease mapping with MeSH.

REQUIRED INVESTIGATION POLICY:
  1. You MUST call at least one of {lookup_chembl_compound, \
query_opentargets_target_diseases, pubmed_search} before producing the final \
verdict — these contain the empirical mechanism evidence.
  2. Plan your queries: start with lookup_chembl_compound on the most likely \
active constituent of the source. Then query_opentargets_target_diseases on \
its top targets. Cross-check with pubmed_search for any specific compound + \
disease pair you want to validate.
  3. Use lookup_mesh_term to verify the malady → mapped disease link is \
clinically sensible.

After ~5-10 rounds of investigation, output the final verdict in the \
structured JSON schema. Be specific and cite the actual compounds, targets, \
and diseases you discovered through tools — don't fabricate evidence and \
don't lean on parametric knowledge if a tool can verify it.

NUMERIC RANGES (strict):
  biological_plausibility ∈ [0.0, 1.0]   0=incoherent, 1=biologically obvious
  evidence_coherence      ∈ [0.0, 1.0]   0=evidence contradicts, 1=internally consistent

VERDICT enum (pick exactly one):
  traditional_only, mechanistic_only, unsupported,
  partial_support, moderate_support, strong_support

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

Use lookup_chembl_compound, query_opentargets_target_diseases, pubmed_search, \
and lookup_mesh_term to investigate. Make at least 2-3 tool calls before \
producing the final verdict JSON. Cite only compounds, targets, and diseases \
that appear in your tool results.
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
RETURN s.name AS source,
       s.aliases AS source_aliases,
       m.name AS malady,
       m.description AS malady_description,
       r.evidence_span AS treats_evidence,
       d.name AS primary_disease
"""


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------

@dataclass
class BaselineToolCallVerdict:
    source: str
    malady: str
    primary_disease: str | None
    treats_evidence: str | None

    verdict: str
    agrees_with_pass1: bool
    verdict_delta: int
    biological_plausibility: float
    evidence_coherence: float

    key_evidence: list[dict]
    concerns: list[dict]
    rationale: str
    requires_human_review: bool

    raw_response: str = ""
    tool_call_log: list[dict] = field(default_factory=list)
    tool_calls_count: int = 0
    rounds_used: int = 0
    error: str | None = None
    skipped: bool = False
    model: str = ""
    duration_s: float = 0.0


# ---------------------------------------------------------------------------
# Tool execution wrappers
# ---------------------------------------------------------------------------

def _exec_chembl(compound_name: str) -> dict:
    """Wrap chembl.fetch_compound_data + fetch_compound_indications."""
    if not compound_name:
        return {"error": "missing compound_name"}
    try:
        compound = {"inchikey": "", "smiles": "", "name": compound_name}
        result = chembl.fetch_compound_data(compound)
        targets = []
        for t in (result.targets or [])[:_TOOL_RESULT_ROW_CAP]:
            targets.append({
                "target_chembl_id": t.target_chembl_id,
                "target_name": t.target_name,
                "gene_symbol": t.gene_symbol,
                "uniprot_id": t.uniprot_id,
                "evidence_type": t.evidence_type,
                "pchembl_score": t.pchembl_score,
                "assay_description": (t.assay_description or "")[:200],
                "mechanism_action": t.mechanism_action,
            })
        # Indications via the separate function (no compound_chembl_id needed
        # — uses molecule resolution again).
        indications: list[dict] = []
        try:
            ir = chembl.fetch_compound_indications(compound)
            for ind in (ir.indications or [])[:_TOOL_RESULT_ROW_CAP]:
                indications.append({
                    "mesh_heading": ind.mesh_heading,
                    "efo_term": ind.efo_term,
                    "max_phase": ind.max_phase,
                })
        except Exception:
            pass
        return {
            "compound_name": compound_name,
            "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
            "targets": targets,
            "n_targets_total": len(result.targets or []),
            "indications": indications,
            "n_indications_total": len(indications),
        }
    except Exception as e:
        logger.warning("ChEMBL tool failed for %r: %s", compound_name, e)
        return {"error": str(e), "compound_name": compound_name}


def _exec_opentargets(target_id: str) -> dict:
    """Wrap open_targets.fetch_target_associations_batch for a single target."""
    if not target_id:
        return {"error": "missing target_id"}
    try:
        # The batch function takes a list of (uniprot_id, ensembl_id) pairs.
        # If the agent gave us a UniProt, resolve it to ENSEMBL first.
        if target_id.startswith("CHEMBL"):
            return {
                "target_id": target_id,
                "error": "Open Targets indexes by ENSEMBL gene ID; pass a UniProt ID (e.g. P23219) instead of a ChEMBL ID.",
                "diseases": [],
            }
        # Treat as UniProt
        ensembl_pairs = open_targets.bulk_resolve_uniprot_to_ensembl([target_id])
        ensembl_id = ensembl_pairs.get(target_id)
        if not ensembl_id:
            return {
                "target_id": target_id,
                "error": "Could not resolve UniProt to Ensembl gene ID.",
                "diseases": [],
            }
        out = open_targets.fetch_target_associations_batch([(target_id, ensembl_id)])
        result = out.get(target_id)
        if result is None:
            return {"target_id": target_id, "error": "no result", "diseases": []}
        diseases = []
        for a in (result.associations or [])[:_TOOL_RESULT_ROW_CAP]:
            diseases.append({
                "disease": a.disease_name,
                "efo_id": a.disease_efo_id,
                "ot_overall_score": a.overall_score,
                "datasource_scores": a.datasource_scores,
            })
        return {
            "target_id": target_id,
            "ensembl_id": ensembl_id,
            "outcome": result.outcome.value if hasattr(result.outcome, "value") else str(result.outcome),
            "diseases": diseases,
            "n_associations_total": len(result.associations or []),
        }
    except Exception as e:
        logger.warning("Open Targets tool failed for %r: %s", target_id, e)
        return {"error": str(e), "target_id": target_id}


def _exec_pubmed(query: str, max_results: int = 5) -> dict:
    """NCBI E-utils esearch + esummary."""
    if not query:
        return {"error": "missing query"}
    max_results = max(1, min(20, int(max_results or 5)))
    try:
        # esearch → PMIDs
        es = requests.get(
            f"{_NCBI_EUTILS}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
            },
            timeout=_HTTP_TIMEOUT_S,
        )
        es.raise_for_status()
        pmids = es.json().get("esearchresult", {}).get("idlist", []) or []
        if not pmids:
            return {"query": query, "results": [], "n": 0}
        # esummary → titles + pubdates
        ss = requests.get(
            f"{_NCBI_EUTILS}/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "json",
            },
            timeout=_HTTP_TIMEOUT_S,
        )
        ss.raise_for_status()
        result = ss.json().get("result", {})
        items = []
        for pmid in pmids:
            entry = result.get(pmid) or {}
            items.append({
                "pmid": pmid,
                "title": (entry.get("title") or "")[:300],
                "pubdate": entry.get("pubdate"),
                "authors": [a.get("name") for a in (entry.get("authors") or [])[:3]],
                "journal": entry.get("fulljournalname") or entry.get("source"),
            })
        return {"query": query, "results": items, "n": len(items)}
    except Exception as e:
        logger.warning("PubMed tool failed for %r: %s", query, e)
        return {"error": str(e), "query": query}


def _exec_mesh(term: str) -> dict:
    """NLM MeSH descriptor lookup."""
    if not term:
        return {"error": "missing term"}
    try:
        r = requests.get(
            _MESH_LOOKUP,
            params={"label": term, "match": "contains", "limit": 10},
            timeout=_HTTP_TIMEOUT_S,
        )
        r.raise_for_status()
        items = r.json() or []
        out = []
        for it in items[:_TOOL_RESULT_ROW_CAP]:
            out.append({
                "mesh_id": (it.get("resource") or "").rsplit("/", 1)[-1],
                "label": it.get("label"),
                "match": it.get("match"),
            })
        return {"term": term, "descriptors": out, "n": len(out)}
    except Exception as e:
        logger.warning("MeSH tool failed for %r: %s", term, e)
        return {"error": str(e), "term": term}


def _execute_tool(fn_name: str, fn_args: dict) -> dict:
    if fn_name == "lookup_chembl_compound":
        return _exec_chembl(fn_args.get("compound_name", ""))
    if fn_name == "query_opentargets_target_diseases":
        return _exec_opentargets(fn_args.get("target_id", ""))
    if fn_name == "pubmed_search":
        return _exec_pubmed(
            fn_args.get("query", ""),
            int(fn_args.get("max_results") or 5),
        )
    if fn_name == "lookup_mesh_term":
        return _exec_mesh(fn_args.get("term", ""))
    return {"error": f"Unknown tool: {fn_name}"}


def _summarize_result(result: dict, *, max_rows_logged: int = _TOOL_RESULT_ROW_CAP) -> dict:
    """Tool result summary kept in the call log for citation auditing.

    Stores up to `max_rows_logged` full rows per result list so the judge
    can verify ALL cited entities against what the agent actually saw.
    """
    summary: dict[str, Any] = {}
    for k, v in result.items():
        if isinstance(v, list):
            rows = v[:max_rows_logged]
            summary[k] = {
                "count": len(v),
                "rows": rows,
                "sample_first": rows[0] if rows else None,
            }
        else:
            summary[k] = v
    return summary


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class BaselineTaskAToolCall(BaseAgent):
    """Baseline D: Gemini with external biomedical APIs (ChEMBL,
    Open Targets, PubMed, MeSH).

    Same model as the cheap baselines (Flash Lite by default). Agent loop
    runs up to MAX_TOOL_ROUNDS turns, calling whichever tools the LLM
    requests; the final round forces structured-JSON output in the
    critic schema. If the LLM tries to skip tools entirely the loop
    pushes back once before forcing the final answer.
    """

    def __init__(
        self,
        client: GraphClient,
        *,
        gemini_model: str | None = None,
        max_rounds: int = MAX_TOOL_ROUNDS,
        max_workers: int = 4,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        # Gemini SDK is thread-safe; client is shared across worker threads.
        self._gemini = make_gemini_client()
        self._model = gemini_model or GEMINI_MODEL
        self._max_rounds = max_rounds
        # ThreadPool size — capped at 4 by default to keep PubMed RPS
        # under the no-key NCBI limit (3 req/sec) when multiple workers
        # invoke pubmed_search in the same window.
        self._max_workers = max_workers

    @property
    def name(self) -> str:
        return "BaselineTaskAToolCall"

    def run(
        self,
        *,
        claims: list[dict] | None = None,
        source_name: str | None = None,
        malady_name: str | None = None,
        limit: int | None = None,
        pass1_lookup: dict[tuple[str, str], str] | None = None,
        checkpoint_path: str | None = None,
        checkpoint_every: int = 5,
        **_: Any,
    ) -> dict:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from pathlib import Path
        from threading import Lock

        target_claims = self._resolve_claims(claims, source_name, malady_name)
        if limit is not None:
            target_claims = target_claims[:limit]

        self._log_progress(
            f"Critiquing {len(target_claims)} claim(s) with external-tools "
            f"baseline (model={self._model}, max_rounds={self._max_rounds}, "
            f"workers={self._max_workers})"
        )

        # Pre-fetch all anchors (cheap, sequential, single Cypher round-trip).
        anchors: list[dict] = []
        for c in target_claims:
            a = self._fetch_anchor(c["source"], c["malady"])
            if a is not None:
                anchors.append(a)

        results: list[BaselineToolCallVerdict] = []
        errors: list[str] = []
        results_lock = Lock()
        completed_counter = {"n": 0}

        def _process_one(anchor: dict) -> BaselineToolCallVerdict:
            t0 = time.time()
            try:
                final_response, tool_log, rounds = self._run_agent_loop(anchor)
                verdict = (final_response or {}).get("verdict") or "unsupported"
                if verdict not in VERDICT_VALUES:
                    verdict = "unsupported"
                cv = BaselineToolCallVerdict(
                    source=anchor["source"],
                    malady=anchor["malady"],
                    primary_disease=anchor.get("primary_disease"),
                    treats_evidence=anchor.get("treats_evidence"),
                    verdict=verdict,
                    agrees_with_pass1=False,
                    verdict_delta=0,
                    biological_plausibility=_clamp01(
                        (final_response or {}).get("biological_plausibility")
                    ),
                    evidence_coherence=_clamp01(
                        (final_response or {}).get("evidence_coherence")
                    ),
                    key_evidence=list((final_response or {}).get("key_evidence") or []),
                    concerns=list((final_response or {}).get("concerns") or []),
                    rationale=str((final_response or {}).get("rationale") or ""),
                    requires_human_review=bool(
                        (final_response or {}).get("requires_human_review")
                    ),
                    raw_response=json.dumps(final_response or {}, ensure_ascii=False),
                    tool_call_log=tool_log,
                    tool_calls_count=len(tool_log),
                    rounds_used=rounds,
                    model=self._model,
                    duration_s=time.time() - t0,
                )
                if pass1_lookup is not None:
                    pass1_v = pass1_lookup.get((anchor["source"], anchor["malady"]))
                    if pass1_v:
                        cv.verdict_delta = _verdict_delta(pass1_v, cv.verdict)
                        cv.agrees_with_pass1 = (cv.verdict_delta == 0)
                return cv
            except Exception as e:
                err_msg = f"{anchor['source']} -> {anchor['malady']}: {e}"
                errors.append(err_msg)
                logger.warning("Worker error: %s", err_msg)
                return BaselineToolCallVerdict(
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
                )

        def _maybe_checkpoint(force: bool = False) -> None:
            if not checkpoint_path:
                return
            n = completed_counter["n"]
            if not force and (n == 0 or n % checkpoint_every != 0):
                return
            tmp = Path(checkpoint_path).with_suffix(
                Path(checkpoint_path).suffix + ".tmp"
            )
            payload = {
                "agent": self.name,
                "model": self._model,
                "claims_total": len(results),
                "errors": list(errors),
                "verdicts": [asdict(r) for r in results],
                "checkpoint": {"n_done": n, "force_flush": force},
                "completed_at": dt.datetime.utcnow().isoformat(),
            }
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=str)
            tmp.replace(checkpoint_path)

        # Parallel claim processing.
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(_process_one, a): a for a in anchors}
            for future in as_completed(futures):
                anchor = futures[future]
                cv = future.result()
                with results_lock:
                    results.append(cv)
                    completed_counter["n"] += 1
                    n = completed_counter["n"]
                self._log_progress(
                    f"[{n}/{len(anchors)}] {anchor['source']} → {anchor['malady']} "
                    f"(rounds={cv.rounds_used}, calls={cv.tool_calls_count}, "
                    f"verdict={cv.verdict}, t={cv.duration_s:.1f}s)"
                )
                _maybe_checkpoint(force=False)

        _maybe_checkpoint(force=True)

        return {
            "agent": self.name,
            "model": self._model,
            "claims_total": len(results),
            "errors": errors,
            "verdicts": [asdict(r) for r in results],
            "completed_at": dt.datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def _run_agent_loop(self, anchor: dict) -> tuple[dict | None, list[dict], int]:
        """Multi-turn function-calling loop with external biomedical tools.

        Returns (final_response_dict, tool_call_log, rounds_used).
        """
        # Per-call (per-thread) rate-limit gate. Avoids races across
        # workers — each thread paces its own Gemini calls.
        last_call = [0.0]
        aliases = anchor.get("source_aliases") or []
        if not isinstance(aliases, list):
            aliases = []
        user_prompt = CLAIM_PROMPT_TEMPLATE.format(
            source=anchor["source"],
            aliases=", ".join(aliases) if aliases else "<none>",
            malady=anchor["malady"],
            primary_disease=anchor.get("primary_disease") or "<not mapped>",
            treats_evidence=anchor.get("treats_evidence") or "<no evidence span>",
            malady_description=anchor.get("malady_description") or "<no description>",
        )

        contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])
        ]
        tool_log: list[dict] = []

        for round_idx in range(self._max_rounds + 1):
            self._rate_limit_local(last_call)
            is_final = round_idx == self._max_rounds

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                tools=None if is_final else _TOOLS,
                tool_config=None if is_final else _TOOL_CONFIG,
                response_mime_type="application/json" if is_final else None,
                response_schema=CRITIC_RESPONSE_SCHEMA if is_final else None,
            )

            resp = self._call_with_retry(contents, config)
            if resp is None:
                return None, tool_log, round_idx

            candidate = resp.candidates[0]
            parts = candidate.content.parts or []
            fn_call_parts = [p for p in parts if getattr(p, "function_call", None)]
            if not fn_call_parts:
                # No more function calls — text-only response.
                contents.append(candidate.content)
                if is_final:
                    return self._parse_json(resp.text), tool_log, round_idx
                # If the agent tried to skip tools entirely, push back.
                if not tool_log:
                    contents.append(types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=(
                                "You haven't called any biomedical tools yet. "
                                "The investigation policy requires at least 2-3 "
                                "tool calls before answering. Please call "
                                "lookup_chembl_compound on the most likely "
                                "active compound, then continue investigating."
                            )
                        )],
                    ))
                    continue
                # Force final JSON after a real investigation.
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="Now produce the final verdict JSON in the required schema."
                    )],
                ))
                self._rate_limit_local(last_call)
                final_resp = self._call_with_retry(
                    contents,
                    types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        response_mime_type="application/json",
                        response_schema=CRITIC_RESPONSE_SCHEMA,
                    ),
                )
                if final_resp is None:
                    return None, tool_log, round_idx
                return (
                    self._parse_json(final_resp.text),
                    tool_log,
                    round_idx + 1,
                )

            # 3) Process function calls — execute tools, append responses.
            contents.append(candidate.content)
            fn_response_parts = []
            for part in fn_call_parts:
                fn = part.function_call
                fn_name = fn.name
                fn_args = dict(fn.args) if fn.args else {}
                self._log_progress(f"  [round {round_idx}] tool: {fn_name}({fn_args})")
                t_tool = time.time()
                result = _execute_tool(fn_name, fn_args)
                tool_log.append({
                    "round": round_idx,
                    "tool": fn_name,
                    "args": fn_args,
                    "duration_s": round(time.time() - t_tool, 3),
                    "result_summary": _summarize_result(result),
                })
                fn_response_parts.append(types.Part.from_function_response(
                    name=fn_name,
                    response=result,
                ))
            contents.append(types.Content(role="user", parts=fn_response_parts))

        return None, tool_log, self._max_rounds

    def _call_with_retry(self, contents, config):
        last_err: Exception | None = None
        for attempt in range(_GEMINI_MAX_RETRIES):
            try:
                return self._gemini.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                last_err = e
                if "429" in str(e) and attempt < _GEMINI_MAX_RETRIES - 1:
                    wait_s = 5 * (2 ** attempt)
                    logger.info("External-tools rate-limited, backoff %ds", wait_s)
                    time.sleep(wait_s)
                    continue
                logger.warning("External-tools call failed: %s", e)
                return None
        if last_err:
            logger.warning("External-tools exhausted retries: %s", last_err)
        return None

    @staticmethod
    def _parse_json(text: str | None) -> dict | None:
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _rate_limit_local(self, last_call: list[float]) -> None:
        """Per-thread rate-limit gate (last_call is a 1-element list mutated
        in place by the caller's thread)."""
        elapsed = time.time() - last_call[0]
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        last_call[0] = time.time()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_claims(self, claims, source, malady):
        if claims is not None:
            return list(claims)
        if source and malady:
            return [{"source": source, "malady": malady}]
        rows = self.client.run(
            "MATCH (s:Source)-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady) "
            "WHERE (s.archived IS NULL OR s.archived = false) "
            "  AND (m.archived IS NULL OR m.archived = false) "
            "RETURN s.name AS source, m.name AS malady ORDER BY source, malady"
        )
        return [{"source": r["source"], "malady": r["malady"]} for r in rows]

    def _fetch_anchor(self, source: str, malady: str) -> dict | None:
        rows = self.client.run(
            _CLAIM_ANCHOR_QUERY, {"source": source, "malady": malady},
        )
        return rows[0] if rows else None
