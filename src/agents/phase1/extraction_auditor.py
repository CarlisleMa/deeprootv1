"""Phase 1 — Extraction Auditor.

Runs immediately AFTER the ExtractionAgent and BEFORE the Phase 1 linking
agents. Three audit jobs:

  1) Source canonicalization + dedup  (generate-then-verify)
       Per-node: ask Gemini for the canonical Latin binomial / chemical
       name (or "uncanonicalized" for ambiguous/generic), then VERIFY
       against an external database — COCONUT for organisms, PubChem for
       chemicals. Each Source ends up with three properties:
         canonical_name           : "Coptis chinensis" (or null)
         canonical_source         : "gemini+coconut", "gemini+pubchem",
                                    "gemini_unverified_organism",
                                    "gemini_unverified_chemical",
                                    "uncanonicalized", "error"
         canonical_raw_response   : the literal Gemini output (audit trail)

       Then GROUP BY canonical_name; groups of 2+ nodes whose members are
       ALL "grounded" (canonical_source ∈ {gemini+coconut, gemini+pubchem})
       AUTO-MERGE. Groups with any "unverified" member go to BORDERLINE.

  2) Evidence-span hallucination check  (deterministic)
       Every extracted node carries an `evidence_span` property. The span
       is checked (whitespace-normalised, case-insensitive) against the
       source chunk text. Spans that don't appear are soft-archived with
       reason="hallucinated_evidence".

  3) Coverage report  (read-only)
       Per-document node counts, confidence distribution, and (new)
       canonicalization-source distribution.

Design notes:
  - Gemini calls are batched (20 names/call) with structured JSON output
    so 558 nodes resolve in ~28 calls.
  - Each node's canonical_name is cached on the node; re-runs only
    canonicalize new (un-canonicalized) Sources, so iterating is cheap.
  - Verification is HTTP for PubChem (free, no key) and a local parquet
    lookup for COCONUT (no network).
  - Soft-delete contract: merged nodes get archived=true,
    archive_reason="merged_into:<canonical>". Outgoing edges are RE-TARGETED
    to the canonical node before archival (Phase 2 queries already filter
    archived=false, so the canonical node ends up with full traversal
    capacity).

Scope notes (v1):
  - Source canonicalization only. Traditional_Malady dedup is deferred —
    no good external DB to verify against, and false-merge cost is high.
  - Only outgoing edges (TREATS_TRADITIONALLY, PREPARED_AS) are migrated.
    Incoming edges (Chemical_Compound -> Source IS_EXTRACTED_FROM) don't
    exist yet at the post-extraction stage. When linking adds them, this
    method will need a small extension.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from google import genai
from google.genai import types

from src.agents.base import BaseAgent
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.data.coconut import _df as _coconut_df
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BATCH_SIZE = 20
_GEMINI_RATE_LIMIT_S = 1.0   # Polite gap between Gemini calls (per batch)
_PUBCHEM_RATE_LIMIT_S = 0.25  # PubChem requests "no more than 5/s"
_PUBCHEM_TIMEOUT_S = 10
_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"

_WS_RE = re.compile(r"\s+")


def _norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", s or "").strip().lower()


# ---------------------------------------------------------------------------
# Gemini canonicalization
# ---------------------------------------------------------------------------

CANONICALIZE_SYSTEM_PROMPT = """\
You are a biomedical name normalizer. For each substance below, return its \
canonical name in one of three forms:

  type="organism"           — for plants, animals, fungi, microbes. Return a \
Latin binomial (Genus species) when known. Use the most widely accepted \
modern binomial.

  type="chemical"           — for inorganic substances or pure chemical \
compounds. Return the standard IUPAC or commonly-accepted chemical name \
(or chemical formula if more recognizable, e.g. "arsenic trisulfide").

  type="uncanonicalized"    — for generic categories ("hot medicinals", \
"superior class medicinals"), ambiguous names that could refer to multiple \
distinct entities, OCR garbage, or anything you cannot confidently assign. \
DO NOT GUESS. When in doubt, choose this.

For type="organism", you must ALSO specify the `part` field — the specific \
organ, tissue, or derivative if the substance name names one. This matters \
for downstream chemistry: chicken feather and chicken egg have very \
different bioactive profiles even though they're the same species. Use \
short lowercase English, for example:

  Plants: "root", "root bark", "rhizome", "leaf", "flower", "seed", \
"fruit", "stem", "bark", "resin", "peel", "tuber", "bulb"
  Animals: "egg", "feather", "fat", "gall", "horn", "antler", "shell", \
"intestine", "gizzard lining", "skin", "bile", "penis", "musk", "carapace", \
"plastron", "egg-case", "exuviae"
  Fungi: "fruiting body", "sclerotium", "mycelium"
  When the original name does NOT specify a part: "whole"

For type="chemical" or type="uncanonicalized", set part to "".

The aliases provided are LLM-extracted hints from historical Chinese medical \
texts; they may be helpful but may also be noisy — use them as suggestions, \
not as ground truth.

Respond with structured JSON only.\
"""

CANONICALIZE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "input_name": {"type": "string"},
                    "canonical": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["organism", "chemical", "uncanonicalized"],
                    },
                    "part": {
                        "type": "string",
                        "description": (
                            "For organisms: the specific organ/tissue/"
                            "derivative if named, lowercase (e.g. 'root', "
                            "'feather', 'gizzard lining'); 'whole' if no "
                            "part named. Empty for chemicals/uncanonicalized."
                        ),
                    },
                },
                "required": ["input_name", "canonical", "type", "part"],
            },
        },
    },
    "required": ["results"],
}


# ---------------------------------------------------------------------------
# External lookups for verification
# ---------------------------------------------------------------------------


def _coconut_has_organism(name: str) -> bool:
    """Return True iff `name` (case-insensitive substring) appears in the
    COCONUT `organisms` column. Coverage is plant-heavy."""
    if not name:
        return False
    try:
        return bool(_coconut_df["organisms"].str.lower().str.contains(
            name.lower(), na=False, regex=False,
        ).any())
    except Exception as e:
        logger.warning("COCONUT lookup failed for %r: %s", name, e)
        return False


def _pubchem_has_compound(name: str, *, last_call: list[float]) -> bool:
    """Return True iff PubChem recognises `name` as a compound. Free API,
    no key required. `last_call` is a single-element list used as a
    mutable timestamp for rate-limiting across calls."""
    if not name:
        return False
    elapsed = time.time() - last_call[0]
    if elapsed < _PUBCHEM_RATE_LIMIT_S:
        time.sleep(_PUBCHEM_RATE_LIMIT_S - elapsed)
    last_call[0] = time.time()

    url = f"{_PUBCHEM_BASE}/{requests.utils.quote(name)}/cids/JSON"
    try:
        r = requests.get(url, timeout=_PUBCHEM_TIMEOUT_S)
    except requests.RequestException as e:
        logger.warning("PubChem request failed for %r: %s", name, e)
        return False
    if r.status_code == 200:
        cids = (r.json().get("IdentifierList") or {}).get("CID") or []
        return bool(cids)
    if r.status_code == 404:
        return False
    logger.warning("PubChem unexpected status %d for %r", r.status_code, name)
    return False


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ExtractionAuditor(BaseAgent):
    """Canonicalize Source names via Gemini+verification, merge by canonical,
    verify evidence spans, and report coverage."""

    def __init__(
        self,
        client: GraphClient,
        *,
        corpus_dir: str | Path = "data/historical_corpus",
        gemini_api_key: str | None = None,
        gemini_model: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(client, **kwargs)
        self._corpus_dir = Path(corpus_dir)
        self._corpus_text: dict[str, str] | None = None  # lazy
        api_key = gemini_api_key or GEMINI_API_KEY
        self._gemini = genai.Client(api_key=api_key) if api_key else None
        self._model = gemini_model or GEMINI_MODEL
        self._last_gemini_call = 0.0
        self._last_pubchem_call = [0.0]  # mutable across calls

    @property
    def name(self) -> str:
        return "ExtractionAuditor"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        dry_run: bool = True,
        skip_canonicalize: bool = False,
        skip_merge: bool = False,
        skip_evidence: bool = False,
        force_recanonicalize: bool = False,
        **_: Any,
    ) -> dict:
        self._log_progress(f"Starting extraction audit (dry_run={dry_run})")

        canon_summary: dict = {}
        merge_summary: dict = {}
        evidence_summary: dict = {}

        canon_records: list[dict] = []
        if not skip_canonicalize:
            canon_summary = self._canonicalize_sources(
                dry_run=dry_run, force=force_recanonicalize,
            )
            canon_records = canon_summary.pop("records", [])
            self._log_progress(
                f"Canonicalization: {canon_summary['written_count']} written, "
                f"{canon_summary['by_source'].get('uncanonicalized', 0)} uncanonicalized"
            )

        if not skip_merge:
            # In dry-run mode the canonical_name property hasn't been written
            # to the graph, so the merge step must operate on the in-memory
            # records produced by canonicalize. In write mode, we prefer the
            # graph as the source of truth (also covers --skip-canonicalize
            # runs that rely on previously cached canonical_name properties).
            merge_summary = self._merge_by_canonical(
                dry_run=dry_run,
                records=canon_records if dry_run else None,
            )
            self._log_progress(
                f"Merge: {merge_summary['auto_merged_groups']} auto-merged, "
                f"{merge_summary['borderline_groups']} borderline"
            )

        if not skip_evidence:
            evidence_summary = self._verify_evidence_spans(dry_run)
            self._log_progress(
                f"Evidence: {evidence_summary['hallucinated_count']} of "
                f"{evidence_summary['checked_count']} checked"
            )

        coverage = self._coverage_report()

        return {
            "dry_run": dry_run,
            "canonicalization": canon_summary,
            "merge": merge_summary,
            "evidence": evidence_summary,
            "coverage": coverage,
        }

    # ==================================================================
    # 1a) Canonicalization (per node, cached)
    # ==================================================================

    def _canonicalize_sources(self, *, dry_run: bool, force: bool) -> dict:
        if not self._gemini:
            return {
                "written_count": 0,
                "by_source": {},
                "warning": "no Gemini client configured",
            }

        # Pull Sources lacking canonical_name (or all, if --force)
        if force:
            cypher = """
                MATCH (s:Source)
                WHERE (s.archived IS NULL OR s.archived = false)
                RETURN s.name AS name, s.aliases AS aliases,
                       s.evidence_span AS evidence_span
                ORDER BY s.name
            """
        else:
            cypher = """
                MATCH (s:Source)
                WHERE (s.archived IS NULL OR s.archived = false)
                  AND (s.canonical_name IS NULL)
                RETURN s.name AS name, s.aliases AS aliases,
                       s.evidence_span AS evidence_span
                ORDER BY s.name
            """
        rows = self.client.run(cypher)
        if not rows:
            return {"written_count": 0, "by_source": {}, "skipped_cached": True}

        self._log_progress(f"Canonicalizing {len(rows)} Source(s) in batches of {_BATCH_SIZE}")

        # Build a lookup from name → original row so we can attach aliases
        # to each canonicalize record (merge step needs them later).
        rows_by_name = {r["name"]: r for r in rows}

        by_source: dict[str, int] = defaultdict(int)
        written = 0
        records: list[dict] = []  # diagnostics + handoff to merge step

        for i in range(0, len(rows), _BATCH_SIZE):
            batch = rows[i : i + _BATCH_SIZE]
            try:
                results = self._gemini_canonicalize_batch(batch)
            except Exception as e:
                logger.warning("Gemini batch %d failed: %s", i // _BATCH_SIZE, e)
                for r in batch:
                    record = self._enrich_record(
                        {
                            "name": r["name"],
                            "canonical_raw": "",
                            "canonical_type": "error",
                            "canonical_name": None,
                            "canonical_part": "",
                            "canonical_source": "error",
                        },
                        rows_by_name,
                    )
                    records.append(record)
                    by_source["error"] += 1
                    if not dry_run:
                        self._write_canonical(record)
                        written += 1
                continue

            for raw in results:
                input_name = raw.get("input_name", "")
                canonical = (raw.get("canonical") or "").strip()
                ctype = raw.get("type", "uncanonicalized")
                part = (raw.get("part") or "").strip().lower()

                if ctype == "uncanonicalized" or not canonical:
                    record = self._record_uncanonicalized(input_name, canonical)
                elif ctype == "organism":
                    confirmed = _coconut_has_organism(canonical)
                    record = self._record_organism(
                        input_name, canonical, part, confirmed
                    )
                elif ctype == "chemical":
                    confirmed = _pubchem_has_compound(
                        canonical, last_call=self._last_pubchem_call
                    )
                    record = self._record_chemical(input_name, canonical, confirmed)
                else:
                    record = self._record_uncanonicalized(input_name, canonical)

                record = self._enrich_record(record, rows_by_name)
                records.append(record)
                by_source[record["canonical_source"]] += 1

                if not dry_run:
                    self._write_canonical(record)
                    written += 1

            self._log_progress(
                f"  Batch {i // _BATCH_SIZE + 1}/"
                f"{(len(rows) + _BATCH_SIZE - 1) // _BATCH_SIZE} done"
            )

        return {
            "input_count": len(rows),
            "written_count": written,
            "by_source": dict(by_source),
            "examples": records[:20],  # preview only
            "records": records,  # full handoff to merge step
        }

    @staticmethod
    def _enrich_record(record: dict, rows_by_name: dict[str, dict]) -> dict:
        """Attach aliases and other row data the merge step needs."""
        original = rows_by_name.get(record["name"], {})
        record["aliases"] = original.get("aliases") or []
        return record

    @staticmethod
    def _record_uncanonicalized(input_name: str, raw: str) -> dict:
        return {
            "name": input_name,
            "canonical_raw": raw,
            "canonical_type": "uncanonicalized",
            "canonical_name": None,
            "canonical_part": "",
            "canonical_source": "uncanonicalized",
        }

    @staticmethod
    def _record_organism(
        input_name: str, canonical: str, part: str, confirmed: bool,
    ) -> dict:
        # Default empty part to "whole" so organisms with no part named
        # still group together (rather than scattering into single-member
        # buckets per node).
        return {
            "name": input_name,
            "canonical_raw": canonical,
            "canonical_type": "organism",
            "canonical_name": canonical,
            "canonical_part": part or "whole",
            "canonical_source": "gemini+coconut" if confirmed else "gemini_unverified_organism",
        }

    @staticmethod
    def _record_chemical(input_name: str, canonical: str, confirmed: bool) -> dict:
        return {
            "name": input_name,
            "canonical_raw": canonical,
            "canonical_type": "chemical",
            "canonical_name": canonical,
            "canonical_part": "",
            "canonical_source": "gemini+pubchem" if confirmed else "gemini_unverified_chemical",
        }

    def _write_canonical(self, record: dict) -> None:
        # canonical_name may be None; Cypher can't SET a literal null via this
        # helper, so we pass an empty string when null and clean up later.
        self.client.set_node_properties(
            "Source",
            {"name": record["name"]},
            {
                "canonical_name": record["canonical_name"] or "",
                "canonical_part": record.get("canonical_part") or "",
                "canonical_source": record["canonical_source"],
                "canonical_type": record["canonical_type"],
                "canonical_raw_response": record["canonical_raw"],
            },
        )

    def _gemini_canonicalize_batch(self, items: list[dict]) -> list[dict]:
        """Send a batch of Source rows to Gemini, return list of canonicalizations."""
        # Polite rate limit
        elapsed = time.time() - self._last_gemini_call
        if elapsed < _GEMINI_RATE_LIMIT_S:
            time.sleep(_GEMINI_RATE_LIMIT_S - elapsed)
        self._last_gemini_call = time.time()

        # Build numbered prompt
        lines = []
        for n, item in enumerate(items, 1):
            aliases = item.get("aliases") or []
            ev = (item.get("evidence_span") or "").strip()
            ev_short = ev[:200] + ("..." if len(ev) > 200 else "")
            lines.append(
                f'{n}. name="{item["name"]}"  '
                f'aliases={aliases[:8] if aliases else []}  '
                f'evidence="{ev_short}"'
            )
        prompt = (
            "Canonicalize each of the following substances:\n\n"
            + "\n".join(lines)
            + "\n\nReturn one result per substance, preserving input_name verbatim."
        )

        response = self._gemini.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=CANONICALIZE_SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=CANONICALIZE_SCHEMA,
            ),
        )
        data = json.loads(response.text)
        return data.get("results", [])

    # ==================================================================
    # 1b) Merge by canonical
    # ==================================================================

    def _merge_by_canonical(
        self, *, dry_run: bool, records: list[dict] | None = None,
    ) -> dict:
        # Preferred path: use in-memory records from the canonicalize step
        # (works in dry_run mode where canonical_name was never persisted).
        # Fallback path: query the graph for nodes whose canonical_name is
        # already cached from a prior --write-graph run.
        if records:
            rows = [
                r for r in records
                if r.get("canonical_name")
            ]
            # No degree info in memory; merge step will pick the keeper
            # alphabetically when degrees are tied at 0.
            for r in rows:
                r.setdefault("degree", 0)
        else:
            rows = self.client.run("""
                MATCH (s:Source)
                WHERE (s.archived IS NULL OR s.archived = false)
                  AND s.canonical_name IS NOT NULL
                  AND s.canonical_name <> ""
                OPTIONAL MATCH (s)-[r]-()
                WITH s, count(r) AS degree
                RETURN s.name AS name,
                       s.aliases AS aliases,
                       s.canonical_name AS canonical_name,
                       s.canonical_part AS canonical_part,
                       s.canonical_source AS canonical_source,
                       degree
                ORDER BY s.name
            """)

        # Group by (canonical_name, canonical_part) so same-species but
        # different-organ Sources stay distinct (e.g. chicken feather vs
        # chicken egg, mulberry root vs mulberry root bark).
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            name_key = _norm_ws(r["canonical_name"])
            part_key = _norm_ws(r.get("canonical_part") or "")
            groups[(name_key, part_key)].append({
                "name": r["name"],
                "aliases": r.get("aliases") or [],
                "canonical_name": r["canonical_name"],
                "canonical_part": r.get("canonical_part") or "",
                "canonical_source": r["canonical_source"],
                "degree": r.get("degree", 0),
            })

        auto_merged: list[dict] = []
        borderline: list[dict] = []
        single_member_groups = 0

        for key, members in groups.items():
            if len(members) < 2:
                single_member_groups += 1
                continue

            sources_seen = {m["canonical_source"] for m in members}
            grounded = sources_seen <= {"gemini+coconut", "gemini+pubchem"}

            decision = self._build_merge_decision(members, grounded)
            if decision["action"] == "auto_merge":
                auto_merged.append(decision)
                if not dry_run:
                    self._apply_merge(decision)
            else:
                borderline.append(decision)

        return {
            "groups_total": len(groups),
            "single_member_groups": single_member_groups,
            "multi_member_groups": len(groups) - single_member_groups,
            "auto_merged_groups": len(auto_merged),
            "borderline_groups": len(borderline),
            "auto_merged": auto_merged,
            "borderline": borderline,
        }

    @staticmethod
    def _build_merge_decision(members: list[dict], grounded: bool) -> dict:
        """Pick canonical node + classify as auto_merge / borderline.

        Choice of canonical node from the cluster, in priority order:
          1. The member whose `name` matches its `canonical_name` (the
             "natural" canonical, no rename needed).
          2. The highest-degree member (preserves the most edges without
             a rewrite).
          3. Alphabetical first.
        """
        # Sort to get deterministic behaviour across runs
        members = sorted(members, key=lambda m: (-m["degree"], m["name"]))

        canon_norm = _norm_ws(members[0]["canonical_name"])
        natural_canonical = next(
            (m for m in members if _norm_ws(m["name"]) == canon_norm),
            None,
        )
        if natural_canonical is not None:
            keeper = natural_canonical
            keeper_reason = "name_matches_canonical"
        else:
            keeper = members[0]
            keeper_reason = "highest_degree_member"

        merged_in = [m["name"] for m in members if m["name"] != keeper["name"]]
        sources_present = sorted({m["canonical_source"] for m in members})

        return {
            "action": "auto_merge" if grounded else "borderline",
            "canonical_name": members[0]["canonical_name"],
            "canonical_part": members[0].get("canonical_part") or "",
            "keeper": keeper["name"],
            "keeper_reason": keeper_reason,
            "merged_in": merged_in,
            "all_members": [m["name"] for m in members],
            "canonical_sources": sources_present,
            "grounded": grounded,
        }

    def _apply_merge(self, decision: dict) -> None:
        keeper = decision["keeper"]
        for old in decision["merged_in"]:
            self._migrate_outgoing_edges(old, keeper, "TREATS_TRADITIONALLY")
            self._migrate_outgoing_edges(old, keeper, "PREPARED_AS")
            self._archive_merged(old, keeper, decision["canonical_name"])

        # Union aliases (incl. merged node names) into the keeper.
        union_aliases: set[str] = set(decision["merged_in"])
        # Pull keeper's current aliases (and any merged-node aliases via the result of recent reads)
        rows = self.client.run(
            "MATCH (s:Source {name: $name}) RETURN s.aliases AS aliases",
            {"name": keeper},
        )
        if rows:
            union_aliases |= set(rows[0].get("aliases") or [])
        # Add aliases of merged nodes (read while they still exist, even archived)
        for old in decision["merged_in"]:
            rs = self.client.run(
                "MATCH (s:Source {name: $name}) RETURN s.aliases AS aliases",
                {"name": old},
            )
            if rs:
                union_aliases |= set(rs[0].get("aliases") or [])
        union_aliases.discard(keeper)
        self.client.set_node_properties(
            "Source", {"name": keeper}, {"aliases": sorted(union_aliases)},
        )

    def _migrate_outgoing_edges(
        self, old: str, keeper: str, rel_type: str,
    ) -> None:
        query = f"""
            MATCH (old:Source {{name: $old}})-[r:{rel_type}]->(target)
            MATCH (keeper:Source {{name: $keeper}})
            WITH old, keeper, r, target, properties(r) AS props
            MERGE (keeper)-[r2:{rel_type}]->(target)
            ON CREATE SET r2 = props,
                          r2.merged_from = $old
            ON MATCH SET r2.confidence_score = CASE
                WHEN coalesce(props.confidence_score, 0) >
                     coalesce(r2.confidence_score, 0)
                THEN props.confidence_score
                ELSE r2.confidence_score END
            DELETE r
        """
        self.client.run_write(query, {"old": old, "keeper": keeper})

    def _archive_merged(self, old: str, keeper: str, canonical_name: str) -> None:
        self.client.set_node_properties(
            "Source",
            {"name": old},
            {
                "archived": True,
                "archive_reason": f"merged_into:{keeper}",
                "merged_canonical_name": canonical_name,
                "reviewed_by": "extraction_auditor",
            },
        )

    # ==================================================================
    # 2) Evidence-span verification
    # ==================================================================

    def _verify_evidence_spans(self, dry_run: bool) -> dict:
        if self._corpus_text is None:
            self._corpus_text = self._load_corpus_text()

        if not self._corpus_text:
            return {
                "checked_count": 0,
                "hallucinated_count": 0,
                "missing_documents": [],
                "examples": [],
                "warning": "no corpus text available; skipped",
            }

        rows = self.client.run("""
            MATCH (n)
            WHERE (n:Source OR n:Traditional_Malady OR n:Preparation_Method)
              AND (n.archived IS NULL OR n.archived = false)
              AND n.evidence_span IS NOT NULL
              AND n.evidence_span <> ""
            RETURN labels(n)[0] AS label, n.name AS name,
                   n.source_document AS doc, n.evidence_span AS span
        """)

        hallucinated: list[dict] = []
        missing_docs: set[str] = set()
        for r in rows:
            doc = r.get("doc") or ""
            text = self._corpus_text.get(doc)
            if text is None:
                missing_docs.add(doc)
                continue
            if _norm_ws(r["span"]) not in _norm_ws(text):
                hallucinated.append({
                    "label": r["label"],
                    "name": r["name"],
                    "doc": doc,
                    "span": r["span"],
                })

        if not dry_run:
            for h in hallucinated:
                self.client.set_node_properties(
                    h["label"],
                    {"name": h["name"]},
                    {
                        "archived": True,
                        "archive_reason": "hallucinated_evidence",
                        "reviewed_by": "extraction_auditor",
                    },
                )

        return {
            "checked_count": len(rows),
            "hallucinated_count": len(hallucinated),
            "missing_documents": sorted(missing_docs),
            "examples": hallucinated[:10],
        }

    def _load_corpus_text(self) -> dict[str, str]:
        if not self._corpus_dir.is_dir():
            self.logger.warning(
                "Corpus dir not found: %s — evidence verification disabled",
                self._corpus_dir,
            )
            return {}
        out: dict[str, str] = {}
        for p in sorted(self._corpus_dir.glob("*.txt")):
            try:
                out[p.stem] = p.read_text(encoding="utf-8")
            except Exception as e:
                self.logger.warning("Failed to read %s: %s", p, e)
        return out

    # ==================================================================
    # 3) Coverage report (read-only)
    # ==================================================================

    def _coverage_report(self) -> dict:
        per_doc = self.client.run("""
            MATCH (n)
            WHERE (n:Source OR n:Traditional_Malady OR n:Preparation_Method)
              AND (n.archived IS NULL OR n.archived = false)
              AND n.source_document IS NOT NULL
            RETURN labels(n)[0] AS label,
                   n.source_document AS doc,
                   count(n) AS cnt
            ORDER BY doc, label
        """)

        confidence_dist = self.client.run("""
            MATCH ()-[r:TREATS_TRADITIONALLY]->()
            WHERE r.archived IS NULL OR r.archived = false
            RETURN
              count(CASE WHEN r.confidence_score = 1.0 THEN 1 END) AS exact_1,
              count(CASE WHEN r.confidence_score >= 0.9 AND r.confidence_score < 1.0 THEN 1 END) AS p9,
              count(CASE WHEN r.confidence_score >= 0.7 AND r.confidence_score < 0.9 THEN 1 END) AS p7,
              count(CASE WHEN r.confidence_score >= 0.5 AND r.confidence_score < 0.7 THEN 1 END) AS p5,
              count(CASE WHEN r.confidence_score < 0.5 THEN 1 END) AS lt5,
              count(r) AS total
        """)

        canonical_source_dist = self.client.run("""
            MATCH (s:Source)
            WHERE (s.archived IS NULL OR s.archived = false)
              AND s.canonical_source IS NOT NULL
            RETURN s.canonical_source AS source, count(s) AS cnt
            ORDER BY cnt DESC
        """)

        return {
            "nodes_per_document": per_doc,
            "treats_confidence_distribution": (
                confidence_dist[0] if confidence_dist else {}
            ),
            "canonical_source_distribution": canonical_source_dist,
        }
