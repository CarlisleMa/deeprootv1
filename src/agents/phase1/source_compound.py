"""Phase 1 — Source → Compound Linking Agent (v2).

Pure routing layer. Consumes the `ExtractionAuditor`'s canonical_name +
canonical_type + canonical_source on each Source and routes to the right
external compound DB:

  organism  →  COCONUT (lookup_exact, falls back to Latin-binomial aliases)
  chemical  →  PubChem (fetch_by_name, falls back to fetch_by_formula)
  uncanonicalized → skip (logged in run summary)

Writes:
  Chemical_Compound  MERGEd by `inchikey` (RDKit-computed from SMILES, or an
                     "unstructured:<slug>" sentinel for compounds without a
                     parseable structure)
  IS_EXTRACTED_FROM  edge per (compound, source) with:
                       confidence_score  : flat prior per evidence_type
                       evidence_type     : routing+verification audit tag
                       source_db         : "COCONUT" | "PubChem"
                       lookup_query      : the literal name we queried
                       coconut_row | pubchem_cid
                       np_likeness | annotation_level (raw COCONUT tags)

Design notes:
  * No LLM calls in this agent. All canonicalization happens upstream in
    `ExtractionAuditor`. If `canonical_source IS NULL`, the linker refuses
    to run on that node (status = SKIPPED_NO_AUDITOR).
  * Idempotent + resumable: each Source is tagged with `linker_status`,
    `linker_attempted_at`, `linker_compound_count`, `linker_evidence_type`.
    Reruns query only the residual (NULL or 'error') status.
  * Confidence priors are flat and per-evidence-type (see CONFIDENCE
    constant). Raw COCONUT scores (np_likeness, annotation_level) are
    preserved on each edge as TAGS so the Critic / Reviewer can re-weight
    using a calibrated downstream model. The choice of flat priors over
    a learned score is documented in the paper's methods section.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.agents.base import BaseAgent
from src.data import coconut, pubchem
from src.data.structures import to_canonical_smiles, to_inchikey, unstructured_key
from src.graph import queries
from src.graph.client import GraphClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class LinkStatus(str, Enum):
    LINKED = "linked"
    NO_COMPOUNDS_FOUND = "no_compounds_found"
    SKIPPED_UNCANONICALIZED = "skipped_uncanonicalized"
    SKIPPED_NO_AUDITOR = "skipped_no_auditor"
    ERROR = "error"


# Flat priors per evidence_type. Documented in paper methods section.
CONFIDENCE: dict[str, float] = {
    "coconut_organism_canonical":   0.7,
    "coconut_organism_alias":       0.55,
    "coconut_organism_unverified":  0.5,
    "pubchem_chemical_canonical":   0.8,
    "pubchem_chemical_unverified":  0.55,
    "pubchem_chemical_formula":     0.5,
}

# Part-specific evidence penalty. COCONUT data is species-level; when the
# Source describes a specific organ ("bark", "root", "seed", ...), the
# COCONUT compound list is weaker evidence for "this compound is in THIS
# part" than for "this compound is in this species somewhere." Applied
# only to organism-routed edges where canonical_part is set and != "whole".
# Keeps the ~76% of organism Sources that are part-specific honest in
# downstream confidence multiplication, without zeroing them out.
PART_SPECIFIC_PENALTY: float = 0.1


def _normalize_part(canonical_part: str | None) -> str:
    """Return a non-empty part label for tagging. Empty / whitespace / None
    all collapse to "whole" so the edge tag is always present and queryable."""
    p = (canonical_part or "").strip().lower()
    return p if p else "whole"


def _is_part_specific(canonical_type: str | None, canonical_part: str | None) -> bool:
    """True iff this Source carries a non-whole-organism part hint that
    weakens the species-level COCONUT evidence."""
    if canonical_type != "organism":
        return False
    return _normalize_part(canonical_part) != "whole"


# Match Genus species (e.g. "Coptis chinensis") in alias strings. Tolerates a
# possible third capitalized authority/varietal token but only takes the first
# two for the actual lookup.
_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s+([a-z]+)\b")


# Recognise things that look like a chemical formula rather than a name, so the
# PubChem fallback knows when to try fastformula. Heuristic: at least one
# uppercase letter followed by digits, and no internal whitespace.
_FORMULA_RE = re.compile(r"^[A-Z][A-Za-z0-9·\.\(\)\[\]]*\d[A-Za-z0-9·\.\(\)\[\]]*$")


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


@dataclass
class _Compound:
    """Normalized view across COCONUT and PubChem responses."""
    name: str
    smiles: str
    molecular_formula: str
    source_db: str
    inchikey: str
    coconut_row: int | None = None
    pubchem_cid: int | None = None
    np_likeness: float | None = None
    annotation_level: int | None = None
    # Set only when the compound came from PubChem's formula endpoint AND
    # the formula matched more than one CID. Used to tag the IS_EXTRACTED_FROM
    # edge so downstream knows this is a non-deterministic structural pick.
    pubchem_formula_candidate_count: int | None = None
    pubchem_formula_candidate_cids: list[int] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _binomial_aliases(aliases: list[str] | None) -> list[str]:
    """Filter aliases down to those that look like Latin binomials."""
    if not aliases:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for a in aliases:
        if not a:
            continue
        m = _BINOMIAL_RE.search(a)
        if not m:
            continue
        binomial = f"{m.group(1)} {m.group(2)}"
        key = binomial.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(binomial)
    return out


def _looks_like_formula(s: str) -> bool:
    if not s or " " in s.strip():
        return False
    return bool(_FORMULA_RE.match(s.strip()))


def _coconut_to_compound(c: coconut.CoconutCompound) -> _Compound | None:
    inchikey = to_inchikey(c.smiles) or unstructured_key(c.name)
    canon_smiles = to_canonical_smiles(c.smiles) or c.smiles
    return _Compound(
        name=c.name,
        smiles=canon_smiles,
        molecular_formula=c.molecular_formula,
        source_db=c.source_db,
        inchikey=inchikey,
        coconut_row=c.coconut_row,
        np_likeness=c.np_likeness,
        annotation_level=c.annotation_level,
    )


def _pubchem_to_compound(
    p: pubchem.PubchemCompound,
    *,
    formula_candidates: list[int] | None = None,
    formula_candidate_count: int | None = None,
) -> _Compound:
    # PubChem ships an InChIKey directly. Re-canonicalize SMILES through RDKit
    # so the `smiles` property matches what we'd get from any other DB.
    # When the lookup came via the formula endpoint and the formula was
    # ambiguous (more than one CID), we attach the candidate list so the
    # IS_EXTRACTED_FROM edge can be tagged as non-deterministic.
    canon_smiles = to_canonical_smiles(p.smiles) or p.smiles
    return _Compound(
        name=p.name or f"PubChem CID {p.cid}",
        smiles=canon_smiles,
        molecular_formula=p.molecular_formula,
        source_db=p.source_db,
        inchikey=p.inchikey or unstructured_key(p.name or str(p.cid)),
        pubchem_cid=p.cid,
        pubchem_formula_candidate_cids=formula_candidates,
        pubchem_formula_candidate_count=formula_candidate_count,
    )


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    return s[idx]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class SourceCompoundLinker(BaseAgent):
    """Route audited Sources to COCONUT/PubChem and write Chemical_Compound +
    IS_EXTRACTED_FROM edges."""

    @property
    def name(self) -> str:
        return "SourceCompoundLinker"

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        dry_run: bool = False,
        retry_misses: bool = False,
        force_relink: bool = False,
        limit: int | None = None,
        **_: Any,
    ) -> dict:
        pubchem.reset_stats()
        coconut.reset_stats()
        t_start = time.time()

        cypher = queries.LINKABLE_SOURCES
        if force_relink:
            cypher = queries.LINKABLE_SOURCES_FORCE
        elif retry_misses:
            cypher = queries.LINKABLE_SOURCES_RETRY_MISSES

        sources = self.client.run(cypher)
        if limit is not None:
            sources = sources[:limit]
        self._log_progress(
            f"Linking {len(sources)} Source(s) "
            f"(dry_run={dry_run}, retry_misses={retry_misses}, force_relink={force_relink})"
        )

        by_status: dict[str, int] = {s.value: 0 for s in LinkStatus}
        by_evidence_type: dict[str, int] = {k: 0 for k in CONFIDENCE}
        by_part_context: dict[str, int] = {}  # part_label -> edge count
        edges_part_specific = 0
        edges_whole_or_compound_level = 0
        compounds_per_source: list[int] = []
        edge_writes = 0
        unique_inchikeys: set[str] = set()
        errors: list[str] = []

        for i, s in enumerate(sources, 1):
            if i % 25 == 0 or i == len(sources):
                self._log_progress(f"  {i}/{len(sources)}")
            try:
                status, compounds, evidence_type = self._link_source(s)
            except Exception as e:
                msg = f"{s.get('name', '?')}: {e}"
                logger.exception(msg)
                errors.append(msg)
                status, compounds, evidence_type = LinkStatus.ERROR, [], None

            by_status[status.value] += 1
            if evidence_type:
                by_evidence_type[evidence_type] += 1

            if status == LinkStatus.LINKED:
                compounds_per_source.append(len(compounds))
                # Track part-context distribution across all written edges
                part_label = _normalize_part(s.get("canonical_part"))
                part_specific = _is_part_specific(
                    s.get("canonical_type"), s.get("canonical_part"),
                )
                by_part_context[part_label] = by_part_context.get(part_label, 0) + len(compounds)
                if part_specific:
                    edges_part_specific += len(compounds)
                else:
                    edges_whole_or_compound_level += len(compounds)

                if not dry_run:
                    for c in compounds:
                        self._write_compound(c)
                        self._write_edge(c, s, evidence_type)
                        edge_writes += 1
                        unique_inchikeys.add(c.inchikey)
                else:
                    edge_writes += len(compounds)
                    for c in compounds:
                        unique_inchikeys.add(c.inchikey)

            if not dry_run:
                self._persist_status(s["name"], status, len(compounds), evidence_type)

        duration_s = round(time.time() - t_start, 2)
        ps = pubchem.get_stats()
        cs = coconut.get_stats()

        return {
            "dry_run": dry_run,
            "sources_total": len(sources),
            "by_status": by_status,
            "by_evidence_type": by_evidence_type,
            "by_part_context": dict(sorted(
                by_part_context.items(), key=lambda kv: -kv[1]
            )),
            "edges_part_specific": edges_part_specific,
            "edges_whole_or_compound_level": edges_whole_or_compound_level,
            "compounds_unique": len(unique_inchikeys),
            # `edge_writes` counts MERGE invocations on IS_EXTRACTED_FROM, NOT
            # newly-created edges. On a first run against an empty graph this
            # equals creations; on `--force-relink` it counts updates too.
            # Capture exact create-vs-update counts via Cypher result counters
            # in a follow-up if needed for paper figures.
            "edge_writes": edge_writes,
            "duration_s": duration_s,
            "external_calls": {
                "coconut_lookup_calls": cs["lookup_calls"],
                "coconut_lookup_hits": cs["lookup_hits"],
                "coconut_lookup_misses": cs["lookup_misses"],
                "coconut_alias_fallback_hits": cs["alias_fallback_hits"],
                "pubchem_name_calls": ps["name_calls"],
                "pubchem_formula_calls": ps["formula_calls"],
                "pubchem_cache_hits": ps["cache_hits"],
                "pubchem_errors": ps["errors"],
            },
            "compound_count_distribution": {
                "p50": _percentile(compounds_per_source, 50),
                "p75": _percentile(compounds_per_source, 75),
                "p95": _percentile(compounds_per_source, 95),
                "max": max(compounds_per_source) if compounds_per_source else 0,
            },
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Per-source routing
    # ------------------------------------------------------------------

    def _link_source(
        self, s: dict,
    ) -> tuple[LinkStatus, list[_Compound], str | None]:
        canonical_source = s.get("canonical_source")
        canonical_type = s.get("canonical_type")
        canonical_name = (s.get("canonical_name") or "").strip()
        aliases = s.get("aliases") or []

        if canonical_source is None:
            return LinkStatus.SKIPPED_NO_AUDITOR, [], None

        if canonical_type == "uncanonicalized" or not canonical_name:
            return LinkStatus.SKIPPED_UNCANONICALIZED, [], None

        if canonical_type == "organism":
            return self._lookup_organism(canonical_name, canonical_source, aliases)

        if canonical_type == "chemical":
            return self._lookup_chemical(canonical_name, canonical_source)

        # Unknown canonical_type (e.g. "error" auditor row) — treat as no result
        return LinkStatus.NO_COMPOUNDS_FOUND, [], None

    def _lookup_organism(
        self, canonical_name: str, canonical_source: str, aliases: list[str],
    ) -> tuple[LinkStatus, list[_Compound], str | None]:
        # Primary: canonical_name from auditor.
        coc = coconut.lookup_exact(canonical_name)
        if coc:
            evidence_type = (
                "coconut_organism_canonical"
                if canonical_source == "gemini+coconut"
                else "coconut_organism_unverified"
            )
            return self._wrap_coconut(coc, evidence_type)

        # Fallback: try Latin-binomial-shaped aliases.
        for alias in _binomial_aliases(aliases):
            if alias.lower() == canonical_name.lower():
                continue
            coc = coconut.lookup_exact(alias)
            if coc:
                coconut.record_alias_fallback_hit()
                return self._wrap_coconut(coc, "coconut_organism_alias")

        return LinkStatus.NO_COMPOUNDS_FOUND, [], None

    def _lookup_chemical(
        self, canonical_name: str, canonical_source: str,
    ) -> tuple[LinkStatus, list[_Compound], str | None]:
        # Primary: PubChem name lookup.
        p = pubchem.fetch_by_name(canonical_name)
        if p:
            evidence_type = (
                "pubchem_chemical_canonical"
                if canonical_source == "gemini+pubchem"
                else "pubchem_chemical_unverified"
            )
            compound = _pubchem_to_compound(p)
            return LinkStatus.LINKED, [compound], evidence_type

        # Fallback: formula lookup if the canonical name looks formula-shaped.
        # Formula lookups are inherently non-unique (anhydrous vs hydrate vs
        # polymorph share a formula); we keep recall by writing the first CID
        # and tag the edge with the candidate count so downstream knows it's
        # ambiguous rather than authoritative.
        if _looks_like_formula(canonical_name):
            fr = pubchem.fetch_by_formula(canonical_name)
            if fr:
                compound = _pubchem_to_compound(
                    fr.compound,
                    formula_candidates=fr.candidate_cids,
                    formula_candidate_count=fr.candidate_count,
                )
                return LinkStatus.LINKED, [compound], "pubchem_chemical_formula"

        return LinkStatus.NO_COMPOUNDS_FOUND, [], None

    @staticmethod
    def _wrap_coconut(
        rows: list[coconut.CoconutCompound], evidence_type: str,
    ) -> tuple[LinkStatus, list[_Compound], str | None]:
        compounds: list[_Compound] = []
        seen: set[str] = set()
        for r in rows:
            c = _coconut_to_compound(r)
            if c is None:
                continue
            if c.inchikey in seen:
                continue
            seen.add(c.inchikey)
            compounds.append(c)
        if not compounds:
            return LinkStatus.NO_COMPOUNDS_FOUND, [], None
        return LinkStatus.LINKED, compounds, evidence_type

    # ------------------------------------------------------------------
    # Graph writes
    # ------------------------------------------------------------------

    def _write_compound(self, c: _Compound) -> None:
        # Identity is `inchikey` (UNIQUE). Other properties are labels/tags;
        # only set them when truthy so unstructured compounds (no SMILES,
        # no formula, etc.) can coexist without colliding on empty-string
        # values. The smiles UNIQUE constraint was demoted to a range index
        # in schema.py for the same reason.
        on_create: dict[str, Any] = {
            "name": c.name,
            "source_db": c.source_db,
            "created_by": "source_compound_linker_v2",
            "created_at": dt.datetime.utcnow().isoformat(),
        }
        if c.smiles:
            on_create["smiles"] = c.smiles
        if c.molecular_formula:
            on_create["molecular_formula"] = c.molecular_formula
        if c.coconut_row is not None:
            on_create["coconut_row"] = c.coconut_row
        if c.pubchem_cid is not None:
            on_create["pubchem_cid"] = c.pubchem_cid
        if c.np_likeness is not None:
            on_create["np_likeness"] = c.np_likeness
        if c.annotation_level is not None:
            on_create["annotation_level"] = c.annotation_level

        self.client.merge_node(
            "Chemical_Compound",
            {"inchikey": c.inchikey},
            extra_on_create=on_create,
            match_key="inchikey",
        )

    def _write_edge(
        self,
        c: _Compound,
        s: dict,
        evidence_type: str,
    ) -> None:
        canonical_type = s.get("canonical_type")
        part_label = _normalize_part(s.get("canonical_part"))
        part_specific = _is_part_specific(canonical_type, s.get("canonical_part"))

        base_conf = CONFIDENCE[evidence_type]
        penalty = PART_SPECIFIC_PENALTY if part_specific else 0.0
        confidence = round(max(0.0, base_conf - penalty), 3)

        # evidence_resolution captures HOW well the source DB resolves the
        # compound to the Source's actual entity:
        #   - "compound_level" : PubChem returned the exact molecule
        #   - "species_level"  : COCONUT lists the compound for the species
        #                        as a whole (true for organisms, even when the
        #                        Source describes a specific part — the
        #                        species-level data can't disambiguate)
        if canonical_type == "organism":
            evidence_resolution = "species_level"
        elif canonical_type == "chemical":
            evidence_resolution = "compound_level"
        else:
            evidence_resolution = "unknown"

        rel_props: dict[str, Any] = {
            "confidence_score": confidence,
            "confidence_base_prior": base_conf,
            "confidence_part_penalty": penalty,
            "evidence_type": evidence_type,
            "evidence_resolution": evidence_resolution,
            "species_part_context": part_label,
            "part_specific": part_specific,
            "source_db": c.source_db,
            "lookup_query": s.get("canonical_name", ""),
            "created_by": "source_compound_linker_v2",
            "created_at": dt.datetime.utcnow().isoformat(),
        }
        if c.coconut_row is not None:
            rel_props["coconut_row"] = c.coconut_row
        if c.pubchem_cid is not None:
            rel_props["pubchem_cid"] = c.pubchem_cid
        if c.np_likeness is not None:
            rel_props["np_likeness"] = c.np_likeness
        if c.annotation_level is not None:
            rel_props["annotation_level"] = c.annotation_level

        # Formula-route ambiguity tags. Set only when the compound came from
        # PubChem's formula endpoint AND the formula matched more than one
        # CID. Downstream can filter on `pubchem_formula_ambiguous` to find
        # edges whose structural choice was non-deterministic.
        if c.pubchem_formula_candidate_count is not None:
            count = c.pubchem_formula_candidate_count
            rel_props["pubchem_formula_candidate_count"] = count
            rel_props["pubchem_formula_ambiguous"] = count > 1
            if c.pubchem_formula_candidate_cids:
                rel_props["pubchem_formula_candidate_cids"] = c.pubchem_formula_candidate_cids

        self.client.merge_edge(
            "Chemical_Compound", {"inchikey": c.inchikey},
            "Source", {"name": s["name"]},
            "IS_EXTRACTED_FROM",
            rel_props,
            from_key="inchikey",
        )

    def _persist_status(
        self,
        source_name: str,
        status: LinkStatus,
        compound_count: int,
        evidence_type: str | None,
    ) -> None:
        props: dict[str, Any] = {
            "linker_status": status.value,
            "linker_attempted_at": dt.datetime.utcnow().isoformat(),
            "linker_compound_count": compound_count,
        }
        if evidence_type:
            props["linker_evidence_type"] = evidence_type
        self.client.set_node_properties(
            "Source", {"name": source_name}, props,
        )
