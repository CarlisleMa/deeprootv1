"""COCONUT 2.0 compound database — exact-match organism lookup.

Loads the trimmed COCONUT parquet snapshot once at import time and builds
an in-memory inverted index from species name → row indices, so the
Source→Compound linker can do O(1) exact lookups instead of substring
scans across the full table.

Public API:
    lookup_exact(species_name) -> list[CoconutCompound]
    CoconutCompound dataclass

Removed (vs. previous version):
    * `clean_name`, `_gemini_fallback` — name canonicalization is now
      owned by `ExtractionAuditor`. The auditor's `canonical_name` is
      passed straight to `lookup_exact`. The data layer no longer
      instantiates a Gemini client at import time.
    * The `(np_likeness * annotation_level) / 25` confidence formula —
      that score conflates "compound annotation quality" with "is this
      compound a constituent of this organism." Raw `np_likeness` and
      `annotation_level` are exposed as tags; the linker assigns flat
      per-evidence-type priors to IS_EXTRACTED_FROM edges.
    * `MAX_COMPOUNDS_PER_SOURCE_TOP_PERCENTILE` truncation — removed.
      All matched compounds are returned; downstream Phase 2 can filter.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CoconutCompound:
    name: str                  # COCONUT 'name' (preferred) or 'iupac_name'
    smiles: str                # COCONUT 'canonical_smiles'
    molecular_formula: str
    coconut_row: int           # parquet row index (stable within a snapshot)
    np_likeness: float         # raw COCONUT score, 0–5 (passed through, NOT confidence)
    annotation_level: int      # raw COCONUT level, 0–5 (passed through, NOT confidence)
    source_db: str = "COCONUT"


# ---------------------------------------------------------------------------
# Parquet load + index build (at module import)
# ---------------------------------------------------------------------------


_PARQUET_PATH = Path(__file__).parent / "coconut_02-2026-trimmed.parquet"
if not _PARQUET_PATH.exists():
    raise FileNotFoundError(
        f"COCONUT snapshot not found at {_PARQUET_PATH}.\n"
        "This ~40 MB derived dataset is not shipped in the repository. "
        "Build it once with:\n"
        "    python scripts/prepare_coconut.py --source <COCONUT_bulk_export>\n"
        "See data/README.md for where to obtain the COCONUT 2.0 source and the "
        "required output schema."
    )
_df = pd.read_parquet(_PARQUET_PATH)
logger.info("Loaded %d compounds from %s", len(_df), _PARQUET_PATH.name)


_WS_RE = re.compile(r"\s+")


def _normalize_species(s: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation/spaces."""
    if not s:
        return ""
    return _WS_RE.sub(" ", s).strip().lower()


def _genus_species(s: str) -> str | None:
    """Return the first two tokens of a species name, lowercased.

    Used as a fallback so "Coptis chinensis" matches an indexed
    "Coptis chinensis Franch." (botanist annotation) and similar.
    Returns None if the input has fewer than two word tokens.
    """
    norm = _normalize_species(s)
    if not norm:
        return None
    tokens = norm.split()
    if len(tokens) < 2:
        return None
    return f"{tokens[0]} {tokens[1]}"


def _split_organisms(cell: str | None) -> list[str]:
    """COCONUT's `organisms` column is `|`-separated for multi-organism rows."""
    if not cell:
        return []
    return [_normalize_species(o) for o in cell.split("|") if o.strip()]


def _build_organism_index() -> dict[str, list[int]]:
    """Build species → list of parquet row indices.

    Indexes both the full normalized species name AND the two-token
    "genus species" prefix, so lookups tolerate trailing botanist
    annotations like "Franch." or "L. ssp. xyz".
    """
    index: dict[str, list[int]] = defaultdict(list)
    organisms_col = _df["organisms"].fillna("")
    for row_idx, cell in enumerate(organisms_col):
        for organism in _split_organisms(cell):
            index[organism].append(row_idx)
            gs = _genus_species(organism)
            if gs and gs != organism:
                index[gs].append(row_idx)
    # Deduplicate the row-index lists (genus+species may collide with full
    # species for single-token-after-genus organisms).
    for key in list(index.keys()):
        index[key] = sorted(set(index[key]))
    logger.info("Built organism index: %d distinct keys", len(index))
    return index


_ORGANISM_INDEX: dict[str, list[int]] = _build_organism_index()


# ---------------------------------------------------------------------------
# Stats — counters exposed for the linker's run summary, mirroring the
# pubchem module so call accounting can be reported accurately per run.
# ---------------------------------------------------------------------------


_stats = {
    "lookup_calls": 0,        # any call to lookup_exact() with a non-empty name
    "lookup_hits": 0,         # calls that returned at least one compound
    "lookup_misses": 0,       # calls that returned []
    "alias_fallback_hits": 0, # caller-incremented when alias resolves after canonical miss
}


def get_stats() -> dict[str, int]:
    """Return a snapshot of call counters for the linker's run summary."""
    return dict(_stats)


def reset_stats() -> None:
    """Reset call counters. Useful between linker runs."""
    for k in _stats:
        _stats[k] = 0


def record_alias_fallback_hit() -> None:
    """Caller-side counter increment: a Latin-binomial alias resolved after
    the auditor's canonical_name missed. Tracked here (instead of in the
    linker) so the data-layer stats stay self-contained."""
    _stats["alias_fallback_hits"] += 1


# ---------------------------------------------------------------------------
# Row → dataclass
# ---------------------------------------------------------------------------


def _row_to_compound(row_idx: int) -> CoconutCompound | None:
    row = _df.iloc[row_idx]
    name = row.get("name") or row.get("iupac_name") or ""
    if not name:
        return None
    return CoconutCompound(
        name=str(name),
        smiles=str(row.get("canonical_smiles") or ""),
        molecular_formula=str(row.get("molecular_formula") or ""),
        coconut_row=int(row_idx),
        np_likeness=float(row.get("np_likeness") or 0.0),
        annotation_level=int(row.get("annotation_level") or 0),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def lookup_exact(species_name: str) -> list[CoconutCompound]:
    """Return all COCONUT compounds known to be derived from this species.

    Lookup is exact match against the inverted index (case-folded,
    whitespace-collapsed). If the full species name misses, a second
    attempt is made on the first two tokens ("genus species" prefix).
    Returns an empty list on miss.
    """
    if not species_name:
        return []

    _stats["lookup_calls"] += 1

    norm = _normalize_species(species_name)
    rows = _ORGANISM_INDEX.get(norm)
    if rows is None:
        gs = _genus_species(species_name)
        if gs:
            rows = _ORGANISM_INDEX.get(gs)

    if not rows:
        _stats["lookup_misses"] += 1
        return []
    _stats["lookup_hits"] += 1

    out: list[CoconutCompound] = []
    seen_keys: set[str] = set()
    for row_idx in rows:
        compound = _row_to_compound(row_idx)
        if compound is None:
            continue
        # Deduplicate by SMILES if present (multiple parquet rows may
        # describe the same molecule under slightly different names);
        # otherwise dedupe by name.
        key = compound.smiles or f"name::{compound.name.lower()}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(compound)
    return out
