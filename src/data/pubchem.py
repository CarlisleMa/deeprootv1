"""PubChem REST client for compound lookup.

Two entry points used by the Source→Compound linker:

  fetch_by_name(name)      → first-CID compound by exact name lookup
  fetch_by_formula(formula) → first-CID compound by molecular formula

Both return a `PubchemCompound` dataclass or None on miss / 404. Results
are persisted to a disk JSON cache so re-runs are free; rate-limiting
follows PubChem's "no more than 5 requests/sec" recommendation.

PubChem REST docs: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
_RATE_LIMIT_S = 0.25      # PubChem: "no more than 5 requests/sec"
_TIMEOUT_S = 10
_PROPERTIES = "SMILES,InChIKey,IUPACName,MolecularFormula"

_CACHE_DIR = Path.home() / ".cache" / "deeproot"
_CACHE_PATH = _CACHE_DIR / "pubchem.json"

# Module-level state for cache + rate limiting. The cache is loaded lazily
# and persisted on every write (the cache is small — bounded by the number
# of distinct names/formulas the linker has ever queried).
_cache: dict[str, dict | None] | None = None
_cache_lock = threading.Lock()
_last_call: list[float] = [0.0]

# Counters exposed for the linker's run summary.
_stats = {
    "name_calls": 0,
    "formula_calls": 0,
    "cache_hits": 0,
    "errors": 0,
}


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class PubchemCompound:
    cid: int
    name: str             # IUPAC name from PubChem
    smiles: str           # CanonicalSMILES from PubChem
    inchikey: str
    molecular_formula: str
    source_db: str = "PubChem"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _load_cache() -> dict[str, dict | None]:
    global _cache
    if _cache is not None:
        return _cache
    if _CACHE_PATH.exists():
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to read PubChem cache (%s); starting fresh", e)
            _cache = {}
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    if _cache is None:
        return
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(_CACHE_PATH)


def _cache_get(key: str) -> tuple[bool, dict | None]:
    """Return (hit, value). value=None means a cached negative result."""
    cache = _load_cache()
    if key in cache:
        return True, cache[key]
    return False, None


def _cache_put(key: str, value: dict | None) -> None:
    with _cache_lock:
        cache = _load_cache()
        cache[key] = value
        _save_cache()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _rate_limit() -> None:
    elapsed = time.time() - _last_call[0]
    if elapsed < _RATE_LIMIT_S:
        time.sleep(_RATE_LIMIT_S - elapsed)
    _last_call[0] = time.time()


def _get_json(url: str) -> dict | None:
    """GET a PubChem URL and return parsed JSON, or None on 404/error."""
    _rate_limit()
    try:
        r = requests.get(url, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        _stats["errors"] += 1
        logger.warning("PubChem GET failed for %s: %s", url, e)
        return None
    if r.status_code == 200:
        try:
            return r.json()
        except ValueError:
            _stats["errors"] += 1
            return None
    if r.status_code == 404:
        return None
    _stats["errors"] += 1
    logger.warning("PubChem unexpected status %d for %s", r.status_code, url)
    return None


def _properties_to_compound(props: dict, cid: int) -> PubchemCompound | None:
    # PubChem currently exposes "SMILES" (modern). Older API used
    # "CanonicalSMILES"; some endpoints also return "ConnectivitySMILES".
    # Try in priority order so the module survives field renames.
    smiles = (
        props.get("SMILES")
        or props.get("CanonicalSMILES")
        or props.get("ConnectivitySMILES")
        or ""
    )
    inchikey = props.get("InChIKey") or ""
    if not smiles or not inchikey:
        # PubChem returned a CID but no usable structure — skip.
        return None
    return PubchemCompound(
        cid=int(cid),
        name=props.get("IUPACName") or "",
        smiles=smiles,
        inchikey=inchikey,
        molecular_formula=props.get("MolecularFormula") or "",
    )


def _fetch_properties_for_cid(cid: int) -> PubchemCompound | None:
    url = f"{_BASE}/cid/{cid}/property/{_PROPERTIES}/JSON"
    data = _get_json(url)
    if not data:
        return None
    table = (data.get("PropertyTable") or {}).get("Properties") or []
    if not table:
        return None
    return _properties_to_compound(table[0], cid)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_by_name(name: str) -> PubchemCompound | None:
    """Look up a compound by exact name. Returns None on 404 or no structure."""
    if not name or not name.strip():
        return None
    cache_key = f"name::{name.strip().lower()}"
    hit, cached = _cache_get(cache_key)
    if hit:
        _stats["cache_hits"] += 1
        return PubchemCompound(**cached) if cached else None

    _stats["name_calls"] += 1
    quoted = requests.utils.quote(name, safe="")
    url = f"{_BASE}/name/{quoted}/property/{_PROPERTIES}/JSON"
    data = _get_json(url)
    if not data:
        _cache_put(cache_key, None)
        return None
    table = (data.get("PropertyTable") or {}).get("Properties") or []
    if not table:
        _cache_put(cache_key, None)
        return None
    first = table[0]
    cid = first.get("CID")
    if not cid:
        _cache_put(cache_key, None)
        return None
    compound = _properties_to_compound(first, int(cid))
    _cache_put(cache_key, asdict(compound) if compound else None)
    return compound


@dataclass
class FormulaResult:
    """Result of a formula-based PubChem lookup. Carries ambiguity context
    so callers can tag edges honestly when a formula matches many CIDs.

    `candidate_cids` is capped at the first 25 to keep edge property size
    bounded; `candidate_count` is the unbounded total reported by PubChem.
    """
    compound: PubchemCompound
    candidate_cids: list[int]   # at most 25, in PubChem's returned order
    candidate_count: int        # total CIDs PubChem returned (may exceed list)


_FORMULA_CIDS_CAP = 25


def fetch_by_formula(formula: str) -> FormulaResult | None:
    """Look up a compound by molecular formula. Multiple compounds typically
    share a formula (anhydrous vs hydrate vs polymorph etc.) — returns a
    FormulaResult that carries the chosen first-CID compound PLUS the
    ambiguity context (count + list of candidates), so the caller can
    annotate the edge as ambiguous when warranted. None on 404 / no result.
    """
    if not formula or not formula.strip():
        return None
    cache_key = f"formula::{formula.strip()}"
    hit, cached = _cache_get(cache_key)
    if hit:
        if cached is None:
            _stats["cache_hits"] += 1
            return None
        # Cache schema changed (PubchemCompound flat dict → nested
        # {compound, candidate_cids, candidate_count}). Guard against legacy
        # entries by detecting the old shape and treating as a miss; the
        # next call will refresh and overwrite with the new shape.
        if not isinstance(cached, dict) or "compound" not in cached:
            pass  # fall through to live fetch below
        else:
            _stats["cache_hits"] += 1
            return FormulaResult(
                compound=PubchemCompound(**cached["compound"]),
                candidate_cids=cached["candidate_cids"],
                candidate_count=cached["candidate_count"],
            )

    _stats["formula_calls"] += 1
    quoted = requests.utils.quote(formula, safe="")
    url = f"{_BASE}/fastformula/{quoted}/cids/JSON"
    data = _get_json(url)
    if not data:
        _cache_put(cache_key, None)
        return None
    cids = (data.get("IdentifierList") or {}).get("CID") or []
    if not cids:
        _cache_put(cache_key, None)
        return None
    compound = _fetch_properties_for_cid(int(cids[0]))
    if compound is None:
        _cache_put(cache_key, None)
        return None

    candidate_cids = [int(c) for c in cids[:_FORMULA_CIDS_CAP]]
    result = FormulaResult(
        compound=compound,
        candidate_cids=candidate_cids,
        candidate_count=len(cids),
    )
    _cache_put(cache_key, {
        "compound": asdict(compound),
        "candidate_cids": candidate_cids,
        "candidate_count": len(cids),
    })
    return result


def get_stats() -> dict[str, int]:
    """Return a snapshot of call counters for the linker's run summary."""
    return dict(_stats)


def reset_stats() -> None:
    """Reset call counters. Useful between linker runs."""
    for k in _stats:
        _stats[k] = 0
