"""ChEMBL REST client for compound→target lookup (v2).

Used by the Compound→Target linker to resolve a graph-side
Chemical_Compound (identified by InChIKey, with SMILES + name fallbacks)
to a normalized list of biological targets.

Three meaningful behaviours for the linker:

  1. Distinguish transient API failure from "compound not in ChEMBL":
       - HTTP 404                → permanent miss (NOT_IN_CHEMBL)
       - network error / 5xx     → TRANSIENT_FAILURE (caller marks `error`,
                                   the compound is re-tried via
                                   --retry-misses)
       - HTTP 200 with no record → permanent miss
     Returning None for both transient and permanent failures was a
     poison-the-data bug: a 503 during a long run silently mis-labels
     compounds as no_targets_found.

  2. Pagination via &limit=1000 on every endpoint. ChEMBL's default
     page size is 20; for target-rich compounds we were missing most
     of the activities and metadata. limit=1000 is the documented
     ChEMBL maximum and is sufficient: no compound, after the agent's
     per-compound cap of 20 targets, has 1000+ qualifying activities.

  3. Multi-target-type support:
       SINGLE PROTEIN   → uniprot_id is the natural display ID
       PROTEIN COMPLEX  → ChEMBL ID owns identity; UniProt is one of N subunits
       PROTEIN FAMILY   → e.g. "Tyrosine-protein kinases"
       ORGANISM         → e.g. Plasmodium falciparum (NCBI tax_id when known)
     The linker keys Biological_Target by `target_chembl_id` (always
     present); `uniprot_id` and `ncbi_tax_id` are stored as informational
     properties when applicable.

Public API:
    fetch_compound_data(compound, *, pchembl_floor, allowed_target_types,
                        min_assay_confidence) -> CompoundResult
    run_batch_search(compounds, *, max_workers, ...) -> list[CompoundResult]

Removed (vs. v1):
    * SINGLE-PROTEIN-only filter (now a parameter)
    * Hardcoded pchembl=5 (parameterized)
    * Single-page reads (now paginated)
    * `print()` debug spam (logging only)
    * 374 lines of commented-out duplicate code at the file bottom
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import quote

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
_TIMEOUT_S = 15
_RATE_LIMIT_S = 0.05      # ChEMBL has no documented hard limit; light politeness
_PAGE_SIZE = 1000         # ChEMBL's documented max page size

# Target types we accept by default. Mineral / cell-line / tissue and
# unchecked types are excluded — they're either too coarse for our
# downstream evidence chain or have data-quality issues.
DEFAULT_ALLOWED_TARGET_TYPES = (
    "SINGLE PROTEIN",
    "PROTEIN COMPLEX",
    "PROTEIN FAMILY",
    "ORGANISM",
)

# ChEMBL `assay_type` codes (one-letter):
#   B = Binding        — affinity measurements (Kd, Ki, IC50, etc.)
#   F = Functional     — activity in functional / cell / organism assays
#   A = ADMET          — pharmacokinetic, not target-binding
#   T = Toxicity       — adverse outcomes, not target-binding
#   P = Physicochemical — solubility/logP, not target-binding
#   U = Unassigned     — uncurated
# Default: Binding + Functional (covers protein binding AND organism /
# whole-cell assays — the latter is critical for parasite/microbe targets).
DEFAULT_ALLOWED_ASSAY_TYPES = ("B", "F")

# `standard_relation` values: "=", ">", "<", ">=", "<=", "~", "<<", ">>",
# "~". We accept exact-equality measurements only — inequality bounds are
# less reliable (the true value could be very different from the bound).
ALLOWED_STANDARD_RELATIONS = {"=", "==", "==", "~", "~="}


# ---------------------------------------------------------------------------
# Module-level session + rate-limit lock (concurrent-callable)
# ---------------------------------------------------------------------------

_session = requests.Session()
_session_lock = threading.Lock()
_last_call = [0.0]


# ---------------------------------------------------------------------------
# HTTP outcome discriminator (fixes the transient-failure-as-no_targets bug)
# ---------------------------------------------------------------------------


class HttpOutcome(str, Enum):
    OK = "ok"                          # 200 + parseable JSON; data may be empty
    NOT_FOUND = "not_found"            # 404 → permanent miss
    TRANSIENT_FAILURE = "transient"    # 5xx, timeouts, network errors


@dataclass
class HttpResponse:
    outcome: HttpOutcome
    data: dict | None = None
    status_code: int | None = None
    error: str | None = None


def _rate_limited_get(url: str) -> HttpResponse:
    """Thread-safe rate-limited GET with discriminated outcomes.

    The lock only covers the rate-limit timestamp arithmetic — NOT the
    HTTP call itself — so 8 workers actually parallelise (~15× speedup
    vs holding the lock through the network round-trip).
    """
    with _session_lock:
        elapsed = time.time() - _last_call[0]
        if elapsed < _RATE_LIMIT_S:
            time.sleep(_RATE_LIMIT_S - elapsed)
        _last_call[0] = time.time()
    try:
        r = _session.get(url, timeout=_TIMEOUT_S)
    except requests.RequestException as e:
        logger.warning("ChEMBL transient failure for %s: %s", url, e)
        return HttpResponse(HttpOutcome.TRANSIENT_FAILURE, error=str(e))

    if r.status_code == 200:
        try:
            return HttpResponse(HttpOutcome.OK, data=r.json(), status_code=200)
        except ValueError as e:
            return HttpResponse(
                HttpOutcome.TRANSIENT_FAILURE,
                status_code=200,
                error=f"non-JSON 200 response: {e}",
            )
    if r.status_code == 404:
        return HttpResponse(HttpOutcome.NOT_FOUND, status_code=404)
    if 500 <= r.status_code < 600:
        return HttpResponse(
            HttpOutcome.TRANSIENT_FAILURE,
            status_code=r.status_code,
            error=f"HTTP {r.status_code}",
        )
    # Other 4xx codes are unexpected — treat as transient so the compound
    # is re-tried rather than mis-labelled as a permanent miss.
    logger.warning("ChEMBL unexpected status %d for %s", r.status_code, url)
    return HttpResponse(
        HttpOutcome.TRANSIENT_FAILURE,
        status_code=r.status_code,
        error=f"HTTP {r.status_code}",
    )


def _fetch_paginated(base_url: str, payload_key: str) -> tuple[list[dict], bool]:
    """Fetch ALL pages of a ChEMBL endpoint. Returns (rows, ok).

    `ok` is False on transient failure at any point — the caller should
    treat that as `error` (retryable), not `not_found`. ChEMBL exposes
    pagination via `meta.next` URLs; we follow them until exhausted.
    """
    # ChEMBL defaults to XML; we always want JSON. Rather than threading
    # &format=json through every call site, set it centrally here.
    sep = "&" if "?" in base_url else "?"
    url = f"{base_url}{sep}limit={_PAGE_SIZE}&format=json"
    all_rows: list[dict] = []
    while url:
        resp = _rate_limited_get(url)
        if resp.outcome == HttpOutcome.NOT_FOUND:
            return all_rows, True  # 404 mid-pagination = end of data
        if resp.outcome != HttpOutcome.OK or resp.data is None:
            return all_rows, False
        all_rows.extend(resp.data.get(payload_key, []) or [])
        next_path = (resp.data.get("page_meta") or {}).get("next")
        if next_path:
            # ChEMBL `next` is a relative path like "/chembl/api/data/..."
            # Strip the host so we route through CHEMBL_API consistently.
            base = "https://www.ebi.ac.uk"
            url = f"{base}{next_path}" if next_path.startswith("/") else next_path
        else:
            url = None
    return all_rows, True


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChemblTarget:
    """A normalized compound→target relationship for the linker.

    Identity unification:
      `target_chembl_id` is ALWAYS present (the MERGE key downstream).
      `uniprot_id` is set only for SINGLE PROTEIN.
      `ncbi_tax_id` is set only for ORGANISM.
    """
    target_chembl_id: str
    target_type: str            # see DEFAULT_ALLOWED_TARGET_TYPES
    target_name: str            # ChEMBL pref_name — display
    uniprot_id: str = ""        # SINGLE PROTEIN only
    gene_symbol: str = ""       # SINGLE PROTEIN only
    ncbi_tax_id: str = ""       # ORGANISM only
    evidence_type: str = ""     # "mechanism" | "activity"
    pchembl_score: float | None = None
    assay_id: str = ""          # mechanism ChEMBL ID or assay ChEMBL ID
    assay_type: str = ""        # "B" | "F" — activities only; "" for mechanisms
    assay_description: str = "" # truncated; activities only
    mechanism_action: str | None = None        # mechanism only


@dataclass
class ChemblIndication:
    """A normalized compound→disease indication for the KNOWN_TREATS linker.

    Sourced from ChEMBL's `drug_indication` endpoint. `mesh_id` and
    `efo_id` are the cross-ontology IDs we use for downstream matching
    against existing Modern_Disease nodes.
    """
    drug_indication_id: str       # ChEMBL's row id (audit)
    molecule_chembl_id: str       # which family member this came from
    mesh_id: str                  # MeSH descriptor, e.g. "D003924"
    mesh_heading: str             # e.g. "Diabetes Mellitus, Type 2"
    efo_id: str                   # e.g. "EFO_0001359" (may be empty)
    efo_term: str                 # e.g. "type II diabetes mellitus"
    max_phase_for_ind: int        # 0 (preclinical) .. 4 (approved)


@dataclass
class IndicationResult:
    """Per-compound indication-lookup outcome.

    Mirrors `CompoundResult`'s discriminated outcomes so the agent can
    distinguish a permanent miss (compound has no indications in ChEMBL)
    from a transient failure (5xx / network) that should be retried.
    """
    origin_inchikey: str
    origin_chembl_id: str
    outcome: str = "ok"           # "ok" | "not_in_chembl" | "transient_failure"
    indications: list[ChemblIndication] = field(default_factory=list)
    family_size: int = 0
    error: str | None = None


@dataclass
class CompoundResult:
    """Per-compound lookup result.

    - `outcome == OK`  → consult `targets` (may be empty for "in ChEMBL but
                         no qualifying targets")
    - `outcome == NOT_IN_CHEMBL`  → permanent miss
    - `outcome == TRANSIENT_FAILURE`  → retryable; caller should NOT mark
                                        as no_targets_found
    """
    origin_inchikey: str
    origin_name: str
    origin_smiles: str
    outcome: str = "ok"            # "ok" | "not_in_chembl" | "transient_failure"
    chembl_id: str = ""
    chembl_pref_name: str = ""
    lookup_method: str = "none"    # "inchikey" | "smiles" | "name" | "none"
    targets: list[ChemblTarget] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Compound resolution (InChIKey → SMILES → name)
# ---------------------------------------------------------------------------


def _resolve_molecule(
    inchikey: str, smiles: str, name: str,
) -> tuple[dict | None, str, bool]:
    """Find the ChEMBL molecule record. Returns (record, method, ok).

    `ok` is False if any attempt hit a transient failure (so we mark the
    compound as `transient_failure`, not `not_in_chembl`).
    """
    transient_seen = False

    if inchikey and not inchikey.startswith("unstructured:"):
        url = (
            f"{CHEMBL_API}/molecule"
            f"?molecule_structures__standard_inchi_key={quote(inchikey)}"
        )
        rows, ok = _fetch_paginated(url, "molecules")
        if rows:
            return rows[0], "inchikey", True
        if not ok:
            transient_seen = True

    if smiles:
        url = (
            f"{CHEMBL_API}/molecule"
            f"?molecule_structures__canonical_smiles__exact={quote(smiles)}"
        )
        rows, ok = _fetch_paginated(url, "molecules")
        if rows:
            return rows[0], "smiles", True
        if not ok:
            transient_seen = True

    if name and name.lower() != "unknown":
        url = (
            f"{CHEMBL_API}/molecule"
            f"?pref_name__iexact={quote(name)}"
        )
        rows, ok = _fetch_paginated(url, "molecules")
        if rows:
            return rows[0], "name", True
        if not ok:
            transient_seen = True

    return None, "none", not transient_seen


def _expand_molecule_hierarchy(parent_id: str) -> tuple[list[str], bool]:
    """Return all ChEMBL IDs in the molecule's hierarchy (parent + salts +
    tautomers). Returns (ids, ok). False ok is transient failure."""
    url = (
        f"{CHEMBL_API}/molecule"
        f"?molecule_hierarchy__parent_chembl_id={parent_id}"
    )
    rows, ok = _fetch_paginated(url, "molecules")
    ids = [m["molecule_chembl_id"] for m in rows]
    if parent_id not in ids:
        ids.append(parent_id)
    return ids, ok


# ---------------------------------------------------------------------------
# Target metadata enrichment (multi-type-aware)
# ---------------------------------------------------------------------------


def _extract_target_identity(target_record: dict) -> dict:
    """Pull the identity-relevant fields from a ChEMBL /target record.

    Returns a dict with:
      target_chembl_id, target_type, target_name, uniprot_id, gene_symbol,
      ncbi_tax_id
    Empty strings for missing fields.
    """
    target_type = target_record.get("target_type") or ""
    out = {
        "target_chembl_id": target_record.get("target_chembl_id", "") or "",
        "target_type": target_type,
        "target_name": target_record.get("pref_name") or "",
        "uniprot_id": "",
        "gene_symbol": "",
        "ncbi_tax_id": "",
    }

    components = target_record.get("target_components") or []

    # SINGLE PROTEIN: take the only component's accession + gene symbol
    if target_type == "SINGLE PROTEIN" and components:
        comp = components[0] or {}
        out["uniprot_id"] = comp.get("accession") or ""
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") == "GENE_SYMBOL":
                out["gene_symbol"] = syn.get("component_synonym") or ""
                break

    # ORGANISM: tax_id lives on the target record itself
    if target_type == "ORGANISM":
        tax_id = target_record.get("tax_id")
        if tax_id is not None:
            out["ncbi_tax_id"] = str(tax_id)

    # PROTEIN COMPLEX / PROTEIN FAMILY: identity is the ChEMBL ID alone;
    # individual subunit accessions would be ambiguous and arbitrary.

    return out


_TARGET_METADATA_CHUNK_SIZE = 100  # cap IDs per /target?target_chembl_id__in= URL


def _fetch_target_metadata(
    target_chembl_ids: set[str], allowed_target_types: tuple[str, ...],
) -> tuple[dict[str, dict], bool]:
    """Bulk-fetch metadata for a set of target ChEMBL IDs. Returns
    ({target_chembl_id: identity_dict}, ok).

    Only targets whose `target_type` is in `allowed_target_types` are
    returned — others are silently dropped so the linker won't write
    them as edges.

    IDs are chunked at `_TARGET_METADATA_CHUNK_SIZE` to bound URL length.
    A curcumin-class compound can have hundreds of unique targets
    after pagination; ChEMBL IDs average ~12 chars, so 100 IDs per chunk
    keeps each URL well under any server's URL-length limit (~8 KB).
    Sets `ok=False` if any chunk hits a transient failure.
    """
    if not target_chembl_ids:
        return {}, True
    out: dict[str, dict] = {}
    overall_ok = True
    sorted_ids = sorted(target_chembl_ids)
    for i in range(0, len(sorted_ids), _TARGET_METADATA_CHUNK_SIZE):
        chunk = sorted_ids[i : i + _TARGET_METADATA_CHUNK_SIZE]
        ids_str = ",".join(chunk)
        url = f"{CHEMBL_API}/target?target_chembl_id__in={ids_str}"
        rows, ok = _fetch_paginated(url, "targets")
        if not ok:
            overall_ok = False
        for t in rows:
            identity = _extract_target_identity(t)
            if identity["target_type"] not in allowed_target_types:
                continue
            if not identity["target_chembl_id"]:
                continue
            out[identity["target_chembl_id"]] = identity
    return out, overall_ok


# ---------------------------------------------------------------------------
# Mechanism + activity collection
# ---------------------------------------------------------------------------


def _fetch_mechanisms(family_ids: list[str]) -> tuple[list[dict], bool]:
    if not family_ids:
        return [], True
    url = (
        f"{CHEMBL_API}/mechanism"
        f"?molecule_chembl_id__in={','.join(family_ids)}"
    )
    return _fetch_paginated(url, "mechanisms")


def _fetch_activities(
    family_ids: list[str], pchembl_floor: float,
) -> tuple[list[dict], bool]:
    """Activity rows with pchembl_value >= floor. Target-type filtering
    happens in the metadata enrichment step (ChEMBL doesn't accept
    target_type__in cleanly, so we filter post-hoc).
    """
    if not family_ids:
        return [], True
    url = (
        f"{CHEMBL_API}/activity"
        f"?molecule_chembl_id__in={','.join(family_ids)}"
        f"&pchembl_value__gte={pchembl_floor:.1f}"
    )
    return _fetch_paginated(url, "activities")


def _fetch_phenotypic_activities(
    family_ids: list[str], allowed_assay_types: tuple[str, ...],
) -> tuple[list[dict], bool]:
    """Functional/phenotypic activity rows that have NO pchembl_value.

    These represent qualitative biological activity (organism-level
    inhibition, antioxidant scavenging, parasite killing) where ChEMBL
    didn't compute a potency. Without these, well-studied natural
    products like myrcene/limonene/stigmasterol — which appear in
    dozens of Sources but have only phenotypic ChEMBL data — silently
    fall out of the graph.

    Filtering pipeline (downstream code applies more):
      - pchembl_value IS NULL (i.e. not already covered by _fetch_activities)
      - assay_type IN allowed (F by default — Functional)
      - target_chembl_id present (handled in caller via target_meta lookup)
      - placeholder targets (NON-PROTEIN TARGET / UNDEFINED / etc.) get
        dropped by the existing allowed_target_types filter in
        _fetch_target_metadata
    """
    if not family_ids:
        return [], True
    # ChEMBL filter syntax: pchembl_value__isnull=true narrows to rows
    # where the field is NULL. Combined with assay_type IN allowed,
    # this isolates phenotypic data.
    url = (
        f"{CHEMBL_API}/activity"
        f"?molecule_chembl_id__in={','.join(family_ids)}"
        f"&pchembl_value__isnull=true"
        f"&assay_type__in={','.join(allowed_assay_types)}"
    )
    return _fetch_paginated(url, "activities")


# ---------------------------------------------------------------------------
# Per-compound assembly
# ---------------------------------------------------------------------------


def fetch_compound_data(
    compound: dict, *,
    pchembl_floor: float = 5.0,
    allowed_target_types: tuple[str, ...] = DEFAULT_ALLOWED_TARGET_TYPES,
    allowed_assay_types: tuple[str, ...] = DEFAULT_ALLOWED_ASSAY_TYPES,
    include_phenotypic: bool = False,
) -> CompoundResult:
    """Look up one compound in ChEMBL and return its normalized targets.

    `compound` keys read: `inchikey`, `smiles`, `name`. Any may be empty;
    the resolver tries InChIKey first, then SMILES, then name.

    Default `pchembl_floor=5.0` writes all activity tiers (weak/moderate/
    strong) and lets downstream consumers filter at query time via the
    `evidence_type` edge property. The agent's per-tier confidence prior
    discriminates them so a `r.confidence_score >= 0.5` filter naturally
    excludes the weak tier without losing the data.

    `allowed_target_types` defaults to {SINGLE PROTEIN, PROTEIN COMPLEX,
    PROTEIN FAMILY, ORGANISM} — ORGANISM is critical for TCM corpora
    that include antiparasitic and antimicrobial remedies.

    `allowed_assay_types` defaults to ("B", "F") — Binding and Functional
    assays. Excludes ADMET, toxicity, and physicochemical measurements.
    Per-row data-validity flags and inequality `standard_relation` values
    are also filtered (we accept "=" and "~" only).

    `include_phenotypic=True` adds a second activity-fetch pass for rows
    with pchembl_value=NULL (qualitative bioactivity). Recovers compounds
    like terpenes / sterols / sugars that have rich phenotypic data
    (antiparasitic, antimicrobial, antifungal organism assays) but no
    quantitative potency. Tagged as evidence_type=`activity` with
    pchembl_score=None — the agent maps these to `chembl_phenotypic` at
    confidence 0.40. Off by default for backward-compatibility with the
    initial live run.
    """
    result = CompoundResult(
        origin_inchikey=compound.get("inchikey", "") or "",
        origin_name=compound.get("name", "") or "",
        origin_smiles=compound.get("smiles", "") or "",
    )

    try:
        molecule, method, ok_resolve = _resolve_molecule(
            result.origin_inchikey, result.origin_smiles, result.origin_name,
        )
        result.lookup_method = method
        if not ok_resolve:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_molecule_resolution"
            return result
        if molecule is None:
            result.outcome = "not_in_chembl"
            return result

        result.chembl_id = molecule.get("molecule_chembl_id", "")
        result.chembl_pref_name = molecule.get("pref_name") or ""
        parent_id = (
            (molecule.get("molecule_hierarchy") or {}).get("parent_chembl_id")
            or result.chembl_id
        )
        family_ids, ok_family = _expand_molecule_hierarchy(parent_id)
        if not ok_family:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_hierarchy_expansion"
            return result

        mechanisms, ok_mech = _fetch_mechanisms(family_ids)
        if not ok_mech:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_mechanism_fetch"
            return result

        activities, ok_act = _fetch_activities(family_ids, pchembl_floor)
        if not ok_act:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_activity_fetch"
            return result

        # Optional second pass for phenotypic / qualitative activities.
        # See _fetch_phenotypic_activities for rationale; gated on
        # include_phenotypic to keep the default reproducibility footprint
        # intact.
        if include_phenotypic:
            phen_activities, ok_phen = _fetch_phenotypic_activities(
                family_ids, allowed_assay_types,
            )
            if not ok_phen:
                result.outcome = "transient_failure"
                result.error = "transient_failure_during_phenotypic_fetch"
                return result
            activities = activities + phen_activities

        # Bulk enrich each unique target (mechanism + activity union)
        target_chembl_ids: set[str] = set()
        for m in mechanisms:
            tid = m.get("target_chembl_id")
            if tid:
                target_chembl_ids.add(tid)
        for a in activities:
            tid = a.get("target_chembl_id")
            if tid:
                target_chembl_ids.add(tid)

        target_meta, ok_meta = _fetch_target_metadata(
            target_chembl_ids, allowed_target_types,
        )
        if not ok_meta:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_target_metadata"
            return result

        # Emit ChemblTarget rows. Mechanisms first (gold standard); skip
        # any target that mechanism already covered, so the same
        # (compound, target) pair never gets two ChemblTarget rows.
        seen_targets: set[str] = set()
        for m in mechanisms:
            tid = m.get("target_chembl_id")
            meta = target_meta.get(tid)
            if not meta or tid in seen_targets:
                continue
            seen_targets.add(tid)
            result.targets.append(ChemblTarget(
                target_chembl_id=tid,
                target_type=meta["target_type"],
                target_name=meta["target_name"],
                uniprot_id=meta["uniprot_id"],
                gene_symbol=meta["gene_symbol"],
                ncbi_tax_id=meta["ncbi_tax_id"],
                evidence_type="mechanism",
                pchembl_score=None,
                assay_id=str(m.get("mec_id") or tid),
                mechanism_action=m.get("action_type"),
            ))

        # Collapse activities per target by max pchembl, with assay-quality
        # filtering (assay_type + standard_relation + data_validity) and
        # per-target dedup vs mechanisms.
        # Preserve `None` for phenotypic activities (no pchembl_value).
        # Converting None → 0.0 here would cause the agent's
        # _evidence_type_for() to mis-tag them as `chembl_activity_weak`
        # instead of `chembl_phenotypic`. Same data, wrong audit label.
        best_act: dict[str, dict] = {}
        for a in activities:
            tid = a.get("target_chembl_id")
            if not tid or tid not in target_meta or tid in seen_targets:
                continue
            # Quality gates
            atype = a.get("assay_type") or ""
            if atype not in allowed_assay_types:
                continue
            relation = a.get("standard_relation") or ""
            if relation and relation not in ALLOWED_STANDARD_RELATIONS:
                continue
            if a.get("data_validity_comment"):
                # Non-empty validity flag = ChEMBL flagged this row as
                # questionable (out-of-range, transcription error, etc.)
                continue
            raw_p = a.get("pchembl_value")
            if raw_p is None or raw_p == "":
                p: float | None = None
                rank: float = -1.0   # phenotypic sorts below all quantitative
            else:
                try:
                    p = float(raw_p)
                    rank = p
                except (TypeError, ValueError):
                    p = None
                    rank = -1.0
            if tid not in best_act or rank > best_act[tid]["rank"]:
                desc = (a.get("assay_description") or "")[:200]
                best_act[tid] = {
                    "pchembl": p,             # None for phenotypic
                    "rank": rank,             # numeric for sorting/dedup
                    "assay_id": a.get("assay_chembl_id") or "",
                    "assay_type": atype,
                    "assay_description": desc,
                }
        for tid, info in best_act.items():
            meta = target_meta[tid]
            result.targets.append(ChemblTarget(
                target_chembl_id=tid,
                target_type=meta["target_type"],
                target_name=meta["target_name"],
                uniprot_id=meta["uniprot_id"],
                gene_symbol=meta["gene_symbol"],
                ncbi_tax_id=meta["ncbi_tax_id"],
                evidence_type="activity",
                pchembl_score=info["pchembl"],   # None for phenotypic
                assay_id=info["assay_id"],
                assay_type=info["assay_type"],
                assay_description=info["assay_description"],
                mechanism_action=None,
            ))

        return result

    except Exception as e:
        logger.exception("ChEMBL lookup raised for %r", result.origin_name)
        result.outcome = "transient_failure"
        result.error = f"unexpected exception: {e}"
        return result


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Drug-indication lookup (KNOWN_TREATS linker)
# ---------------------------------------------------------------------------


def _fetch_drug_indications(family_ids: list[str]) -> tuple[list[dict], bool]:
    """Fetch all rows from `/drug_indication` for the given molecule IDs.

    Indications are typically attached to the parent ChEMBL ID (the drug
    proper, not its salt forms or tautomers), so the caller passes the
    full molecule hierarchy and we paginate over the union.
    """
    if not family_ids:
        return [], True
    url = (
        f"{CHEMBL_API}/drug_indication"
        f"?molecule_chembl_id__in={','.join(family_ids)}"
    )
    return _fetch_paginated(url, "drug_indications")


def _normalize_indication(row: dict) -> ChemblIndication | None:
    """Parse a raw ChEMBL drug_indication row into a `ChemblIndication`.

    Rows lacking both `mesh_id` and `efo_id` are dropped — without at
    least one cross-ontology ID, downstream matching has nothing to
    work with.
    """
    mesh_id = (row.get("mesh_id") or "").strip()
    efo_id = (row.get("efo_id") or "").strip()
    if not mesh_id and not efo_id:
        return None
    raw_phase = row.get("max_phase_for_ind")
    try:
        phase = int(float(raw_phase)) if raw_phase is not None else 0
    except (TypeError, ValueError):
        phase = 0
    return ChemblIndication(
        drug_indication_id=str(row.get("drug_indication_id") or ""),
        molecule_chembl_id=row.get("molecule_chembl_id") or "",
        mesh_id=mesh_id,
        mesh_heading=(row.get("mesh_heading") or "").strip(),
        efo_id=efo_id,
        efo_term=(row.get("efo_term") or "").strip(),
        max_phase_for_ind=phase,
    )


def fetch_compound_indications(
    chembl_id: str, *, inchikey: str = "",
) -> IndicationResult:
    """Look up a compound's clinical indications in ChEMBL.

    Walks the molecule hierarchy starting from `chembl_id` and queries
    `/drug_indication` for every family member, so an indication
    attached to the parent (typical case) is still found when the
    cached ID is a salt or tautomer.

    Per (compound, mesh_id) pair, multiple ChEMBL rows can exist at
    different `max_phase_for_ind` values — keep the row with the
    highest phase. The agent does the same dedup again at the
    Modern_Disease level (after matching a mesh_id to a graph node)
    in case two distinct mesh_ids land on the same disease node.
    """
    result = IndicationResult(
        origin_inchikey=inchikey or "",
        origin_chembl_id=chembl_id or "",
    )
    if not chembl_id:
        result.outcome = "not_in_chembl"
        return result

    try:
        # Resolve the molecule first to get its actual parent_chembl_id.
        # The cached `linker_chembl_id` may be a salt or tautomer (a
        # child in ChEMBL's hierarchy). Calling _expand_molecule_hierarchy
        # on a child would return only that child — missing every
        # indication attached to the parent. Drug indications are
        # almost always recorded on the parent, so this resolution is
        # load-bearing for recall.
        # ChEMBL defaults to XML; force JSON so _rate_limited_get's r.json()
        # parse succeeds. The paginated endpoints handle this centrally in
        # _fetch_paginated, but this is a direct (non-paginated) call.
        url = f"{CHEMBL_API}/molecule/{quote(chembl_id)}?format=json"
        resp = _rate_limited_get(url)
        if resp.outcome == HttpOutcome.NOT_FOUND:
            result.outcome = "not_in_chembl"
            return result
        if resp.outcome != HttpOutcome.OK or resp.data is None:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_molecule_lookup"
            return result
        molecule = resp.data
        parent_id = (
            (molecule.get("molecule_hierarchy") or {}).get("parent_chembl_id")
            or chembl_id
        )

        family_ids, ok_family = _expand_molecule_hierarchy(parent_id)
        if not ok_family:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_hierarchy_expansion"
            return result
        result.family_size = len(family_ids)

        rows, ok = _fetch_drug_indications(family_ids)
        if not ok:
            result.outcome = "transient_failure"
            result.error = "transient_failure_during_indication_fetch"
            return result
        if not rows:
            result.outcome = "not_in_chembl"
            return result

        # Per-mesh_id dedup at max-phase. Falls back to efo_id when
        # mesh_id is empty so EFO-only indications are preserved.
        best: dict[str, ChemblIndication] = {}
        for row in rows:
            ind = _normalize_indication(row)
            if ind is None:
                continue
            key = ind.mesh_id or f"efo:{ind.efo_id}"
            if key not in best or ind.max_phase_for_ind > best[key].max_phase_for_ind:
                best[key] = ind

        result.indications = list(best.values())
        return result

    except Exception as e:
        logger.exception("ChEMBL indication lookup raised for %r", chembl_id)
        result.outcome = "transient_failure"
        result.error = f"unexpected exception: {e}"
        return result


# ---------------------------------------------------------------------------
# Bulk drug-indication lookup (batched KNOWN_TREATS linker)
# ---------------------------------------------------------------------------
#
# Per-compound calls dominate wall clock at our scale (~5K compounds × 3 HTTP
# round trips × ~300ms = tens of minutes). ChEMBL's `__in=` filter accepts
# ~100 IDs per URL, so collapsing to bulk calls gives ~25-50× speedup.
#
# Three primitives, each with `(parent_dict, ok)` return shape mirroring
# `_fetch_paginated`. `ok=False` is a coarse transient-failure signal — the
# caller treats the entire batch as transient. Re-running with --retry-misses
# picks up the failed compounds.

_BULK_CHUNK_SIZE = 100        # IDs per __in= URL (URL-length bound)
_BULK_MAX_WORKERS = 8         # parallel chunk fetches per primitive


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _bulk_fetch_concurrent(
    chunks: list[list[str]],
    url_builder,                      # callable[chunk] -> url
    payload_key: str,
    max_workers: int = _BULK_MAX_WORKERS,
) -> tuple[list[dict], bool]:
    """Run paginated GETs across `chunks` concurrently.
    Returns (all_rows_flattened, ok). ok=False if ANY chunk failed."""
    if not chunks:
        return [], True
    overall_ok = True
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_fetch_paginated, url_builder(c), payload_key): c
                for c in chunks}
        for fut in as_completed(futs):
            try:
                rows, ok = fut.result()
            except Exception as e:
                logger.warning("bulk chunk failed: %s", e)
                overall_ok = False
                continue
            if not ok:
                overall_ok = False
            all_rows.extend(rows or [])
    return all_rows, overall_ok


def bulk_resolve_molecules(
    chembl_ids: list[str],
) -> tuple[dict[str, str], bool]:
    """Map each input chembl_id → its parent_chembl_id (or itself if it IS
    the parent). Used to canonicalize cached `linker_chembl_id` values
    that might be salts/tautomers before hierarchy expansion.

    Returns ({chembl_id: parent_chembl_id}, ok). Missing inputs (ChEMBL
    has no record) are absent from the dict — caller treats as
    `not_in_chembl`.
    """
    unique_ids = list({i for i in chembl_ids if i})
    if not unique_ids:
        return {}, True
    chunks = _chunked(unique_ids, _BULK_CHUNK_SIZE)
    rows, ok = _bulk_fetch_concurrent(
        chunks,
        lambda c: f"{CHEMBL_API}/molecule?molecule_chembl_id__in={','.join(c)}",
        "molecules",
    )
    out: dict[str, str] = {}
    for m in rows:
        cid = m.get("molecule_chembl_id")
        if not cid:
            continue
        parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id") or cid
        out[cid] = parent
    return out, ok


def bulk_expand_hierarchy(
    parent_ids: list[str],
) -> tuple[dict[str, list[str]], bool]:
    """Map each parent_chembl_id → all family members (parent + salts +
    tautomers). Each parent maps to at least [parent_id] even when no
    siblings exist.

    Returns ({parent_chembl_id: [family_chembl_ids]}, ok).
    """
    unique_parents = list({p for p in parent_ids if p})
    if not unique_parents:
        return {}, True
    out: dict[str, list[str]] = {p: [p] for p in unique_parents}
    chunks = _chunked(unique_parents, _BULK_CHUNK_SIZE)
    rows, ok = _bulk_fetch_concurrent(
        chunks,
        lambda c: (
            f"{CHEMBL_API}/molecule"
            f"?molecule_hierarchy__parent_chembl_id__in={','.join(c)}"
        ),
        "molecules",
    )
    for m in rows:
        mid = m.get("molecule_chembl_id")
        parent = (m.get("molecule_hierarchy") or {}).get("parent_chembl_id")
        if not mid or not parent or parent not in out:
            continue
        if mid not in out[parent]:
            out[parent].append(mid)
    return out, ok


def bulk_fetch_indications(
    family_ids: list[str],
) -> tuple[dict[str, list[ChemblIndication]], bool]:
    """Map each family_chembl_id → all indications attached to it.
    Indications are usually attached to the parent ChEMBL ID (the drug
    proper, not its salt forms), but we fan over the whole family for
    safety — exact mirror of the per-compound code path.

    Returns ({family_chembl_id: [ChemblIndication]}, ok). Family members
    with no indications are absent from the dict.
    """
    unique_ids = list({i for i in family_ids if i})
    if not unique_ids:
        return {}, True
    chunks = _chunked(unique_ids, _BULK_CHUNK_SIZE)
    rows, ok = _bulk_fetch_concurrent(
        chunks,
        lambda c: (
            f"{CHEMBL_API}/drug_indication"
            f"?molecule_chembl_id__in={','.join(c)}"
        ),
        "drug_indications",
    )
    out: dict[str, list[ChemblIndication]] = {}
    for row in rows:
        ind = _normalize_indication(row)
        if ind is None:
            continue
        cid = ind.molecule_chembl_id
        if not cid:
            continue
        out.setdefault(cid, []).append(ind)
    return out, ok


def fetch_compound_indications_batch(
    inchikey_to_chembl_id: dict[str, str],
) -> dict[str, IndicationResult]:
    """Bulk-fetch indications for many compounds at once.

    Composes the three bulk primitives + per-compound assembly. Returns
    `{inchikey: IndicationResult}` — same `IndicationResult` shape the
    per-compound `fetch_compound_indications` returns, so downstream
    matching/writing code is identical.

    Coarse failure semantics: if any of the three bulk phases hits a
    transient failure, EVERY compound in the batch is marked
    `transient_failure`. The caller's `--retry-misses` pass picks them
    up next run. This is the intentional simplicity tradeoff vs the
    per-compound version's fine-grained discrimination — for the scale
    we operate at, ChEMBL is reliable enough that a whole-batch retry
    is rarely needed.

    Per-(compound, mesh_id|efo_id) max-phase dedup at the assembly
    step. Same logic as `fetch_compound_indications`.
    """
    if not inchikey_to_chembl_id:
        return {}

    cached_ids = list({cid for cid in inchikey_to_chembl_id.values() if cid})

    # Phase 1: resolve to parent IDs
    cid_to_parent, ok1 = bulk_resolve_molecules(cached_ids)
    if not ok1:
        return _all_transient(
            inchikey_to_chembl_id, "transient_failure_during_bulk_resolve",
        )

    # Phase 2: expand hierarchies
    parent_ids = list({p for p in cid_to_parent.values() if p})
    parent_to_family, ok2 = bulk_expand_hierarchy(parent_ids)
    if not ok2:
        return _all_transient(
            inchikey_to_chembl_id, "transient_failure_during_bulk_hierarchy",
        )

    # Phase 3: fetch indications
    all_family_ids = [
        fid for family in parent_to_family.values() for fid in family
    ]
    family_to_inds, ok3 = bulk_fetch_indications(all_family_ids)
    if not ok3:
        return _all_transient(
            inchikey_to_chembl_id, "transient_failure_during_bulk_indications",
        )

    # Phase 4: assemble per-compound results with max-phase dedup
    results: dict[str, IndicationResult] = {}
    for inchikey, cached_id in inchikey_to_chembl_id.items():
        if not cached_id:
            results[inchikey] = IndicationResult(
                origin_inchikey=inchikey, origin_chembl_id="",
                outcome="not_in_chembl",
            )
            continue
        parent = cid_to_parent.get(cached_id)
        if parent is None:
            results[inchikey] = IndicationResult(
                origin_inchikey=inchikey, origin_chembl_id=cached_id,
                outcome="not_in_chembl",
            )
            continue
        family = parent_to_family.get(parent, [parent])
        all_inds: list[ChemblIndication] = []
        for fid in family:
            all_inds.extend(family_to_inds.get(fid, []))
        if not all_inds:
            results[inchikey] = IndicationResult(
                origin_inchikey=inchikey, origin_chembl_id=cached_id,
                outcome="not_in_chembl", family_size=len(family),
            )
            continue
        # Per-mesh_id max-phase dedup (falls back to efo_id when mesh_id empty)
        best: dict[str, ChemblIndication] = {}
        for ind in all_inds:
            key = ind.mesh_id or f"efo:{ind.efo_id}"
            if key not in best or ind.max_phase_for_ind > best[key].max_phase_for_ind:
                best[key] = ind
        results[inchikey] = IndicationResult(
            origin_inchikey=inchikey, origin_chembl_id=cached_id,
            outcome="ok", indications=list(best.values()),
            family_size=len(family),
        )

    return results


def _all_transient(
    inchikey_to_chembl_id: dict[str, str], error: str,
) -> dict[str, IndicationResult]:
    """Build a `{inchikey: transient_failure}` map for whole-batch failure."""
    return {
        ik: IndicationResult(
            origin_inchikey=ik, origin_chembl_id=cid or "",
            outcome="transient_failure", error=error,
        )
        for ik, cid in inchikey_to_chembl_id.items()
    }


# ---------------------------------------------------------------------------
# Target subunit lookup (PROTEIN COMPLEX subunit fan-out)
# ---------------------------------------------------------------------------
#
# Used by TargetDiseaseLinker for PROTEIN COMPLEX targets — we want each
# subunit's UniProt accession so we can query Open Targets for each
# component independently and aggregate (max-score per disease) into a
# single complex-level RELATES_TO edge.


def bulk_fetch_target_components(
    target_chembl_ids: list[str],
) -> tuple[dict[str, list[str]], bool]:
    """Map each target_chembl_id → list of subunit UniProt accessions.

    Returns ({target_chembl_id: [uniprot_accessions]}, ok). Targets
    that ChEMBL returns with no `target_components` (or that aren't
    actually multi-subunit) get an empty list.

    Used for PROTEIN COMPLEX subunit fan-out — single-protein target
    callers should keep using `_fetch_target_metadata` directly.
    """
    unique_ids = list({t for t in target_chembl_ids if t})
    if not unique_ids:
        return {}, True
    out: dict[str, list[str]] = {}
    overall_ok = True
    for chunk in _chunked(unique_ids, _BULK_CHUNK_SIZE):
        url = (
            f"{CHEMBL_API}/target?target_chembl_id__in={','.join(chunk)}"
        )
        rows, ok = _fetch_paginated(url, "targets")
        if not ok:
            overall_ok = False
        for t in rows:
            tid = t.get("target_chembl_id") or ""
            if not tid:
                continue
            uniprots: list[str] = []
            seen: set[str] = set()
            for comp in (t.get("target_components") or []):
                acc = (comp or {}).get("accession") or ""
                if acc and acc not in seen:
                    seen.add(acc)
                    uniprots.append(acc)
            out[tid] = uniprots
    return out, overall_ok


def run_batch_search(
    compounds: list[dict], *,
    max_workers: int = 8,
    pchembl_floor: float = 5.0,
    allowed_target_types: tuple[str, ...] = DEFAULT_ALLOWED_TARGET_TYPES,
    allowed_assay_types: tuple[str, ...] = DEFAULT_ALLOWED_ASSAY_TYPES,
) -> list[CompoundResult]:
    """Parallel ChEMBL lookup. Results returned in completion order, NOT
    input order. Each item carries `origin_inchikey/name/smiles` so the
    caller can match back to graph nodes."""
    results: list[CompoundResult] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(
                fetch_compound_data, c,
                pchembl_floor=pchembl_floor,
                allowed_target_types=allowed_target_types,
                allowed_assay_types=allowed_assay_types,
            ): c
            for c in compounds
        }
        for fut in as_completed(futs):
            results.append(fut.result())
    return results
