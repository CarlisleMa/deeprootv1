"""Open Targets GraphQL client for target → disease association lookup.

Used by the TargetDiseaseLinker to resolve graph-side Biological_Target
nodes (identified by `target_chembl_id`, with `uniprot_id` for SINGLE
PROTEIN) to a normalized list of associated diseases with confidence
scores.

Three meaningful behaviours for the linker:

  1. Bulk-fetch primitives. Per-target queries are slow at scale; the
     linker passes lists and gets back dicts keyed by the input ID.
     Parallel chunked fetches under a thread pool.

  2. Discriminated outcomes. A network blip on one chunk should mark
     ONLY those targets as transient (caller flips them to `error`,
     they get re-tried via --retry-misses) — never as
     `no_associations` (which would mis-label them as permanent
     misses).

  3. UniProt → ENSEMBL resolution. OT's primary key is ENSEMBL gene ID;
     our targets have UniProt. Resolution is one search query per
     UniProt, parallelized.

What this module is NOT:
  * Not a generic OT search client (that's the job of OT's GraphQL
    explorer at https://platform.opentargets.org/api).
  * Not a per-pair scorer. The `targets→associated diseases` query
    returns the full disease list per target with scores in one
    paginated call; per-pair queries would multiply API calls by N.
  * Not an LLM-backed semantic mapper. The legacy implementation
    fell back to mapping symptoms to high-level therapeutic-area EFO
    IDs via Gemini, which produced semantically wrong edges (target
    ↔ "Nervous System Disease" rather than no edge). Removed.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OT_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"
_TIMEOUT_S = 20
_RATE_LIMIT_S = 0.05      # OT has no documented hard limit; light politeness
_PAGE_SIZE = 200          # per associatedDiseases page
_BULK_MAX_WORKERS = 8     # parallel target queries


_session = requests.Session()
_session_lock = threading.Lock()
_last_call = [0.0]


# ---------------------------------------------------------------------------
# Discriminated HTTP outcome
# ---------------------------------------------------------------------------


class OtOutcome(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"           # 200 with empty data, or 404
    TRANSIENT_FAILURE = "transient"   # 5xx, timeouts, network errors


@dataclass
class OtResponse:
    outcome: OtOutcome
    data: dict | None = None
    status_code: int | None = None
    error: str | None = None


def _rate_limited_post(query: str, variables: dict | None) -> OtResponse:
    """Thread-safe rate-limited POST with discriminated outcomes.

    The lock only covers the rate-limit timestamp arithmetic — NOT
    the HTTP call itself — so workers actually parallelise. Mirrors
    the pattern in chembl.py.
    """
    with _session_lock:
        elapsed = time.time() - _last_call[0]
        if elapsed < _RATE_LIMIT_S:
            time.sleep(_RATE_LIMIT_S - elapsed)
        _last_call[0] = time.time()
    try:
        r = _session.post(
            OT_GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            timeout=_TIMEOUT_S,
        )
    except requests.RequestException as e:
        logger.warning("Open Targets transient failure: %s", e)
        return OtResponse(OtOutcome.TRANSIENT_FAILURE, error=str(e))

    if r.status_code == 200:
        try:
            payload = r.json()
        except ValueError as e:
            return OtResponse(
                OtOutcome.TRANSIENT_FAILURE, status_code=200,
                error=f"non-JSON 200 response: {e}",
            )
        # GraphQL embeds errors in the body even with 200 OK
        if payload.get("errors"):
            err_msg = "; ".join(
                e.get("message", "?") for e in payload.get("errors", [])
            )
            logger.warning("Open Targets GraphQL error: %s", err_msg)
            return OtResponse(
                OtOutcome.TRANSIENT_FAILURE, status_code=200, error=err_msg,
            )
        return OtResponse(OtOutcome.OK, data=payload.get("data") or {}, status_code=200)
    if r.status_code == 404:
        return OtResponse(OtOutcome.NOT_FOUND, status_code=404)
    if 500 <= r.status_code < 600:
        return OtResponse(
            OtOutcome.TRANSIENT_FAILURE, status_code=r.status_code,
            error=f"HTTP {r.status_code}",
        )
    logger.warning("Open Targets unexpected status %d", r.status_code)
    return OtResponse(
        OtOutcome.TRANSIENT_FAILURE, status_code=r.status_code,
        error=f"HTTP {r.status_code}",
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OtAssociation:
    """A normalized target→disease association from Open Targets."""
    disease_id: str             # EFO_xxxxx or MONDO_xxxxx
    disease_name: str           # OT's preferred label
    score: float                # OT overall_score 0..1
    datasource_scores: dict     # {datasource_id: score}
    target_ensembl_id: str      # which target this came from


@dataclass
class OtTargetResult:
    """Per-target associated-disease lookup result.

    Mirrors chembl.IndicationResult's discriminated outcomes so the
    linker can distinguish transient failures from permanent misses.
    """
    origin_uniprot_id: str
    target_ensembl_id: str = ""
    outcome: str = "ok"            # "ok" | "no_target" | "transient_failure"
    associations: list[OtAssociation] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# UniProt → ENSEMBL resolution
# ---------------------------------------------------------------------------


_RESOLVE_QUERY = """
query resolveByUniProt($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 1}) {
    hits {
      id
      name
      object {
        ... on Target {
          id
          approvedSymbol
        }
      }
    }
  }
}
"""


def _resolve_one_uniprot(uniprot_id: str) -> tuple[str, str, bool]:
    """Return (uniprot_id, ensembl_id_or_empty, ok).

    `ok=False` ONLY when the underlying HTTP call hit a transient
    failure — the caller treats that target as `transient_failure`
    rather than mis-labeling it as a permanent miss (no_target).
    A successful 200 with an empty hit list returns ("", True) —
    a legitimate "OT doesn't know this UniProt" answer.

    REGRESSION fix: the prior version returned ("", "") for both
    transient AND genuine miss. Under --force-relink, that flowed
    into _plan_protein_target's no_target branch and queued the
    target for delete_rows — a single OT outage during a full-graph
    force-relink would wipe every existing RELATES_TO edge.
    """
    if not uniprot_id:
        return uniprot_id, "", True
    resp = _rate_limited_post(_RESOLVE_QUERY, {"q": uniprot_id})
    if resp.outcome == OtOutcome.TRANSIENT_FAILURE:
        return uniprot_id, "", False
    if resp.outcome != OtOutcome.OK or resp.data is None:
        return uniprot_id, "", True   # NOT_FOUND / 200-with-no-data — clean miss
    hits = (resp.data.get("search") or {}).get("hits") or []
    if not hits:
        return uniprot_id, "", True
    obj = hits[0].get("object") or {}
    return uniprot_id, obj.get("id") or "", True


def bulk_resolve_uniprot_to_ensembl(
    uniprot_ids: list[str], *, max_workers: int = _BULK_MAX_WORKERS,
) -> tuple[dict[str, str], set[str], bool]:
    """Map a batch of UniProt IDs → ENSEMBL gene IDs via OT search.

    Returns ({uniprot: ensembl}, transient_uniprots_set, overall_ok).
    `transient_uniprots_set` is the set of UniProt IDs that hit a
    transient failure (caller marks those as transient, not no_target).
    `overall_ok` is False if ANY transient occurred.

    Missing-but-not-transient inputs (OT returned no hit) get an empty
    string in the dict and are NOT in the transient set — treat as
    legitimate `no_target`.
    """
    unique_ids = list({u for u in uniprot_ids if u})
    if not unique_ids:
        return {}, set(), True
    out: dict[str, str] = {}
    transient: set[str] = set()
    overall_ok = True
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_resolve_one_uniprot, u): u for u in unique_ids}
        for fut in as_completed(futs):
            try:
                uniprot, ensembl, ok = fut.result()
                out[uniprot] = ensembl
                if not ok:
                    transient.add(uniprot)
                    overall_ok = False
            except Exception as e:
                logger.warning("uniprot resolve failed: %s", e)
                transient.add(futs[fut])
                overall_ok = False
                out[futs[fut]] = ""
    return out, transient, overall_ok


# ---------------------------------------------------------------------------
# Associated diseases per target
# ---------------------------------------------------------------------------


_ASSOC_QUERY = """
query targetAssoc($id: String!, $index: Int!, $size: Int!) {
  target(ensemblId: $id) {
    id
    associatedDiseases(page: {index: $index, size: $size}) {
      count
      rows {
        score
        datasourceScores { id, score }
        disease {
          id
          name
        }
      }
    }
  }
}
"""


def _fetch_associations_for_target(
    ensembl_id: str, *, min_score: float = 0.0,
) -> tuple[list[OtAssociation], bool]:
    """Paginated fetch of all associated diseases for one target.

    Returns (associations, ok). `min_score` filters at fetch time to
    avoid pulling tens of thousands of low-score rows for promiscuous
    targets. ok=False on any transient failure mid-pagination.
    """
    all_assocs: list[OtAssociation] = []
    page_index = 0
    while True:
        resp = _rate_limited_post(
            _ASSOC_QUERY,
            {"id": ensembl_id, "index": page_index, "size": _PAGE_SIZE},
        )
        if resp.outcome == OtOutcome.NOT_FOUND:
            return all_assocs, True
        if resp.outcome != OtOutcome.OK or resp.data is None:
            return all_assocs, False
        target_block = resp.data.get("target")
        if target_block is None:
            return all_assocs, True   # OT doesn't know this target — clean miss
        assoc_block = target_block.get("associatedDiseases") or {}
        rows = assoc_block.get("rows") or []
        if not rows:
            break
        for row in rows:
            score = row.get("score")
            if score is None or score < min_score:
                continue
            disease = row.get("disease") or {}
            ds_scores = {
                ds.get("id"): ds.get("score")
                for ds in (row.get("datasourceScores") or [])
                if ds.get("id")
            }
            all_assocs.append(OtAssociation(
                disease_id=disease.get("id", ""),
                disease_name=disease.get("name", "") or "",
                score=float(score),
                datasource_scores=ds_scores,
                target_ensembl_id=ensembl_id,
            ))
        # OT's pagination: count is total available; if rows < page size,
        # we got the last page. Otherwise advance.
        if len(rows) < _PAGE_SIZE:
            break
        page_index += 1
        if page_index > 50:    # defensive cap — 10K associations is plenty
            logger.warning("Pagination cap hit for ENSG %s", ensembl_id)
            break
    return all_assocs, True


def bulk_fetch_associated_diseases(
    ensembl_ids: list[str], *,
    min_score: float = 0.0,
    max_workers: int = _BULK_MAX_WORKERS,
) -> tuple[dict[str, list[OtAssociation]], set[str], bool]:
    """Map a batch of ENSEMBL IDs → list of associated-disease rows.

    Returns ({ensembl_id: [OtAssociation]}, transient_ensembls_set, overall_ok).

    Each target is paginated independently; targets are processed in
    parallel. `transient_ensembls_set` is the set of ENSEMBL IDs whose
    fetch hit a transient failure AT ANY POINT during pagination —
    including partial-completion failures. The caller treats those
    targets as `transient_failure`, NOT as `ok` with partial data.

    REGRESSION fix: the prior version stored partial-paginated rows
    in the output dict regardless of `ok`. Downstream then saw the
    ENSEMBL key present and treated it as a successful response with
    truncated data — overwriting prior edges with incomplete info
    under --force-relink.
    """
    unique_ids = list({i for i in ensembl_ids if i})
    if not unique_ids:
        return {}, set(), True
    out: dict[str, list[OtAssociation]] = {}
    transient: set[str] = set()
    overall_ok = True
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_fetch_associations_for_target, eid, min_score=min_score): eid
            for eid in unique_ids
        }
        for fut in as_completed(futs):
            eid = futs[fut]
            try:
                assocs, ok = fut.result()
                if not ok:
                    transient.add(eid)
                    overall_ok = False
                    # Drop partial assocs — better to retry than to
                    # write incomplete data as if it were complete.
                    out[eid] = []
                else:
                    out[eid] = assocs
            except Exception as e:
                logger.warning("association fetch failed for %s: %s", eid, e)
                transient.add(eid)
                overall_ok = False
                out[eid] = []
    return out, transient, overall_ok


def fetch_target_associations_batch(
    uniprot_to_target_id: dict[str, str], *, min_score: float = 0.0,
) -> dict[str, OtTargetResult]:
    """End-to-end: bulk-resolve UniProts → ENSEMBLs → fetch associated
    diseases. Returns {target_chembl_id: OtTargetResult}.

    Per-target failure semantics: a transient failure in either phase
    marks ONLY the affected targets as `transient_failure`. The caller's
    --retry-misses run picks them up. Targets whose lookup completed
    cleanly are returned as `ok` (or `no_target` if OT genuinely had
    no record). This is the per-target discrimination the v1 batched
    code was missing — without it, partial OT outages either wiped
    every target's prior edges (when treated as transient) or stamped
    every target as `no_target` (when treated as a clean miss).
    """
    if not uniprot_to_target_id:
        return {}

    # Phase 1: resolve UniProts to ENSEMBL IDs
    uniprots = list({u for u in uniprot_to_target_id.values() if u})
    uniprot_to_ensembl, transient_uniprots, _ok1 = (
        bulk_resolve_uniprot_to_ensembl(uniprots)
    )

    # Phase 2: fetch associations for resolved ENSEMBL IDs
    ensembl_ids = [eid for eid in uniprot_to_ensembl.values() if eid]
    ensembl_to_assocs, transient_ensembls, _ok2 = (
        bulk_fetch_associated_diseases(ensembl_ids, min_score=min_score)
    )

    # Phase 3: assemble per-target results, preserving per-target outcome
    results: dict[str, OtTargetResult] = {}
    for tcid, uniprot in uniprot_to_target_id.items():
        # Transient resolve → THIS target is transient (not no_target)
        if uniprot in transient_uniprots:
            results[tcid] = OtTargetResult(
                origin_uniprot_id=uniprot,
                outcome="transient_failure",
                error="transient_during_uniprot_resolve",
            )
            continue
        ensembl = uniprot_to_ensembl.get(uniprot, "")
        if not ensembl:
            # Clean miss: OT returned 200 with no hits for this UniProt
            results[tcid] = OtTargetResult(
                origin_uniprot_id=uniprot, outcome="no_target",
            )
            continue
        # Transient association fetch → THIS target is transient
        if ensembl in transient_ensembls:
            results[tcid] = OtTargetResult(
                origin_uniprot_id=uniprot,
                target_ensembl_id=ensembl,
                outcome="transient_failure",
                error="transient_during_association_fetch",
            )
            continue
        assocs = ensembl_to_assocs.get(ensembl, [])
        results[tcid] = OtTargetResult(
            origin_uniprot_id=uniprot,
            target_ensembl_id=ensembl,
            outcome="ok",
            associations=assocs,
        )
    return results
