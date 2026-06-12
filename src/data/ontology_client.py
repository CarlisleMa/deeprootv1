"""Rate-limited HTTP client for medical ontology APIs.

Wraps NLM Clinical Tables (ICD-10-CM), MeSH RDF Lookup, and EBI OLS4
(SNOMED CT) behind a single interface with connection pooling, automatic
retries, and a token-bucket rate limiter.

Usage:
    from src.data.ontology_client import OntologyClient

    oc = OntologyClient()
    icd_results = oc.search_icd10("malaria")
    mesh_results = oc.search_mesh("malaria")
    snomed_results = oc.search_snomed("malaria")
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config import (
    ICD10_API_BASE,
    MESH_LOOKUP_BASE,
    OLS4_API_BASE,
    ONTOLOGY_API_RATE_LIMIT,
)

logger = logging.getLogger(__name__)


class OntologyClient:
    """Rate-limited HTTP client for medical ontology APIs."""

    def __init__(self, max_requests_per_second: int | None = None):
        rps = max_requests_per_second or ONTOLOGY_API_RATE_LIMIT
        self._min_interval = 1.0 / rps
        self._last_request_time = 0.0
        # Guards _last_request_time AND the requests.Session — both are
        # used concurrently by the malady mapper's parallel verifier.
        self._lock = threading.Lock()

        self._session = requests.Session()

        # Retry on transient errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> OntologyClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Rate-limited GET
    # ------------------------------------------------------------------

    def _rate_limited_get(
        self, url: str, params: dict | None = None, timeout: int = 15
    ) -> requests.Response:
        """GET with thread-safe token-bucket rate limiting.

        The lock only covers the rate-limit timestamp arithmetic — NOT the
        HTTP call itself — so concurrent workers actually parallelize
        (~10× speedup vs holding the lock through the network round-trip).
        Mirrors the pattern in chembl.py's _rate_limited_get. requests.Session
        is documented thread-safe for read operations (GETs), and the
        connection pool internal to the session handles concurrent dispatch.
        """
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_request_time = time.time()
        return self._session.get(url, params=params, timeout=timeout)

    # ------------------------------------------------------------------
    # ICD-10-CM  (NLM Clinical Tables)
    # ------------------------------------------------------------------

    def search_icd10(self, term: str, max_results: int = 10) -> list[dict]:
        """Search NLM Clinical Tables for ICD-10-CM codes.

        Returns list of dicts: {name, icd10_code, source, match_type, confidence}
        """
        try:
            resp = self._rate_limited_get(
                ICD10_API_BASE,
                params={"sf": "code,name", "terms": term, "maxList": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("ICD-10 API error for %r: %s", term, e)
            return []

        # Response format: [total_count, [codes], extra_data, [[code, name], ...]]
        candidates: list[dict] = []
        if isinstance(data, list) and len(data) >= 4 and data[3]:
            for code, name in data[3]:
                is_exact = name.lower().strip() == term.lower().strip()
                candidates.append(
                    {
                        "name": name,
                        "icd10_code": code,
                        "mesh_id": "",
                        "snomed_id": "",
                        "source": "icd10",
                        "match_type": "exact" if is_exact else "fuzzy",
                        "confidence": 0.95 if is_exact else 0.65,
                    }
                )
        return candidates

    # ------------------------------------------------------------------
    # MeSH  (NLM MeSH RDF Lookup)
    # ------------------------------------------------------------------

    def search_mesh(self, term: str, limit: int = 10) -> list[dict]:
        """Search NLM MeSH RDF Lookup for descriptor matches.

        Returns list of dicts: {name, mesh_id, source, match_type, confidence}
        """
        try:
            resp = self._rate_limited_get(
                MESH_LOOKUP_BASE,
                params={"label": term, "match": "contains", "limit": limit},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("MeSH API error for %r: %s", term, e)
            return []

        candidates: list[dict] = []
        results = data if isinstance(data, list) else []
        for r in results:
            label = r.get("label", "")
            resource = r.get("resource", "")
            mesh_id = resource.split("/")[-1] if resource else ""
            is_exact = label.lower().strip() == term.lower().strip()
            candidates.append(
                {
                    "name": label,
                    "icd10_code": "",
                    "mesh_id": mesh_id,
                    "snomed_id": "",
                    "source": "mesh",
                    "match_type": "exact" if is_exact else "fuzzy",
                    "confidence": 0.95 if is_exact else 0.65,
                }
            )
        return candidates

    # ------------------------------------------------------------------
    # SNOMED CT  (EBI OLS4)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # MeSH synonym lookup (ChEMBL-side expansion for KNOWN_TREATS)
    # ------------------------------------------------------------------

    def get_mesh_synonyms(self, mesh_id: str) -> list[str]:
        """Return all entry-term labels (preferred + non-preferred) for a
        MeSH descriptor ID, e.g. "D003924" → ["Diabetes Mellitus, Type 2",
        "Type 2 Diabetes Mellitus", "T2DM", ...].

        Used by the Compound→Disease and Target→Disease linkers as
        tier-3 of disease matching: when an external ID ships a mesh_id
        but our graph node has no MeSH coverage, we expand the descriptor's
        synonym list and try to match each synonym (normalized) against
        existing Modern_Disease names. This bridges the gap when the
        surface forms differ.

        IMPORTANT: the correct endpoint is `/mesh/lookup/details` (NOT
        `/mesh/lookup/term`, which expects a label-search query and
        returns 400 for descriptor-keyed lookups). The `descriptor`
        parameter must be the FULL URI form
        (`http://id.nlm.nih.gov/mesh/D003924`) — bare IDs return 400.
        Supplementary records (C-prefix like C535275) return 200 with
        an empty `terms` array — that's the expected structural
        difference, not an error.

        Returns an empty list on any failure — tier 3 silently skips.
        """
        if not mesh_id:
            return []
        url = "https://id.nlm.nih.gov/mesh/lookup/details"
        try:
            resp = self._rate_limited_get(
                url,
                params={"descriptor": f"http://id.nlm.nih.gov/mesh/{mesh_id}"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("MeSH synonym lookup failed for %r: %s", mesh_id, e)
            return []

        terms = (data or {}).get("terms") or []
        if not isinstance(terms, list):
            return []
        labels: list[str] = []
        seen: set[str] = set()
        for entry in terms:
            label = (entry or {}).get("label", "")
            if not label:
                continue
            key = label.lower()
            if key in seen:
                continue
            seen.add(key)
            labels.append(label)
        return labels

    # ------------------------------------------------------------------
    # Pathogen → disease lookup (MONDO + DOID via OLS4 v2 relatedFrom)
    # Used by TargetDiseaseLinker for ORGANISM targets — maps an NCBI
    # tax_id to the disease classes that pathogen causes. Each disease
    # class has an OWL `has_material_basis_in` (MONDO) or `caused_by`
    # (DOID) restriction pointing to the NCBITaxon class. OLS4 v2's
    # `relatedFrom` endpoint walks that axiom in reverse — give it
    # the NCBITaxon IRI, get back every disease class whose restriction
    # points at it.
    #
    # NOTE: I originally implemented this as an OLS4 fulltext search
    # for the string "NCBITaxon_{id}" within EFO+DOID. That returned
    # 0 hits for every canonical pathogen because OLS4's search index
    # doesn't tokenize axiom-references. Switching to v2 relatedFrom
    # against MONDO+DOID (NOT EFO — EFO doesn't carry these axioms;
    # MONDO does) returns the actual disease classes.
    #
    # Each returned record includes the class's `dbXRefs` — typically
    # ~10 cross-references including MeSH, ICD-10, SNOMED, EFO, DOID,
    # UMLS. This lets the caller's matcher tier-1 match by ANY of
    # those IDs against the in-graph Modern_Disease universe, not
    # just by name.
    # ------------------------------------------------------------------

    _OLS_V2_BASE = "https://www.ebi.ac.uk/ols4/api/v2"

    def get_pathogen_diseases_efo(self, tax_id: str) -> list[dict]:
        """Return MONDO disease entries that have a `has_material_basis_in`
        relation pointing to NCBITaxon_{tax_id}.

        Despite the method name (kept for backwards compatibility with
        the agent's call sites), this queries MONDO — EFO inherits
        from MONDO but doesn't carry the pathogen axioms directly,
        so MONDO is the authoritative source. The returned records
        carry MONDO IDs which can be matched against Modern_Disease
        via the mondo_id field, OR via any of the dbXRefs (mesh_id,
        icd10_code, etc.).
        """
        return self._pathogen_disease_v2(
            tax_id, ontology="mondo", source_label="efo",
        )

    def get_pathogen_diseases_doid(self, tax_id: str) -> list[dict]:
        """Return DOID disease entries that reference NCBITaxon_{tax_id}.

        DOID exposes the same caused_by relations independently from
        MONDO, so querying both gives us cross-validation when both
        agree (the "consensus" tier in TargetDiseaseLinker)."""
        return self._pathogen_disease_v2(
            tax_id, ontology="doid", source_label="doid",
        )

    # Cap on ancestors per relatedFrom hit. OLS4's /ancestors returns
    # the full chain to root; we typically want the immediate broader
    # disease class (e.g. "Plasmodium falciparum malaria" → "malaria")
    # but NOT the universal classes ("infectious disease", "disease").
    # Cap=5 keeps the candidate list manageable while almost always
    # including the relevant parent (which sorts near the top in OLS4).
    _ANCESTOR_CAP = 5

    def _pathogen_disease_v2(
        self, tax_id: str, *, ontology: str, source_label: str,
    ) -> list[dict]:
        """Query OLS4 v2 relatedFrom for disease classes pointing at
        NCBITaxon_{tax_id}, walk ancestors via /ancestors, and return
        every class along the chain as a separate candidate.

        Why ancestor walking:
          OLS4 returns the SPECIFIC disease class whose axiom references
          the pathogen (e.g. "Plasmodium falciparum malaria" with
          MeSH:D016778). Our in-graph Modern_Disease nodes are usually
          at the BROADER level (e.g. "Malaria" with MeSH:D008288 — the
          parent MeSH descriptor). Tier-1 ID matching would miss
          because the IDs differ. The /ancestors endpoint returns the
          full chain in one call; each ancestor's dbXRefs include the
          ID of that broader MeSH/ICD/SNOMED descriptor — letting
          tier-1 fire on whichever level matches the in-graph node.

          Cap on ancestors (`_ANCESTOR_CAP`) keeps overly-broad
          classes ("disease", "infectious disease") from inflating
          the candidate list. OLS4 sorts ancestors by distance from
          the start class, so the first 5 are always the closest
          parents.

        Per-tax_id cost: 1 relatedFrom + (per hit) class-detail + 1
        ancestors call + (per ancestor, capped at 5) class-detail.
        For canonical pathogens with 1 relatedFrom hit, ~7-8 HTTP
        calls total. Cached at the agent level (one method
        invocation per tax_id per run).
        """
        if not tax_id:
            return []
        from urllib.parse import quote
        ncbi_iri = f"http://purl.obolibrary.org/obo/NCBITaxon_{tax_id}"
        encoded_ncbi = quote(quote(ncbi_iri, safe=""), safe="")

        related_url = (
            f"{self._OLS_V2_BASE}/ontologies/{ontology}"
            f"/classes/{encoded_ncbi}/relatedFrom"
        )
        try:
            resp = self._rate_limited_get(related_url, params={"size": 100})
            resp.raise_for_status()
            elements = resp.json().get("elements") or []
        except Exception as e:
            logger.warning(
                "%s relatedFrom failed for tax_id %r: %s",
                ontology, tax_id, e,
            )
            return []

        out: list[dict] = []
        seen_curies: set[str] = set()

        def _emit(curie: str, label: str, iri: str) -> None:
            """Build a candidate record from a class IRI; skip duplicates."""
            if not curie or curie in seen_curies:
                return
            seen_curies.add(curie)
            xrefs = self._fetch_class_xrefs(ontology, iri) if iri else {}
            out.append({
                "name": label,
                "id": curie,
                "iri": iri,
                "source": source_label,
                "mesh_id": xrefs.get("MESH", ""),
                "icd10_code": xrefs.get("ICD10CM") or xrefs.get("ICD10WHO", ""),
                "snomed_id": xrefs.get("SCTID", ""),
                "all_xrefs": xrefs,
            })

        for el in elements:
            curie = el.get("curie") or el.get("shortForm") or ""
            label = el.get("label", "")
            iri = el.get("iri") or ""
            if isinstance(label, list):
                label = label[0] if label else ""
            if not curie or not label:
                continue
            # Emit the specific class first
            _emit(curie, label, iri)
            # Then walk ancestors (one call returns the whole chain)
            for a_curie, a_label, a_iri in self._fetch_class_ancestors(
                ontology, iri,
            ):
                _emit(a_curie, a_label, a_iri)
        return out

    def _fetch_class_ancestors(
        self, ontology: str, iri: str,
    ) -> list[tuple[str, str, str]]:
        """Return ancestors (full chain to root) as [(curie, label, iri), ...].

        Uses OLS4 v2's `/ancestors` endpoint. Cap at `_ANCESTOR_CAP`.
        Empty list on any failure. The ENDPOINT IS `ancestors`, not
        `parents` — `/parents` returns 404 in current OLS4.
        """
        if not iri:
            return []
        from urllib.parse import quote
        encoded = quote(quote(iri, safe=""), safe="")
        url = (
            f"{self._OLS_V2_BASE}/ontologies/{ontology}"
            f"/classes/{encoded}/ancestors"
        )
        try:
            resp = self._rate_limited_get(
                url, params={"size": self._ANCESTOR_CAP},
            )
            resp.raise_for_status()
            elements = resp.json().get("elements") or []
        except Exception as e:
            logger.debug("ancestors lookup failed for %s: %s", iri, e)
            return []
        out: list[tuple[str, str, str]] = []
        for el in elements:
            curie = el.get("curie") or el.get("shortForm") or ""
            label = el.get("label", "")
            if isinstance(label, list):
                label = label[0] if label else ""
            a_iri = el.get("iri") or ""
            if curie and label:
                out.append((curie, label, a_iri))
        return out

    def _fetch_class_xrefs(self, ontology: str, iri: str) -> dict[str, str]:
        """Pull dbXRefs from an OLS4 v2 class entry. Returns
        {prefix: id} mapping the cross-reference list, e.g.
        {"MESH": "D008288", "ICD10CM": "B53", "SCTID": "61462000"}.

        Empty dict on failure or when the class has no dbXRefs."""
        from urllib.parse import quote
        encoded = quote(quote(iri, safe=""), safe="")
        url = f"{self._OLS_V2_BASE}/ontologies/{ontology}/classes/{encoded}"
        try:
            resp = self._rate_limited_get(url)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug("xref lookup failed for %s: %s", iri, e)
            return {}
        # OLS4 v2 stores dbXRefs under the OBO annotation property URI
        raw = data.get(
            "http://www.geneontology.org/formats/oboInOwl#hasDbXref"
        ) or data.get("dbXRefs") or []
        if not isinstance(raw, list):
            raw = [raw]
        out: dict[str, str] = {}
        for x in raw:
            if isinstance(x, dict):
                x = x.get("value", "")
            if not isinstance(x, str) or ":" not in x:
                continue
            prefix, _, value = x.partition(":")
            # First occurrence wins (don't overwrite when a class has
            # multiple cross-refs to the same source)
            if prefix and value and prefix not in out:
                out[prefix] = value
        return out

    # ------------------------------------------------------------------
    # SNOMED CT  (EBI OLS4)
    # ------------------------------------------------------------------

    def search_snomed(self, term: str, rows: int = 10) -> list[dict]:
        """Search EBI OLS4 for SNOMED CT concepts.

        Returns list of dicts: {name, snomed_id, source, match_type, confidence}
        """
        try:
            resp = self._rate_limited_get(
                OLS4_API_BASE,
                params={"q": term, "ontology": "snomed", "rows": rows},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("SNOMED/OLS4 API error for %r: %s", term, e)
            return []

        candidates: list[dict] = []
        docs = data.get("response", {}).get("docs", [])
        for doc in docs:
            label = doc.get("label", "")
            short_form = doc.get("short_form", "")
            # OLS4 short_form may look like "SNOMED_12345" or "SCT_12345"
            snomed_id = short_form
            for prefix in ("SNOMED_", "SCT_", "SCTID_"):
                if short_form.startswith(prefix):
                    snomed_id = short_form[len(prefix) :]
                    break
            is_exact = label.lower().strip() == term.lower().strip()
            candidates.append(
                {
                    "name": label,
                    "icd10_code": "",
                    "mesh_id": "",
                    "snomed_id": snomed_id,
                    "source": "snomed",
                    "match_type": "exact" if is_exact else "fuzzy",
                    "confidence": 0.95 if is_exact else 0.65,
                }
            )
        return candidates
