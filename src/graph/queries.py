"""Reusable Cypher query templates for the two inference tasks.

These are the core metapath queries used by Phase 2 agents.
Import and call via GraphClient.run(QUERY, params).
"""

# ===========================================================================
# Task A: Validate Source → Traditional_Malady claims
# ===========================================================================

TASK_A_VALIDATE_CLAIM = """
// Task A: Score mechanistic support for Source → Traditional_Malady pairs.
// Returns one row per (source, malady, disease) triple with converging
// evidence paths and an aggregate confidence score.
// Filters out archived nodes from Reviewer Agent.
MATCH (s:Source)-[r_tt:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)
MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)-[r_rel:RELATES_TO]->(d:Modern_Disease)
WHERE (d.archived IS NULL OR d.archived = false)
MATCH (m)-[r_map:MAPS_TO]->(d)
WHERE r_map.is_primary = true OR r_map.is_primary IS NULL
WITH s, m, d,
     c.name AS compound, t.name AS target,
     r_ex.confidence_score * r_tgt.confidence_score *
       r_rel.confidence_score * r_map.confidence_score AS path_score
RETURN s.name AS source,
       m.name AS traditional_malady,
       d.name AS modern_disease,
       COLLECT({compound: compound, target: target, score: path_score}) AS evidence_paths,
       COUNT(*) AS num_converging_paths,
       AVG(path_score) AS avg_path_confidence
ORDER BY num_converging_paths DESC, avg_path_confidence DESC
"""

TASK_A_FOR_SOURCE = """
// Task A variant: validate claims for a specific source.
MATCH (s:Source {name: $source_name})-[r_tt:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)
MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)-[r_rel:RELATES_TO]->(d:Modern_Disease)
WHERE (d.archived IS NULL OR d.archived = false)
MATCH (m)-[r_map:MAPS_TO]->(d)
WHERE r_map.is_primary = true OR r_map.is_primary IS NULL
WITH s, m, d,
     c.name AS compound, t.name AS target,
     r_ex.confidence_score * r_tgt.confidence_score *
       r_rel.confidence_score * r_map.confidence_score AS path_score
RETURN s.name AS source,
       m.name AS traditional_malady,
       d.name AS modern_disease,
       COLLECT({compound: compound, target: target, score: path_score}) AS evidence_paths,
       COUNT(*) AS num_converging_paths,
       AVG(path_score) AS avg_path_confidence
ORDER BY num_converging_paths DESC, avg_path_confidence DESC
"""

# ===========================================================================
# Task B: Novel compound discovery for modern diseases
# ===========================================================================

TASK_B_NOVEL_COMPOUNDS = """
// Task B: Find novel compound candidates for a given modern disease.
// Traverses backwards through historical knowledge, cross-validates via
// Target → Disease, and filters out compounds with known direct links.
// Filters out archived nodes from Reviewer Agent.
MATCH (d:Modern_Disease {name: $disease_name})
WHERE (d.archived IS NULL OR d.archived = false)
MATCH (m:Traditional_Malady)-[r_map:MAPS_TO]->(d)
WHERE (m.archived IS NULL OR m.archived = false)
  AND (r_map.is_primary = true OR r_map.is_primary IS NULL)
MATCH (s:Source)-[r_tt:TREATS_TRADITIONALLY]->(m)
WHERE (s.archived IS NULL OR s.archived = false)
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)
MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)
// Cross-validate: does the target relate back to the query disease?
OPTIONAL MATCH (t)-[r_rel:RELATES_TO]->(d)
// Novelty filter: exclude compounds already known to treat this disease
WHERE NOT EXISTS {
  MATCH (c)-[:KNOWN_TREATS]->(d)
}
WITH c, d, s, m, t,
     r_map.confidence_score * r_tt.confidence_score *
       r_ex.confidence_score * r_tgt.confidence_score AS discovery_path_score,
     CASE WHEN r_rel IS NOT NULL
          THEN r_rel.confidence_score ELSE 0.0 END AS cross_validation_score
RETURN c.name AS compound,
       s.name AS source,
       m.name AS traditional_malady,
       t.name AS target,
       discovery_path_score,
       cross_validation_score,
       discovery_path_score * (1 + cross_validation_score) AS composite_score
ORDER BY composite_score DESC
LIMIT $limit
"""

# ===========================================================================
# Utility queries
# ===========================================================================

UNSUPPORTED_CLAIMS = """
// Find Source → Malady claims with NO mechanistic evidence path.
// Filters out archived nodes from Reviewer Agent.
MATCH (s:Source)-[r1:TREATS_TRADITIONALLY]->(m:Traditional_Malady)
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)
  AND EXISTS {
    MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s)
  }
  AND NOT EXISTS {
    MATCH (c2:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s)
    MATCH (c2)-[:TARGETS]->(:Biological_Target)-[:RELATES_TO]->(:Modern_Disease)
          <-[r_map_us:MAPS_TO]-(m)
    WHERE r_map_us.is_primary = true OR r_map_us.is_primary IS NULL
  }
RETURN s.name AS source,
       m.name AS traditional_malady,
       r1.confidence_score AS claim_confidence,
       "No mechanistic support found" AS flag
ORDER BY r1.confidence_score DESC
"""

GRAPH_STATS = """
// Quick overview of active (non-archived) graph contents.
CALL {
  MATCH (n) WHERE n.archived IS NULL OR n.archived = false
  RETURN count(n) AS total_nodes
}
CALL {
  MATCH ()-[r]->() WHERE r.archived IS NULL OR r.archived = false
  RETURN count(r) AS total_edges
}
CALL {
  MATCH (s:Source) WHERE s.archived IS NULL OR s.archived = false
  RETURN count(s) AS sources
}
CALL {
  MATCH (c:Chemical_Compound) WHERE c.archived IS NULL OR c.archived = false
  RETURN count(c) AS compounds
}
CALL {
  MATCH (t:Biological_Target) WHERE t.archived IS NULL OR t.archived = false
  RETURN count(t) AS targets
}
CALL {
  MATCH (d:Modern_Disease) WHERE d.archived IS NULL OR d.archived = false
  RETURN count(d) AS diseases
}
RETURN total_nodes, total_edges, sources, compounds, targets, diseases
"""

UNLINKED_SOURCES = """
// DEPRECATED — used by the v1 Source→Compound linker. The v2 linker
// (`source_compound_linker_v2`) consumes `LINKABLE_SOURCES` instead so
// it gets the auditor's canonical_name + canonical_type and can route
// each Source to COCONUT vs PubChem appropriately.
MATCH (s:Source)
WHERE (s.archived IS NULL OR s.archived = false)
  AND NOT EXISTS {
    MATCH (:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s)
  }
RETURN s.name AS source, s.aliases AS aliases
ORDER BY s.name
"""

LINKABLE_SOURCES = """
// Sources eligible for the v2 Source→Compound linker. Requires the
// ExtractionAuditor to have set `canonical_source` first (the linker
// refuses to run on un-canonicalized Sources). Resumability is via
// `linker_status`: NULL = never tried; 'error' = retry. Other terminal
// statuses ('linked', 'no_compounds_found', 'skipped_uncanonicalized')
// are skipped on default reruns; the CLI flags `--retry-misses` and
// `--force-relink` widen this filter.
MATCH (s:Source)
WHERE (s.archived IS NULL OR s.archived = false)
  AND s.canonical_source IS NOT NULL
  AND (s.linker_status IS NULL OR s.linker_status = 'error')
RETURN s.name AS name,
       s.aliases AS aliases,
       s.canonical_name AS canonical_name,
       s.canonical_source AS canonical_source,
       s.canonical_type AS canonical_type,
       s.canonical_part AS canonical_part
ORDER BY s.name
"""

LINKABLE_SOURCES_RETRY_MISSES = """
// Same as LINKABLE_SOURCES but ALSO retries `no_compounds_found` rows.
// Used after a PubChem cache update or a COCONUT snapshot refresh.
MATCH (s:Source)
WHERE (s.archived IS NULL OR s.archived = false)
  AND s.canonical_source IS NOT NULL
  AND (
       s.linker_status IS NULL
    OR s.linker_status = 'error'
    OR s.linker_status = 'no_compounds_found'
  )
RETURN s.name AS name,
       s.aliases AS aliases,
       s.canonical_name AS canonical_name,
       s.canonical_source AS canonical_source,
       s.canonical_type AS canonical_type,
       s.canonical_part AS canonical_part
ORDER BY s.name
"""

LINKABLE_SOURCES_FORCE = """
// Same as LINKABLE_SOURCES but ignores `linker_status` entirely. Use to
// re-link every Source after a confidence-model or routing-rule change.
MATCH (s:Source)
WHERE (s.archived IS NULL OR s.archived = false)
  AND s.canonical_source IS NOT NULL
RETURN s.name AS name,
       s.aliases AS aliases,
       s.canonical_name AS canonical_name,
       s.canonical_source AS canonical_source,
       s.canonical_type AS canonical_type,
       s.canonical_part AS canonical_part
ORDER BY s.name
"""

UNLINKED_COMPOUNDS = """
// DEPRECATED — used by the v1 Compound→Target linker. The v2 linker
// (`compound_target_linker_v2`) consumes `LINKABLE_COMPOUNDS_FOR_TARGETS`
// instead so it gets the InChIKey + SMILES + name for ChEMBL routing
// and respects the `target_linker_status` resume contract.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND NOT EXISTS {
    MATCH (c)-[:TARGETS]->(:Biological_Target)
  }
RETURN c.name AS compound, c.smiles AS smiles
ORDER BY c.name
"""

# ---------------------------------------------------------------------------
# Compound → Target Linker (ChEMBL routing)
# ---------------------------------------------------------------------------
# Resumability is via `target_linker_status` set by the agent:
#   NULL                = never linked
#   "linked"            = at least one TARGETS edge written
#   "no_targets_found"  = ChEMBL returned no qualifying targets
#                         (compound not in ChEMBL, or all activities below
#                          the pchembl floor)
#   "error"             = ChEMBL call raised
# Three filter variants drive default vs --retry-misses vs --force-relink.

LINKABLE_COMPOUNDS_FOR_TARGETS = """
// Default: compounds the linker hasn't touched yet.
// NULL inchikey is excluded — the linker MERGEs by inchikey, so a NULL
// would crash the per-compound write. Legacy compounds without an
// InChIKey should be repaired by SourceCompoundLinker first.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
  AND c.target_linker_status IS NULL
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.smiles AS smiles,
       c.source_db AS source_db
ORDER BY c.inchikey
"""

LINKABLE_COMPOUNDS_RETRY = """
// --retry-misses: also re-process compounds the linker previously
// errored on or found no targets for. Useful after a ChEMBL snapshot
// refresh, a pchembl-floor change, or a transient API outage.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
  AND (
       c.target_linker_status IS NULL
    OR c.target_linker_status = 'error'
    OR c.target_linker_status = 'no_targets_found'
  )
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.smiles AS smiles,
       c.source_db AS source_db
ORDER BY c.inchikey
"""

LINKABLE_COMPOUNDS_FORCE = """
// --force-relink: ignore target_linker_status entirely. Used after a
// confidence-prior or routing-rule change. Combine with --rebuild for a
// clean slate.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.smiles AS smiles,
       c.source_db AS source_db
ORDER BY c.inchikey
"""

# ---------------------------------------------------------------------------
# Compound → Disease KNOWN_TREATS Linker (ChEMBL drug_indication routing)
# ---------------------------------------------------------------------------
# Resumability is via `kt_linker_status` set by the agent:
#   NULL                 = never linked
#   "linked"             = at least one KNOWN_TREATS edge written
#   "no_indications"     = ChEMBL has no qualifying indications for this
#                          compound (or indications exist but none matched
#                          an in-graph Modern_Disease — match-only mode)
#   "error"              = ChEMBL call raised; retry via --retry-misses
#
# Filter only compounds with linker_chembl_id set (set by CompoundTargetLinker).
# Without it we can't query drug_indication, so processing them would just
# burn API calls.

LINKABLE_COMPOUNDS_FOR_INDICATIONS = """
// Default: compounds the indication linker hasn't touched yet.
// linker_chembl_id is required — we look up indications by molecule
// ChEMBL ID, so compounds that never resolved in ChEMBL can't have any.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
  AND c.linker_chembl_id IS NOT NULL
  AND c.linker_chembl_id <> ''
  AND c.kt_linker_status IS NULL
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.linker_chembl_id AS chembl_id
ORDER BY c.inchikey
"""

LINKABLE_COMPOUNDS_FOR_INDICATIONS_RETRY = """
// --retry-misses: also re-process compounds the indication linker
// previously errored on or found no in-graph matches for. Useful after
// the Modern_Disease universe expands (more MaladyDiseaseMapper runs)
// or after a ChEMBL drug_indication snapshot refresh.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
  AND c.linker_chembl_id IS NOT NULL
  AND c.linker_chembl_id <> ''
  AND (
       c.kt_linker_status IS NULL
    OR c.kt_linker_status = 'error'
    OR c.kt_linker_status = 'no_indications'
  )
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.linker_chembl_id AS chembl_id
ORDER BY c.inchikey
"""

LINKABLE_COMPOUNDS_FOR_INDICATIONS_FORCE = """
// --force-relink: ignore kt_linker_status entirely. Used after a
// confidence-prior or matching-rule change. Combine with --rebuild
// for a clean slate.
MATCH (c:Chemical_Compound)
WHERE (c.archived IS NULL OR c.archived = false)
  AND c.inchikey IS NOT NULL
  AND c.linker_chembl_id IS NOT NULL
  AND c.linker_chembl_id <> ''
RETURN c.inchikey AS inchikey,
       c.name AS name,
       c.linker_chembl_id AS chembl_id
ORDER BY c.inchikey
"""

ALL_MODERN_DISEASES = """
// Snapshot of the existing Modern_Disease universe. Loaded once at the
// start of a per-target/per-compound linker run to build the in-memory
// match index. Match-only mode means we never create new Modern_Disease
// nodes from external sources — if it's not in this list, the candidate
// is dropped.
//
// Returns all available cross-ontology IDs so each linker can do tier-1
// matching against whichever ontology its data source uses (EFO from
// Open Targets, MeSH from ChEMBL, MONDO from EFO xrefs, DOID from
// Disease Ontology, etc.). Properties absent on a node return NULL.
MATCH (d:Modern_Disease)
WHERE d.archived IS NULL OR d.archived = false
RETURN d.name AS name,
       d.mesh_id AS mesh_id,
       d.icd10_code AS icd10_code,
       d.snomed_id AS snomed_id,
       d.efo_id AS efo_id,
       d.mondo_id AS mondo_id,
       d.doid_id AS doid_id
"""

# ---------------------------------------------------------------------------
# Target → Disease RELATES_TO Linker (batched — Open Targets + EFO/DOID/LLM)
# ---------------------------------------------------------------------------
# Resumability is via `td_linker_status` set by the agent:
#   NULL                       = never linked
#   "linked"                   = at least one RELATES_TO edge written
#   "no_associations"          = OT has no qualifying associations or
#                                EFO/DOID/LLM didn't map the pathogen
#                                (or matched diseases aren't in the graph)
#   "skipped_no_uniprot"       = SINGLE PROTEIN target with no uniprot_id
#                                (data quality issue)
#   "skipped_no_subunits"      = PROTEIN COMPLEX with no fetchable subunit
#                                UniProts from ChEMBL
#   "skipped_protein_family"   = PROTEIN FAMILY (intentionally not linked)
#   "skipped_no_tax_id"        = ORGANISM with no ncbi_tax_id
#   "skipped_unknown_type"     = unrecognized target_type
#   "error"                    = OT/OLS/LLM call raised; retry candidate

LINKABLE_TARGETS_FOR_DISEASES = """
// Default: targets the disease linker hasn't touched yet.
// Returns target_type so the agent can dispatch to the right code path.
MATCH (t:Biological_Target)
WHERE (t.archived IS NULL OR t.archived = false)
  AND t.target_chembl_id IS NOT NULL
  AND t.td_linker_status IS NULL
RETURN t.target_chembl_id AS target_chembl_id,
       t.name AS name,
       t.target_type AS target_type,
       t.uniprot_id AS uniprot_id,
       t.gene_symbol AS gene_symbol,
       t.ncbi_tax_id AS ncbi_tax_id,
       t.target_pref_name AS target_pref_name
ORDER BY t.target_chembl_id
"""

LINKABLE_TARGETS_FOR_DISEASES_RETRY = """
// --retry-misses: also re-process targets the linker errored on or
// found no associations for. Useful after the Modern_Disease universe
// expands, after an OT snapshot refresh, or after curating new
// pathogen mappings.
MATCH (t:Biological_Target)
WHERE (t.archived IS NULL OR t.archived = false)
  AND t.target_chembl_id IS NOT NULL
  AND (
       t.td_linker_status IS NULL
    OR t.td_linker_status = 'error'
    OR t.td_linker_status = 'no_associations'
  )
RETURN t.target_chembl_id AS target_chembl_id,
       t.name AS name,
       t.target_type AS target_type,
       t.uniprot_id AS uniprot_id,
       t.gene_symbol AS gene_symbol,
       t.ncbi_tax_id AS ncbi_tax_id,
       t.target_pref_name AS target_pref_name
ORDER BY t.target_chembl_id
"""

LINKABLE_TARGETS_FOR_DISEASES_FORCE = """
// --force-relink: ignore td_linker_status entirely. Used after a
// confidence-prior or matching-rule change. Combine with --rebuild
// for a clean slate.
MATCH (t:Biological_Target)
WHERE (t.archived IS NULL OR t.archived = false)
  AND t.target_chembl_id IS NOT NULL
RETURN t.target_chembl_id AS target_chembl_id,
       t.name AS name,
       t.target_type AS target_type,
       t.uniprot_id AS uniprot_id,
       t.gene_symbol AS gene_symbol,
       t.ncbi_tax_id AS ncbi_tax_id,
       t.target_pref_name AS target_pref_name
ORDER BY t.target_chembl_id
"""

UNMAPPED_MALADIES = """
// Find active Traditional_Malady nodes without Modern_Disease mappings.
// Used by the Malady→Disease Mapper.
MATCH (m:Traditional_Malady)
WHERE (m.archived IS NULL OR m.archived = false)
  AND NOT EXISTS {
    MATCH (m)-[:MAPS_TO]->(:Modern_Disease)
  }
RETURN m.name AS malady, m.description AS description
ORDER BY m.name
"""

# ---------------------------------------------------------------------------
# Malady → Disease Mapper (generate-then-verify)
# ---------------------------------------------------------------------------
# Resumability is via `mapper_status` set by the agent:
#   NULL                 = never mapped
#   "linked"             = at least one MAPS_TO edge written
#   "tcm_no_equivalent"  = explicit refusal (no edge written)
#   "error"              = Gemini call failed; retry candidate
# Three filter variants drive default vs --retry-misses vs --force-remap.

MAPPABLE_MALADIES = """
// Default: maladies the mapper hasn't touched yet.
MATCH (m:Traditional_Malady)
WHERE (m.archived IS NULL OR m.archived = false)
  AND m.mapper_status IS NULL
RETURN m.name AS name,
       m.description AS description,
       m.evidence_span AS evidence_span,
       m.aliases AS aliases,
       m.source_document AS source_document
ORDER BY m.name
"""

MAPPABLE_MALADIES_RETRY = """
// --retry-misses: also re-process maladies the mapper previously skipped,
// errored on, or couldn't verify. Useful after a prompt, model, or
// ontology snapshot change. Excludes 'linked' so successful mappings
// don't get re-touched.
MATCH (m:Traditional_Malady)
WHERE (m.archived IS NULL OR m.archived = false)
  AND (
       m.mapper_status IS NULL
    OR m.mapper_status = 'error'
    OR m.mapper_status = 'tcm_no_equivalent'
    OR m.mapper_status = 'unverified'
  )
RETURN m.name AS name,
       m.description AS description,
       m.evidence_span AS evidence_span,
       m.aliases AS aliases,
       m.source_document AS source_document
ORDER BY m.name
"""

MAPPABLE_MALADIES_FORCE = """
// --force-remap: ignore mapper_status entirely. Used after a confidence
// model or routing-rule change. Combine with --rebuild for a clean slate.
MATCH (m:Traditional_Malady)
WHERE (m.archived IS NULL OR m.archived = false)
RETURN m.name AS name,
       m.description AS description,
       m.evidence_span AS evidence_span,
       m.aliases AS aliases,
       m.source_document AS source_document
ORDER BY m.name
"""

# ===========================================================================
# Reviewer Agent queries
# ===========================================================================

ALL_SOURCES_WITH_DEGREE = """
// Return all Source nodes with their edge count (degree).
// Used by the Reviewer Agent to find orphaned/low-degree nodes.
MATCH (s:Source)
WHERE s.archived IS NULL OR s.archived = false
OPTIONAL MATCH (s)-[r]-()
WITH s, count(r) AS degree
RETURN s.name AS name, s.aliases AS aliases, degree
ORDER BY degree ASC, s.name
"""

ALL_MALADIES_WITH_MAPPING = """
// Return all Traditional_Malady nodes with their PRIMARY mapped disease
// (if any). Syndrome components are visible via dedicated critic queries.
// Used by the Reviewer Agent to check mapping quality.
MATCH (m:Traditional_Malady)
WHERE m.archived IS NULL OR m.archived = false
OPTIONAL MATCH (m)-[r:MAPS_TO]->(d:Modern_Disease)
  WHERE r.is_primary = true OR r.is_primary IS NULL
RETURN m.name AS malady, m.description AS description,
       d.name AS disease, r.confidence_score AS conf, r.mapping_source AS method
ORDER BY m.name
"""

OVERMAPPED_DISEASES = """
// Find Modern_Disease nodes with too many incoming MAPS_TO edges.
// Suggests over-generalization in the mapping. Counts only primary edges
// so a syndrome that legitimately fans out to 3 components doesn't
// inflate every component's incoming count.
MATCH (m:Traditional_Malady)-[r:MAPS_TO]->(d:Modern_Disease)
WHERE (d.archived IS NULL OR d.archived = false)
  AND (r.is_primary = true OR r.is_primary IS NULL)
WITH d, count(m) AS malady_count,
     collect(m.name) AS malady_names
WHERE malady_count > $threshold
RETURN d.name AS disease, malady_count,
       malady_names[0..10] AS sample_maladies
ORDER BY malady_count DESC
"""

ARCHIVED_STATS = """
// Count archived nodes by label.
MATCH (n)
WHERE n.archived = true
RETURN labels(n)[0] AS label,
       n.archive_reason AS reason,
       count(n) AS cnt
ORDER BY cnt DESC
"""

# ===========================================================================
# Critic Agent queries (Phase 2)
# ===========================================================================

CRITIC_FULL_EVIDENCE_CHAIN = """
// Return ALL evidence chain data for a given (Source, Malady) pair.
// Uses OPTIONAL MATCH so partial paths (no targets) are still returned.
// NOTE: RELATES_TO uses 'confidence' not 'confidence_score'.
MATCH (s:Source {name: $source_name})-[r_tt:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady_name})
WHERE (s.archived IS NULL OR s.archived = false)
  AND (m.archived IS NULL OR m.archived = false)

// Malady → Disease mapping (primary only — components seen via critic queries)
MATCH (m)-[r_map:MAPS_TO]->(d:Modern_Disease)
WHERE (d.archived IS NULL OR d.archived = false)
  AND (r_map.is_primary = true OR r_map.is_primary IS NULL)

// All compounds from source
MATCH (c:Chemical_Compound)-[r_ex:IS_EXTRACTED_FROM]->(s)

// Targets (optional)
OPTIONAL MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)

// Target → Disease (optional)
OPTIONAL MATCH (t)-[r_rel:RELATES_TO]->(d2:Modern_Disease)

RETURN
  s.name AS source, m.name AS malady,
  r_tt.confidence_score AS tt_conf, r_tt.evidence_span AS tt_evidence,

  d.name AS mapped_disease, r_map.confidence_score AS map_conf,
  r_map.mapping_source AS map_method,

  c.name AS compound, c.smiles AS smiles,
  r_ex.confidence_score AS ex_conf,

  t.name AS target, t.uniprot_id AS uniprot, t.gene_symbol AS gene,
  r_tgt.pchembl_score AS pchembl, r_tgt.evidence_type AS tgt_type,
  r_tgt.confidence_score AS tgt_conf,

  d2.name AS reached_disease,
  r_rel.confidence AS rel_conf, r_rel.ot_overall_score AS ot_score,
  r_rel.ot_resolved_id AS ot_id, r_rel.rationale AS rationale,

  CASE WHEN d2.name = d.name THEN true ELSE false END AS loop_closed
"""

CRITIC_TARGET_GENERICITY = """
// Count distinct diseases each target relates to.
// Used to penalize generic targets (PTPN1 → 313 diseases).
MATCH (t:Biological_Target)-[:RELATES_TO]->(d:Modern_Disease)
RETURN t.name AS target, t.uniprot_id AS uniprot,
       count(DISTINCT d) AS disease_count
ORDER BY disease_count DESC
"""

CRITIC_COMPOUND_UBIQUITY = """
// Count distinct active sources each compound is extracted from.
// Used to penalize ubiquitous compounds (BETA-SITOSTEROL → 55 sources).
MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s:Source)
WHERE s.archived IS NULL OR s.archived = false
RETURN c.name AS compound, count(DISTINCT s) AS source_count
ORDER BY source_count DESC
"""

CRITIC_CROSS_SOURCE_COMPOUND = """
// Pattern A: Same compound in a different source that also treats the same malady.
MATCH (s1:Source {name: $source_name})-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady_name})
MATCH (c:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s1)
MATCH (c)-[:IS_EXTRACTED_FROM]->(s2:Source)
MATCH (s2)-[:TREATS_TRADITIONALLY]->(m)
WHERE s2.name <> $source_name
  AND (s2.archived IS NULL OR s2.archived = false)
RETURN c.name AS shared_compound, s2.name AS other_source
"""

CRITIC_TARGET_CONVERGENCE = """
// Pattern B: Different compounds from different sources converge on same target
// for the same malady.
MATCH (s1:Source {name: $source_name})-[:TREATS_TRADITIONALLY]->(m:Traditional_Malady {name: $malady_name})
MATCH (c1:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s1)
MATCH (c1)-[:TARGETS]->(t:Biological_Target)
MATCH (c2:Chemical_Compound)-[:IS_EXTRACTED_FROM]->(s2:Source)
MATCH (s2)-[:TREATS_TRADITIONALLY]->(m)
MATCH (c2)-[:TARGETS]->(t)
WHERE s2.name <> $source_name
  AND c2.name <> c1.name
  AND (s2.archived IS NULL OR s2.archived = false)
RETURN t.name AS converged_target, t.uniprot_id AS uniprot,
       c1.name AS compound_from_source, c2.name AS compound_from_other,
       s2.name AS other_source
"""

CRITIC_COMPOUND_PROFILE = """
// Full neighborhood of a compound for nomination report.
MATCH (c:Chemical_Compound {name: $compound_name})
OPTIONAL MATCH (c)-[r_ex:IS_EXTRACTED_FROM]->(s:Source)
  WHERE s.archived IS NULL OR s.archived = false
OPTIONAL MATCH (c)-[r_tgt:TARGETS]->(t:Biological_Target)
OPTIONAL MATCH (t)-[r_rel:RELATES_TO]->(d:Modern_Disease)
OPTIONAL MATCH (c)-[r_kt:KNOWN_TREATS]->(d2:Modern_Disease)
RETURN c.name AS compound, c.smiles AS smiles,
       collect(DISTINCT s.name) AS sources,
       collect(DISTINCT {
           target: t.name, uniprot: t.uniprot_id,
           pchembl: r_tgt.pchembl_score, evidence_type: r_tgt.evidence_type
       }) AS targets,
       collect(DISTINCT {
           disease: d.name, rel_conf: r_rel.confidence,
           ot_score: r_rel.ot_overall_score, ot_id: r_rel.ot_resolved_id
       }) AS diseases_via_targets,
       collect(DISTINCT {
           disease: d2.name, clinical_phase: r_kt.clinical_phase
       }) AS known_treats
"""
