"""Neo4j schema initialization.

Run once to set up constraints, indexes, and the ontology structure.
Safe to run multiple times (uses IF NOT EXISTS).

Usage:
    from src.graph.client import GraphClient
    from src.graph.schema import init_schema

    with GraphClient() as client:
        init_schema(client)
"""

from __future__ import annotations

from src.graph.client import GraphClient

# -- Node labels ---------------------------------------------------------------

NODE_LABELS = [
    "Source",
    "Traditional_Malady",
    "Modern_Disease",
    "Chemical_Compound",
    "Biological_Target",
    "Preparation_Method",
]

# -- Relationship types --------------------------------------------------------

RELATIONSHIP_TYPES = [
    ("TREATS_TRADITIONALLY", "Source", "Traditional_Malady"),
    ("MAPS_TO", "Traditional_Malady", "Modern_Disease"),
    ("IS_EXTRACTED_FROM", "Chemical_Compound", "Source"),
    ("TARGETS", "Chemical_Compound", "Biological_Target"),
    ("RELATES_TO", "Biological_Target", "Modern_Disease"),
    ("PREPARED_AS", "Source", "Preparation_Method"),
    # Used by the novelty filter in Task B — marks known drug-disease pairs
    ("KNOWN_TREATS", "Chemical_Compound", "Modern_Disease"),
]

# -- Uniqueness constraints ----------------------------------------------------

CONSTRAINTS = [
    ("source_name", "Source", "name"),
    ("trad_malady_name", "Traditional_Malady", "name"),
    ("modern_disease_name", "Modern_Disease", "name"),
    # Chemical_Compound identity is structural, not nominal. The MERGE key is
    # `inchikey` (computed via RDKit from SMILES, or an "unstructured:" sentinel
    # for compounds without parseable structure). `name` is kept as a label
    # property, NOT unique — multiple historical names may legitimately point at
    # the same molecule, and the same name string may be used for different
    # molecules across DBs. `smiles` is also NOT unique (demoted to a regular
    # range index below): InChIKey owns identity, and we want unstructured
    # compounds to be able to coexist without false-collide on missing/empty
    # SMILES.
    ("compound_inchikey", "Chemical_Compound", "inchikey"),
    # Biological_Target identity is the ChEMBL target ID. This is the
    # unified key across all target_type values we accept:
    #   SINGLE PROTEIN   → also has uniprot_id property (single accession)
    #   PROTEIN COMPLEX  → multiple subunits; ChEMBL ID is unambiguous
    #   PROTEIN FAMILY   → e.g. "Tyrosine-protein kinases"
    #   ORGANISM         → e.g. Plasmodium falciparum (also has ncbi_tax_id)
    # UniProt was tried as the identity in an earlier iteration but only
    # applies cleanly to SINGLE PROTEIN; PROTEIN COMPLEX has multiple
    # equally-valid UniProt accessions, and ORGANISM has none.
    ("target_chembl_id", "Biological_Target", "target_chembl_id"),
    ("prep_method_name", "Preparation_Method", "name"),
]

# Constraints that previously existed and must be DROPPED before init_schema()
# can succeed on graphs that were created with the v1 schema.
DEPRECATED_CONSTRAINTS = [
    "compound_name",     # superseded by compound_inchikey
    "compound_smiles",   # demoted to a regular index (see RANGE_INDEXES)
    "target_name",       # superseded by target_chembl_id
    "target_uniprot_id", # short-lived; superseded by target_chembl_id
]

# -- Indexes for performance ---------------------------------------------------

# Regular (range) indexes for fast lookup on non-unique properties. Unlike
# uniqueness constraints, these allow duplicate / null values.
RANGE_INDEXES = [
    ("compound_smiles_idx", "Chemical_Compound", "smiles"),
    # Speeds up Phase 2 queries that join targets by display name.
    ("target_name_idx", "Biological_Target", "name"),
    # UniProt is informational on protein targets; range index lets us
    # query "all targets with a UniProt" cheaply (e.g. for filtering
    # ORGANISM/COMPLEX out of a Phase 2 traversal that wants only
    # single proteins).
    ("target_uniprot_idx", "Biological_Target", "uniprot_id"),
]

INDEXES = [
    # Full-text indexes for fuzzy search during entity resolution
    ("source_ft", "Source", ["name", "aliases"]),
    ("compound_ft", "Chemical_Compound", ["name", "smiles"]),
    ("malady_ft", "Traditional_Malady", ["name", "description"]),
]


def init_schema(client: GraphClient) -> None:
    """Create all constraints and indexes. Safe to run repeatedly."""

    # Drop legacy constraints first so the new ones can take their place.
    for cname in DEPRECATED_CONSTRAINTS:
        try:
            client.run_write(f"DROP CONSTRAINT {cname} IF EXISTS")
        except Exception as e:
            print(f"  (info) could not drop legacy constraint {cname}: {e}")

    # Uniqueness constraints (also create implicit B-tree indexes)
    for cname, label, prop in CONSTRAINTS:
        client.run_write(
            f"CREATE CONSTRAINT {cname} IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
        )

    # Range (B-tree) indexes for non-unique searchable properties.
    for iname, label, prop in RANGE_INDEXES:
        client.run_write(
            f"CREATE INDEX {iname} IF NOT EXISTS "
            f"FOR (n:{label}) ON (n.{prop})"
        )

    # Full-text indexes for fuzzy matching
    for iname, label, props in INDEXES:
        prop_list = ", ".join(f"n.{p}" for p in props)
        try:
            client.run_write(
                f'CREATE FULLTEXT INDEX {iname} IF NOT EXISTS '
                f'FOR (n:{label}) ON EACH [{prop_list}]'
            )
        except Exception:
            # Full-text index creation syntax varies across Neo4j versions;
            # not critical — agents can fall back to exact matching.
            pass

    print(f"Schema initialized: {len(CONSTRAINTS)} constraints, "
          f"{len(INDEXES)} full-text indexes")


def print_schema_summary(client: GraphClient) -> None:
    """Print current graph statistics."""
    print("=== Graph Schema Summary ===")
    for label in NODE_LABELS:
        count = client.count_nodes(label)
        print(f"  {label:25s} {count:>6d} nodes")
    print("  ---")
    for rel_type, from_l, to_l in RELATIONSHIP_TYPES:
        count = client.count_edges(rel_type)
        print(f"  {from_l} -[{rel_type}]-> {to_l}: {count} edges")
