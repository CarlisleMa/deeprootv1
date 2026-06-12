"""Pydantic models for Neo4j nodes and edges.

All agents use these models as the shared contract for what goes into the graph.
When creating a node or edge, instantiate the model first — it validates fields
and provides the dict for the Cypher MERGE query.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

class Source(BaseModel):
    """A remedy organism or substance extracted from a historical text."""
    name: str = Field(..., description="Canonical organism/substance name")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    source_document: str = Field(..., description="Title of the historical text")
    evidence_span: str = Field("", description="Exact text span where this entity appears")
    created_by: str = "extraction_agent"

    neo4j_label: str = "Source"


class TraditionalMalady(BaseModel):
    """A historical symptom or disease description."""
    name: str = Field(..., description="Malady name as stated in the text")
    description: str = Field("", description="Longer description if available")
    source_document: str = Field("", description="Which text this comes from")
    evidence_span: str = Field("", description="Exact text span")
    created_by: str = "extraction_agent"

    neo4j_label: str = "Traditional_Malady"


class ModernDisease(BaseModel):
    """A standardized modern disease/phenotype."""
    name: str = Field(..., description="Standardized disease name")
    icd10_code: str = Field("", description="ICD-10 code if available")
    mesh_id: str = Field("", description="MeSH descriptor ID if available")
    snomed_id: str = Field("", description="SNOMED CT concept ID if available")
    created_by: str = "malady_disease_mapper"

    neo4j_label: str = "Modern_Disease"


class ChemicalCompound(BaseModel):
    """A natural product or active ingredient."""
    name: str = Field(..., description="Compound name")
    smiles: str = Field("", description="SMILES string")
    molecular_formula: str = Field("", description="Molecular formula")
    coconut_id: str = Field("", description="COCONUT database ID")
    duke_id: str = Field("", description="Dr. Duke's database ID")
    source_db: str = Field("", description="Which database this came from")
    created_by: str = "source_compound_linker"

    neo4j_label: str = "Chemical_Compound"


class BiologicalTarget(BaseModel):
    """A molecular target or mechanism."""
    name: str = Field(..., description="Target name (e.g., protein, receptor)")
    uniprot_id: str = Field("", description="UniProt accession")
    gene_symbol: str = Field("", description="HGNC gene symbol")
    target_type: str = Field("", description="e.g., protein, enzyme, receptor")
    created_by: str = "compound_target_linker"

    neo4j_label: str = "Biological_Target"


class PreparationMethod(BaseModel):
    """How a remedy is prepared/administered."""
    name: str = Field(..., description="Preparation method (e.g., decoction, powder)")
    route: str = Field("", description="Administration route (oral, topical, etc.)")
    evidence_span: str = Field("", description="Exact text span")
    created_by: str = "extraction_agent"

    neo4j_label: str = "Preparation_Method"


# ---------------------------------------------------------------------------
# Edge models
# ---------------------------------------------------------------------------

class Edge(BaseModel):
    """Base edge model. All edges carry a confidence score."""
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    evidence_span: str = Field("", description="Text evidence for this edge")
    created_by: str = Field(...)
    created_at: Optional[datetime] = None
    reviewed: bool = False


class TreatsTraditionally(Edge):
    """Source → Traditional_Malady"""
    rel_type: str = "TREATS_TRADITIONALLY"
    created_by: str = "extraction_agent"


class MapsTo(Edge):
    """Traditional_Malady → Modern_Disease"""
    rel_type: str = "MAPS_TO"
    mapping_method: str = Field("", description="e.g., snomed_exact, icd10_fuzzy, llm")
    created_by: str = "malady_disease_mapper"


class IsExtractedFrom(Edge):
    """Chemical_Compound → Source"""
    rel_type: str = "IS_EXTRACTED_FROM"
    source_db: str = Field("", description="COCONUT, Duke, etc.")
    created_by: str = "source_compound_linker"


class Targets(Edge):
    """Chemical_Compound → Biological_Target"""
    rel_type: str = "TARGETS"
    
    # We keep these for ChEMBL-specific metadata
    evidence_type: str = Field("", description="mechanism or activity")
    assay_id: str = Field("", description="ChEMBL target or assay ID")
    pchembl_score: Optional[float] = Field(None, description="The raw strength (e.g. 7.4)")
    created_by: str = "compound_target_linker"


class RelatesTo(Edge):
    """Biological_Target → Modern_Disease

    Written by TargetDiseaseLinker. Sourced from Open Targets for protein
    targets (SINGLE PROTEIN, PROTEIN COMPLEX) and from EFO/DOID/LLM
    for ORGANISM targets (mapping pathogen NCBI tax_id → infectious
    disease MeSH ID).

    Phase 2's CRITIC_FULL_EVIDENCE_CHAIN uses `confidence` (NOT
    `confidence_score` like other edge types). Keep that name stable.
    """
    rel_type: str = "RELATES_TO"
    confidence: float = Field(0.0, description="Flat per-evidence-type prior (0..1)")
    evidence_type: str = Field(
        "",
        description=(
            "Tier label: ot_association_strong/moderate/weak (SINGLE PROTEIN), "
            "ot_association_complex_aggregate (PROTEIN COMPLEX), "
            "ncbi_pathogen_efo / ncbi_pathogen_doid / ncbi_pathogen_consensus / "
            "ncbi_pathogen_llm_verified (ORGANISM)"
        ),
    )
    source_db: str = Field("", description="OpenTargets, EFO, DOID, etc.")
    ot_overall_score: Optional[float] = Field(None, description="Raw OT score 0..1")
    ot_resolved_id: str = Field("", description="EFO/MONDO ID OT used")
    ot_target_ensembl_id: str = Field("", description="ENSEMBL ID OT used for the target")
    ot_top_subunit_uniprot: str = Field(
        "",
        description="For PROTEIN COMPLEX: which subunit's score won the per-disease max",
    )
    ot_datasource_scores: str = Field(
        "",
        description="JSON dump of OT's datasource breakdown (audit)",
    )
    pathogen_lookup_source: str = Field(
        "",
        description="ORGANISM only: which sources confirmed the pathogen→disease "
                    "mapping (efo|doid|efo+doid|llm_verified)",
    )
    rationale: str = Field("", description="Deterministic template; LLM rationale for ORGANISM")
    requires_review: bool = Field(False, description="True for unverified LLM mappings")
    created_by: str = "target_disease_linker"


class PreparedAs(Edge):
    """Source → Preparation_Method"""
    rel_type: str = "PREPARED_AS"
    created_by: str = "extraction_agent"


#No class for the known treats edge was initially created
#This edge is a negative filter
class KnownTreats(Edge):
    """
    Chemical_Compound → Modern_Disease
    Represents established clinical knowledge from CHEMBL.
    """
    rel_type: str = "KNOWN_TREATS"
    clinical_phase: str = Field("", description="e.g., Phase 3, Approved")
    source_db: str = Field("", description="e.g. CHEMBL")
    created_by: str = "compound_target_linker"   
