"""RDKit wrappers for canonical SMILES and InChIKey.

Used by the Source→Compound linker to give every Chemical_Compound node a
structural identity (InChIKey) so that the same molecule reached via
different name strings or different reference DBs (COCONUT, PubChem)
converges on a single graph node.
"""

from __future__ import annotations

from rdkit import Chem
from rdkit.RDLogger import DisableLog

# RDKit is chatty on stderr for unparseable SMILES. Silence it; callers
# already handle None returns explicitly.
DisableLog("rdApp.*")


def to_inchikey(smiles: str | None) -> str | None:
    """Compute the InChIKey for a SMILES string. Returns None on failure."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def to_canonical_smiles(smiles: str | None) -> str | None:
    """Re-canonicalize a SMILES string via RDKit. Returns None on failure."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def unstructured_key(name: str) -> str:
    """Build a sentinel inchikey for compounds without a parseable structure.

    Uses the literal name lowercased and whitespace-collapsed. Two different
    "unstructured" compounds with different names get different keys; two
    extractions of the same unstructured name converge.
    """
    slug = "_".join((name or "").strip().lower().split())
    return f"unstructured:{slug}"
