"""RDKit molecular graph features for ChemFluor graph neural models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chemfluor.data_standardization import canonicalize_smiles

try:
    from rdkit import Chem, RDLogger
except ImportError as exc:  # pragma: no cover - exercised only without RDKit.
    Chem = None
    RDLogger = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")


ATOM_FEATURE_DIM = 45
BOND_FEATURE_DIM = 12


@dataclass(frozen=True)
class GraphData:
    """Numpy graph tensors for one molecule."""

    x: np.ndarray
    edge_index: np.ndarray
    edge_attr: np.ndarray
    num_nodes: int
    canonical_smiles: str


def require_rdkit() -> None:
    """Raise a helpful error if RDKit is unavailable."""
    if Chem is None:
        raise ImportError("RDKit is required for graph feature generation.") from _RDKIT_IMPORT_ERROR


def one_hot_with_unknown(value: object, choices: list[object]) -> list[float]:
    """One-hot encode a value with the final bucket reserved for unknown."""
    values = [0.0] * (len(choices) + 1)
    try:
        index = choices.index(value)
    except ValueError:
        index = len(choices)
    values[index] = 1.0
    return values


def atom_features(atom: object) -> list[float]:
    """Return stable atom features for graph models."""
    atomic_number_choices = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
    degree_choices = [0, 1, 2, 3, 4, 5]
    formal_charge_choices = [-2, -1, 0, 1, 2]
    chiral_choices = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]
    hydrogen_choices = [0, 1, 2, 3, 4]
    hybridization_choices = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
    ]
    features = [
        *one_hot_with_unknown(atom.GetAtomicNum(), atomic_number_choices),
        *one_hot_with_unknown(atom.GetTotalDegree(), degree_choices),
        *one_hot_with_unknown(atom.GetFormalCharge(), formal_charge_choices),
        *one_hot_with_unknown(atom.GetChiralTag(), chiral_choices),
        *one_hot_with_unknown(atom.GetTotalNumHs(), hydrogen_choices),
        *one_hot_with_unknown(atom.GetHybridization(), hybridization_choices),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        float(atom.GetMass() * 0.01),
    ]
    if len(features) != ATOM_FEATURE_DIM:
        raise RuntimeError(f"Atom feature dimension changed unexpectedly: {len(features)}")
    return features


def bond_features(bond: object) -> list[float]:
    """Return stable bond features for graph models."""
    bond_type_choices = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ]
    stereo_choices = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    features = [
        *one_hot_with_unknown(bond.GetBondType(), bond_type_choices),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *one_hot_with_unknown(bond.GetStereo(), stereo_choices),
    ]
    if len(features) != BOND_FEATURE_DIM:
        raise RuntimeError(f"Bond feature dimension changed unexpectedly: {len(features)}")
    return features


def mol_to_graph(smiles: str) -> GraphData | None:
    """Convert a SMILES string into atom and bond graph tensors."""
    require_rdkit()
    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return None
    mol = Chem.MolFromSmiles(canonical)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    node_features = np.asarray(
        [atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32
    )
    edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        features = bond_features(bond)
        edges.extend([(begin, end), (end, begin)])
        edge_features.extend([features, features])

    if edges:
        edge_index = np.asarray(edges, dtype=np.int64).T
        edge_attr = np.asarray(edge_features, dtype=np.float32)
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
        edge_attr = np.zeros((0, BOND_FEATURE_DIM), dtype=np.float32)

    return GraphData(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=int(node_features.shape[0]),
        canonical_smiles=canonical,
    )
