"""Named conformer geometry sets and pooling components for later ablations."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdMolAlign

from .geometry_cache import GEOMETRY_SCHEMA_VERSION, payload_checksum, validate_geometry_entry

RDLogger.DisableLog("rdApp.*")

CONFORMER_SET_SCHEMA_VERSION = "uniprop_named_conformer_set_v1"
CONFORMER_GEOMETRY_VARIANTS = (
    "rdkit_mmff_single",
    "xtb_single",
    "rdkit_multi_conformer",
    "rdkit_multi_equal_pooling",
    "rdkit_multi_energy_weighted_pooling",
    "rdkit_multi_solvent_conditioned_pooling",
)


@dataclass(frozen=True)
class XtbEnvironment:
    available: bool
    executable: str | None
    version: str | None
    detail: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def conformer_set_cache_path(cache_dir: Path, molecule_id: str, geometry_set_name: str) -> Path:
    """Return a cache path that allows multiple named geometry sets per molecule."""
    return cache_dir / molecule_id / f"{geometry_set_name}.json"


def detect_xtb_environment(executable: str = "xtb") -> XtbEnvironment:
    """Detect optional xTB without making it a required dependency."""
    resolved = shutil.which(executable)
    if resolved is None:
        return XtbEnvironment(False, None, None, "xTB executable not found on PATH.")
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return XtbEnvironment(False, resolved, None, str(exc))
    version = (completed.stdout or completed.stderr).strip().splitlines()
    return XtbEnvironment(completed.returncode == 0, resolved, version[0] if version else None)


def _checksum(payload: dict[str, Any]) -> str:
    without_checksum = {key: value for key, value in payload.items() if key != "checksum"}
    data = json.dumps(without_checksum, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _heavy_signature(mol: Chem.Mol) -> dict[str, Any]:
    heavy = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(heavy)
    bonds = []
    for bond in heavy.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        bonds.append([min(a, b), max(a, b), str(bond.GetBondType()), int(bond.GetIsAromatic())])
    return {
        "atom_symbols": [atom.GetSymbol() for atom in heavy.GetAtoms()],
        "atomic_numbers": [int(atom.GetAtomicNum()) for atom in heavy.GetAtoms()],
        "bond_signature": sorted(bonds),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in heavy.GetAtoms())),
        "heavy_atom_count": int(heavy.GetNumHeavyAtoms()),
    }


def _coordinates(mol: Chem.Mol, conformer_id: int) -> list[list[float]]:
    conf = mol.GetConformer(conformer_id)
    return [
        [
            round(float(conf.GetAtomPosition(index).x), 8),
            round(float(conf.GetAtomPosition(index).y), 8),
            round(float(conf.GetAtomPosition(index).z), 8),
        ]
        for index in range(mol.GetNumAtoms())
    ]


def _optimize_conformer(mol_with_h: Chem.Mol, conformer_id: int, mmff_variant: str) -> tuple[str, float, bool, int]:
    props = AllChem.MMFFGetMoleculeProperties(mol_with_h, mmffVariant=mmff_variant)
    if props is not None:
        result = int(AllChem.MMFFOptimizeMolecule(mol_with_h, confId=conformer_id, mmffVariant=mmff_variant, maxIters=500))
        ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, props, confId=conformer_id)
        return mmff_variant, float(ff.CalcEnergy()), result == 0, result
    if not AllChem.UFFHasAllMoleculeParams(mol_with_h):
        raise ValueError("No MMFF or UFF parameters available for molecule.")
    result = int(AllChem.UFFOptimizeMolecule(mol_with_h, confId=conformer_id, maxIters=500))
    ff = AllChem.UFFGetMoleculeForceField(mol_with_h, confId=conformer_id)
    return "UFF", float(ff.CalcEnergy()), result == 0, result


def generate_rdkit_conformer_set(
    molecule_id: str,
    canonical_smiles: str,
    *,
    geometry_set_name: str = "rdkit_multi_conformer",
    num_conformers: int = 8,
    mmff_variant: str = "MMFF94s",
    prune_rms_thresh: float = 0.05,
    remove_hydrogens: bool = True,
) -> dict[str, Any]:
    """Generate a deterministic named set of RDKit conformers sorted by energy."""
    if num_conformers < 1:
        raise ValueError("num_conformers must be at least 1.")
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {canonical_smiles}")
    Chem.SanitizeMol(mol)
    expected = _heavy_signature(mol)
    mol_with_h = Chem.AddHs(mol, addCoords=False)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(int(hashlib.sha256(f"{molecule_id}|{geometry_set_name}".encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF)
    params.pruneRmsThresh = float(prune_rms_thresh)
    params.useRandomCoords = False
    params.clearConfs = True
    conf_ids = list(AllChem.EmbedMultipleConfs(mol_with_h, numConfs=int(num_conformers), params=params))
    if not conf_ids:
        raise ValueError("RDKit conformer embedding produced no conformers.")

    optimized = []
    for conf_id in conf_ids:
        method, energy, converged, optimizer_code = _optimize_conformer(mol_with_h, int(conf_id), mmff_variant)
        optimized.append(
            {
                "source_conf_id": int(conf_id),
                "method": method,
                "energy": energy,
                "convergence_status": {
                    "converged": bool(converged),
                    "optimizer_code": int(optimizer_code),
                    "embedding_code": 0,
                },
            }
        )
    optimized.sort(key=lambda item: (float(item["energy"]), int(item["source_conf_id"])))
    output_mol = Chem.RemoveHs(mol_with_h, sanitize=True) if remove_hydrogens else mol_with_h
    observed = _heavy_signature(output_mol)
    if observed != expected:
        raise ValueError("Conformer generation changed heavy-atom graph, topology, charge, or atom order.")
    if output_mol.GetNumConformers() > 1:
        rdMolAlign.AlignMolConformers(output_mol)
    conformers = []
    for rank, item in enumerate(optimized):
        source_conf_id = int(item["source_conf_id"])
        conformers.append(
            {
                "conformer_id": f"{geometry_set_name}_{rank:03d}",
                "rank": rank,
                "source_conf_id": source_conf_id,
                "method": item["method"],
                "energy": float(item["energy"]),
                "coordinates": _coordinates(output_mol, source_conf_id),
                "convergence_status": item["convergence_status"],
            }
        )
    entry: dict[str, Any] = {
        "schema_version": CONFORMER_SET_SCHEMA_VERSION,
        "base_geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "molecule_id": molecule_id,
        "canonical_smiles": canonical_smiles,
        "geometry_set_name": geometry_set_name,
        "geometry_variant": "rdkit_multi_conformer" if num_conformers > 1 else "rdkit_mmff_single",
        "conformer_count": len(conformers),
        "atom_symbols": observed["atom_symbols"],
        "atomic_numbers": observed["atomic_numbers"],
        "topology_signature": observed,
        "rdkit_version": rdBase.rdkitVersion,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "generation": {
            "requested_conformers": int(num_conformers),
            "mmff_variant": mmff_variant,
            "prune_rms_thresh": float(prune_rms_thresh),
            "hydrogen_policy": "explicit hydrogens optimized; hydrogens removed after optimization"
            if remove_hydrogens
            else "explicit hydrogens retained after optimization",
        },
        "conformers": conformers,
    }
    entry["checksum"] = _checksum(entry)
    validate_conformer_set_entry(entry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
    return entry


def migrate_single_geometry_entry(entry: dict[str, Any], *, geometry_set_name: str = "rdkit_mmff_single") -> dict[str, Any]:
    """Wrap an original v1 single-geometry cache entry as a named conformer set."""
    validate_geometry_entry(entry, molecule_id=entry.get("molecule_id"), canonical_smiles=entry.get("canonical_smiles"))
    migrated = {
        "schema_version": CONFORMER_SET_SCHEMA_VERSION,
        "base_geometry_schema_version": entry["schema_version"],
        "molecule_id": entry["molecule_id"],
        "canonical_smiles": entry["canonical_smiles"],
        "geometry_set_name": geometry_set_name,
        "geometry_variant": "rdkit_mmff_single",
        "conformer_count": 1,
        "atom_symbols": entry["atom_symbols"],
        "atomic_numbers": entry["atomic_numbers"],
        "topology_signature": entry["topology_signature"],
        "rdkit_version": entry.get("rdkit_version"),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "generation": {"migration_source_schema": entry["schema_version"], "source_checksum": payload_checksum(entry)},
        "conformers": [
            {
                "conformer_id": f"{geometry_set_name}_000",
                "rank": 0,
                "source_conf_id": 0,
                "method": entry["optimization_method"],
                "energy": entry["energy"],
                "coordinates": entry["coordinates"],
                "convergence_status": entry["convergence_status"],
            }
        ],
    }
    migrated["checksum"] = _checksum(migrated)
    validate_conformer_set_entry(migrated, molecule_id=entry["molecule_id"], canonical_smiles=entry["canonical_smiles"])
    return migrated


def validate_conformer_set_entry(
    entry: dict[str, Any],
    *,
    molecule_id: str | None = None,
    canonical_smiles: str | None = None,
) -> None:
    """Validate a named conformer-set cache entry."""
    if entry.get("schema_version") != CONFORMER_SET_SCHEMA_VERSION:
        raise ValueError("Invalid conformer-set schema version.")
    if molecule_id is not None and entry.get("molecule_id") != molecule_id:
        raise ValueError("Conformer-set molecule_id does not match.")
    if canonical_smiles is not None and entry.get("canonical_smiles") != canonical_smiles:
        raise ValueError("Conformer-set canonical_smiles does not match.")
    if entry.get("checksum") != _checksum(entry):
        raise ValueError("Conformer-set checksum mismatch.")
    conformers = entry.get("conformers")
    if not isinstance(conformers, list) or not conformers:
        raise ValueError("Conformer-set must contain at least one conformer.")
    if int(entry.get("conformer_count", -1)) != len(conformers):
        raise ValueError("Conformer count does not match conformer list.")
    atom_count = len(entry.get("atom_symbols", []))
    energies = []
    conformer_ids = []
    for conformer in conformers:
        conformer_ids.append(str(conformer.get("conformer_id")))
        energy = float(conformer["energy"])
        energies.append(energy)
        coords = conformer.get("coordinates")
        if not isinstance(coords, list) or len(coords) != atom_count:
            raise ValueError("Conformer energy and geometry alignment failed.")
        if any(not isinstance(row, list) or len(row) != 3 for row in coords):
            raise ValueError("Conformer coordinates must be Nx3.")
        if "method" not in conformer or "convergence_status" not in conformer:
            raise ValueError("Conformer method and convergence are required.")
    if len(conformer_ids) != len(set(conformer_ids)):
        raise ValueError("Conformer IDs must be unique.")
    if energies != sorted(energies):
        raise ValueError("Conformers must be ordered deterministically by energy.")
    if canonical_smiles is not None:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError("Invalid canonical SMILES for conformer set.")
        if entry.get("topology_signature") != _heavy_signature(mol):
            raise ValueError("Conformer-set topology signature does not match canonical input.")


def read_valid_conformer_set(path: Path, molecule_id: str, canonical_smiles: str) -> dict[str, Any]:
    entry = json.loads(path.read_text(encoding="utf-8"))
    validate_conformer_set_entry(entry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
    return entry


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for conformer pooling components.") from exc
    return torch


def equal_pool(conformer_embeddings: Any) -> Any:
    """Permutation-invariant equal pooling over conformer embeddings."""
    return conformer_embeddings.mean(dim=-2)


def energy_weighted_pool(conformer_embeddings: Any, energies: Any, temperature: float = 1.0) -> tuple[Any, Any]:
    """Permutation-invariant Boltzmann-style pooling over conformer embeddings."""
    torch = _require_torch()
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    shifted = energies - energies.min(dim=-1, keepdim=True).values
    weights = torch.softmax(-shifted / float(temperature), dim=-1)
    pooled = (conformer_embeddings * weights.unsqueeze(-1)).sum(dim=-2)
    return pooled, weights


class SolventConditionedConformerAttention:
    """Isolated solvent-conditioned conformer weighting component."""

    @staticmethod
    def build(torch: Any, conformer_dim: int, solvent_dim: int) -> Any:
        nn = torch.nn

        class Attention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conformer_score = nn.Linear(conformer_dim, conformer_dim, bias=False)
                self.solvent_query = nn.Linear(solvent_dim, conformer_dim, bias=False)

            def forward(self, conformer_embeddings: Any, solvent_embedding: Any) -> tuple[Any, Any]:
                query = self.solvent_query(solvent_embedding).unsqueeze(-2)
                keys = self.conformer_score(conformer_embeddings)
                scores = (keys * query).sum(dim=-1) / float(conformer_embeddings.shape[-1]) ** 0.5
                weights = torch.softmax(scores, dim=-1)
                pooled = (conformer_embeddings * weights.unsqueeze(-1)).sum(dim=-2)
                return pooled, weights

        return Attention()
