"""Deterministic RDKit geometry cache for UniProp molecules."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

GEOMETRY_SCHEMA_VERSION = "uniprop_geometry_cache_v2"
LEGACY_GEOMETRY_SCHEMA_VERSIONS = {"uniprop_geometry_cache_v1"}
DEFAULT_MANIFEST = Path("data/processed/uniprop/molecule_manifest.csv")
DEFAULT_CACHE_DIR = Path("data/processed/uniprop/geometry_cache")


@dataclass(frozen=True)
class GeometryResult:
    """One cache operation result."""

    molecule_id: str
    status: str
    cache_path: Path | None
    failure_reason: str | None = None
    detail: str | None = None


def molecule_seed(molecule_id: str) -> int:
    """Derive an RDKit-compatible deterministic seed from a stable molecule ID."""
    return int(hashlib.sha256(molecule_id.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def cache_path(cache_dir: Path, molecule_id: str) -> Path:
    """Return the JSON cache path for one molecule."""
    return cache_dir / f"{molecule_id}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_ready(payload: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def payload_checksum(payload: dict[str, Any]) -> str:
    """Checksum an entry excluding its checksum field."""
    without_checksum = {key: value for key, value in payload.items() if key != "checksum"}
    data = json.dumps(_json_ready(without_checksum), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _bond_signature(mol: Chem.Mol) -> list[list[int | str]]:
    signature = []
    for bond in mol.GetBonds():
        begin = int(bond.GetBeginAtomIdx())
        end = int(bond.GetEndAtomIdx())
        signature.append(
            [
                min(begin, end),
                max(begin, end),
                str(bond.GetBondType()),
                int(bond.GetIsAromatic()),
            ]
        )
    return sorted(signature)


def _formal_charge(mol: Chem.Mol) -> int:
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _heavy_graph_signature(mol: Chem.Mol) -> dict[str, Any]:
    heavy = Chem.RemoveHs(mol, sanitize=False)
    Chem.SanitizeMol(heavy)
    return {
        "atom_symbols": [atom.GetSymbol() for atom in heavy.GetAtoms()],
        "atomic_numbers": [int(atom.GetAtomicNum()) for atom in heavy.GetAtoms()],
        "bond_signature": _bond_signature(heavy),
        "formal_charge": _formal_charge(heavy),
        "heavy_atom_count": int(heavy.GetNumHeavyAtoms()),
    }


def _coordinates(mol: Chem.Mol) -> list[list[float]]:
    conformer = mol.GetConformer()
    return [
        [
            round(float(conformer.GetAtomPosition(index).x), 8),
            round(float(conformer.GetAtomPosition(index).y), 8),
            round(float(conformer.GetAtomPosition(index).z), 8),
        ]
        for index in range(mol.GetNumAtoms())
    ]


def _finite_coordinates(mol: Chem.Mol) -> bool:
    try:
        conformer = mol.GetConformer()
    except ValueError:
        return False
    for index in range(mol.GetNumAtoms()):
        pos = conformer.GetAtomPosition(index)
        if not all(math.isfinite(float(value)) for value in [pos.x, pos.y, pos.z]):
            return False
    return True


def _embedding_attempt(params: Any, label: str, mol_with_h: Chem.Mol) -> dict[str, Any]:
    code = int(AllChem.EmbedMolecule(mol_with_h, params))
    return {
        "method": label,
        "code": code,
        "max_attempts": int(getattr(params, "maxAttempts", 0)) if hasattr(params, "maxAttempts") else None,
        "use_random_coords": bool(getattr(params, "useRandomCoords", False)),
    }


def _embed_molecule(mol_with_h: Chem.Mol, seed: int) -> tuple[int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    settings = [
        ("ETKDGv3", 20, False),
        ("ETKDGv3_retry", 200, False),
        ("ETKDGv3_random_coords", 200, True),
    ]
    for label, max_attempts, random_coords in settings:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useRandomCoords = bool(random_coords)
        try:
            params.maxAttempts = int(max_attempts)
        except AttributeError:
            pass
        params.clearConfs = True
        attempt = _embedding_attempt(params, label, mol_with_h)
        attempt["requested_max_attempts"] = int(max_attempts)
        attempts.append(attempt)
        if attempt["code"] == 0:
            return 0, attempts
    return int(attempts[-1]["code"]), attempts


def _optimize(mol_with_h: Chem.Mol, mmff_variant: str) -> tuple[str, float | None, bool, int, list[dict[str, Any]], bool, bool]:
    attempts: list[dict[str, Any]] = []
    props = AllChem.MMFFGetMoleculeProperties(mol_with_h, mmffVariant=mmff_variant)
    if props is not None:
        for max_iters in [500, 2000]:
            result = int(AllChem.MMFFOptimizeMolecule(mol_with_h, mmffVariant=mmff_variant, maxIters=max_iters))
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, props)
            energy = float(ff.CalcEnergy()) if ff is not None else None
            attempts.append(
                {
                    "method": mmff_variant,
                    "max_iters": max_iters,
                    "optimizer_code": result,
                    "converged": result == 0,
                    "energy": energy,
                }
            )
            if result == 0:
                return mmff_variant, energy, True, result, attempts, True, bool(AllChem.UFFHasAllMoleculeParams(mol_with_h))
            if result != 1:
                break

    uff_ok = bool(AllChem.UFFHasAllMoleculeParams(mol_with_h))
    if not uff_ok:
        if props is None:
            raise ValueError("No MMFF or UFF parameters available for molecule.")
        raise ValueError(f"{mmff_variant} did not converge and UFF parameters are unavailable.")
    result = int(AllChem.UFFOptimizeMolecule(mol_with_h, maxIters=2000))
    ff = AllChem.UFFGetMoleculeForceField(mol_with_h)
    energy = float(ff.CalcEnergy()) if ff is not None else None
    attempts.append(
        {
            "method": "UFF",
            "max_iters": 2000,
            "optimizer_code": result,
            "converged": result == 0,
            "energy": energy,
        }
    )
    return "UFF", energy, result == 0, result, attempts, props is not None, uff_ok


def generate_geometry_entry(
    molecule_id: str,
    canonical_smiles: str,
    *,
    mmff_variant: str = "MMFF94s",
    remove_hydrogens: bool = True,
) -> dict[str, Any]:
    """Generate one deterministic geometry cache entry."""
    mol = Chem.MolFromSmiles(str(canonical_smiles))
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES: {canonical_smiles}")
    Chem.SanitizeMol(mol)
    expected = _heavy_graph_signature(mol)
    seed = molecule_seed(molecule_id)

    mol_with_h = Chem.AddHs(mol, addCoords=False)
    embed_result, embedding_attempts = _embed_molecule(mol_with_h, seed)
    if embed_result != 0:
        raise ValueError(f"RDKit ETKDGv3 embedding failed with code {embed_result}.")

    method, energy, converged, optimize_code, optimization_attempts, mmff_available, uff_available = _optimize(mol_with_h, mmff_variant)
    if not _finite_coordinates(mol_with_h):
        raise ValueError("RDKit generated non-finite coordinates.")
    if not converged:
        raise ValueError(f"{method} did not converge with code {optimize_code}.")
    output_mol = Chem.RemoveHs(mol_with_h, sanitize=True) if remove_hydrogens else mol_with_h
    if not _finite_coordinates(output_mol):
        raise ValueError("Output geometry contains non-finite coordinates.")
    observed = _heavy_graph_signature(output_mol)
    if observed != expected:
        raise ValueError("Optimized geometry changed heavy-atom graph, topology, charge, or atom order.")
    quality = "mmff_converged" if method.startswith("MMFF") else "uff_converged"

    entry: dict[str, Any] = {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "molecule_id": molecule_id,
        "canonical_smiles": canonical_smiles,
        "geometry_status": "success",
        "geometry_support_status": "supported",
        "force_field_support_status": "mmff_supported" if mmff_available else "uff_only",
        "geometry_quality": quality,
        "model_vocabulary_status": "not_evaluated",
        "atom_symbols": [atom.GetSymbol() for atom in output_mol.GetAtoms()],
        "atomic_numbers": [int(atom.GetAtomicNum()) for atom in output_mol.GetAtoms()],
        "coordinates": _coordinates(output_mol),
        "optimization_method": method,
        "mmff_available": bool(mmff_available),
        "uff_available": bool(uff_available),
        "energy": energy,
        "convergence_status": {
            "converged": bool(converged),
            "optimizer_code": optimize_code,
            "embedding_code": embed_result,
        },
        "embedding_attempts": embedding_attempts,
        "optimization_attempts": optimization_attempts,
        "rdkit_version": rdBase.rdkitVersion,
        "seed": seed,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "hydrogen_policy": (
            "explicit hydrogens used for ETKDGv3 embedding and force-field optimization; "
            "hydrogens removed after optimization"
            if remove_hydrogens
            else "explicit hydrogens retained after optimization"
        ),
        "topology_signature": observed,
    }
    entry["checksum"] = payload_checksum(entry)
    return entry


def validate_geometry_entry(
    entry: dict[str, Any],
    *,
    molecule_id: str | None = None,
    canonical_smiles: str | None = None,
) -> None:
    """Validate a cache entry and raise a clear error when it is unusable."""
    schema = entry.get("schema_version")
    if schema != GEOMETRY_SCHEMA_VERSION and schema not in LEGACY_GEOMETRY_SCHEMA_VERSIONS:
        raise ValueError("Invalid geometry schema version.")
    if molecule_id is not None and entry.get("molecule_id") != molecule_id:
        raise ValueError("Cache molecule_id does not match manifest row.")
    if canonical_smiles is not None and entry.get("canonical_smiles") != canonical_smiles:
        raise ValueError("Cache canonical_smiles does not match manifest row.")
    if entry.get("checksum") != payload_checksum(entry):
        raise ValueError("Cache checksum mismatch.")
    atom_symbols = entry.get("atom_symbols")
    atomic_numbers = entry.get("atomic_numbers")
    coordinates = entry.get("coordinates")
    if not isinstance(atom_symbols, list) or not isinstance(atomic_numbers, list):
        raise ValueError("Cache atom arrays are missing.")
    if not isinstance(coordinates, list):
        raise ValueError("Cache coordinate array is missing.")
    if len(atom_symbols) != len(atomic_numbers) or len(atom_symbols) != len(coordinates):
        raise ValueError("Atom and coordinate array lengths differ.")
    for xyz in coordinates:
        if not isinstance(xyz, list) or len(xyz) != 3:
            raise ValueError("Coordinate rows must contain exactly three values.")
        if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in xyz):
            raise ValueError("Coordinate rows must contain finite numeric values.")
    converged = bool(entry.get("convergence_status", {}).get("converged"))
    if not converged:
        raise ValueError("Cache geometry is finite but nonconverged.")
    method = str(entry.get("optimization_method", ""))
    if method == "UFF":
        if schema == GEOMETRY_SCHEMA_VERSION and entry.get("geometry_quality") not in {"uff_converged"}:
            raise ValueError("UFF cache entry has incompatible geometry quality.")
    elif not method.startswith("MMFF"):
        raise ValueError("Cache optimization method is unsupported.")
    if canonical_smiles is not None:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            raise ValueError("Manifest canonical SMILES is invalid.")
        expected = _heavy_graph_signature(mol)
        if entry.get("topology_signature") != expected:
            raise ValueError("Cache topology signature does not match canonical input.")


def read_valid_cache(path: Path, molecule_id: str, canonical_smiles: str) -> dict[str, Any]:
    """Read and validate an existing cache entry."""
    entry = json.loads(path.read_text(encoding="utf-8"))
    validate_geometry_entry(entry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
    if entry.get("schema_version") in LEGACY_GEOMETRY_SCHEMA_VERSIONS:
        method = str(entry.get("optimization_method", ""))
        entry = {
            **entry,
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "geometry_status": "success",
            "geometry_support_status": "supported",
            "force_field_support_status": "mmff_supported" if method.startswith("MMFF") else "uff_only",
            "geometry_quality": "mmff_converged" if method.startswith("MMFF") else "uff_converged",
            "model_vocabulary_status": "not_evaluated",
            "mmff_available": bool(method.startswith("MMFF")),
            "uff_available": True,
            "embedding_attempts": entry.get("embedding_attempts", []),
            "optimization_attempts": entry.get("optimization_attempts", []),
            "updated_at": _utc_now(),
        }
        entry["checksum"] = payload_checksum(entry)
    return entry


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON through a temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_molecule_manifest(path: Path) -> pd.DataFrame:
    """Load the molecule manifest without reading experimental target labels."""
    rows = pd.read_csv(path)
    required = ["molecule_id", "canonical_isomeric_smiles"]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Molecule manifest is missing required column(s): {missing}")
    if rows["molecule_id"].duplicated().any():
        raise ValueError("Molecule manifest contains duplicate molecule_id values.")
    return rows


def select_manifest_rows(
    manifest: pd.DataFrame,
    *,
    molecule_ids: Iterable[str] | None = None,
    limit: int | None = None,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> pd.DataFrame:
    """Select a stable subset of molecule manifest rows."""
    rows = manifest.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    if molecule_ids:
        wanted = set(molecule_ids)
        rows = rows[rows["molecule_id"].isin(wanted)].copy()
    if shard_index is not None or shard_count is not None:
        if shard_index is None or shard_count is None:
            raise ValueError("Both shard_index and shard_count are required for sharding.")
        if shard_count < 1 or not 0 <= shard_index < shard_count:
            raise ValueError("Shard index/count are out of range.")
        positions = pd.Series(range(len(rows)), index=rows.index)
        rows = rows[positions % shard_count == shard_index].copy()
    if limit is not None:
        rows = rows.head(limit).copy()
    return rows.reset_index(drop=True)


def process_one_molecule(
    row: dict[str, Any],
    cache_dir: Path,
    *,
    resume: bool,
    overwrite_invalid: bool,
    mmff_variant: str,
    remove_hydrogens: bool,
) -> GeometryResult:
    """Validate or generate one cache entry."""
    molecule_id = str(row["molecule_id"])
    canonical_smiles = str(row["canonical_isomeric_smiles"])
    path = cache_path(cache_dir, molecule_id)
    if resume and path.exists():
        try:
            read_valid_cache(path, molecule_id, canonical_smiles)
            return GeometryResult(molecule_id, "hit", path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            if not overwrite_invalid:
                return GeometryResult(molecule_id, "invalid_cache", path, "invalid_cache", str(exc))

    try:
        entry = generate_geometry_entry(
            molecule_id,
            canonical_smiles,
            mmff_variant=mmff_variant,
            remove_hydrogens=remove_hydrogens,
        )
        validate_geometry_entry(entry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
        atomic_write_json(path, entry)
        read_valid_cache(path, molecule_id, canonical_smiles)
        return GeometryResult(molecule_id, "generated", path)
    except (RuntimeError, ValueError, OSError) as exc:
        return GeometryResult(molecule_id, "failed", path, "generation_failed", str(exc))


def build_geometry_cache(
    manifest_path: Path,
    cache_dir: Path,
    *,
    limit: int | None = None,
    molecule_ids: Iterable[str] | None = None,
    workers: int = 1,
    resume: bool = True,
    overwrite_invalid: bool = False,
    fail_fast: bool = False,
    mmff_variant: str = "MMFF94s",
    remove_hydrogens: bool = True,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> list[GeometryResult]:
    """Build or validate cache entries for selected manifest molecules."""
    manifest = load_molecule_manifest(manifest_path)
    selected = select_manifest_rows(
        manifest,
        molecule_ids=molecule_ids,
        limit=limit,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    records = selected.to_dict("records")
    results: list[GeometryResult] = []
    if workers <= 1:
        for row in records:
            result = process_one_molecule(
                row,
                cache_dir,
                resume=resume,
                overwrite_invalid=overwrite_invalid,
                mmff_variant=mmff_variant,
                remove_hydrogens=remove_hydrogens,
            )
            results.append(result)
            if fail_fast and result.status in {"failed", "invalid_cache"}:
                break
        return results

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_one_molecule,
                row,
                cache_dir,
                resume=resume,
                overwrite_invalid=overwrite_invalid,
                mmff_variant=mmff_variant,
                remove_hydrogens=remove_hydrogens,
            ): str(row["molecule_id"])
            for row in records
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if fail_fast and result.status in {"failed", "invalid_cache"}:
                break
    return sorted(results, key=lambda item: item.molecule_id)


def status_summary(results: list[GeometryResult], expected_total: int | None = None) -> dict[str, Any]:
    """Summarize cache operation statuses."""
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return {
        "schema_version": GEOMETRY_SCHEMA_VERSION,
        "expected_total": expected_total if expected_total is not None else len(results),
        "processed_total": len(results),
        "status_counts": counts,
        "reconciles": (expected_total is None or expected_total == len(results)),
    }


def write_failure_reports(results: list[GeometryResult], json_path: Path | None, csv_path: Path | None) -> None:
    """Write structured failure reports."""
    failures = [
        {
            "molecule_id": result.molecule_id,
            "status": result.status,
            "cache_path": str(result.cache_path) if result.cache_path else None,
            "failure_reason": result.failure_reason,
            "detail": result.detail,
        }
        for result in results
        if result.status in {"failed", "invalid_cache"}
    ]
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8")
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["molecule_id", "status", "cache_path", "failure_reason", "detail"],
            )
            writer.writeheader()
            writer.writerows(failures)
