"""Stable molecule and row manifests for UniProp preparation.

This module intentionally stops before geometry generation. It builds stable
identifiers, leakage-safe split assignments, audit reports, and train-only
normalization summaries from the existing processed FluorCast dataset.

Stereochemical policy:

- ``canonical_isomeric_smiles`` preserves RDKit isomeric canonicalization.
- ``canonical_nonisomeric_smiles`` removes stereochemical distinctions and is
  used only as an auxiliary grouping/reporting field.
- ``molecule_id`` is derived from the isomeric canonical SMILES when available,
  making enantiomers/diastereomers deterministic separate molecules.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DEFAULT_PROCESSED_DIR = Path("data/processed/fluodb_lite")
DEFAULT_OUTPUT_DIR = Path("data/processed/uniprop")
DEFAULT_TARGET_COLUMNS = [
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
    "stokes_shift_nm",
]
SPLIT_FAMILIES = ["random", "molecule", "scaffold", "solvent", "double_cold_start"]
MANIFEST_SCHEMA_VERSION = "uniprop_manifest_v1"


@dataclass(frozen=True)
class ManifestBundle:
    """In-memory manifest products."""

    source_path: Path
    molecule_manifest: pd.DataFrame
    row_manifest: pd.DataFrame
    metadata: dict[str, Any]


def resolve_authoritative_dataset(path: Path | None = None) -> Path:
    """Return the processed FluorCast dataset used as UniProp source of truth."""
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {path}")
        return path

    with_stokes = DEFAULT_PROCESSED_DIR / "combined_deduplicated_with_stokes.csv"
    base = DEFAULT_PROCESSED_DIR / "combined_deduplicated.csv"
    if with_stokes.exists():
        return with_stokes
    if base.exists():
        return base
    raise FileNotFoundError(
        "Could not find authoritative processed FluorCast dataset. Expected "
        f"{with_stokes} or {base}."
    )


def stable_hash(prefix: str, *parts: object, length: int = 16) -> str:
    """Return a compact deterministic ID from structured string parts."""
    payload = json.dumps(
        ["" if pd.isna(part) else str(part) for part in parts],
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def _stable_unit_interval(*parts: object) -> float:
    digest = stable_hash("h", *parts, length=16).split("_", 1)[1]
    return int(digest, 16) / float(16**16)


def _canonicalize(smiles: object, *, isomeric: bool) -> tuple[str | None, str]:
    if pd.isna(smiles):
        return None, "missing"
    text = str(smiles).strip()
    if not text:
        return None, "missing"
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None, "invalid"
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric), "ok"
    except (RuntimeError, ValueError):
        return None, "invalid"


def _mol_from_canonical(smiles: object) -> Chem.Mol | None:
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(str(smiles))
    except (RuntimeError, ValueError):
        return None


def _inchikey(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except (ValueError, RuntimeError):
        return None


def _formal_charge(mol: Chem.Mol | None) -> int | None:
    if mol is None:
        return None
    return int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))


def _atom_count(mol: Chem.Mol | None) -> int | None:
    if mol is None:
        return None
    return int(mol.GetNumAtoms())


def _heavy_atom_count(mol: Chem.Mol | None) -> int | None:
    if mol is None:
        return None
    return int(mol.GetNumHeavyAtoms())


def _scaffold(smiles: object) -> str:
    mol = _mol_from_canonical(smiles)
    if mol is None:
        return "<INVALID>"
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except (RuntimeError, ValueError):
        return "<INVALID>"
    return scaffold or "<ACYCLIC>"


def _target_columns(rows: pd.DataFrame, target_columns: Iterable[str] | None) -> list[str]:
    requested = list(target_columns or DEFAULT_TARGET_COLUMNS)
    return [column for column in requested if column in rows.columns]


def _read_processed_dataset(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    required = [
        "chromophore_smiles",
        "canonical_chromophore_smiles",
        "canonical_solvent_smiles",
        "source_dataset",
    ]
    missing = [column for column in required if column not in rows.columns]
    if missing:
        raise ValueError(f"Processed dataset is missing required column(s): {missing}")
    if rows["canonical_chromophore_smiles"].isna().any():
        raise ValueError("Processed dataset contains missing canonical chromophore SMILES.")
    return rows


def build_manifests(
    dataset_path: Path | None = None,
    target_columns: Iterable[str] | None = None,
    compute_inchikey: bool = False,
    compute_rdkit_properties: bool = False,
    compute_nonisomeric: bool = False,
) -> ManifestBundle:
    """Build molecule and row manifests from the authoritative processed table."""
    source_path = resolve_authoritative_dataset(dataset_path)
    rows = _read_processed_dataset(source_path).copy()
    targets = _target_columns(rows, target_columns)
    rows["_source_row_number"] = np.arange(len(rows), dtype=int)

    rows["_canonical_isomeric_smiles"] = rows["canonical_chromophore_smiles"].astype(str)
    unique_canonicals = sorted(set(rows["_canonical_isomeric_smiles"]))
    nonisomeric_by_isomeric: dict[str, str | None] = {}
    status_by_isomeric: dict[str, str] = {}
    for smiles in unique_canonicals:
        if compute_nonisomeric:
            nonisomeric, status = _canonicalize(smiles, isomeric=False)
            nonisomeric_by_isomeric[smiles] = nonisomeric
            status_by_isomeric[smiles] = "ok_processed_canonical" if status == "ok" else "invalid_processed_canonical"
        else:
            nonisomeric_by_isomeric[smiles] = smiles
            status_by_isomeric[smiles] = "ok_processed_canonical_nonisomeric_not_computed"
    rows["_canonical_nonisomeric_smiles"] = rows["_canonical_isomeric_smiles"].map(nonisomeric_by_isomeric)
    rows["_canonicalization_status"] = rows["_canonical_isomeric_smiles"].map(status_by_isomeric)
    if rows["_canonical_isomeric_smiles"].isna().any():
        bad = int(rows["_canonical_isomeric_smiles"].isna().sum())
        raise ValueError(f"Processed dataset contains {bad} invalid chromophore SMILES row(s).")

    rows["_molecule_id"] = rows["_canonical_isomeric_smiles"].map(
        lambda smiles: stable_hash("mol", MANIFEST_SCHEMA_VERSION, smiles)
    )
    rows["_solvent_key"] = rows["canonical_solvent_smiles"].map(
        lambda value: "<MISSING_SOLVENT>" if pd.isna(value) else str(value)
    )
    rows["_solvent_id"] = rows["_solvent_key"].map(
        lambda smiles: stable_hash("solv", MANIFEST_SCHEMA_VERSION, smiles)
    )

    molecule_rows = []
    grouped = rows.sort_values(
        ["_canonical_isomeric_smiles", "_source_row_number"], kind="mergesort"
    ).groupby("_molecule_id", sort=True, dropna=False)
    for molecule_id, group in grouped:
        smiles = str(group["_canonical_isomeric_smiles"].iloc[0])
        mol = _mol_from_canonical(smiles) if compute_rdkit_properties or compute_inchikey else None
        molecule_rows.append(
            {
                "molecule_id": molecule_id,
                "original_smiles": str(group["chromophore_smiles"].iloc[0]),
                "canonical_isomeric_smiles": smiles,
                "canonical_nonisomeric_smiles": group["_canonical_nonisomeric_smiles"].iloc[0],
                "inchikey": _inchikey(mol) if compute_inchikey else None,
                "formal_charge": _formal_charge(mol) if compute_rdkit_properties else None,
                "atom_count": _atom_count(mol) if compute_rdkit_properties else None,
                "heavy_atom_count": _heavy_atom_count(mol) if compute_rdkit_properties else None,
                "canonicalization_status": ",".join(sorted(set(group["_canonicalization_status"]))),
                "source_row_count": int(len(group)),
                "deterministic_molecule_seed": int(
                    hashlib.sha256(str(molecule_id).encode("utf-8")).hexdigest()[:8],
                    16,
                ),
            }
        )
    molecule_manifest = pd.DataFrame(molecule_rows).sort_values("molecule_id").reset_index(drop=True)

    row_id_columns = ["_molecule_id", "_solvent_id", "source_dataset", *targets]
    row_ids = []
    for values in rows[row_id_columns].itertuples(index=False, name=None):
        molecule_id, solvent_id, source_dataset, *target_values = values
        row_ids.append(
            stable_hash(
                "row",
                MANIFEST_SCHEMA_VERSION,
                molecule_id,
                solvent_id,
                source_dataset,
                *[
                    "<NA>" if pd.isna(value) else f"{float(value):.12g}"
                    for value in target_values
                ],
            )
        )
    row_manifest = pd.DataFrame(
        {
            "row_id": row_ids,
            "molecule_id": rows["_molecule_id"].to_numpy(),
            "solvent_id": rows["_solvent_id"].to_numpy(),
            "canonical_solvent_smiles": rows["canonical_solvent_smiles"].to_numpy(),
            "source_dataset": rows["source_dataset"].to_numpy(),
            "source_row_number": rows["_source_row_number"].to_numpy(),
        }
    )
    for target in targets:
        row_manifest[target] = rows[target]
        row_manifest[f"{target}_available"] = rows[target].notna()

    metadata = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "authoritative_dataset": str(source_path),
        "source_rows": int(len(rows)),
        "manifest_rows": int(len(row_manifest)),
        "unique_molecules": int(row_manifest["molecule_id"].nunique()),
        "unique_solvents": int(row_manifest["solvent_id"].nunique()),
        "target_columns": targets,
        "stereochemical_policy": (
            "molecule_id uses RDKit canonical isomeric SMILES; "
            "canonical_nonisomeric_smiles is retained only for auxiliary grouping."
        ),
        "inchikey_policy": (
            "inchikey is emitted when explicitly computed with --compute-inchikey; "
            "bulk default leaves it missing to keep manifest inspection lightweight."
        ),
        "rdkit_property_policy": (
            "formal_charge, atom_count, and heavy_atom_count are emitted when "
            "computed with --compute-rdkit-properties; bulk default leaves them "
            "missing to avoid blocking manifest inspection on difficult structures."
        ),
        "canonical_nonisomeric_policy": (
            "canonical_nonisomeric_smiles is computed with --compute-nonisomeric; "
            "the default mirrors canonical_isomeric_smiles for fast contract builds."
        ),
    }
    return ManifestBundle(source_path, molecule_manifest, row_manifest, metadata)


def _stable_group_split(
    rows: pd.DataFrame,
    group_column: str,
    test_size: float,
    seed: int,
    family: str,
) -> pd.Series:
    groups = rows[group_column].astype(str).to_numpy()
    group_sizes = rows.groupby(group_column, sort=True).size().reset_index(name="rows")
    if len(group_sizes) < 2:
        raise ValueError(f"Split requires at least two unique {group_column} values.")
    group_sizes["_score"] = group_sizes[group_column].map(
        lambda group: _stable_unit_interval(MANIFEST_SCHEMA_VERSION, family, seed, group)
    )
    group_sizes = group_sizes.sort_values(["_score", group_column], kind="mergesort")
    target_test_rows = max(1, int(round(len(rows) * test_size)))
    test_groups: set[str] = set()
    test_rows = 0
    for _, group in group_sizes.iterrows():
        test_groups.add(str(group[group_column]))
        test_rows += int(group["rows"])
        if test_rows >= target_test_rows:
            break
    if len(test_groups) == len(group_sizes):
        last_group = str(group_sizes.iloc[-1][group_column])
        test_groups.remove(last_group)
    assignment = pd.Series("train", index=rows.index, dtype="object")
    assignment.loc[rows[group_column].astype(str).isin(test_groups)] = "test"
    return assignment


def make_split_assignments(
    row_manifest: pd.DataFrame,
    molecule_manifest: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    compute_rdkit_scaffolds: bool = False,
) -> pd.DataFrame:
    """Create deterministic split-family assignments for manifest rows."""
    if len(row_manifest) < 2:
        raise ValueError("At least two manifest rows are required for splitting.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    rows = row_manifest.reset_index(drop=True).copy()
    scaffold_map = dict(
        zip(
            molecule_manifest["molecule_id"],
            (
                molecule_manifest["canonical_isomeric_smiles"].map(_scaffold)
                if compute_rdkit_scaffolds
                else molecule_manifest["canonical_nonisomeric_smiles"].fillna(
                    molecule_manifest["canonical_isomeric_smiles"]
                )
            ),
            strict=True,
        )
    )
    rows["scaffold_group"] = rows["molecule_id"].map(scaffold_map)

    splits = pd.DataFrame({"row_id": rows["row_id"]})
    ordered_row_ids = (
        rows[["row_id"]]
        .assign(_score=rows["row_id"].map(lambda row_id: _stable_unit_interval(MANIFEST_SCHEMA_VERSION, "random", seed, row_id)))
        .sort_values(["_score", "row_id"], kind="mergesort")
    )
    target_random_test = max(1, int(round(len(rows) * test_size)))
    random_assignment = pd.Series("train", index=rows.index, dtype="object")
    random_test_ids = set(ordered_row_ids.head(target_random_test)["row_id"])
    random_assignment.loc[rows["row_id"].isin(random_test_ids)] = "test"
    splits["random"] = random_assignment
    splits["molecule"] = _stable_group_split(rows, "molecule_id", test_size, seed, "molecule")
    splits["scaffold"] = _stable_group_split(rows, "scaffold_group", test_size, seed, "scaffold")
    splits["solvent"] = _stable_group_split(rows, "solvent_id", test_size, seed, "solvent")

    double_assignment = _double_cold_start_split(rows, test_size, seed)
    splits["double_cold_start"] = double_assignment
    return splits


def _double_cold_start_split(rows: pd.DataFrame, test_size: float, seed: int) -> pd.Series:
    ordered = (
        rows[["row_id", "molecule_id", "solvent_id"]]
        .assign(_score=rows["row_id"].map(lambda row_id: _stable_unit_interval(MANIFEST_SCHEMA_VERSION, "double_cold_start", seed, row_id)))
        .sort_values(["_score", "row_id"], kind="mergesort")
        .reset_index(drop=False)
    )
    target_test_rows = max(1, int(round(len(rows) * test_size)))
    small_counts = list(range(1, min(len(ordered), 25) + 1))
    scaled_counts = [
        max(1, int(round(target_test_rows * factor)))
        for factor in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    ]
    candidate_counts = [
        count for count in [*small_counts, *scaled_counts, len(ordered) - 1]
        if 0 < count < len(ordered)
    ]
    for candidate_count in dict.fromkeys(candidate_counts):
        selected = ordered.head(candidate_count)
        molecule_holdout = set(selected["molecule_id"])
        solvent_holdout = set(selected["solvent_id"])
        heldout_mol = rows["molecule_id"].isin(molecule_holdout)
        heldout_solvent = rows["solvent_id"].isin(solvent_holdout)
        assignment = pd.Series("heldout_boundary", index=rows.index, dtype="object")
        assignment.loc[~heldout_mol & ~heldout_solvent] = "train"
        assignment.loc[heldout_mol & heldout_solvent] = "test"
        train_rows = int((assignment == "train").sum())
        test_rows = int((assignment == "test").sum())
        if train_rows and test_rows and test_rows >= target_test_rows:
            return assignment
    raise ValueError("Double-cold-start split produced an empty train or test partition.")


def audit_split_leakage(
    row_manifest: pd.DataFrame,
    molecule_manifest: pd.DataFrame,
    split_assignments: pd.DataFrame,
    compute_rdkit_scaffolds: bool = False,
) -> pd.DataFrame:
    """Return leakage diagnostics for every split family."""
    rows = row_manifest.merge(split_assignments, on="row_id", how="left", validate="one_to_one")
    scaffold_map = dict(
        zip(
            molecule_manifest["molecule_id"],
            (
                molecule_manifest["canonical_isomeric_smiles"].map(_scaffold)
                if compute_rdkit_scaffolds
                else molecule_manifest["canonical_nonisomeric_smiles"].fillna(
                    molecule_manifest["canonical_isomeric_smiles"]
                )
            ),
            strict=True,
        )
    )
    rows["scaffold_group"] = rows["molecule_id"].map(scaffold_map)

    audits = [
        {
            "split_family": "random",
            "train_rows": int((rows["random"] == "train").sum()),
            "test_rows": int((rows["random"] == "test").sum()),
            "heldout_boundary_rows": 0,
            "passed": True,
            "overlapping_molecule_ids": math.nan,
            "overlapping_scaffold_groups": math.nan,
            "overlapping_solvent_ids": math.nan,
            "note": "not_applicable_random_row_split",
        }
    ]
    checks = {
        "molecule": ("molecule_id",),
        "scaffold": ("scaffold_group",),
        "solvent": ("solvent_id",),
        "double_cold_start": ("molecule_id", "solvent_id"),
    }
    for family, columns in checks.items():
        train = rows[rows[family] == "train"]
        test = rows[rows[family] == "test"]
        row: dict[str, Any] = {
            "split_family": family,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "heldout_boundary_rows": int((rows[family] == "heldout_boundary").sum()),
            "passed": True,
            "note": "",
        }
        for column in columns:
            overlap = set(train[column].astype(str)).intersection(set(test[column].astype(str)))
            row[f"overlapping_{column}s"] = int(len(overlap))
            row["passed"] = bool(row["passed"] and not overlap)
        audits.append(row)
    return pd.DataFrame(audits)


def split_statistics(
    row_manifest: pd.DataFrame,
    molecule_manifest: pd.DataFrame,
    split_assignments: pd.DataFrame,
    target_columns: Iterable[str],
    compute_rdkit_scaffolds: bool = False,
) -> pd.DataFrame:
    """Return row/group counts and target coverage for split partitions."""
    rows = row_manifest.merge(split_assignments, on="row_id", how="left", validate="one_to_one")
    scaffold_map = dict(
        zip(
            molecule_manifest["molecule_id"],
            (
                molecule_manifest["canonical_isomeric_smiles"].map(_scaffold)
                if compute_rdkit_scaffolds
                else molecule_manifest["canonical_nonisomeric_smiles"].fillna(
                    molecule_manifest["canonical_isomeric_smiles"]
                )
            ),
            strict=True,
        )
    )
    rows["scaffold_group"] = rows["molecule_id"].map(scaffold_map)
    stats = []
    for family in SPLIT_FAMILIES:
        for partition, subset in rows.groupby(family, dropna=False):
            entry: dict[str, Any] = {
                "split_family": family,
                "partition": partition,
                "rows": int(len(subset)),
                "molecules": int(subset["molecule_id"].nunique()),
                "solvents": int(subset["solvent_id"].nunique()),
                "scaffolds": int(subset["scaffold_group"].nunique()),
            }
            for target in target_columns:
                available = f"{target}_available"
                if available in subset.columns:
                    entry[f"{target}_available_rows"] = int(subset[available].sum())
                    entry[f"{target}_coverage_fraction"] = float(subset[available].mean()) if len(subset) else math.nan
            stats.append(entry)
    return pd.DataFrame(stats).sort_values(["split_family", "partition"]).reset_index(drop=True)


def training_normalization_statistics(
    row_manifest: pd.DataFrame,
    split_assignments: pd.DataFrame,
    target_columns: Iterable[str],
) -> pd.DataFrame:
    """Compute target normalization statistics from training rows only."""
    rows = row_manifest.merge(split_assignments, on="row_id", how="left", validate="one_to_one")
    stats = []
    for family in SPLIT_FAMILIES:
        train = rows[rows[family] == "train"]
        for target in target_columns:
            values = pd.to_numeric(train[target], errors="coerce").dropna()
            stats.append(
                {
                    "split_family": family,
                    "target": target,
                    "train_available_rows": int(len(values)),
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "std": float(values.std(ddof=0)) if len(values) else math.nan,
                }
            )
    return pd.DataFrame(stats)


def validate_manifest_reconciliation(bundle: ManifestBundle) -> None:
    """Raise when manifest counts do not reconcile with the processed table."""
    row_manifest = bundle.row_manifest
    molecule_manifest = bundle.molecule_manifest
    if len(row_manifest) != bundle.metadata["source_rows"]:
        raise ValueError("Row manifest count does not match source row count.")
    if row_manifest["row_id"].nunique() != len(row_manifest):
        raise ValueError("Row manifest contains duplicate row_id values.")
    if molecule_manifest["molecule_id"].nunique() != len(molecule_manifest):
        raise ValueError("Molecule manifest contains duplicate molecule_id values.")
    source_counts = row_manifest.groupby("molecule_id").size().rename("actual").reset_index()
    expected = molecule_manifest[["molecule_id", "source_row_count"]]
    compared = expected.merge(source_counts, on="molecule_id", how="left")
    if not (compared["source_row_count"] == compared["actual"]).all():
        raise ValueError("Molecule source_row_count values do not match row manifest.")


def write_manifest_outputs(
    output_dir: Path,
    bundle: ManifestBundle,
    split_assignments: pd.DataFrame,
    leakage_audit: pd.DataFrame,
    stats: pd.DataFrame,
    normalization: pd.DataFrame,
) -> None:
    """Persist all manifest artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle.molecule_manifest.to_csv(output_dir / "molecule_manifest.csv", index=False)
    bundle.row_manifest.to_csv(output_dir / "row_manifest.csv", index=False)
    split_assignments.to_csv(output_dir / "split_assignments.csv", index=False)
    leakage_audit.to_csv(output_dir / "split_leakage_audit.csv", index=False)
    stats.to_csv(output_dir / "split_statistics.csv", index=False)
    normalization.to_csv(output_dir / "training_normalization_statistics.csv", index=False)
    metadata = dict(bundle.metadata)
    metadata["all_leakage_audits_passed"] = bool(leakage_audit["passed"].all())
    (output_dir / "manifest_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
