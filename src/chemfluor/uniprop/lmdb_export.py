"""Export FluorCast UniProp manifests to upstream-compatible LMDB files."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from rdkit import Chem

from .geometry_cache import GEOMETRY_SCHEMA_VERSION, read_valid_cache

LMDB_SCHEMA_VERSION = "fluorcast_uniprop_lmdb_v1"
DEFAULT_TARGET_COLUMNS = [
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
    "stokes_shift_nm",
]
UPSTREAM_REVISION_FILE = Path("third_party/nablacolors.REVISION")


def _require_lmdb() -> Any:
    try:
        import lmdb  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "lmdb is required to export or validate UniProp LMDB files. "
            "Install it in the active environment before running this step."
        ) from exc
    return lmdb


def encode_int_key(index: int, nbytes: int = 8) -> bytes:
    """Encode deterministic integer LMDB keys in upstream-compatible big-endian form."""
    if index < 0:
        raise ValueError("LMDB key index must be non-negative.")
    return int(index).to_bytes(nbytes, byteorder="big", signed=False)


def decode_int_key(key: bytes) -> int:
    return int.from_bytes(key, byteorder="big", signed=False)


def file_sha256(path: Path) -> str:
    """Hash one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def table_hash(path: Path) -> str:
    return file_sha256(path)


def upstream_revision(revision_file: Path = UPSTREAM_REVISION_FILE) -> str | None:
    if not revision_file.exists():
        return None
    for line in revision_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("commit="):
            return line.split("=", 1)[1].strip()
    return None


allowable_features = {
    "possible_atomic_num_list": list(range(1, 119)) + ["misc"],
    "possible_chirality_list": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
        "CHI_SQUAREPLANAR",
        "CHI_OTHER",
    ],
    "possible_degree_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "misc"],
    "possible_formal_charge_list": [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "misc"],
    "possible_numH_list": [0, 1, 2, 3, 4, 5, 6, 7, 8, "misc"],
    "possible_number_radical_e_list": [0, 1, 2, 3, 4, "misc"],
    "possible_hybridization_list": ["SP", "SP2", "SP3", "SP3D", "SP3D2", "misc"],
    "possible_is_aromatic_list": [False, True],
    "possible_is_in_ring_list": [False, True],
    "possible_bond_type_list": ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "misc"],
    "possible_bond_stereo_list": [
        "STEREONONE",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
        "STEREOANY",
    ],
    "possible_is_conjugated_list": [False, True],
}


def safe_index(values: Iterable[Any], element: Any) -> int:
    options = list(values)
    try:
        return options.index(element)
    except ValueError:
        return len(options) - 1


def atom_to_feature_vector(atom: Chem.Atom) -> list[int]:
    """Encode atom features exactly like pinned nablaColors get_3d_lmdb.py."""
    return [
        safe_index(allowable_features["possible_atomic_num_list"], atom.GetAtomicNum()),
        allowable_features["possible_chirality_list"].index(str(atom.GetChiralTag())),
        safe_index(allowable_features["possible_degree_list"], atom.GetTotalDegree()),
        safe_index(allowable_features["possible_formal_charge_list"], atom.GetFormalCharge()),
        safe_index(allowable_features["possible_numH_list"], atom.GetTotalNumHs()),
        safe_index(
            allowable_features["possible_number_radical_e_list"],
            atom.GetNumRadicalElectrons(),
        ),
        safe_index(allowable_features["possible_hybridization_list"], str(atom.GetHybridization())),
        allowable_features["possible_is_aromatic_list"].index(atom.GetIsAromatic()),
        allowable_features["possible_is_in_ring_list"].index(atom.IsInRing()),
    ]


def bond_to_feature_vector(bond: Chem.Bond) -> list[int]:
    """Encode bond features exactly like pinned nablaColors get_3d_lmdb.py."""
    return [
        safe_index(allowable_features["possible_bond_type_list"], str(bond.GetBondType())),
        allowable_features["possible_bond_stereo_list"].index(str(bond.GetStereo())),
        allowable_features["possible_is_conjugated_list"].index(bond.GetIsConjugated()),
    ]


def get_graph(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return upstream-compatible node_attr, edge_index, edge_attr arrays."""
    node_attr = np.asarray(
        [atom_to_feature_vector(atom) for atom in mol.GetAtoms()],
        dtype=np.int32,
    )
    if mol.GetNumBonds() == 0:
        return node_attr, np.empty((2, 0), dtype=np.int32), np.empty((0, 3), dtype=np.int32)

    edges: list[tuple[int, int]] = []
    edge_features: list[list[int]] = []
    for bond in mol.GetBonds():
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        features = bond_to_feature_vector(bond)
        edges.append((i, j))
        edge_features.append(features)
        edges.append((j, i))
        edge_features.append(features)
    return (
        node_attr,
        np.asarray(edges, dtype=np.int32).T,
        np.asarray(edge_features, dtype=np.int32),
    )


@dataclass(frozen=True)
class ExportConfig:
    """LMDB export configuration."""

    row_manifest_path: Path
    molecule_manifest_path: Path
    split_assignments_path: Path
    geometry_cache_dir: Path
    output_dir: Path
    split_family: str
    seed: int
    target_columns: tuple[str, ...]
    map_size: int
    batch_size: int
    overwrite: bool
    resume: bool
    valid_size: float = 0.1


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def load_export_inputs(config: ExportConfig) -> pd.DataFrame:
    """Load and join manifests; targets come only from row_manifest."""
    rows = _read_csv(config.row_manifest_path, "row manifest")
    molecules = _read_csv(config.molecule_manifest_path, "molecule manifest")
    splits = _read_csv(config.split_assignments_path, "split assignments")
    required_rows = {"row_id", "molecule_id", "solvent_id", "canonical_solvent_smiles"}
    missing = sorted(required_rows.difference(rows.columns))
    if missing:
        raise ValueError(f"Row manifest is missing required column(s): {missing}")
    required_molecules = {"molecule_id", "canonical_isomeric_smiles"}
    missing_mols = sorted(required_molecules.difference(molecules.columns))
    if missing_mols:
        raise ValueError(f"Molecule manifest is missing required column(s): {missing_mols}")
    if config.split_family not in splits.columns:
        raise ValueError(f"Split assignments are missing split family: {config.split_family}")

    merged = rows.merge(
        molecules[["molecule_id", "canonical_isomeric_smiles"]],
        on="molecule_id",
        how="left",
        validate="many_to_one",
    ).merge(
        splits[["row_id", config.split_family]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if merged["canonical_isomeric_smiles"].isna().any():
        raise ValueError("Some row manifest molecule IDs are missing from molecule_manifest.")
    if merged[config.split_family].isna().any():
        raise ValueError("Some row IDs are missing split assignments.")
    merged["lmdb_partition"] = merged[config.split_family].map(
        lambda value: "test" if value == "test" else "train"
    )
    if "valid" not in set(merged["lmdb_partition"]):
        train_rows = merged[merged["lmdb_partition"] == "train"].copy()
        if len(train_rows) and config.valid_size > 0:
            n_valid = max(1, int(round(len(train_rows) * config.valid_size)))
            scored = train_rows[["row_id"]].copy()
            scored["_score"] = scored["row_id"].map(
                lambda row_id: hashlib.sha256(
                    f"{LMDB_SCHEMA_VERSION}|valid|{config.seed}|{row_id}".encode("utf-8")
                ).hexdigest()
            )
            valid_ids = set(
                scored.sort_values(["_score", "row_id"], kind="mergesort")
                .head(n_valid)["row_id"]
                .astype(str)
            )
            merged.loc[merged["row_id"].astype(str).isin(valid_ids), "lmdb_partition"] = "valid"
    return merged.sort_values("row_id", kind="mergesort").reset_index(drop=True)


def _target_values(row: pd.Series, target_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    values = []
    mask = []
    for target in target_columns:
        value = row[target] if target in row.index else np.nan
        if pd.isna(value):
            values.append(np.nan)
            mask.append(False)
        else:
            values.append(float(value))
            mask.append(True)
    return np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.bool_)


def _geometry_record(row: pd.Series, geometry_cache_dir: Path) -> dict[str, Any]:
    molecule_id = str(row["molecule_id"])
    smiles = str(row["canonical_isomeric_smiles"])
    path = geometry_cache_dir / f"{molecule_id}.json"
    return read_valid_cache(path, molecule_id, smiles)


def build_lmdb_record(
    row: pd.Series,
    geometry: dict[str, Any],
    target_columns: tuple[str, ...],
    integer_id: int,
) -> dict[str, Any]:
    """Build one gzip-pickle record matching the upstream PCQ loader fields."""
    smiles = str(row["canonical_isomeric_smiles"])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid canonical chromophore SMILES: {smiles}")
    mol = Chem.RemoveHs(mol)
    atoms = np.asarray(geometry["atom_symbols"])
    coordinates = np.asarray(geometry["coordinates"], dtype=np.float32)
    if coordinates.shape != (len(atoms), 3):
        raise ValueError(f"Bad geometry coordinate shape for {row['row_id']}: {coordinates.shape}")

    node_attr, edge_index, edge_attr = get_graph(mol)
    if node_attr.shape[0] != len(atoms):
        raise ValueError("Graph node count does not match cached atom count.")
    target, target_mask = _target_values(row, target_columns)
    solvent = row["canonical_solvent_smiles"]
    solvent_smi = "" if pd.isna(solvent) else str(solvent)
    return {
        "atoms": atoms,
        "input_pos": [coordinates],
        "label_pos": coordinates,
        "smi": smiles,
        "solvent_smi": solvent_smi,
        "node_attr": node_attr.astype(np.int32, copy=False),
        "edge_index": edge_index.astype(np.int32, copy=False),
        "edge_attr": edge_attr.astype(np.int32, copy=False),
        "target": target,
        "target_mask": target_mask,
        "target_columns": np.asarray(target_columns),
        "row_id": str(row["row_id"]),
        "molecule_id": str(row["molecule_id"]),
        "solvent_id": str(row["solvent_id"]),
        "id": int(integer_id),
        "geometry_cache_schema": geometry.get("schema_version"),
    }


def _write_lmdb(path: Path, records: list[dict[str, Any]], map_size: int, batch_size: int) -> None:
    lmdb = _require_lmdb()
    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(map_size),
    )
    try:
        txn = env.begin(write=True)
        for index, record in enumerate(records):
            key = encode_int_key(int(record["id"]))
            value = gzip.compress(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            txn.put(key, value, overwrite=False)
            if (index + 1) % batch_size == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        env.sync()
    finally:
        env.close()


def _atomic_replace_lmdb(tmp_lmdb: Path, final_lmdb: Path) -> None:
    os.replace(tmp_lmdb, final_lmdb)


def _completion_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".complete")


def _metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".metadata.json")


def _partition_name(name: str) -> str:
    return "valid" if name == "validation" else name


def export_uniprop_lmdb(config: ExportConfig) -> dict[str, Any]:
    """Export train/valid/test LMDB files for one split family."""
    rows = load_export_inputs(config)
    partitions = {"train": "train", "valid": "valid", "test": "test"}
    config.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "schema_version": LMDB_SCHEMA_VERSION,
        "split_family": config.split_family,
        "seed": int(config.seed),
        "target_columns": list(config.target_columns),
        "partitions": {},
    }
    for partition in partitions:
        subset = rows[rows["lmdb_partition"] == partition].copy()
        out_path = config.output_dir / f"{partition}.lmdb"
        marker = _completion_marker(out_path)
        if config.resume and marker.exists() and out_path.exists():
            validation = validate_lmdb(
                out_path,
                expected_row_ids=set(subset["row_id"].astype(str)),
                target_columns=config.target_columns,
            )
            if validation["valid"]:
                results["partitions"][partition] = validation
                continue
        if out_path.exists() and not config.overwrite:
            raise FileExistsError(f"LMDB exists and --overwrite was not set: {out_path}")

        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        records = []
        for integer_id, (_, row) in enumerate(subset.iterrows()):
            geometry = _geometry_record(row, config.geometry_cache_dir)
            records.append(build_lmdb_record(row, geometry, config.target_columns, integer_id))
        _write_lmdb(tmp_path, records, config.map_size, config.batch_size)
        validation = validate_lmdb(
            tmp_path,
            expected_row_ids=set(subset["row_id"].astype(str)),
            target_columns=config.target_columns,
            require_complete_marker=False,
        )
        if not validation["valid"]:
            raise ValueError(f"Validation failed for temporary LMDB {tmp_path}: {validation}")
        _atomic_replace_lmdb(tmp_path, out_path)
        marker.write_text(json.dumps({"completed_at": _utc_now(), "rows": len(records)}), encoding="utf-8")
        results["partitions"][partition] = validate_lmdb(
            out_path,
            expected_row_ids=set(subset["row_id"].astype(str)),
            target_columns=config.target_columns,
        )

    metadata = build_metadata(config, rows, results)
    (config.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def build_metadata(config: ExportConfig, rows: pd.DataFrame, results: dict[str, Any]) -> dict[str, Any]:
    """Build the LMDB export sidecar metadata."""
    target_counts = {
        target: int(pd.to_numeric(rows[target], errors="coerce").notna().sum())
        for target in config.target_columns
        if target in rows.columns
    }
    return {
        "schema_version": LMDB_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "source_dataset_hash": table_hash(config.row_manifest_path),
        "manifest_hash": table_hash(config.molecule_manifest_path),
        "split_hash": table_hash(config.split_assignments_path),
        "geometry_cache_schema": GEOMETRY_SCHEMA_VERSION,
        "upstream_revision": upstream_revision(),
        "row_counts": {
            "total_selected": int(len(rows)),
            **{
                partition: int((rows["lmdb_partition"] == partition).sum())
                for partition in ["train", "valid", "test"]
            },
        },
        "target_counts": target_counts,
        "creation_config": {
            "row_manifest_path": str(config.row_manifest_path),
            "molecule_manifest_path": str(config.molecule_manifest_path),
            "split_assignments_path": str(config.split_assignments_path),
            "geometry_cache_dir": str(config.geometry_cache_dir),
            "output_dir": str(config.output_dir),
            "split_family": config.split_family,
            "seed": config.seed,
            "target_columns": list(config.target_columns),
            "map_size": config.map_size,
            "batch_size": config.batch_size,
            "valid_size": config.valid_size,
        },
        "partition_reports": results["partitions"],
    }


def read_lmdb_records(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    """Read all gzip-pickled records from one LMDB in key order."""
    lmdb = _require_lmdb()
    if not path.exists():
        raise FileNotFoundError(f"LMDB not found: {path}")
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin(write=False) as txn:
            records = []
            for key, value in txn.cursor():
                records.append((bytes(key), pickle.loads(gzip.decompress(bytes(value)))))
            return records
    finally:
        env.close()


def validate_record(record: dict[str, Any], target_columns: tuple[str, ...]) -> list[str]:
    """Validate one decoded UniProp LMDB record."""
    errors = []
    required = {
        "atoms",
        "input_pos",
        "label_pos",
        "smi",
        "solvent_smi",
        "node_attr",
        "edge_index",
        "edge_attr",
        "target",
        "target_mask",
        "row_id",
        "molecule_id",
    }
    missing = sorted(required.difference(record))
    if missing:
        errors.append(f"missing keys: {missing}")
        return errors
    atoms = np.asarray(record["atoms"])
    input_pos = record["input_pos"]
    label_pos = np.asarray(record["label_pos"])
    node_attr = np.asarray(record["node_attr"])
    edge_index = np.asarray(record["edge_index"])
    edge_attr = np.asarray(record["edge_attr"])
    target = np.asarray(record["target"])
    target_mask = np.asarray(record["target_mask"])
    if len(input_pos) < 1:
        errors.append("input_pos must contain at least one conformer")
    else:
        first_pos = np.asarray(input_pos[0])
        if first_pos.shape != (len(atoms), 3):
            errors.append(f"input_pos shape {first_pos.shape} does not match atom count {len(atoms)}")
    if label_pos.shape != (len(atoms), 3):
        errors.append(f"label_pos shape {label_pos.shape} does not match atom count {len(atoms)}")
    if node_attr.dtype != np.int32 or node_attr.shape != (len(atoms), 9):
        errors.append(f"node_attr shape/dtype invalid: {node_attr.shape}/{node_attr.dtype}")
    if edge_index.dtype != np.int32 or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        errors.append(f"edge_index shape/dtype invalid: {edge_index.shape}/{edge_index.dtype}")
    if edge_attr.dtype != np.int32 or edge_attr.ndim != 2 or edge_attr.shape[1] != 3:
        errors.append(f"edge_attr shape/dtype invalid: {edge_attr.shape}/{edge_attr.dtype}")
    if target.shape != (len(target_columns),):
        errors.append(f"target shape {target.shape} does not match target columns")
    if target_mask.shape != (len(target_columns),):
        errors.append(f"target_mask shape {target_mask.shape} does not match target columns")
    if not np.asarray(target_mask).dtype == np.bool_:
        errors.append("target_mask must be bool")
    if np.any((~target_mask.astype(bool)) & ~np.isnan(target.astype(float))):
        errors.append("masked missing targets must be NaN, not numeric substitutes")
    return errors


def validate_lmdb(
    path: Path,
    *,
    expected_row_ids: set[str] | None = None,
    target_columns: tuple[str, ...] = tuple(DEFAULT_TARGET_COLUMNS),
    require_complete_marker: bool = True,
) -> dict[str, Any]:
    """Validate one exported LMDB."""
    marker = _completion_marker(path)
    errors = []
    if require_complete_marker and not marker.exists() and path.exists():
        errors.append("completion marker is missing")
    records = read_lmdb_records(path)
    row_ids: list[str] = []
    keys: list[int] = []
    for key, record in records:
        keys.append(decode_int_key(key))
        row_ids.append(str(record.get("row_id")))
        errors.extend(f"{record.get('row_id', '<unknown>')}: {error}" for error in validate_record(record, target_columns))
    if keys != sorted(keys):
        errors.append("LMDB keys are not sorted")
    if len(row_ids) != len(set(row_ids)):
        errors.append("duplicate row IDs detected")
    if expected_row_ids is not None and set(row_ids) != expected_row_ids:
        errors.append("row IDs do not reconcile with expected manifest partition")
    target_counts = {target: 0 for target in target_columns}
    for _, record in records:
        mask = np.asarray(record["target_mask"]).astype(bool)
        for index, target in enumerate(target_columns):
            if index < len(mask) and mask[index]:
                target_counts[target] += 1
    return {
        "valid": not errors,
        "path": str(path),
        "rows": len(records),
        "row_ids": row_ids,
        "target_counts": target_counts,
        "errors": errors,
    }
