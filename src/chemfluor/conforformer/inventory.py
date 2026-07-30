"""Inventory construction for full-dataset ConforFormer embedding runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import pandas as pd


INVENTORY_SCHEMA_VERSION = 1
DEFAULT_DATASET = Path("data/processed/fluodb_lite/combined_deduplicated.csv")
DEFAULT_SMILES_COLUMN = "canonical_chromophore_smiles"
DEFAULT_SOLVENT_COLUMN = "canonical_solvent_smiles"
DEFAULT_SHARD_SIZE = 128


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def fluorcast_git_commit(root: Path | str | None = None) -> str:
    repo = Path.cwd() if root is None else Path(root)
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def atomic_write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
        Path(temp_name).replace(path)
    except Exception:
        if temp_name is not None:
            tmp = Path(temp_name)
            if tmp.exists():
                tmp.unlink()
        raise
    return path


@dataclass(frozen=True)
class InventoryBuildResult:
    inventory_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


def molecule_id(canonical_smiles: str) -> str:
    return "mol_" + hashlib.sha256(canonical_smiles.encode("utf-8")).hexdigest()[:24]


def build_inventory_frame(
    rows: pd.DataFrame,
    *,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_molecules: int | None = None,
) -> pd.DataFrame:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if max_molecules is not None and max_molecules <= 0:
        raise ValueError("max_molecules must be positive")
    if smiles_column not in rows.columns:
        raise ValueError(f"missing required column: {smiles_column}")

    valid = rows[rows[smiles_column].notna()].copy()
    valid[smiles_column] = valid[smiles_column].astype(str).str.strip()
    valid = valid[valid[smiles_column] != ""]
    counts = valid.groupby(smiles_column, sort=True).size().rename("source_row_count")
    inventory = counts.reset_index().sort_values(smiles_column, kind="mergesort").reset_index(drop=True)
    if max_molecules is not None:
        inventory = inventory.iloc[:max_molecules].copy()
    inventory.insert(0, "molecule_index", range(len(inventory)))
    inventory.insert(1, "molecule_id", inventory[smiles_column].map(molecule_id))
    inventory["shard_index"] = inventory["molecule_index"] // int(shard_size)
    return inventory[
        [
            "molecule_index",
            "molecule_id",
            smiles_column,
            "shard_index",
            "source_row_count",
        ]
    ]


def build_inventory(
    *,
    source_csv: Path | str = DEFAULT_DATASET,
    output_dir: Path | str,
    smiles_column: str = DEFAULT_SMILES_COLUMN,
    solvent_column: str = DEFAULT_SOLVENT_COLUMN,
    shard_size: int = DEFAULT_SHARD_SIZE,
    max_molecules: int | None = None,
    git_root: Path | str | None = None,
) -> InventoryBuildResult:
    source_csv = Path(source_csv)
    output_dir = Path(output_dir)
    if not source_csv.exists():
        raise FileNotFoundError(f"source CSV not found: {source_csv}")
    rows = pd.read_csv(source_csv, low_memory=False)
    inventory = build_inventory_frame(
        rows,
        smiles_column=smiles_column,
        shard_size=shard_size,
        max_molecules=max_molecules,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "molecule_inventory.csv"
    manifest_path = output_dir / "inventory_manifest.json"

    tmp_csv = inventory_path.with_name(f".{inventory_path.name}.tmp")
    inventory.to_csv(tmp_csv, index=False)
    tmp_csv.replace(inventory_path)

    manifest = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "source_csv_path": str(source_csv),
        "source_csv_sha256": sha256_file(source_csv),
        "inventory_csv_sha256": sha256_file(inventory_path),
        "source_row_count": int(len(rows)),
        "unique_molecule_count": int(len(inventory)),
        "shard_size": int(shard_size),
        "shard_count": int(inventory["shard_index"].max() + 1) if len(inventory) else 0,
        "columns": {
            "chromophore": smiles_column,
            "solvent": solvent_column,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fluorcast_git_commit": fluorcast_git_commit(git_root),
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return InventoryBuildResult(inventory_path=inventory_path, manifest_path=manifest_path, manifest=manifest)


def load_inventory(run_root: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    inventory_dir = Path(run_root) / "inventory"
    inventory_path = inventory_dir / "molecule_inventory.csv"
    manifest_path = inventory_dir / "inventory_manifest.json"
    if not inventory_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"missing inventory artifacts under {inventory_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256_file(inventory_path)
    expected = manifest.get("inventory_csv_sha256")
    if actual != expected:
        raise ValueError("inventory CSV SHA-256 does not match inventory manifest")
    inventory = pd.read_csv(inventory_path)
    required = {"molecule_index", "molecule_id", DEFAULT_SMILES_COLUMN, "shard_index", "source_row_count"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"inventory CSV missing columns: {sorted(missing)}")
    return inventory, manifest

