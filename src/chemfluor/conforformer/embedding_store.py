"""Atomic ConforFormer embedding shard storage and finalization."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .inventory import atomic_write_text, fluorcast_git_commit, load_inventory, sha256_file, sha256_payload
from .pooling import pool_all, pooling_configuration


EMBEDDING_SHARD_SCHEMA_VERSION = 1
FINAL_EMBEDDING_SCHEMA_VERSION = 1
EXPECTED_EMBEDDING_DIM = 512


class EmbeddingStoreError(RuntimeError):
    """Raised when embedding artifacts fail integrity or provenance checks."""


def shard_npz_path(run_root: Path | str, shard_index: int) -> Path:
    return Path(run_root) / "embeddings" / f"shard_{int(shard_index):05d}.npz"


def shard_done_path(run_root: Path | str, shard_index: int) -> Path:
    return Path(run_root) / "embeddings" / f"shard_{int(shard_index):05d}.done.json"


def status_path(run_root: Path | str, shard_index: int) -> Path:
    return Path(run_root) / "conformer_status" / f"shard_{int(shard_index):05d}.json"


def expected_identity(
    *,
    inventory_manifest: dict[str, Any],
    checkpoint_sha256: str,
    dictionary_sha256: str,
    upstream_commit: str,
    architecture_payload: dict[str, Any],
    preprocessing_payload: dict[str, Any],
    conformer_config_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_sha256": inventory_manifest["source_csv_sha256"],
        "inventory_sha256": inventory_manifest["inventory_csv_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
        "dictionary_sha256": dictionary_sha256,
        "pinned_upstream_conforformer_commit": upstream_commit,
        "architecture_configuration_identity": architecture_payload,
        "preprocessing_version": preprocessing_payload.get("preprocess_version"),
        "preprocessing_configuration": preprocessing_payload,
        "conformer_configuration_hash": sha256_payload(conformer_config_payload),
        "conformer_configuration": conformer_config_payload,
        "pooling_configuration": pooling_configuration(),
    }


def validate_done_manifest(
    manifest: dict[str, Any],
    *,
    shard_index: int,
    expected_molecule_count: int,
    identity: dict[str, Any],
    npz_path: Path,
) -> None:
    if manifest.get("schema_version") != EMBEDDING_SHARD_SCHEMA_VERSION:
        raise EmbeddingStoreError("unsupported embedding shard schema version")
    if int(manifest.get("shard_index", -1)) != int(shard_index):
        raise EmbeddingStoreError("embedding shard index mismatch")
    if int(manifest.get("expected_molecule_count", -1)) != int(expected_molecule_count):
        raise EmbeddingStoreError("embedding shard expected molecule count mismatch")
    if int(manifest.get("embedding_dimension", -1)) != EXPECTED_EMBEDDING_DIM:
        raise EmbeddingStoreError("embedding dimension must be 512")
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise EmbeddingStoreError(f"embedding shard provenance mismatch: {key}")
    if not npz_path.exists():
        raise EmbeddingStoreError("embedding NPZ is missing")
    if manifest.get("npz_sha256") != sha256_file(npz_path):
        raise EmbeddingStoreError("embedding NPZ SHA-256 mismatch")
    if manifest.get("status") != "complete":
        raise EmbeddingStoreError("embedding shard is not complete")


def load_valid_embedding_shard(
    run_root: Path | str,
    shard_index: int,
    *,
    expected_molecule_count: int,
    identity: dict[str, Any],
) -> tuple[np.lib.npyio.NpzFile, dict[str, Any]]:
    npz_path = shard_npz_path(run_root, shard_index)
    done_path = shard_done_path(run_root, shard_index)
    if not done_path.exists():
        raise EmbeddingStoreError("embedding done manifest is missing")
    manifest = json.loads(done_path.read_text(encoding="utf-8"))
    validate_done_manifest(
        manifest,
        shard_index=shard_index,
        expected_molecule_count=expected_molecule_count,
        identity=identity,
        npz_path=npz_path,
    )
    data = np.load(npz_path, allow_pickle=False)
    validate_npz_arrays(data, expected_molecule_count=expected_molecule_count)
    return data, manifest


def shard_is_complete(run_root: Path | str, shard_index: int, *, expected_molecule_count: int, identity: dict[str, Any]) -> bool:
    try:
        data, _ = load_valid_embedding_shard(
            run_root,
            shard_index,
            expected_molecule_count=expected_molecule_count,
            identity=identity,
        )
        data.close()
        return True
    except Exception:
        return False


def validate_npz_arrays(data: Any, *, expected_molecule_count: int) -> None:
    required = {
        "molecule_ids",
        "canonical_smiles",
        "molecule_offsets",
        "conformer_ids",
        "conformer_embeddings",
        "conformer_energies_kcal_mol",
        "mean_embeddings",
        "lowest_energy_embeddings",
        "boltzmann_298k_embeddings",
        "boltzmann_fallback_reasons",
        "statuses",
        "failure_codes",
        "failure_messages",
    }
    missing = required.difference(data.files)
    if missing:
        raise EmbeddingStoreError(f"embedding NPZ missing arrays: {sorted(missing)}")
    molecule_ids = data["molecule_ids"]
    offsets = data["molecule_offsets"]
    conformer_embeddings = data["conformer_embeddings"]
    energies = data["conformer_energies_kcal_mol"]
    if molecule_ids.shape[0] != expected_molecule_count:
        raise EmbeddingStoreError("molecule count mismatch")
    if offsets.shape != (expected_molecule_count + 1,):
        raise EmbeddingStoreError("molecule_offsets shape mismatch")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise EmbeddingStoreError("molecule_offsets must be monotonic and start at zero")
    if conformer_embeddings.ndim != 2 or conformer_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise EmbeddingStoreError("conformer_embeddings must have shape [n, 512]")
    if conformer_embeddings.shape[0] != int(offsets[-1]):
        raise EmbeddingStoreError("offsets do not match conformer embedding rows")
    if energies.shape != (conformer_embeddings.shape[0],):
        raise EmbeddingStoreError("energy count does not match conformer embeddings")
    for name in ("mean_embeddings", "lowest_energy_embeddings", "boltzmann_298k_embeddings"):
        array = data[name]
        if array.shape != (expected_molecule_count, EXPECTED_EMBEDDING_DIM):
            raise EmbeddingStoreError(f"{name} shape mismatch")
    statuses = data["statuses"].astype(str)
    success_mask = statuses == "success"
    if not np.isfinite(conformer_embeddings).all():
        raise EmbeddingStoreError("conformer embeddings contain non-finite values")
    for name in ("mean_embeddings", "lowest_energy_embeddings", "boltzmann_298k_embeddings"):
        if not np.isfinite(data[name][success_mask]).all():
            raise EmbeddingStoreError(f"{name} contains non-finite successful embeddings")
    if int(np.sum((statuses == "success") | (statuses == "terminal_failure"))) != expected_molecule_count:
        raise EmbeddingStoreError("each molecule must be success or terminal_failure")


def write_embedding_shard(
    *,
    run_root: Path | str,
    shard_index: int,
    rows: pd.DataFrame,
    conformer_ids_by_molecule: list[list[str]],
    embeddings_by_molecule: list[np.ndarray | None],
    energies_by_molecule: list[np.ndarray | None],
    failure_codes: list[str | None],
    failure_messages: list[str | None],
    identity: dict[str, Any],
) -> tuple[Path, Path]:
    run_root = Path(run_root)
    npz_path = shard_npz_path(run_root, shard_index)
    done_path = shard_done_path(run_root, shard_index)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    molecule_ids = rows["molecule_id"].astype(str).to_numpy()
    canonical_smiles = rows["canonical_chromophore_smiles"].astype(str).to_numpy()
    offsets = [0]
    all_conformer_ids: list[str] = []
    all_embeddings: list[np.ndarray] = []
    all_energies: list[float] = []
    mean_rows: list[np.ndarray] = []
    lowest_rows: list[np.ndarray] = []
    boltz_rows: list[np.ndarray] = []
    fallback_reasons: list[str] = []
    statuses: list[str] = []
    codes: list[str] = []
    messages: list[str] = []

    zero = np.zeros(EXPECTED_EMBEDDING_DIM, dtype=np.float32)
    for idx, _row in rows.reset_index(drop=True).iterrows():
        emb = embeddings_by_molecule[idx]
        energies = energies_by_molecule[idx]
        ids = conformer_ids_by_molecule[idx]
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32)
        if emb is not None and emb.ndim == 2 and emb.shape[1] == EXPECTED_EMBEDDING_DIM and emb.shape[0] > 0:
            if not np.isfinite(emb).all():
                raise EmbeddingStoreError("successful embeddings must be finite")
            energy = np.asarray(energies if energies is not None else np.full(emb.shape[0], np.nan), dtype=np.float64)
            pooled = pool_all(emb, energy)
            statuses.append("success")
            codes.append("")
            messages.append("")
            all_conformer_ids.extend(ids)
            all_embeddings.append(emb)
            all_energies.extend(float(value) for value in energy)
            mean_rows.append(pooled.mean)
            lowest_rows.append(pooled.lowest_energy)
            boltz_rows.append(pooled.boltzmann_298k)
            fallback_reasons.append(pooled.boltzmann_fallback_reason or "")
            offsets.append(offsets[-1] + emb.shape[0])
        else:
            statuses.append("terminal_failure")
            codes.append(str(failure_codes[idx] or "embedding_failed"))
            messages.append(str(failure_messages[idx] or "embedding failed"))
            mean_rows.append(zero)
            lowest_rows.append(zero)
            boltz_rows.append(zero)
            fallback_reasons.append("terminal_failure")
            offsets.append(offsets[-1])

    conformer_embeddings = (
        np.vstack(all_embeddings).astype(np.float32)
        if all_embeddings
        else np.empty((0, EXPECTED_EMBEDDING_DIM), dtype=np.float32)
    )
    arrays = {
        "molecule_ids": molecule_ids.astype("U64"),
        "canonical_smiles": canonical_smiles.astype("U2048"),
        "molecule_offsets": np.asarray(offsets, dtype=np.int64),
        "conformer_ids": np.asarray(all_conformer_ids, dtype="U128"),
        "conformer_embeddings": conformer_embeddings,
        "conformer_energies_kcal_mol": np.asarray(all_energies, dtype=np.float64),
        "mean_embeddings": np.vstack(mean_rows).astype(np.float32),
        "lowest_energy_embeddings": np.vstack(lowest_rows).astype(np.float32),
        "boltzmann_298k_embeddings": np.vstack(boltz_rows).astype(np.float32),
        "boltzmann_fallback_reasons": np.asarray(fallback_reasons, dtype="U128"),
        "statuses": np.asarray(statuses, dtype="U32"),
        "failure_codes": np.asarray(codes, dtype="U128"),
        "failure_messages": np.asarray(messages, dtype="U1024"),
    }

    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=npz_path.parent, delete=False, prefix=f".{npz_path.name}.", suffix=".tmp") as handle:
            temp_name = handle.name
        np.savez_compressed(temp_name, **arrays)
        saved = Path(temp_name)
        if saved.suffix != ".npz":
            saved_npz = Path(str(saved) + ".npz")
            saved_npz.replace(npz_path)
            saved.unlink(missing_ok=True)
        else:
            saved.replace(npz_path)
    except Exception:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
            Path(str(temp_name) + ".npz").unlink(missing_ok=True)
        raise

    manifest = {
        "schema_version": EMBEDDING_SHARD_SCHEMA_VERSION,
        "shard_index": int(shard_index),
        "expected_molecule_count": int(len(rows)),
        "processed_molecule_count": int(len(rows)),
        "success_count": int(statuses.count("success")),
        "terminal_failure_count": int(statuses.count("terminal_failure")),
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        **identity,
        "npz_sha256": sha256_file(npz_path),
        "fluorcast_git_commit": fluorcast_git_commit(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
    }
    atomic_write_text(done_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return npz_path, done_path


def finalize_embeddings(
    *,
    run_root: Path | str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    run_root = Path(run_root)
    inventory, inventory_manifest = load_inventory(run_root)
    expected_shards = int(inventory_manifest["shard_count"])
    seen: set[str] = set()
    index_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0

    for shard_index in range(expected_shards):
        shard_rows = inventory[inventory["shard_index"] == shard_index].reset_index(drop=True)
        data, manifest = load_valid_embedding_shard(
            run_root,
            shard_index,
            expected_molecule_count=len(shard_rows),
            identity=identity,
        )
        validate_npz_arrays(data, expected_molecule_count=len(shard_rows))
        molecule_ids = data["molecule_ids"].astype(str)
        canonical = data["canonical_smiles"].astype(str)
        offsets = data["molecule_offsets"]
        statuses = data["statuses"].astype(str)
        for local_idx, molecule_id_value in enumerate(molecule_ids):
            if molecule_id_value in seen:
                raise EmbeddingStoreError(f"duplicate molecule in shards: {molecule_id_value}")
            seen.add(molecule_id_value)
            if molecule_id_value not in set(shard_rows["molecule_id"].astype(str)):
                raise EmbeddingStoreError(f"unknown molecule in shard {shard_index}: {molecule_id_value}")
            row = {
                "molecule_id": molecule_id_value,
                "canonical_chromophore_smiles": canonical[local_idx],
                "shard_index": shard_index,
                "molecule_row": local_idx,
                "conformer_start": int(offsets[local_idx]),
                "conformer_end": int(offsets[local_idx + 1]),
                "status": statuses[local_idx],
                "embedding_npz": str(shard_npz_path(run_root, shard_index)),
            }
            index_rows.append(row)
            if statuses[local_idx] == "success":
                success_count += 1
            else:
                failure_count += 1
                failure_rows.append(
                    {
                        **row,
                        "failure_code": str(data["failure_codes"][local_idx]),
                        "failure_message": str(data["failure_messages"][local_idx]),
                    }
                )
        data.close()

    expected_ids = set(inventory["molecule_id"].astype(str))
    if seen != expected_ids:
        raise EmbeddingStoreError("finalized molecule set does not match inventory")
    if success_count + failure_count != len(inventory):
        raise EmbeddingStoreError("success plus terminal failure count does not equal inventory count")

    pd.DataFrame(index_rows).sort_values("molecule_id").to_csv(run_root / "embedding_index.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(run_root / "failed_molecules.csv", index=False)
    summary = {
        "schema_version": FINAL_EMBEDDING_SCHEMA_VERSION,
        "inventory_molecule_count": int(len(inventory)),
        "success_count": int(success_count),
        "terminal_failure_count": int(failure_count),
        "shard_count": expected_shards,
    }
    manifest = {
        **summary,
        **identity,
        "inventory_manifest_sha256": sha256_file(run_root / "inventory" / "inventory_manifest.json"),
        "embedding_index_sha256": sha256_file(run_root / "embedding_index.csv"),
        "failed_molecules_sha256": sha256_file(run_root / "failed_molecules.csv"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fluorcast_git_commit": fluorcast_git_commit(),
    }
    atomic_write_text(run_root / "embedding_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    atomic_write_text(run_root / "embedding_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

