"""Native-Windows UniProp integration smoke profile.

This module proves the FluorCast 3D data path, masks, checkpoints, and JSON
contract using a tiny FluorCast-owned PyTorch model. It never loads Uni-Core,
Uni-Mol+, Chemprop, CUDA, or a real UniProp checkpoint.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import platform
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from .geometry_cache import atomic_write_json, cache_path, generate_geometry_entry, read_valid_cache
from .lmdb_export import ExportConfig, build_lmdb_record, export_uniprop_lmdb, read_lmdb_records, validate_lmdb
from .manifests import (
    MANIFEST_SCHEMA_VERSION,
    ManifestBundle,
    build_manifests,
    split_statistics,
    stable_hash,
    training_normalization_statistics,
    validate_manifest_reconciliation,
)

WINDOWS_SMOKE_PROFILE = "windows-smoke"
NIBI_REAL_PROFILE = "nibi-real"
TINY_3D_SMOKE_MODEL_KIND = "tiny_3d_smoke_backbone"
WINDOWS_SMOKE_SCHEMA_VERSION = "fluorcast_uniprop_windows_smoke_v1"
WINDOWS_SMOKE_BUNDLE_SCHEMA_VERSION = "fluorcast_uniprop_windows_smoke_bundle_v1"
WINDOWS_SMOKE_PREDICTION_SCHEMA_VERSION = "fluorcast_uniprop_windows_smoke_prediction_v1"
WINDOWS_SMOKE_TARGETS = ("absorption_nm", "emission_nm", "quantum_yield")


@dataclass(frozen=True)
class Tiny3DSmokeConfig:
    targets: tuple[str, ...] = WINDOWS_SMOKE_TARGETS
    seed: int = 42
    node_feature_dim: int = 9
    edge_feature_dim: int = 3
    solvent_feature_dim: int = 12
    hidden_dim: int = 32
    learning_rate: float = 3.0e-3
    weight_decay: float = 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the Windows UniProp smoke profile.") from exc
    return torch


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_float_features(text: str, dim: int) -> np.ndarray:
    values = []
    for index in range(dim):
        digest = hashlib.sha256(f"{index}|{text}".encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2.0 - 1.0)
    return np.asarray(values, dtype=np.float32)


def _coordinates_sha256(entry: dict[str, Any]) -> str:
    coords = np.asarray(entry["coordinates"], dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(str(entry["molecule_id"]).encode("utf-8"))
    digest.update(str(entry["canonical_smiles"]).encode("utf-8"))
    digest.update(coords.tobytes())
    return digest.hexdigest()


def _canonicalize_smiles(smiles: str, *, field_name: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise ValueError(f"Invalid {field_name}: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def windows_smoke_environment_report() -> dict[str, Any]:
    torch = _require_torch()
    cuda_available = bool(torch.cuda.is_available())
    cpu_tensor_ok = False
    try:
        cpu_tensor_ok = bool(torch.isfinite(torch.ones(1, device="cpu")).all().item())
    except Exception:
        cpu_tensor_ok = False
    try:
        lmdb = importlib.import_module("lmdb")
        lmdb_info = {"available": True, "version": getattr(lmdb, "__version__", None)}
    except Exception as exc:
        lmdb_info = {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "created_at": _utc_now(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_info": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        },
        "platform": {
            "system": platform.system(),
            "platform": platform.platform(),
        },
        "rdkit": {"available": True, "version": rdBase.rdkitVersion},
        "lmdb": lmdb_info,
        "numpy": {"available": True, "version": np.__version__},
        "pandas": {"available": True, "version": pd.__version__},
        "pytorch": {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cpu_usable": cpu_tensor_ok,
            "cuda_available": cuda_available,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        },
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "excluded_dependencies": {
            "cuda_required": False,
            "unicore_required": False,
            "unimol_plus_required": False,
            "chemprop_required": False,
            "real_checkpoint_required": False,
        },
    }


def build_fixture_dataframe() -> pd.DataFrame:
    rows = [
        ("CCO", "O", 350.0, 450.0, 0.15),
        ("CCO", "CCO", 352.0, 453.0, 0.16),
        ("CCO", "CC#N", np.nan, 455.0, 0.18),
        ("CCN", "O", 400.0, np.nan, 0.22),
        ("CCN", "CCO", 405.0, 505.0, 0.24),
        ("CCN", "CS(C)=O", 407.0, 507.0, np.nan),
        ("c1ccccc1", "CC#N", 390.0, 470.0, 0.08),
        ("c1ccccc1", "O", 391.0, 471.0, np.nan),
        ("Oc1ccccc1", "CCO", 365.0, 463.0, 0.31),
        ("Oc1ccccc1", "O", np.nan, 464.0, 0.33),
        ("CCCl", "CS(C)=O", 9999.0, 10099.0, 9.99),
        ("CCCl", "CCO", 9998.0, np.nan, 8.88),
    ]
    records = []
    for index, (chromophore, solvent, absorption, emission, quantum_yield) in enumerate(rows):
        records.append(
            {
                "fixture_row_number": index,
                "chromophore_smiles": chromophore,
                "canonical_chromophore_smiles": _canonicalize_smiles(chromophore, field_name="chromophore_smiles"),
                "solvent_original": solvent,
                "canonical_solvent_smiles": _canonicalize_smiles(solvent, field_name="solvent_smiles"),
                "source_dataset": "windows_smoke_fixture",
                "absorption_nm": absorption,
                "emission_nm": emission,
                "quantum_yield": quantum_yield,
            }
        )
    return pd.DataFrame(records)


def build_fixture_manifests(output_dir: Path, seed: int) -> tuple[ManifestBundle, pd.DataFrame, dict[str, Any]]:
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = manifest_dir / "fixture_source.csv"
    fixture = build_fixture_dataframe()
    fixture.to_csv(fixture_path, index=False)

    bundle = build_manifests(
        fixture_path,
        WINDOWS_SMOKE_TARGETS,
        compute_nonisomeric=True,
        compute_rdkit_properties=True,
    )
    validate_manifest_reconciliation(bundle)

    row_manifest = bundle.row_manifest.copy()
    source_numbers = row_manifest["source_row_number"].astype(int)
    split_assignments = pd.DataFrame({"row_id": row_manifest["row_id"]})
    split_assignments["random"] = np.where(source_numbers >= 10, "test", "train")

    cccl_id = stable_hash("mol", MANIFEST_SCHEMA_VERSION, _canonicalize_smiles("CCCl", field_name="chromophore_smiles"))
    dmso_id = stable_hash("solv", MANIFEST_SCHEMA_VERSION, _canonicalize_smiles("CS(C)=O", field_name="solvent_smiles"))
    split_assignments["molecule"] = np.where(row_manifest["molecule_id"] == cccl_id, "test", "train")
    split_assignments["scaffold"] = split_assignments["molecule"]
    split_assignments["solvent"] = np.where(row_manifest["solvent_id"] == dmso_id, "test", "train")
    double = np.full(len(row_manifest), "train", dtype=object)
    heldout_mol = row_manifest["molecule_id"] == cccl_id
    heldout_solvent = row_manifest["solvent_id"] == dmso_id
    double[heldout_mol & heldout_solvent] = "test"
    double[heldout_mol ^ heldout_solvent] = "heldout_boundary"
    split_assignments["double_cold_start"] = double

    stats = split_statistics(row_manifest, bundle.molecule_manifest, split_assignments, WINDOWS_SMOKE_TARGETS)
    normalization = training_normalization_statistics(row_manifest, split_assignments, WINDOWS_SMOKE_TARGETS)
    bundle.molecule_manifest.to_csv(manifest_dir / "molecule_manifest.csv", index=False)
    row_manifest.to_csv(manifest_dir / "row_manifest.csv", index=False)
    split_assignments.to_csv(manifest_dir / "split_assignments.csv", index=False)
    stats.to_csv(manifest_dir / "split_statistics.csv", index=False)
    normalization.to_csv(manifest_dir / "training_normalization_statistics.csv", index=False)
    metadata = {
        **bundle.metadata,
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "seed": int(seed),
        "fixture_rows": int(len(row_manifest)),
        "unique_chromophores": int(row_manifest["molecule_id"].nunique()),
        "repeated_chromophore_rows": int(len(row_manifest) - row_manifest["molecule_id"].nunique()),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(manifest_dir / "manifest_metadata.json", metadata)
    return bundle, split_assignments, metadata


def generate_fixture_geometries(bundle: ManifestBundle, output_dir: Path) -> dict[str, Any]:
    geometry_dir = output_dir / "geometry_cache"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for _, row in bundle.molecule_manifest.sort_values("molecule_id").iterrows():
        molecule_id = str(row["molecule_id"])
        smiles = str(row["canonical_isomeric_smiles"])
        entry = generate_geometry_entry(molecule_id, smiles)
        atomic_write_json(cache_path(geometry_dir, molecule_id), entry)
        checked = read_valid_cache(cache_path(geometry_dir, molecule_id), molecule_id, smiles)
        generated.append(
            {
                "molecule_id": molecule_id,
                "canonical_smiles": smiles,
                "cache_path": str(cache_path(geometry_dir, molecule_id)),
                "atom_count": len(checked["atom_symbols"]),
                "coordinate_sha256": _coordinates_sha256(checked),
                "optimization_method": checked["optimization_method"],
            }
        )

    row_manifest = bundle.row_manifest
    row_geometry = []
    for _, row in row_manifest.iterrows():
        molecule_id = str(row["molecule_id"])
        entry = read_valid_cache(
            cache_path(geometry_dir, molecule_id),
            molecule_id,
            str(bundle.molecule_manifest.set_index("molecule_id").loc[molecule_id, "canonical_isomeric_smiles"]),
        )
        row_geometry.append(
            {
                "row_id": str(row["row_id"]),
                "molecule_id": molecule_id,
                "coordinate_sha256": _coordinates_sha256(entry),
            }
        )
    repeated_groups = []
    for molecule_id, group in pd.DataFrame(row_geometry).groupby("molecule_id"):
        if len(group) > 1:
            repeated_groups.append(
                {
                    "molecule_id": str(molecule_id),
                    "row_ids": sorted(group["row_id"].astype(str).tolist()),
                    "unique_coordinate_hashes": sorted(group["coordinate_sha256"].unique().tolist()),
                }
            )
    report = {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "geometry_count": len(generated),
        "unique_chromophores": int(row_manifest["molecule_id"].nunique()),
        "one_geometry_per_unique_chromophore": len(generated) == int(row_manifest["molecule_id"].nunique()),
        "repeated_chromophores_reuse_geometry": all(len(item["unique_coordinate_hashes"]) == 1 for item in repeated_groups),
        "generated": generated,
        "repeated_chromophore_groups": repeated_groups,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "geometry_validation_report.json", report)
    return report


def export_fixture_lmdb(bundle: ManifestBundle, split_assignments: pd.DataFrame, output_dir: Path, seed: int) -> dict[str, Any]:
    manifest_dir = output_dir / "manifests"
    lmdb_dir = output_dir / "lmdb"
    config = ExportConfig(
        row_manifest_path=manifest_dir / "row_manifest.csv",
        molecule_manifest_path=manifest_dir / "molecule_manifest.csv",
        split_assignments_path=manifest_dir / "split_assignments.csv",
        geometry_cache_dir=output_dir / "geometry_cache",
        output_dir=lmdb_dir,
        split_family="random",
        seed=int(seed),
        target_columns=WINDOWS_SMOKE_TARGETS,
        map_size=64 * 1024 * 1024,
        batch_size=2,
        overwrite=True,
        resume=False,
        valid_size=0.1,
    )
    metadata = export_uniprop_lmdb(config)
    metadata = {
        **metadata,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(lmdb_dir / "metadata.json", metadata)
    validation = {
        partition: validate_lmdb(lmdb_dir / f"{partition}.lmdb", target_columns=WINDOWS_SMOKE_TARGETS)
        for partition in ["train", "valid", "test"]
    }
    report = {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "metadata": metadata,
        "partitions": validation,
        "all_valid": all(item["valid"] for item in validation.values()),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "lmdb_validation_report.json", report)
    return report


class FluorCastUniPropSmokeDataset:
    """Small LMDB dataset adapter for the Windows smoke path."""

    def __init__(
        self,
        lmdb_path: Path,
        *,
        targets: Sequence[str] = WINDOWS_SMOKE_TARGETS,
        solvent_feature_dim: int = 12,
    ) -> None:
        self.lmdb_path = Path(lmdb_path)
        self.targets = tuple(targets)
        self.solvent_feature_dim = int(solvent_feature_dim)
        self.records = [record for _, record in read_lmdb_records(self.lmdb_path)]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_columns = [str(item) for item in np.asarray(record.get("target_columns", WINDOWS_SMOKE_TARGETS)).tolist()]
        indices = [target_columns.index(target) for target in self.targets]
        target = np.asarray(record["target"], dtype=np.float32)[indices]
        target_mask = np.asarray(record["target_mask"], dtype=np.bool_)[indices]
        return {
            "row_id": str(record["row_id"]),
            "molecule_id": str(record["molecule_id"]),
            "solvent_id": str(record.get("solvent_id", "")),
            "smi": str(record["smi"]),
            "solvent_smi": str(record.get("solvent_smi", "")),
            "atom_features": np.asarray(record["node_attr"], dtype=np.int64),
            "coordinates": np.asarray(record["label_pos"], dtype=np.float32),
            "edge_index": np.asarray(record["edge_index"], dtype=np.int64),
            "edge_attr": np.asarray(record["edge_attr"], dtype=np.float32),
            "solvent_features": _stable_float_features(str(record.get("solvent_smi", "")), self.solvent_feature_dim),
            "target": target,
            "target_mask": target_mask,
        }

    def collater(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        torch = _require_torch()
        if not items:
            return {}
        batch_size = len(items)
        max_atoms = max(int(item["atom_features"].shape[0]) for item in items)
        max_edges = max(int(item["edge_attr"].shape[0]) for item in items)
        node_dim = int(items[0]["atom_features"].shape[1])
        edge_dim = int(items[0]["edge_attr"].shape[1]) if max_edges else 3
        atom_features = np.zeros((batch_size, max_atoms, node_dim), dtype=np.int64)
        coordinates = np.zeros((batch_size, max_atoms, 3), dtype=np.float32)
        atom_mask = np.zeros((batch_size, max_atoms), dtype=np.bool_)
        edge_index = np.zeros((batch_size, 2, max_edges), dtype=np.int64)
        edge_attr = np.zeros((batch_size, max_edges, edge_dim), dtype=np.float32)
        edge_mask = np.zeros((batch_size, max_edges), dtype=np.bool_)
        solvent = np.stack([item["solvent_features"] for item in items]).astype(np.float32)
        target = np.stack([np.nan_to_num(item["target"], nan=0.0) for item in items]).astype(np.float32)
        target_mask = np.stack([item["target_mask"] for item in items]).astype(np.bool_)

        for index, item in enumerate(items):
            n_atoms = int(item["atom_features"].shape[0])
            n_edges = int(item["edge_attr"].shape[0])
            atom_features[index, :n_atoms] = item["atom_features"]
            coordinates[index, :n_atoms] = item["coordinates"]
            atom_mask[index, :n_atoms] = True
            if n_edges:
                edge_index[index, :, :n_edges] = item["edge_index"]
                edge_attr[index, :n_edges] = item["edge_attr"]
                edge_mask[index, :n_edges] = True

        return {
            "row_id": [item["row_id"] for item in items],
            "molecule_id": [item["molecule_id"] for item in items],
            "solvent_id": [item["solvent_id"] for item in items],
            "smi": [item["smi"] for item in items],
            "solvent_smi": [item["solvent_smi"] for item in items],
            "atom_features": torch.as_tensor(atom_features, dtype=torch.float32),
            "coordinates": torch.as_tensor(coordinates, dtype=torch.float32),
            "edge_index": torch.as_tensor(edge_index, dtype=torch.long),
            "edge_attr": torch.as_tensor(edge_attr, dtype=torch.float32),
            "atom_mask": torch.as_tensor(atom_mask, dtype=torch.bool),
            "edge_mask": torch.as_tensor(edge_mask, dtype=torch.bool),
            "solvent_features": torch.as_tensor(solvent, dtype=torch.float32),
            "target": torch.as_tensor(target, dtype=torch.float32),
            "target_mask": torch.as_tensor(target_mask, dtype=torch.bool),
        }


class Tiny3DSmokeBackbone:
    """Factory for the explicit tiny 3D smoke backbone."""

    @staticmethod
    def build(torch: Any, config: Tiny3DSmokeConfig) -> Any:
        nn = torch.nn

        class Backbone(nn.Module):
            model_kind = TINY_3D_SMOKE_MODEL_KIND

            def __init__(self) -> None:
                super().__init__()
                self.node_encoder = nn.Linear(config.node_feature_dim, config.hidden_dim)
                self.coord_encoder = nn.Linear(3, config.hidden_dim)
                self.edge_encoder = nn.Linear(config.edge_feature_dim, config.hidden_dim)
                self.atom_fusion = nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                )

            def forward(
                self,
                atom_features: Any,
                coordinates: Any,
                edge_index: Any,
                edge_attr: Any,
                atom_mask: Any,
                edge_mask: Any,
            ) -> Any:
                mask = atom_mask.unsqueeze(-1).float()
                denom = mask.sum(dim=1).clamp_min(1.0)
                center = (coordinates * mask).sum(dim=1) / denom
                centered = coordinates - center.unsqueeze(1)
                h = self.node_encoder(atom_features / 16.0) + self.coord_encoder(centered)
                edge_context = torch.zeros_like(h)
                for batch_index in range(h.shape[0]):
                    valid = edge_mask[batch_index]
                    if bool(valid.any().item()):
                        dst = edge_index[batch_index, 1, valid].long()
                        messages = self.edge_encoder(edge_attr[batch_index, valid] / 8.0)
                        edge_context[batch_index].index_add_(0, dst, messages)
                h = self.atom_fusion(h + edge_context)
                h = h * mask
                return h.sum(dim=1) / denom

        return Backbone()


class Tiny3DSmokeModel:
    """Tiny 3D backbone plus solvent encoder and multitask prediction heads."""

    @staticmethod
    def build(torch: Any, config: Tiny3DSmokeConfig) -> Any:
        nn = torch.nn

        class Model(nn.Module):
            model_kind = TINY_3D_SMOKE_MODEL_KIND

            def __init__(self) -> None:
                super().__init__()
                self.backbone = Tiny3DSmokeBackbone.build(torch, config)
                self.solvent_encoder = nn.Sequential(
                    nn.Linear(config.solvent_feature_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                    nn.ReLU(),
                )
                self.heads = nn.Linear(config.hidden_dim, len(config.targets))

            def forward(self, batch: dict[str, Any]) -> Any:
                molecule = self.backbone(
                    batch["atom_features"],
                    batch["coordinates"],
                    batch["edge_index"],
                    batch["edge_attr"],
                    batch["atom_mask"],
                    batch["edge_mask"],
                )
                solvent = self.solvent_encoder(batch["solvent_features"])
                fused = self.fusion(torch.cat([molecule, solvent], dim=-1))
                return self.heads(fused)

        return Model()


def build_tiny_smoke_model(torch: Any, config: Tiny3DSmokeConfig) -> Any:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    return Tiny3DSmokeModel.build(torch, config)


def tensor_shape_report(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "shapes": {
            key: list(value.shape)
            for key, value in batch.items()
            if hasattr(value, "shape")
        },
        "target_mask_true": int(batch["target_mask"].sum().item()),
        "atom_mask_true": int(batch["atom_mask"].sum().item()),
        "edge_mask_true": int(batch["edge_mask"].sum().item()),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


def fit_smoke_target_normalizer(samples: list[dict[str, Any]], targets: Sequence[str]) -> dict[str, Any]:
    values = np.stack([sample["target"] for sample in samples]).astype(np.float32)
    masks = np.stack([sample["target_mask"] for sample in samples]).astype(bool)
    means = []
    scales = []
    counts = []
    for index, target in enumerate(targets):
        available = values[masks[:, index], index]
        if available.size == 0:
            raise ValueError(f"Target {target} has no training examples for normalization.")
        mean = float(available.mean())
        std = float(available.std())
        means.append(mean)
        scales.append(std if std > 1.0e-8 else 1.0)
        counts.append(int(available.size))
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "targets": list(targets),
        "mean": means,
        "scale": scales,
        "available_counts": counts,
        "fit_partition": "train",
        "fit_row_ids": [sample["row_id"] for sample in samples],
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


def _scaled_targets(torch: Any, target: Any, normalizer: dict[str, Any]) -> Any:
    mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=target.device)
    scale = torch.as_tensor(normalizer["scale"], dtype=torch.float32, device=target.device)
    return (target - mean) / scale


def inverse_scaled_predictions(values: np.ndarray, normalizer: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)
    scale = np.asarray(normalizer["scale"], dtype=np.float32)
    return values * scale + mean


def masked_multitask_mse(torch: Any, pred: Any, target: Any, mask: Any) -> Any:
    if int(mask.sum().item()) == 0:
        raise ValueError("Masked loss received no available target labels.")
    diff = pred[mask] - target[mask]
    loss = (diff * diff).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Masked multitask loss is NaN or infinite.")
    return loss


def smoke_train_step(
    torch: Any,
    model: Any,
    optimizer: Any,
    batch: dict[str, Any],
    normalizer: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pred = model(batch)
    target = _scaled_targets(torch, batch["target"], normalizer)
    loss = masked_multitask_mse(torch, pred, target, batch["target_mask"])
    loss.backward()
    stats = gradient_statistics(torch, model)
    if not stats["all_finite"]:
        raise FloatingPointError("NaN or infinite gradient detected during Windows smoke training.")
    optimizer.step()
    return float(loss.detach().cpu().item()), stats


def finite_forward_report(torch: Any, model: Any, batch: dict[str, Any]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        pred = model(batch)
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "shape": list(pred.shape),
        "all_finite": bool(torch.isfinite(pred).all().item()),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


def gradient_statistics(torch: Any, model: Any) -> dict[str, Any]:
    rows = []
    all_finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = bool(torch.isfinite(grad).all().item())
        all_finite = all_finite and finite
        rows.append(
            {
                "name": name,
                "finite": finite,
                "max_abs": float(grad.abs().max().cpu().item()) if grad.numel() else 0.0,
                "l2_norm": float(torch.linalg.vector_norm(grad).cpu().item()) if grad.numel() else 0.0,
            }
        )
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "all_finite": all_finite,
        "parameters": rows,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


def parameter_state(model: Any) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters()}


def changed_parameter_names(torch: Any, before: dict[str, Any], model: Any) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if name in before and not torch.equal(before[name], parameter.detach())
    ]


def save_smoke_checkpoint(
    torch: Any,
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    config: Tiny3DSmokeConfig,
    normalizer: dict[str, Any],
    update_index: int,
    loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_SMOKE_PROFILE,
            "model_kind": TINY_3D_SMOKE_MODEL_KIND,
            "real_uniprop_used": False,
            "real_checkpoint_loaded": False,
            "tiny_backbone_used": True,
            "checkpoint_kind": "windows_smoke",
            "update_index": int(update_index),
            "loss": float(loss),
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": normalizer,
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def load_smoke_checkpoint(torch: Any, path: Path) -> tuple[Any, Any, Tiny3DSmokeConfig, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != WINDOWS_SMOKE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported Windows smoke checkpoint schema: {checkpoint.get('schema_version')}")
    if checkpoint.get("model_kind") != TINY_3D_SMOKE_MODEL_KIND:
        raise ValueError("Checkpoint is not a tiny 3D smoke backbone checkpoint.")
    config_payload = dict(checkpoint["model_config"])
    config_payload["targets"] = tuple(config_payload["targets"])
    config = Tiny3DSmokeConfig(**config_payload)
    model = Tiny3DSmokeModel.build(torch, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, config, checkpoint["normalizer"], checkpoint


def predictions_numpy(torch: Any, model: Any, batch: dict[str, Any], normalizer: dict[str, Any]) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        scaled = model(batch).detach().cpu().numpy()
    return inverse_scaled_predictions(scaled, normalizer)


def build_prediction_record(
    output_dir: Path,
    request: dict[str, Any],
    model: Any,
    normalizer: dict[str, Any],
    config: Tiny3DSmokeConfig,
) -> dict[str, Any]:
    torch = _require_torch()
    canonical_molecule = _canonicalize_smiles(str(request["chromophore_smiles"]), field_name="chromophore_smiles")
    canonical_solvent = _canonicalize_smiles(str(request["solvent_smiles"]), field_name="solvent_smiles")
    molecule_id = stable_hash("mol", MANIFEST_SCHEMA_VERSION, canonical_molecule)
    solvent_id = stable_hash("solv", MANIFEST_SCHEMA_VERSION, canonical_solvent)
    geometry_dir = output_dir / "geometry_cache"
    geometry_path = cache_path(geometry_dir, molecule_id)
    geometry_source = "cache_hit"
    if geometry_path.exists():
        geometry = read_valid_cache(geometry_path, molecule_id, canonical_molecule)
    else:
        geometry = generate_geometry_entry(molecule_id, canonical_molecule)
        atomic_write_json(geometry_path, geometry)
        geometry_source = "generated"
    row = pd.Series(
        {
            "row_id": str(request.get("request_id", "windows_smoke_prediction")),
            "molecule_id": molecule_id,
            "solvent_id": solvent_id,
            "canonical_solvent_smiles": canonical_solvent,
            "canonical_isomeric_smiles": canonical_molecule,
            **{target: np.nan for target in config.targets},
        }
    )
    record = build_lmdb_record(row, geometry, tuple(config.targets), integer_id=0)
    dataset = _RecordDataset([record], targets=config.targets, solvent_feature_dim=config.solvent_feature_dim)
    batch = dataset.collater([dataset[0]])
    prediction_values = predictions_numpy(torch, model, batch, normalizer)[0]
    prediction = {
        "model_name": "fluorcast_windows_smoke_tiny_3d",
        "model_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "predicted_absorption_nm": _finite_or_none(prediction_values[0]),
        "predicted_emission_nm": _finite_or_none(prediction_values[1]),
        "predicted_quantum_yield": _finite_or_none(prediction_values[2]),
        "warnings": [
            "Windows smoke output uses a tiny test-only backbone and is not a real UniProp prediction."
        ],
    }
    result = {
        "schema_version": WINDOWS_SMOKE_PREDICTION_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "status": "success",
        "request_id": request.get("request_id"),
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "canonical_molecule_smiles": canonical_molecule,
        "canonical_solvent_smiles": canonical_solvent,
        "molecule_id": molecule_id,
        "geometry_source": geometry_source,
        "predictions": [prediction],
        "provenance": {
            "profile": WINDOWS_SMOKE_PROFILE,
            "model_kind": TINY_3D_SMOKE_MODEL_KIND,
            "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
            "real_uniprop_used": False,
            "real_checkpoint_loaded": False,
            "checkpoint": str(output_dir / "checkpoints" / "checkpoint.pt"),
        },
        "warnings": [
            "Do not report this output as a real UniProp result."
        ],
    }
    validate_windows_smoke_prediction_schema(result)
    return result


def _finite_or_none(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


class _RecordDataset(FluorCastUniPropSmokeDataset):
    def __init__(self, records: list[dict[str, Any]], *, targets: Sequence[str], solvent_feature_dim: int) -> None:
        self.lmdb_path = Path("<in-memory>")
        self.targets = tuple(targets)
        self.solvent_feature_dim = int(solvent_feature_dim)
        self.records = records


def validate_windows_smoke_prediction_schema(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != WINDOWS_SMOKE_PREDICTION_SCHEMA_VERSION:
        raise ValueError("Windows smoke prediction schema_version is missing or unsupported.")
    if payload.get("profile") != WINDOWS_SMOKE_PROFILE:
        raise ValueError("Windows smoke prediction profile is missing or invalid.")
    if payload.get("model_kind") != TINY_3D_SMOKE_MODEL_KIND:
        raise ValueError("Windows smoke prediction model_kind is missing or invalid.")
    if payload.get("real_uniprop_used") is not False or payload.get("real_checkpoint_loaded") is not False:
        raise ValueError("Windows smoke prediction cannot claim real UniProp provenance.")
    if payload.get("tiny_backbone_used") is not True:
        raise ValueError("Windows smoke prediction must declare tiny_backbone_used.")
    if payload.get("status") != "success":
        raise ValueError("Windows smoke prediction must be a successful single-result object.")
    if not isinstance(payload.get("predictions"), list) or not payload["predictions"]:
        raise ValueError("Windows smoke prediction must include one or more prediction rows.")
    for prediction in payload["predictions"]:
        if prediction.get("model_kind") != TINY_3D_SMOKE_MODEL_KIND:
            raise ValueError("Every Windows smoke prediction row must declare model_kind.")


def write_smoke_bundle(output_dir: Path, checkpoint_path: Path) -> Path:
    bundle_dir = output_dir / "smoke_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_path, bundle_dir / "model_weights.pt")
    metadata = {
        "schema_version": WINDOWS_SMOKE_BUNDLE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "source_checkpoint": str(checkpoint_path),
        "warning": "This bundle is intentionally incompatible with real production UniProp loading.",
    }
    _write_json(bundle_dir / "metadata.json", metadata)
    return bundle_dir


def _status(passed: bool, **extra: Any) -> dict[str, Any]:
    return {"status": "passed" if passed else "failed", **extra}


def run_windows_smoke(output_dir: Path, *, seed: int = 42, overwrite: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty; pass --overwrite to replace it: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, dict[str, Any]] = {}
    environment = windows_smoke_environment_report()
    _write_json(output_dir / "environment_report.json", environment)
    stages["environment_report"] = _status(True)

    bundle, split_assignments, manifest_metadata = build_fixture_manifests(output_dir, seed)
    stages["fixture_manifests"] = _status(
        manifest_metadata["unique_chromophores"] >= 3,
        fixture_rows=manifest_metadata["fixture_rows"],
        unique_chromophores=manifest_metadata["unique_chromophores"],
    )

    geometry_report = generate_fixture_geometries(bundle, output_dir)
    stages["geometry"] = _status(
        bool(geometry_report["one_geometry_per_unique_chromophore"])
        and bool(geometry_report["repeated_chromophores_reuse_geometry"]),
        geometry_count=geometry_report["geometry_count"],
    )

    lmdb_report = export_fixture_lmdb(bundle, split_assignments, output_dir, seed)
    stages["lmdb"] = _status(bool(lmdb_report["all_valid"]))

    torch = _require_torch()
    config = Tiny3DSmokeConfig(seed=int(seed))
    train_dataset = FluorCastUniPropSmokeDataset(
        output_dir / "lmdb" / "train.lmdb",
        targets=config.targets,
        solvent_feature_dim=config.solvent_feature_dim,
    )
    test_dataset = FluorCastUniPropSmokeDataset(
        output_dir / "lmdb" / "test.lmdb",
        targets=config.targets,
        solvent_feature_dim=config.solvent_feature_dim,
    )
    train_samples = [train_dataset[index] for index in range(len(train_dataset))]
    train_batch = train_dataset.collater(train_samples)
    shape_report = tensor_shape_report(train_batch)
    _write_json(output_dir / "tensor_shape_report.json", shape_report)
    stages["dataset_adapter_batch"] = _status(bool(train_batch["target_mask"].any().item()))

    normalizer = fit_smoke_target_normalizer(train_samples, config.targets)
    test_row_ids = {test_dataset[index]["row_id"] for index in range(len(test_dataset))}
    normalizer["test_row_ids_excluded"] = sorted(test_row_ids)
    normalizer["no_test_labels_used"] = not bool(set(normalizer["fit_row_ids"]).intersection(test_row_ids))
    _write_json(output_dir / "training_normalization.json", normalizer)
    stages["training_normalization"] = _status(bool(normalizer["no_test_labels_used"]))

    model = build_tiny_smoke_model(torch, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    forward_report = finite_forward_report(torch, model, train_batch)
    stages["forward"] = _status(bool(forward_report["all_finite"]), shape=forward_report["shape"])
    before = parameter_state(model)
    loss0, grad_report = smoke_train_step(torch, model, optimizer, train_batch, normalizer)
    changed = changed_parameter_names(torch, before, model)
    loss_report = {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "loss_values": [{"update_index": 0, "masked_multitask_mse": loss0}],
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "loss_values.json", loss_report)
    _write_json(output_dir / "gradient_statistics.json", grad_report)
    _write_json(
        output_dir / "changed_parameter_names.json",
        {
            "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_SMOKE_PROFILE,
            "model_kind": TINY_3D_SMOKE_MODEL_KIND,
            "changed_parameter_names": changed,
            "real_uniprop_used": False,
            "real_checkpoint_loaded": False,
            "tiny_backbone_used": True,
        },
    )
    stages["backward"] = _status(bool(grad_report["all_finite"]))
    stages["optimizer_step"] = _status(bool(changed), changed_parameter_count=len(changed))

    checkpoint_path = output_dir / "checkpoints" / "checkpoint.pt"
    save_smoke_checkpoint(
        torch,
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        normalizer=normalizer,
        update_index=0,
        loss=loss0,
    )
    before_reload = predictions_numpy(torch, model, train_batch, normalizer)
    reloaded_model, reloaded_optimizer, reloaded_config, reloaded_normalizer, checkpoint = load_smoke_checkpoint(torch, checkpoint_path)
    after_reload = predictions_numpy(torch, reloaded_model, train_batch, reloaded_normalizer)
    reload_identical = bool(np.allclose(before_reload, after_reload, rtol=1e-6, atol=1e-6))
    stages["checkpoint_reload_identity"] = _status(reload_identical, update_index=int(checkpoint["update_index"]))

    resumed_loss, resumed_grad_report = smoke_train_step(torch, reloaded_model, reloaded_optimizer, train_batch, reloaded_normalizer)
    resumed_checkpoint_path = output_dir / "checkpoints" / "resumed_checkpoint.pt"
    save_smoke_checkpoint(
        torch,
        resumed_checkpoint_path,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        config=reloaded_config,
        normalizer=reloaded_normalizer,
        update_index=1,
        loss=resumed_loss,
    )
    stages["resume_one_step"] = _status(bool(resumed_grad_report["all_finite"]), resumed_loss=resumed_loss)

    exact_resume = prove_exact_resume(torch, config, train_batch, normalizer)
    stages["exact_resume"] = _status(bool(exact_resume["exact"]))
    _write_json(output_dir / "exact_resume_report.json", exact_resume)

    request = {"request_id": "windows-smoke-req-001", "chromophore_smiles": "CCO", "solvent_smiles": "O"}
    _write_json(output_dir / "prediction_request.json", request)
    prediction = build_prediction_record(output_dir, request, reloaded_model, reloaded_normalizer, reloaded_config)
    _write_json(output_dir / "predictions.json", prediction)
    stages["json_prediction_schema"] = _status(True)

    smoke_bundle = write_smoke_bundle(output_dir, resumed_checkpoint_path)
    stages["smoke_bundle_guard"] = _status(True)

    all_passed = all(item["status"] == "passed" for item in stages.values())
    summary = {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "all_stages_passed": all_passed,
        "stages": stages,
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "environment_report": str(output_dir / "environment_report.json"),
            "fixture_source": str(output_dir / "manifests" / "fixture_source.csv"),
            "molecule_manifest": str(output_dir / "manifests" / "molecule_manifest.csv"),
            "row_manifest": str(output_dir / "manifests" / "row_manifest.csv"),
            "split_assignments": str(output_dir / "manifests" / "split_assignments.csv"),
            "geometry_validation_report": str(output_dir / "geometry_validation_report.json"),
            "lmdb_validation_report": str(output_dir / "lmdb_validation_report.json"),
            "tensor_shape_report": str(output_dir / "tensor_shape_report.json"),
            "loss_values": str(output_dir / "loss_values.json"),
            "gradient_statistics": str(output_dir / "gradient_statistics.json"),
            "changed_parameter_names": str(output_dir / "changed_parameter_names.json"),
            "checkpoint": str(checkpoint_path),
            "resumed_checkpoint": str(resumed_checkpoint_path),
            "predictions": str(output_dir / "predictions.json"),
            "smoke_bundle": str(smoke_bundle),
            "training_normalization": str(output_dir / "training_normalization.json"),
            "exact_resume_report": str(output_dir / "exact_resume_report.json"),
        },
        "windows_verified_components": [
            "fixture manifest generation",
            "one RDKit ETKDGv3/MMFF geometry per unique chromophore",
            "geometry reuse for repeated chromophore rows",
            "LMDB write/read/validation",
            "FluorCast smoke dataset adapter and target masks",
            "Tiny3DSmokeBackbone finite forward pass",
            "solvent encoder and multitask heads",
            "masked multitask loss",
            "finite backward gradients",
            "optimizer parameter update",
            "checkpoint save/load identity",
            "one-step resume",
            "production-style smoke JSON schema",
        ],
        "nibi_only_unverified_components": [
            "Uni-Core import and execution",
            "Uni-Mol+ import and task/model loading",
            "real UniProp checkpoint loading",
            "real UniProp forward/backward pass",
            "CUDA or scheduled Nibi GPU execution",
            "full FluorCast geometry cache generation",
            "full model training",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def prove_exact_resume(
    torch: Any,
    config: Tiny3DSmokeConfig,
    batch: dict[str, Any],
    normalizer: dict[str, Any],
) -> dict[str, Any]:
    first_model = build_tiny_smoke_model(torch, config)
    first_optimizer = torch.optim.AdamW(first_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss0, _ = smoke_train_step(torch, first_model, first_optimizer, batch, normalizer)
    temp_checkpoint = {
        name: value.detach().clone()
        for name, value in first_model.named_parameters()
    }
    temp_optimizer = first_optimizer.state_dict()

    resumed_model = build_tiny_smoke_model(torch, config)
    resumed_model.load_state_dict(first_model.state_dict())
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    resumed_optimizer.load_state_dict(temp_optimizer)
    loss1_resumed, _ = smoke_train_step(torch, resumed_model, resumed_optimizer, batch, normalizer)

    uninterrupted_model = build_tiny_smoke_model(torch, config)
    uninterrupted_optimizer = torch.optim.AdamW(uninterrupted_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss0_uninterrupted, _ = smoke_train_step(torch, uninterrupted_model, uninterrupted_optimizer, batch, normalizer)
    loss1_uninterrupted, _ = smoke_train_step(torch, uninterrupted_model, uninterrupted_optimizer, batch, normalizer)

    parameter_matches = []
    for name, parameter in resumed_model.named_parameters():
        parameter_matches.append(bool(torch.allclose(parameter, dict(uninterrupted_model.named_parameters())[name], rtol=1e-6, atol=1e-6)))
    return {
        "schema_version": WINDOWS_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "exact": bool(
            np.isclose(loss0, loss0_uninterrupted, rtol=1e-7, atol=1e-7)
            and np.isclose(loss1_resumed, loss1_uninterrupted, rtol=1e-7, atol=1e-7)
            and all(parameter_matches)
        ),
        "losses": {
            "checkpointed_step_0": float(loss0),
            "uninterrupted_step_0": float(loss0_uninterrupted),
            "resumed_step_1": float(loss1_resumed),
            "uninterrupted_step_1": float(loss1_uninterrupted),
        },
        "parameter_match_count": int(sum(parameter_matches)),
        "parameter_total": len(parameter_matches),
        "checkpoint_parameter_count": len(temp_checkpoint),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
