"""Nibi-only real UniProp checkpoint gate for FluorCast.

The gate intentionally stays tiny: one molecular-solvent FluorCast record, one
real upstream checkpoint, one forward/backward/optimizer/reload cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from .geometry_cache import (
    atomic_write_json,
    cache_path,
    generate_geometry_entry,
    read_valid_cache,
)
from .lmdb_export import DEFAULT_TARGET_COLUMNS, build_lmdb_record, file_sha256
from .manifests import (
    MANIFEST_SCHEMA_VERSION,
    build_manifests,
    resolve_authoritative_dataset,
    stable_hash,
    validate_manifest_reconciliation,
)
from .windows_smoke import TINY_3D_SMOKE_MODEL_KIND


REAL_GATE_SCHEMA_VERSION = "fluorcast_uniprop_real_checkpoint_gate_v1"
REAL_GATE_PROFILE = "nibi-real-checkpoint-gate"
DEFAULT_REVISION_FILE = Path("third_party/nablacolors.REVISION")
DEFAULT_UPSTREAM_DIR = Path("third_party/nablacolors")
DEFAULT_CHECKPOINT_MANIFEST = Path("configs/uniprop/checkpoint_manifest.json")
DEFAULT_FEATURE_SCHEMA = Path("configs/uniprop/feature_schema.json")
DEFAULT_FEATURE_SCHEMA_SHA256 = "93e2a5aaf19617b7420a0020cea3c4d5a8550680fe4d2fd410b16d17081577f8"
DEFAULT_OUTPUT_DIR = Path("outputs/uniprop_real_checkpoint_gate")
DEFAULT_GEOMETRY_CACHE_DIR = Path("data/processed/uniprop/geometry_cache")
DEFAULT_CHECKPOINT_ID = "uniprop_rdkit_to_dft_implicit.pt"
KNOWN_FORCE_FIELD_FAILURES = {"mol_8448c4b2500ac36b", "mol_dbd6ed348b1a0518"}
FAILURE_CATEGORIES = {
    "missing_dependency",
    "wrong_upstream_commit",
    "missing_checkpoint",
    "checkpoint_hash_mismatch",
    "checkpoint_fixture",
    "missing_feature_schema",
    "feature_schema_hash_mismatch",
    "upstream_source_hash_mismatch",
    "feature_schema_fixture",
    "unsupported_feature_schema",
    "audit_not_ready",
    "preprocessing_incompatibility",
    "checkpoint_key_incompatibility",
    "cuda_unavailable",
    "out_of_memory",
    "nonfinite_forward_output",
    "nonfinite_loss",
    "missing_gradients",
    "optimizer_no_op",
    "reload_mismatch",
}


class GateFailure(RuntimeError):
    def __init__(self, category: str, message: str) -> None:
        if category not in FAILURE_CATEGORIES:
            raise ValueError(f"Unknown gate failure category: {category}")
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class GateConfig:
    output_dir: Path = DEFAULT_OUTPUT_DIR
    dataset: Path | None = None
    geometry_cache_dir: Path = DEFAULT_GEOMETRY_CACHE_DIR
    checkpoint_manifest: Path = DEFAULT_CHECKPOINT_MANIFEST
    checkpoint_dir: Path | None = None
    checkpoint_id: str = DEFAULT_CHECKPOINT_ID
    feature_schema: Path = DEFAULT_FEATURE_SCHEMA
    expected_feature_schema_hash: str | None = DEFAULT_FEATURE_SCHEMA_SHA256
    revision_file: Path = DEFAULT_REVISION_FILE
    upstream_dir: Path = DEFAULT_UPSTREAM_DIR
    device: str = "cpu"
    seed: int = 42
    learning_rate: float = 1.0e-5
    reload_rtol: float = 1.0e-5
    reload_atol: float = 1.0e-5
    overwrite: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def read_revision_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
    return values


def git_commit(path: Path) -> str | None:
    if not path.exists() or not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def ensure_upstream_paths(upstream_dir: Path) -> None:
    for rel in ["Uni-Core", "unimol_plus"]:
        path = upstream_dir / rel
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def module_details(module_name: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "path": None, "version": None}
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "path": getattr(module, "__file__", None),
            "version": getattr(module, "__version__", None),
        }
    except Exception as exc:
        return {"available": False, "path": None, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def environment_audit(config: GateConfig) -> dict[str, Any]:
    ensure_upstream_paths(config.upstream_dir)
    torch_info: dict[str, Any]
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        torch_info = {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        }
    except Exception as exc:
        torch_info = {"available": False, "version": None, "cuda_available": False, "cuda_runtime": None, "gpu_name": None, "error": f"{type(exc).__name__}: {exc}"}
    try:
        from rdkit import rdBase

        rdkit = {"available": True, "version": rdBase.rdkitVersion}
    except Exception as exc:
        rdkit = {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}
    checkpoint = resolve_checkpoint(config, allow_missing=True)
    schema = feature_schema_report(config.feature_schema, config.expected_feature_schema_hash, allow_missing=True)
    source_verification = {"all_sources_match": False, "files": []}
    if schema["present"] and schema["hash_matches"]:
        source_verification = verify_upstream_sources(config.upstream_dir, load_feature_schema(config.feature_schema, config.expected_feature_schema_hash))
    upstream_commit = git_commit(config.upstream_dir)
    pinned = read_revision_file(config.revision_file)
    ready = (
        torch_info["available"]
        and rdkit["available"]
        and module_details("lmdb")["available"]
        and module_details("unicore")["available"]
        and module_details("unimol_plus")["available"]
        and upstream_commit == pinned.get("commit")
        and source_verification["all_sources_match"]
        and checkpoint["present"]
        and checkpoint["hash_matches"]
        and schema["present"]
        and schema["hash_matches"]
        and (config.device != "cuda" or torch_info["cuda_available"])
    )
    return {
        "hostname": socket.gethostname(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_info": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
            "virtualenv": os.environ.get("VIRTUAL_ENV") or (sys.prefix if sys.prefix != sys.base_prefix else None),
        },
        "platform": platform.platform(),
        "pytorch": torch_info,
        "unicore": {**module_details("unicore"), "unicore_train": shutil.which("unicore-train")},
        "unimol_plus": module_details("unimol_plus"),
        "upstream_uniprop": {**module_details("unimol_plus.models.uniprop"), "commit": upstream_commit},
        "upstream_sources": source_verification,
        "imported_symbols": {
            "model_class": "unimol_plus.models.uniprop.UniPropModel",
            "preprocessing_function": "unimol_plus.data.pcq_dataset.get_graph_features",
            "embedding_function": "unimol_plus.data.data_utils.convert_to_single_emb",
            "checkpoint_loader": "unicore.checkpoint_utils.load_checkpoint_to_cpu",
        },
        "upstream_pinned": pinned,
        "rdkit": rdkit,
        "lmdb": module_details("lmdb"),
        "checkpoint": checkpoint,
        "feature_schema": schema,
        "selected_device": config.device,
        "environment_ready": bool(ready),
    }


def load_checkpoint_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise GateFailure("missing_checkpoint", f"Checkpoint manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_dir(config: GateConfig, manifest: dict[str, Any]) -> Path:
    if config.checkpoint_dir is not None:
        return config.checkpoint_dir
    env_name = str(manifest.get("checkpoint_dir_env", "FLUORCAST_UNIPROP_CHECKPOINT_DIR"))
    return Path(os.environ.get(env_name, str(manifest.get("default_checkpoint_dir", "assets/uniprop/checkpoints"))))


def select_checkpoint_manifest_item(manifest: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    for item in manifest.get("checkpoints", []):
        if item.get("filename") == checkpoint_id or item.get("id") == checkpoint_id:
            return item
    raise GateFailure("missing_checkpoint", f"Checkpoint is not in manifest: {checkpoint_id}")


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(config: GateConfig, *, allow_missing: bool = False) -> dict[str, Any]:
    manifest = load_checkpoint_manifest(config.checkpoint_manifest)
    item = select_checkpoint_manifest_item(manifest, config.checkpoint_id)
    root = _checkpoint_dir(config, manifest)
    path = root / str(item["filename"])
    algorithm = str(item.get("checksum_type") or manifest.get("checksum_type") or "md5")
    expected_hash = str(item.get("checksum"))
    expected_size = int(item["expected_size_bytes"])
    size_is_exact = bool(item.get("size_is_exact", True))
    if not path.exists():
        if allow_missing:
            return {
                "path": str(path),
                "filename": item["filename"],
                "present": False,
                "expected_size_bytes": expected_size,
                "size_is_exact": size_is_exact,
                "actual_size_bytes": None,
                "size_matches": None,
                "hash_type": algorithm,
                "expected_hash": expected_hash,
                "actual_hash": None,
                "actual_sha256": None,
                "hash_matches": False,
                "source": item.get("source"),
                "fixture": bool(item.get("fixture", False)),
            }
        raise GateFailure("missing_checkpoint", f"Checkpoint file is missing: {path}")
    actual_size = path.stat().st_size
    actual_hash = file_hash(path, algorithm)
    actual_sha256 = file_hash(path, "sha256")
    size_matches = actual_size == expected_size
    if size_is_exact and not size_matches and not allow_missing:
        raise GateFailure("checkpoint_hash_mismatch", f"Checkpoint size mismatch for {path}: expected {expected_size}, got {actual_size}")
    if actual_hash != expected_hash and not allow_missing:
        raise GateFailure("checkpoint_hash_mismatch", f"Checkpoint {algorithm} mismatch for {path}: expected {expected_hash}, got {actual_hash}")
    return {
        "path": str(path),
        "filename": item["filename"],
        "present": True,
        "expected_size_bytes": expected_size,
        "size_is_exact": size_is_exact,
        "actual_size_bytes": actual_size,
        "size_matches": size_matches,
        "hash_type": algorithm,
        "expected_hash": expected_hash,
        "actual_hash": actual_hash,
        "actual_sha256": actual_sha256,
        "hash_matches": actual_hash == expected_hash,
        "source": item.get("source"),
        "fixture": bool(item.get("fixture", False)),
    }


def feature_schema_report(path: Path, expected_hash: str | None, *, allow_missing: bool = False) -> dict[str, Any]:
    if not path.exists():
        if allow_missing:
            return {"path": str(path), "present": False, "sha256": None, "expected_sha256": expected_hash, "hash_matches": False}
        raise GateFailure("missing_feature_schema", f"Feature schema is missing: {path}")
    actual = file_sha256(path)
    if expected_hash is not None and actual != expected_hash and not allow_missing:
        raise GateFailure("feature_schema_hash_mismatch", f"Feature schema SHA-256 mismatch for {path}: expected {expected_hash}, got {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "present": True,
        "sha256": actual,
        "expected_sha256": expected_hash,
        "hash_matches": expected_hash is None or actual == expected_hash,
        "schema_version": payload.get("schema_version"),
        "schema_kind": payload.get("schema_kind"),
        "atom_channels": [channel.get("name") for channel in payload.get("atom_channels", [])],
        "edge_channels": [channel.get("name") for channel in payload.get("edge_channels", [])],
        "source_files": payload.get("upstream_source_files", []),
    }


def load_feature_schema(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    report = feature_schema_report(path, expected_hash)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_kind") != "categorical_feature_schema":
        raise GateFailure("unsupported_feature_schema", f"Unsupported UniProp feature schema kind: {payload.get('schema_kind')}")
    return {**payload, "path": str(path), "sha256": report["sha256"]}


def verify_upstream_sources(upstream_dir: Path, schema: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for source in schema.get("upstream_source_files", []):
        relative = Path(str(source["path"]))
        path = upstream_dir / relative
        expected = str(source["sha256"]).lower()
        present = path.exists()
        actual = file_sha256(path).lower() if present else None
        rows.append(
            {
                "path": str(relative).replace("\\", "/"),
                "absolute_path": str(path),
                "role": source.get("role"),
                "present": present,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "hash_matches": present and actual == expected,
            }
        )
    return {"all_sources_match": bool(rows) and all(row["hash_matches"] for row in rows), "files": rows}


def _channel_cardinalities(schema: dict[str, Any], key: str) -> list[int]:
    return [int(channel["cardinality"]) for channel in schema.get(key, [])]


def validate_feature_indices(node_attr: Any, edge_attr: Any, schema: dict[str, Any]) -> dict[str, Any]:
    node = np.asarray(node_attr)
    edge = np.asarray(edge_attr)
    atom_sizes = _channel_cardinalities(schema, "atom_channels")
    edge_sizes = _channel_cardinalities(schema, "edge_channels")
    errors: list[str] = []
    if node.ndim != 2 or node.shape[1] != len(atom_sizes):
        errors.append(f"node_attr shape {node.shape} does not match {len(atom_sizes)} atom channels")
    if edge.ndim != 2 or edge.shape[1] != len(edge_sizes):
        errors.append(f"edge_attr shape {edge.shape} does not match {len(edge_sizes)} edge channels")
    if not np.issubdtype(node.dtype, np.integer):
        errors.append(f"node_attr dtype is not integer: {node.dtype}")
    if not np.issubdtype(edge.dtype, np.integer):
        errors.append(f"edge_attr dtype is not integer: {edge.dtype}")
    atom_channel_errors = []
    edge_channel_errors = []
    if not errors:
        for index, (channel, size) in enumerate(zip(schema["atom_channels"], atom_sizes)):
            values = node[:, index]
            bad = values[(values < 0) | (values >= size)]
            if bad.size:
                atom_channel_errors.append({"name": channel["name"], "cardinality": size, "bad_values": sorted({int(value) for value in bad.tolist()})})
        for index, (channel, size) in enumerate(zip(schema["edge_channels"], edge_sizes)):
            values = edge[:, index] if edge.size else np.asarray([], dtype=np.int32)
            bad = values[(values < 0) | (values >= size)]
            if bad.size:
                edge_channel_errors.append({"name": channel["name"], "cardinality": size, "bad_values": sorted({int(value) for value in bad.tolist()})})
    compatible = not errors and not atom_channel_errors and not edge_channel_errors
    if not compatible:
        raise GateFailure("unsupported_feature_schema", f"Generated UniProp feature indices are out of schema bounds: errors={errors}, atom={atom_channel_errors}, edge={edge_channel_errors}")
    return {
        "compatible": True,
        "atom_channel_names": [channel["name"] for channel in schema["atom_channels"]],
        "atom_channel_cardinalities": atom_sizes,
        "edge_channel_names": [channel["name"] for channel in schema["edge_channels"]],
        "edge_channel_cardinalities": edge_sizes,
        "node_attr_shape": list(node.shape),
        "edge_attr_shape": list(edge.shape),
        "node_attr_min": node.min(axis=0).astype(int).tolist() if node.size else [],
        "node_attr_max": node.max(axis=0).astype(int).tolist() if node.size else [],
        "edge_attr_min": edge.min(axis=0).astype(int).tolist() if edge.size else [None for _ in edge_sizes],
        "edge_attr_max": edge.max(axis=0).astype(int).tolist() if edge.size else [None for _ in edge_sizes],
    }


def select_supported_record(config: GateConfig, schema: dict[str, Any]) -> tuple[pd.Series, dict[str, Any], dict[str, Any]]:
    dataset_path = resolve_authoritative_dataset(config.dataset)
    bundle = build_manifests(dataset_path, tuple(DEFAULT_TARGET_COLUMNS), compute_inchikey=False, compute_rdkit_properties=False, compute_nonisomeric=False)
    validate_manifest_reconciliation(bundle)
    rows = bundle.row_manifest.copy()
    rows = rows[~rows["molecule_id"].astype(str).isin(KNOWN_FORCE_FIELD_FAILURES)].copy()
    if "environment_type" in rows.columns:
        rows = rows[rows["environment_type"].astype("string").eq("molecular_solvent")].copy()
    rows = rows[rows["canonical_solvent_smiles"].notna()].copy()
    rows["_score"] = rows["row_id"].astype(str).map(lambda row_id: hashlib.sha256(f"{config.seed}|real-gate|{row_id}".encode("utf-8")).hexdigest())
    molecules = bundle.molecule_manifest.set_index("molecule_id")
    failures: list[dict[str, Any]] = []
    for _, row in rows.sort_values(["_score", "row_id"], kind="mergesort").iterrows():
        molecule_id = str(row["molecule_id"])
        smiles = str(molecules.loc[molecule_id, "canonical_isomeric_smiles"])
        path = cache_path(config.geometry_cache_dir, molecule_id)
        try:
            if path.exists():
                geometry = read_valid_cache(path, molecule_id, smiles)
                geometry_source = "cache_hit"
            else:
                geometry = generate_geometry_entry(molecule_id, smiles)
                atomic_write_json(path, geometry)
                geometry_source = "generated"
            record = build_lmdb_record(row, geometry, tuple(DEFAULT_TARGET_COLUMNS), integer_id=0)
            features = validate_feature_indices(record["node_attr"], record["edge_attr"], schema)
            if not features["compatible"]:
                failures.append({"molecule_id": molecule_id, "category": "unsupported_feature_schema"})
                continue
            return row, {**geometry, "cache_path": str(path), "geometry_source": geometry_source}, features
        except (ValueError, OSError) as exc:
            failures.append({"molecule_id": molecule_id, "category": "geometry_failure", "detail": str(exc)})
    if failures and any(item["category"] == "unsupported_feature_schema" for item in failures):
        raise GateFailure("unsupported_feature_schema", f"No compatible molecule found; examples: {failures[:5]}")
    raise GateFailure("preprocessing_incompatibility", f"No usable molecular-solvent geometry found; examples: {failures[:5]}")


def upstream_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        arch="uniprop_small",
        multitarget=True,
        seed=int(seed),
        label_prob=0.0,
        mid_prob=0.0,
        mid_lower=0.4,
        mid_upper=0.6,
        noise_scale=0.0,
        finetune_lora=False,
    )


def build_real_uniprop_model(seed: int, upstream_dir: Path) -> Any:
    ensure_upstream_paths(upstream_dir)
    try:
        import unimol_plus.models  # noqa: F401
        from unimol_plus.models.uniprop import UniPropModel
    except Exception as exc:
        raise GateFailure("missing_dependency", f"Could not import real UniProp architecture: {type(exc).__name__}: {exc}") from exc
    model = UniPropModel(upstream_args(seed))
    if getattr(model, "model_kind", None) == TINY_3D_SMOKE_MODEL_KIND:
        raise GateFailure("checkpoint_key_incompatibility", "Tiny3DSmokeBackbone cannot satisfy the real checkpoint gate.")
    return model


class FluorCastRealUniPropAdapter:
    @staticmethod
    def build(torch: Any, backbone: Any) -> Any:
        nn = torch.nn

        class Adapter(nn.Module):
            model_kind = "real_uniprop_fluorcast_adapter"

            def __init__(self) -> None:
                super().__init__()
                self.backbone = backbone
                self.fluorcast_task_head = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, len(DEFAULT_TARGET_COLUMNS)))

            def forward(self, batch: dict[str, Any]) -> Any:
                output = self.backbone(batch["batched_data"])
                pred = output[0] if isinstance(output, tuple) else output
                if pred.ndim == 1:
                    pred = pred.view(-1, 1).repeat(1, 3)
                return self.fluorcast_task_head(pred.float())

        return Adapter()


def checkpoint_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    if isinstance(state.get("ema"), dict) and isinstance(state["ema"].get("params"), dict):
        return state["ema"]["params"]
    if isinstance(state.get("model"), dict):
        return state["model"]
    if isinstance(state.get("model_state_dict"), dict):
        return state["model_state_dict"]
    if all(hasattr(value, "shape") for value in state.values()):
        return state
    raise GateFailure("checkpoint_key_incompatibility", "Checkpoint does not contain ema.params, model, or model_state_dict.")


def load_pretrained_backbone(torch: Any, model: Any, checkpoint_path: Path) -> dict[str, Any]:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = checkpoint_state_dict(state)
    model_state = model.state_dict()
    missing_required = sorted(key for key in model_state if key not in pretrained)
    unexpected = sorted(key for key in pretrained if key not in model_state)
    shape_mismatches = []
    matching: dict[str, Any] = {}
    for key, tensor in pretrained.items():
        if key not in model_state:
            continue
        expected_tensor = model_state[key]
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            shape_mismatches.append(
                {
                    "key": key,
                    "checkpoint_shape": list(tensor.shape),
                    "model_shape": list(expected_tensor.shape),
                }
            )
            continue
        matching[key] = tensor
    loaded_parameter_count = sum(int(tensor.numel()) for tensor in matching.values())
    model_parameter_count = sum(int(tensor.numel()) for tensor in model_state.values())
    fraction_loaded = loaded_parameter_count / model_parameter_count if model_parameter_count else 0.0
    if not matching:
        raise GateFailure("checkpoint_key_incompatibility", "Zero real backbone checkpoint tensors matched the model.")
    if missing_required or shape_mismatches or unexpected:
        raise GateFailure(
            "checkpoint_key_incompatibility",
            "Real UniProp checkpoint does not satisfy explicit backbone key policy. "
            f"Missing={missing_required[:20]} Shape={shape_mismatches[:5]} Unexpected={unexpected[:20]}",
        )
    result = model.load_state_dict(matching, strict=True)
    return {
        "total_checkpoint_tensor_keys": len(pretrained),
        "real_model_keys": len(model_state),
        "loaded_matching_keys": len(matching),
        "missing_required_backbone_keys": list(result.missing_keys),
        "intentionally_missing_fluorcast_head_keys": [],
        "unexpected_keys": list(result.unexpected_keys),
        "shape_mismatches": shape_mismatches,
        "checkpoint_tensor_keys": sorted(pretrained),
        "real_model_key_names": sorted(model_state),
        "loaded_matching_key_names": sorted(matching),
        "loaded_backbone_parameter_count": loaded_parameter_count,
        "real_backbone_parameter_count": model_parameter_count,
        "loaded_backbone_parameter_fraction": fraction_loaded,
    }


def build_preprocessed_batch(torch: Any, row: pd.Series, geometry: dict[str, Any], device: Any, upstream_dir: Path = DEFAULT_UPSTREAM_DIR, feature_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    ensure_upstream_paths(upstream_dir)
    try:
        from unimol_plus.data.pcq_dataset import get_graph_features
        from unimol_plus.data import data_utils
    except Exception as exc:
        raise GateFailure("preprocessing_incompatibility", f"Could not import upstream PCQ preprocessing: {type(exc).__name__}: {exc}") from exc
    record = build_lmdb_record(row, geometry, tuple(DEFAULT_TARGET_COLUMNS), integer_id=0)
    if feature_schema is not None:
        validate_feature_indices(record["node_attr"], record["edge_attr"], feature_schema)
    try:
        feat = get_graph_features(record)
        feat["pos"] = torch.as_tensor(record["label_pos"], dtype=torch.float32)
        feat["pos"] = feat["pos"] - feat["pos"].mean(0, keepdim=True)
        feat["target"] = torch.as_tensor(np.nan_to_num(record["target"], nan=0.0), dtype=torch.float32)
        feat["target_mask"] = torch.as_tensor(record["target_mask"], dtype=torch.bool)
        feat["id"] = torch.as_tensor(record["id"], dtype=torch.long)
        feat["solvent_smi"] = str(record["solvent_smi"])
        max_node_num = (int(feat["atom_mask"].shape[0]) + 1 + 3) // 4 * 4 - 1
        batched = {}
        pad_fns = {
            "atom_feat": data_utils.pad_1d_feat,
            "atom_mask": data_utils.pad_1d,
            "edge_feat": data_utils.pad_2d_feat,
            "shortest_path": data_utils.pad_2d,
            "degree": data_utils.pad_1d,
            "pos": data_utils.pad_1d_feat,
            "pair_type": data_utils.pad_2d_feat,
            "attn_bias": data_utils.pad_attn_bias,
        }
        for key, func in pad_fns.items():
            batched[key] = func([feat[key]], max_node_num).to(device)
        batched["target"] = feat["target"].view(1, -1).to(device)
        batched["target_mask"] = feat["target_mask"].view(1, -1).to(device)
        batched["id"] = feat["id"].view(1).to(device)
        batched["solvent_smi"] = [feat["solvent_smi"]]
        return {"batched_data": batched, "target": batched["target"], "target_mask": batched["target_mask"]}
    except AssertionError as exc:
        raise GateFailure("unsupported_feature_schema", f"Upstream preprocessing rejected feature schema indices: {exc}") from exc
    except Exception as exc:
        raise GateFailure("preprocessing_incompatibility", f"Upstream preprocessing failed: {type(exc).__name__}: {exc}") from exc


def tensor_shapes(batch: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in batch["batched_data"].items():
        if hasattr(value, "shape"):
            result[key] = list(value.shape)
    return result


def masked_mse(torch: Any, pred: Any, target: Any, mask: Any) -> Any:
    if int(mask.sum().item()) == 0:
        raise GateFailure("nonfinite_loss", "Tiny batch has no available FluorCast targets.")
    loss = ((pred[mask] - target[mask]) ** 2).mean()
    if not bool(torch.isfinite(loss).item()):
        raise GateFailure("nonfinite_loss", "Masked six-target loss is non-finite.")
    return loss


def gradient_report(torch: Any, model: Any) -> dict[str, Any]:
    rows = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        finite = bool(torch.isfinite(parameter.grad).all().item())
        nonzero = bool((parameter.grad.detach().abs() > 0).any().item()) if finite else False
        rows.append({"name": name, "finite": finite, "nonzero": nonzero})
    if not rows:
        raise GateFailure("missing_gradients", "No gradients were produced.")
    if not all(row["finite"] for row in rows):
        raise GateFailure("missing_gradients", f"Non-finite gradient(s): {[row['name'] for row in rows if not row['finite']][:20]}")
    if not any(row["nonzero"] for row in rows):
        raise GateFailure("missing_gradients", "No nonzero gradients were produced.")
    return {
        "parameter_count_with_gradients": len(rows),
        "finite_gradient_count": sum(1 for row in rows if row["finite"]),
        "nonzero_gradient_count": sum(1 for row in rows if row["nonzero"]),
        "parameters": rows,
    }


def parameter_state(model: Any) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def changed_parameters(torch: Any, before: dict[str, Any], model: Any) -> list[str]:
    changed = []
    current = dict(model.named_parameters())
    for name, value in before.items():
        if name in current and not bool(torch.equal(value, current[name].detach())):
            changed.append(name)
    if not changed:
        raise GateFailure("optimizer_no_op", "Optimizer step did not change any intended trainable parameter.")
    return changed


def validate_real_gate_report(report: dict[str, Any]) -> None:
    required = {
        "real_uniprop_used",
        "real_checkpoint_loaded",
        "upstream_commits",
        "upstream_sources",
        "imported_real_model",
        "environment_ready",
        "device",
        "checkpoint",
        "feature_schema_hash",
        "selected_molecule_id",
        "selected_solvent_id",
        "geometry",
        "feature_schema_compatibility",
        "preprocessing_tensor_shapes",
        "checkpoint_key_policy",
        "forward_output_shape",
        "finite_forward_outputs",
        "available_target_mask",
        "finite_loss",
        "parameter_count_with_gradients",
        "finite_gradient_count",
        "nonzero_gradient_count",
        "changed_parameter_names",
        "reload_agreement",
        "final_gate_status",
        "all_stages_passed",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise ValueError(f"Real gate JSON report is missing field(s): {missing}")
    if report["real_uniprop_used"] is not True:
        raise ValueError("Real gate report must declare real_uniprop_used=true.")
    if report.get("model_kind") == TINY_3D_SMOKE_MODEL_KIND or report.get("tiny_backbone_used") is True:
        raise ValueError("Tiny3DSmokeBackbone cannot satisfy the real gate.")
    if report.get("checkpoint_key_policy", {}).get("loaded_backbone_parameter_count", 0) <= 0:
        raise ValueError("Real gate report must include loaded real backbone parameters.")


def run_real_checkpoint_gate(config: GateConfig) -> dict[str, Any]:
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        if not config.overwrite:
            raise FileExistsError(f"Output directory is not empty; pass --overwrite: {config.output_dir}")
        shutil.rmtree(config.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    env = environment_audit(config)
    write_json(config.output_dir / "environment_audit.json", env)
    if not env["environment_ready"]:
        raise GateFailure("audit_not_ready", "Real checkpoint gate refused to run because environment audit readiness is false.")
    if config.device == "cuda" and not env["pytorch"]["cuda_available"]:
        raise GateFailure("cuda_unavailable", "CUDA was requested but PyTorch reports no CUDA device.")
    pinned = read_revision_file(config.revision_file)
    actual_commit = git_commit(config.upstream_dir)
    if actual_commit != pinned.get("commit"):
        raise GateFailure("wrong_upstream_commit", f"Wrong upstream commit: expected {pinned.get('commit')}, got {actual_commit}")
    checkpoint = resolve_checkpoint(config)
    if "fixture" in Path(checkpoint["path"]).name.lower() or checkpoint.get("fixture") is True:
        raise GateFailure("checkpoint_fixture", "Fixture checkpoints are not allowed in real mode.")
    schema = load_feature_schema(config.feature_schema, config.expected_feature_schema_hash)
    if "fixture" in str(config.feature_schema).lower() or schema.get("fixture") is True:
        raise GateFailure("feature_schema_fixture", "Fixture feature schemas are not allowed in real mode.")
    source_verification = verify_upstream_sources(config.upstream_dir, schema)
    if not source_verification["all_sources_match"]:
        raise GateFailure("upstream_source_hash_mismatch", f"Pinned upstream source hash verification failed: {source_verification}")
    row, geometry, features = select_supported_record(config, schema)

    import torch

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if config.device == "cuda" else "cpu")
    batch = build_preprocessed_batch(torch, row, geometry, device, config.upstream_dir, schema)
    backbone = build_real_uniprop_model(config.seed, config.upstream_dir)
    model_module = importlib.import_module(backbone.__class__.__module__)
    model_source = getattr(model_module, "__file__", None)
    if model_source is None or str(config.upstream_dir.resolve()) not in str(Path(model_source).resolve()):
        raise GateFailure("missing_dependency", f"Real UniProp model was not imported from the pinned upstream checkout: {model_source}")
    load_report = load_pretrained_backbone(torch, backbone, Path(checkpoint["path"]))
    model = FluorCastRealUniPropAdapter.build(torch, backbone).to(device)
    if getattr(model, "model_kind", None) == TINY_3D_SMOKE_MODEL_KIND:
        raise GateFailure("checkpoint_key_incompatibility", "Tiny3DSmokeBackbone cannot satisfy the real checkpoint gate.")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate)
    model.train()
    before = parameter_state(model)
    try:
        output = model(batch)
        if not bool(torch.isfinite(output).all().item()):
            raise GateFailure("nonfinite_forward_output", "Real UniProp adapter produced non-finite outputs.")
        loss = masked_mse(torch, output, batch["target"], batch["target_mask"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = gradient_report(torch, model)
        optimizer.step()
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise GateFailure("out_of_memory", str(exc)) from exc
        raise
    changed = changed_parameters(torch, before, model)
    model.eval()
    with torch.no_grad():
        before_reload = model(batch).detach().cpu()
    temp_checkpoint = config.output_dir / "temporary_gate_checkpoint.pt"
    torch.save({"schema_version": REAL_GATE_SCHEMA_VERSION, "model_state_dict": model.state_dict()}, temp_checkpoint)
    reloaded_backbone = build_real_uniprop_model(config.seed, config.upstream_dir)
    reloaded = FluorCastRealUniPropAdapter.build(torch, reloaded_backbone).to(device)
    saved = torch.load(temp_checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(saved["model_state_dict"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        after_reload = reloaded(batch).detach().cpu()
    reload_ok = bool(torch.allclose(before_reload, after_reload, rtol=config.reload_rtol, atol=config.reload_atol))
    if not reload_ok:
        raise GateFailure("reload_mismatch", f"Reload output mismatch above tolerance rtol={config.reload_rtol} atol={config.reload_atol}")

    named = list(model.named_parameters())
    report = {
        "schema_version": REAL_GATE_SCHEMA_VERSION,
        "profile": REAL_GATE_PROFILE,
        "model_kind": "real_uniprop_fluorcast_adapter",
        "real_uniprop_used": True,
        "real_checkpoint_loaded": True,
        "tiny_backbone_used": False,
        "upstream_commits": {"nablacolors": actual_commit, "expected_nablacolors": pinned.get("commit")},
        "upstream_sources": source_verification,
        "imported_real_model": {
            "module": backbone.__class__.__module__,
            "class_name": backbone.__class__.__name__,
            "source_path": model_source,
            "upstream_commit": actual_commit,
        },
        "imported_preprocessing": {
            "function": "unimol_plus.data.pcq_dataset.get_graph_features",
            "source_path": str((config.upstream_dir / "unimol_plus/unimol_plus/data/pcq_dataset.py").resolve()),
            "upstream_commit": actual_commit,
        },
        "environment_ready": bool(env["environment_ready"]),
        "environment_audit": str(config.output_dir / "environment_audit.json"),
        "device": str(device),
        "checkpoint": checkpoint,
        "feature_schema_path": str(config.feature_schema),
        "feature_schema_hash": schema["sha256"],
        "feature_schema_version": schema.get("schema_version"),
        "selected_row_id": str(row["row_id"]),
        "selected_molecule_id": str(row["molecule_id"]),
        "selected_solvent_id": str(row["solvent_id"]),
        "geometry": {
            "method": geometry.get("optimization_method"),
            "quality": geometry.get("geometry_quality"),
            "source": geometry.get("geometry_source"),
            "cache_path": geometry.get("cache_path"),
        },
        "feature_schema_compatibility": features,
        "preprocessing_tensor_shapes": tensor_shapes(batch),
        "checkpoint_key_policy": {**load_report, "intentionally_missing_fluorcast_head_keys": [name for name, _ in named if name.startswith("fluorcast_task_head.")]},
        "checkpoint_missing_keys": load_report["missing_required_backbone_keys"],
        "checkpoint_unexpected_keys": load_report["unexpected_keys"],
        "intentionally_unmatched_head_parameters": [name for name, _ in named if name.startswith("fluorcast_task_head.")],
        "pretrained_checkpoint_parameters": load_report["total_checkpoint_tensor_keys"],
        "checkpoint_loaded_backbone_parameters": load_report["loaded_backbone_parameter_count"],
        "checkpoint_loaded_backbone_parameter_fraction": load_report["loaded_backbone_parameter_fraction"],
        "newly_initialized_fluorcast_task_heads": [name for name, _ in named if name.startswith("fluorcast_task_head.")],
        "frozen_parameters": [name for name, p in named if not p.requires_grad],
        "trainable_parameters": [name for name, p in named if p.requires_grad],
        "forward_output_shape": list(output.shape),
        "finite_forward_outputs": True,
        "available_target_mask": batch["target_mask"].detach().cpu().numpy().tolist(),
        "loss_value": float(loss.detach().cpu().item()),
        "finite_loss": True,
        "parameter_count_with_gradients": grads["parameter_count_with_gradients"],
        "finite_gradient_count": grads["finite_gradient_count"],
        "nonzero_gradient_count": grads["nonzero_gradient_count"],
        "changed_parameter_names": changed,
        "reload_agreement": {"passed": True, "rtol": config.reload_rtol, "atol": config.reload_atol},
        "temporary_checkpoint": str(temp_checkpoint),
        "final_gate_status": "passed",
        "all_stages_passed": True,
    }
    validate_real_gate_report(report)
    write_json(config.output_dir / "summary.json", report)
    return report


def failure_report(config: GateConfig, exc: BaseException) -> dict[str, Any]:
    category = exc.category if isinstance(exc, GateFailure) else "preprocessing_incompatibility"
    report = {
        "schema_version": REAL_GATE_SCHEMA_VERSION,
        "profile": REAL_GATE_PROFILE,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": False,
        "all_stages_passed": False,
        "final_gate_status": "failed",
        "failure_category": category,
        "reason_for_failure": str(exc),
    }
    write_json(config.output_dir / "summary.json", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--geometry-cache-dir", type=Path, default=DEFAULT_GEOMETRY_CACHE_DIR)
    parser.add_argument("--checkpoint-manifest", type=Path, default=DEFAULT_CHECKPOINT_MANIFEST)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-id", default=DEFAULT_CHECKPOINT_ID)
    parser.add_argument("--feature-schema", type=Path, default=DEFAULT_FEATURE_SCHEMA)
    parser.add_argument("--expected-feature-schema-sha256")
    parser.add_argument("--revision-file", type=Path, default=DEFAULT_REVISION_FILE)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GateConfig:
    return GateConfig(
        output_dir=args.output_dir,
        dataset=args.dataset,
        geometry_cache_dir=args.geometry_cache_dir,
        checkpoint_manifest=args.checkpoint_manifest,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_id=args.checkpoint_id,
        feature_schema=args.feature_schema,
        expected_feature_schema_hash=args.expected_feature_schema_sha256 or DEFAULT_FEATURE_SCHEMA_SHA256,
        revision_file=args.revision_file,
        upstream_dir=args.upstream_dir,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        report = environment_audit(config)
        write_json(config.output_dir / "environment_audit.json", report)
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
        return 0
    try:
        report = run_real_checkpoint_gate(config)
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
        return 0
    except (GateFailure, FileExistsError, ImportError, ValueError, RuntimeError, OSError) as exc:
        report = failure_report(config, exc)
        print(json.dumps(_jsonable(report), indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
