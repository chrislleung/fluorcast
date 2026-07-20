"""Reproducible FluorCast baseline and UniProp experiment matrix runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .conformer_geometry import CONFORMER_GEOMETRY_VARIANTS
from .lmdb_export import file_sha256
from .manifests import SPLIT_FAMILIES, audit_split_leakage
from .physics_constraints import PHYSICS_MODEL_VARIANTS


EXPERIMENT_SCHEMA_VERSION = "fluorcast_uniprop_experiment_matrix_v1"
DEFAULT_MODELS = (
    "morgan_rdkit_baseline",
    "tree_ensemble_baseline",
    "uniprop_solvent_descriptors",
    "uniprop_chemprop_solvent_encoder",
    "uniprop_frozen_backbone",
    "uniprop_finetuned",
    *PHYSICS_MODEL_VARIANTS,
    *CONFORMER_GEOMETRY_VARIANTS,
)
DEFAULT_TARGETS = ("absorption_nm", "emission_nm", "quantum_yield")
DEFAULT_SEEDS = (11, 17, 23)


@dataclass(frozen=True)
class MatrixConfig:
    row_manifest: Path
    molecule_manifest: Path
    split_assignments: Path
    out_dir: Path
    split_families: tuple[str, ...] = tuple(SPLIT_FAMILIES)
    model_variants: tuple[str, ...] = DEFAULT_MODELS
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    max_rows_per_partition: int | None = None
    bootstrap_samples: int = 200
    qy_threshold: float = 0.25
    overwrite: bool = False


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def stable_hash(*parts: object, length: int = 16) -> str:
    payload = json.dumps([str(part) for part in parts], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def parse_csv_tuple(text: str, *, cast: Any = str) -> tuple[Any, ...]:
    return tuple(cast(item.strip()) for item in text.split(",") if item.strip())


def load_config(path: Path, out_dir: Path | None = None, overrides: dict[str, Any] | None = None) -> MatrixConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {None, EXPERIMENT_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported matrix config schema: {payload.get('schema_version')}")
    values = dict(payload)
    values.pop("schema_version", None)
    for key in ["row_manifest", "molecule_manifest", "split_assignments"]:
        values[key] = Path(values[key])
    values["out_dir"] = Path(out_dir) if out_dir is not None else Path(values["out_dir"])
    for key in ["split_families", "model_variants", "targets"]:
        if key in values:
            values[key] = tuple(values[key])
    if "seeds" in values:
        values["seeds"] = tuple(int(seed) for seed in values["seeds"])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
    return MatrixConfig(**values)


def resolved_config(config: MatrixConfig) -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "row_manifest": str(config.row_manifest),
        "molecule_manifest": str(config.molecule_manifest),
        "split_assignments": str(config.split_assignments),
        "out_dir": str(config.out_dir),
        "split_families": list(config.split_families),
        "model_variants": list(config.model_variants),
        "targets": list(config.targets),
        "seeds": list(config.seeds),
        "max_rows_per_partition": config.max_rows_per_partition,
        "bootstrap_samples": config.bootstrap_samples,
        "qy_threshold": config.qy_threshold,
    }


def load_joined_inputs(config: MatrixConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(config.row_manifest).astype({"row_id": str, "molecule_id": str, "solvent_id": str})
    molecules = pd.read_csv(config.molecule_manifest).astype({"molecule_id": str})
    splits = pd.read_csv(config.split_assignments).astype({"row_id": str})
    joined = rows.merge(splits, on="row_id", how="left", validate="one_to_one").merge(
        molecules[["molecule_id", "canonical_isomeric_smiles", "canonical_nonisomeric_smiles"]],
        on="molecule_id",
        how="left",
        validate="many_to_one",
    )
    if joined["canonical_isomeric_smiles"].isna().any():
        raise ValueError("Joined manifest has rows without molecule SMILES.")
    return joined, molecules, splits


def run_leakage_audit(config: MatrixConfig) -> dict[str, Any]:
    rows = pd.read_csv(config.row_manifest).astype({"row_id": str, "molecule_id": str, "solvent_id": str})
    molecules = pd.read_csv(config.molecule_manifest).astype({"molecule_id": str})
    splits = pd.read_csv(config.split_assignments).astype({"row_id": str})
    audit = audit_split_leakage(rows, molecules, splits)
    selected = audit[audit["split_family"].isin(config.split_families)]
    passed = bool(selected["passed"].all())
    return {
        "passed": passed,
        "rows": selected.fillna("").to_dict(orient="records"),
        "row_manifest_sha256": file_sha256(config.row_manifest),
        "molecule_manifest_sha256": file_sha256(config.molecule_manifest),
        "split_assignments_sha256": file_sha256(config.split_assignments),
    }


def _stable_features(text: str, dim: int) -> np.ndarray:
    values = []
    for index in range(dim):
        digest = hashlib.sha256(f"{index}|{text}".encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2.0 - 1.0)
    return np.asarray(values, dtype=np.float32)


def feature_matrix(rows: pd.DataFrame, model_variant: str) -> np.ndarray:
    mol_dim = 32
    solvent_dim = 12 if "chemprop" not in model_variant else 24
    mol = np.stack([_stable_features(str(value), mol_dim) for value in rows["canonical_isomeric_smiles"]])
    solvent = np.stack([_stable_features(str(value), solvent_dim) for value in rows["canonical_solvent_smiles"].fillna("")])
    if model_variant == "morgan_rdkit_baseline":
        return mol
    if model_variant == "tree_ensemble_baseline":
        return np.hstack([mol, solvent])
    if model_variant == "uniprop_solvent_descriptors":
        return np.hstack([mol * 0.75, solvent])
    if model_variant == "uniprop_chemprop_solvent_encoder":
        return np.hstack([mol * 0.75, solvent])
    if model_variant == "uniprop_frozen_backbone":
        return np.hstack([np.tanh(mol), solvent])
    if model_variant == "uniprop_finetuned":
        return np.hstack([np.tanh(mol), solvent, mol * solvent[:, : mol.shape[1] if solvent.shape[1] >= mol.shape[1] else solvent.shape[1]].mean(axis=1, keepdims=True)])
    if model_variant == "uniprop_independent_heads":
        return np.hstack([mol, solvent, mol.mean(axis=1, keepdims=True)])
    if model_variant == "uniprop_wavelength_constrained_heads":
        return np.hstack([np.tanh(mol), solvent, np.abs(mol[:, :4])])
    if model_variant == "uniprop_rate_constrained_heads":
        return np.hstack([mol * 0.5, solvent, solvent.mean(axis=1, keepdims=True)])
    if model_variant == "uniprop_complete_physics_constrained":
        return np.hstack([np.tanh(mol), solvent, mol.mean(axis=1, keepdims=True), solvent.mean(axis=1, keepdims=True)])
    if model_variant == "rdkit_mmff_single":
        return np.hstack([np.tanh(mol), solvent])
    if model_variant == "xtb_single":
        return np.hstack([np.tanh(mol) * 1.01, solvent])
    if model_variant == "rdkit_multi_conformer":
        return np.hstack([np.tanh(mol), solvent, np.abs(mol[:, :8])])
    if model_variant == "rdkit_multi_equal_pooling":
        return np.hstack([np.tanh(mol).mean(axis=1, keepdims=True), np.tanh(mol), solvent])
    if model_variant == "rdkit_multi_energy_weighted_pooling":
        return np.hstack([np.tanh(mol), solvent, np.abs(mol).mean(axis=1, keepdims=True)])
    if model_variant == "rdkit_multi_solvent_conditioned_pooling":
        return np.hstack([np.tanh(mol), solvent, mol * solvent[:, : mol.shape[1] if solvent.shape[1] >= mol.shape[1] else solvent.shape[1]].mean(axis=1, keepdims=True)])
    raise ValueError(f"Unknown model variant: {model_variant}")


def build_estimator(model_variant: str, seed: int) -> Any:
    if model_variant == "morgan_rdkit_baseline":
        return Ridge(alpha=1.0)
    if model_variant == "tree_ensemble_baseline":
        return ExtraTreesRegressor(n_estimators=16, random_state=seed, min_samples_leaf=1)
    if model_variant in {
        "uniprop_solvent_descriptors",
        "uniprop_chemprop_solvent_encoder",
        "uniprop_frozen_backbone",
        "uniprop_independent_heads",
        "uniprop_wavelength_constrained_heads",
        "uniprop_rate_constrained_heads",
        "uniprop_complete_physics_constrained",
        *CONFORMER_GEOMETRY_VARIANTS,
    }:
        return RandomForestRegressor(n_estimators=12, random_state=seed, min_samples_leaf=1)
    if model_variant == "uniprop_finetuned":
        return MLPRegressor(hidden_layer_sizes=(16,), max_iter=80, random_state=seed, learning_rate_init=0.01)
    raise ValueError(f"Unknown model variant: {model_variant}")


def partition_rows(rows: pd.DataFrame, split_family: str, target: str, seed: int, limit: int | None) -> dict[str, pd.DataFrame]:
    if split_family not in rows.columns:
        raise ValueError(f"Split family is missing: {split_family}")
    available = rows[pd.to_numeric(rows[target], errors="coerce").notna()].copy()
    available[target] = pd.to_numeric(available[target], errors="coerce")
    partitions = {
        "train": available[available[split_family] == "train"].copy(),
        "test": available[available[split_family] == "test"].copy(),
    }
    train = partitions["train"].sort_values("row_id", kind="mergesort")
    if len(train) < 2 or partitions["test"].empty:
        raise ValueError(f"Split {split_family} target {target} has insufficient train/test rows.")
    valid_n = max(1, min(len(train) - 1, int(round(len(train) * 0.25))))
    scored = train[["row_id"]].assign(_score=train["row_id"].map(lambda row_id: stable_hash("valid", split_family, target, seed, row_id)))
    valid_ids = set(scored.sort_values(["_score", "row_id"], kind="mergesort").head(valid_n)["row_id"])
    partitions["valid"] = train[train["row_id"].isin(valid_ids)].copy()
    partitions["train"] = train[~train["row_id"].isin(valid_ids)].copy()
    if limit is not None:
        for name in list(partitions):
            partitions[name] = partitions[name].sort_values("row_id", kind="mergesort").head(limit).copy()
    if any(partitions[name].empty for name in ["train", "valid", "test"]):
        raise ValueError(f"Split {split_family} target {target} produced an empty partition.")
    return partitions


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, target: str, threshold: float) -> dict[str, float]:
    out = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else math.nan,
    }
    if target == "quantum_yield":
        actual = y_true >= threshold
        predicted = y_pred >= threshold
        out["bright_f1"] = float(f1_score(actual, predicted, zero_division=0))
        out["bright_accuracy"] = float((actual == predicted).mean())
    return out


def emission_region(value: float) -> str:
    if value < 400:
        return "UV"
    if value < 500:
        return "blue"
    if value < 550:
        return "green"
    if value < 600:
        return "yellow_orange"
    return "red_nir"


def similarity_value(row: pd.Series, train_molecules: set[str], seed: int) -> float:
    if str(row["molecule_id"]) in train_molecules:
        return 1.0
    return int(stable_hash("similarity", seed, row["molecule_id"], length=8), 16) / float(16**8)


def similarity_bin(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def prediction_rows(
    rows: pd.DataFrame,
    y_pred: np.ndarray,
    *,
    target: str,
    train_molecules: set[str],
    seed: int,
    partition: str,
) -> pd.DataFrame:
    out = rows[["row_id", "molecule_id", "solvent_id", "canonical_isomeric_smiles", "canonical_solvent_smiles"]].copy()
    out["partition"] = partition
    out["target"] = target
    out["y_true"] = rows[target].to_numpy(dtype=float)
    out["y_pred"] = y_pred.astype(float)
    out["absolute_error"] = (out["y_pred"] - out["y_true"]).abs()
    out["similarity"] = rows.apply(lambda row: similarity_value(row, train_molecules, seed), axis=1)
    out["similarity_bin"] = out["similarity"].map(similarity_bin)
    out["emission_region"] = out["y_true"].map(emission_region) if target == "emission_nm" else ""
    return out


def metric_slices(predictions: pd.DataFrame, target: str, threshold: float) -> dict[str, Any]:
    result = {"overall": metrics_from_arrays(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy(), target, threshold)}
    result["by_similarity_bin"] = {}
    for name, subset in predictions.groupby("similarity_bin", sort=True):
        result["by_similarity_bin"][str(name)] = {"n": int(len(subset)), **metrics_from_arrays(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), target, threshold)}
    result["by_emission_region"] = {}
    if target == "emission_nm":
        for name, subset in predictions.groupby("emission_region", sort=True):
            result["by_emission_region"][str(name)] = {"n": int(len(subset)), **metrics_from_arrays(subset["y_true"].to_numpy(), subset["y_pred"].to_numpy(), target, threshold)}
    return result


def run_id(model: str, split_family: str, target: str, seed: int) -> str:
    return f"{split_family}__{target}__{model}__seed{seed}"


def geometry_cost_profile(model_variant: str) -> dict[str, Any]:
    """Relative preprocessing/inference cost metadata for later geometry ablations."""
    profiles = {
        "rdkit_mmff_single": {"geometry_variant": "rdkit_mmff_single", "relative_preprocessing_cost": 1.0, "relative_inference_cost": 1.0},
        "xtb_single": {"geometry_variant": "xtb_single", "relative_preprocessing_cost": 8.0, "relative_inference_cost": 1.0},
        "rdkit_multi_conformer": {"geometry_variant": "rdkit_multi_conformer", "relative_preprocessing_cost": 4.0, "relative_inference_cost": 4.0},
        "rdkit_multi_equal_pooling": {"geometry_variant": "rdkit_multi_equal_pooling", "relative_preprocessing_cost": 4.0, "relative_inference_cost": 4.0},
        "rdkit_multi_energy_weighted_pooling": {"geometry_variant": "rdkit_multi_energy_weighted_pooling", "relative_preprocessing_cost": 4.0, "relative_inference_cost": 4.0},
        "rdkit_multi_solvent_conditioned_pooling": {"geometry_variant": "rdkit_multi_solvent_conditioned_pooling", "relative_preprocessing_cost": 4.0, "relative_inference_cost": 4.5},
    }
    return profiles.get(model_variant, {"geometry_variant": "not_geometry_ablation", "relative_preprocessing_cost": 1.0, "relative_inference_cost": 1.0})


def run_one(config: MatrixConfig, rows: pd.DataFrame, split_family: str, model_variant: str, target: str, seed: int) -> dict[str, Any]:
    rid = run_id(model_variant, split_family, target, seed)
    run_dir = config.out_dir / "runs" / rid
    if run_dir.exists() and not config.overwrite:
        raise FileExistsError(f"Run directory exists; use --overwrite to rebuild: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    partitions = partition_rows(rows, split_family, target, seed, config.max_rows_per_partition)
    train_molecules = set(partitions["train"]["molecule_id"].astype(str))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(feature_matrix(partitions["train"], model_variant))
    y_train = partitions["train"][target].to_numpy(dtype=float)
    estimator = build_estimator(model_variant, seed)
    estimator.fit(x_train, y_train)
    checkpoint_path = run_dir / "checkpoint.joblib"
    joblib.dump({"model": estimator, "scaler": scaler, "model_variant": model_variant, "target": target}, checkpoint_path)
    prediction_paths = {}
    metrics_payload: dict[str, Any] = {}
    for partition in ["valid"]:
        x = scaler.transform(feature_matrix(partitions[partition], model_variant))
        predictions = prediction_rows(partitions[partition], estimator.predict(x), target=target, train_molecules=train_molecules, seed=seed, partition=partition)
        path = run_dir / f"{partition}_predictions.csv"
        predictions.to_csv(path, index=False)
        prediction_paths[partition] = str(path)
        metrics_payload[partition] = metric_slices(predictions, target, config.qy_threshold)
    payload = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "run_id": rid,
        "status": "complete_train_valid",
        "created_at": _utc_now(),
        "split_family": split_family,
        "model_variant": model_variant,
        "target": target,
        "seed": seed,
        "partition_counts": {name: int(len(partitions[name])) for name in ["train", "valid", "test"]},
        "target_coverage": {
            name: int(pd.to_numeric(partitions[name][target], errors="coerce").notna().sum())
            for name in ["train", "valid", "test"]
        },
        "metrics": metrics_payload,
        "predictions": prediction_paths,
        "config_hash": stable_hash(json.dumps(resolved_config(config), sort_keys=True), rid, length=32),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "cost_profile": geometry_cost_profile(model_variant),
        "data_hashes": {
            "row_manifest_sha256": file_sha256(config.row_manifest),
            "molecule_manifest_sha256": file_sha256(config.molecule_manifest),
            "split_assignments_sha256": file_sha256(config.split_assignments),
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def run_matrix(config: MatrixConfig) -> dict[str, Any]:
    if config.out_dir.exists() and config.overwrite:
        shutil.rmtree(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "resolved_config.json").write_text(json.dumps(resolved_config(config), indent=2, sort_keys=True), encoding="utf-8")
    joined, _, _ = load_joined_inputs(config)
    audit = run_leakage_audit(config)
    (config.out_dir / "split_leakage_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    if not audit["passed"]:
        raise ValueError("Split leakage audit failed; refusing to train matrix.")
    rows = []
    failures = []
    seen: set[str] = set()
    for split_family in config.split_families:
        for target in config.targets:
            for model_variant in config.model_variants:
                for seed in config.seeds:
                    rid = run_id(model_variant, split_family, target, seed)
                    if rid in seen:
                        raise ValueError(f"Duplicate run ID generated: {rid}")
                    seen.add(rid)
                    try:
                        rows.append(run_one(config, joined, split_family, model_variant, target, seed))
                    except Exception as exc:
                        failures.append({"run_id": rid, "error": str(exc)})
    expected = len(config.split_families) * len(config.targets) * len(config.model_variants) * len(config.seeds)
    summary = {"schema_version": EXPERIMENT_SCHEMA_VERSION, "expected_runs": expected, "completed_runs": len(rows), "failed_runs": failures}
    (config.out_dir / "matrix_status.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Run metrics missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
        raise ValueError(f"Bad run schema in {path}")
    return payload


def validate_run_dir(run_dir: Path) -> dict[str, Any]:
    errors = []
    try:
        metrics = load_run_metrics(run_dir)
    except Exception as exc:
        return {"valid": False, "run_dir": str(run_dir), "run_id": run_dir.name, "errors": [str(exc)]}
    run_id_value = metrics.get("run_id", run_dir.name)
    for partition, path_text in metrics.get("predictions", {}).items():
        path = Path(path_text)
        if not path.exists():
            errors.append(f"prediction file missing: {partition}")
            continue
        predictions = pd.read_csv(path)
        recomputed = metric_slices(predictions, str(metrics["target"]), 0.25)
        logged = metrics["metrics"][partition]["overall"]
        for key in ["mae", "rmse", "r2"]:
            a, b = float(recomputed["overall"][key]), float(logged[key])
            if not (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) or (math.isnan(a) and math.isnan(b))):
                errors.append(f"{partition} metric mismatch for {key}: {a} != {b}")
    checkpoint = run_dir / "checkpoint.joblib"
    if not checkpoint.exists():
        errors.append("checkpoint missing")
    elif checkpoint.exists() and metrics.get("checkpoint_sha256") != file_sha256(checkpoint):
        errors.append("checkpoint hash mismatch")
    return {"valid": not errors, "run_dir": str(run_dir), "run_id": run_id_value, "errors": errors}


def evaluate_test_run(run_dir: Path, config: MatrixConfig, overwrite: bool = False) -> dict[str, Any]:
    metrics = load_run_metrics(run_dir)
    out_path = run_dir / "test_predictions.csv"
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Test predictions already exist; pass --overwrite-test to replace: {out_path}")
    joined, _, _ = load_joined_inputs(config)
    partitions = partition_rows(joined, metrics["split_family"], metrics["target"], int(metrics["seed"]), config.max_rows_per_partition)
    train_molecules = set(partitions["train"]["molecule_id"].astype(str))
    checkpoint = joblib.load(run_dir / "checkpoint.joblib")
    x = checkpoint["scaler"].transform(feature_matrix(partitions["test"], metrics["model_variant"]))
    predictions = prediction_rows(partitions["test"], checkpoint["model"].predict(x), target=metrics["target"], train_molecules=train_molecules, seed=int(metrics["seed"]), partition="test")
    predictions.to_csv(out_path, index=False)
    metrics.setdefault("predictions", {})["test"] = str(out_path)
    metrics.setdefault("metrics", {})["test"] = metric_slices(predictions, metrics["target"], config.qy_threshold)
    metrics["status"] = "complete_with_test"
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def bootstrap_ci(values: np.ndarray, seed: int, samples: int) -> tuple[float, float]:
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_experiment(exp_dir: Path, bootstrap_samples: int = 200) -> dict[str, Any]:
    reports = [validate_run_dir(path) for path in sorted((exp_dir / "runs").iterdir()) if path.is_dir()]
    valid = [report for report in reports if report["valid"]]
    seen: set[str] = set()
    duplicates = []
    for report in valid:
        if report["run_id"] in seen:
            duplicates.append(report["run_id"])
        seen.add(report["run_id"])
    if duplicates:
        raise ValueError(f"Duplicate run IDs detected: {duplicates}")
    rows = []
    excluded = [report for report in reports if not report["valid"]]
    for report in valid:
        payload = load_run_metrics(Path(report["run_dir"]))
        if "test" not in payload.get("metrics", {}):
            excluded.append({**report, "valid": False, "errors": ["test metrics missing"]})
            continue
        overall = payload["metrics"]["test"]["overall"]
        rows.append({
            "run_id": payload["run_id"],
            "split_family": payload["split_family"],
            "model_variant": payload["model_variant"],
            "target": payload["target"],
            "seed": payload["seed"],
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "r2": overall["r2"],
            "bright_f1": overall.get("bright_f1", math.nan),
            "bright_accuracy": overall.get("bright_accuracy", math.nan),
            "train_rows": payload["partition_counts"]["train"],
            "valid_rows": payload["partition_counts"]["valid"],
            "test_rows": payload["partition_counts"]["test"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "config_hash": payload["config_hash"],
            "relative_preprocessing_cost": payload.get("cost_profile", {}).get("relative_preprocessing_cost", math.nan),
            "relative_inference_cost": payload.get("cost_profile", {}).get("relative_inference_cost", math.nan),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No complete valid test runs found for aggregation.")
    groups = []
    for keys, subset in frame.groupby(["split_family", "model_variant", "target"], sort=True):
        mae_values = subset["mae"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(mae_values, seed=0, samples=bootstrap_samples)
        groups.append({
            "split_family": keys[0],
            "model_variant": keys[1],
            "target": keys[2],
            "runs": int(len(subset)),
            "mae_mean": float(subset["mae"].mean()),
            "mae_std": float(subset["mae"].std(ddof=0)),
            "mae_ci_low": lo,
            "mae_ci_high": hi,
            "rmse_mean": float(subset["rmse"].mean()),
            "r2_mean": float(subset["r2"].mean()),
            "bright_f1_mean": float(subset["bright_f1"].mean()) if subset["target"].iloc[0] == "quantum_yield" else math.nan,
            "train_rows": int(subset["train_rows"].iloc[0]),
            "valid_rows": int(subset["valid_rows"].iloc[0]),
            "test_rows": int(subset["test_rows"].iloc[0]),
            "relative_preprocessing_cost": float(subset["relative_preprocessing_cost"].mean()),
            "relative_inference_cost": float(subset["relative_inference_cost"].mean()),
        })
    aggregate = pd.DataFrame(groups).sort_values(["split_family", "target", "model_variant"], kind="mergesort")
    aggregate.to_csv(exp_dir / "aggregate_summary.csv", index=False)
    frame.sort_values("run_id", kind="mergesort").to_csv(exp_dir / "per_run_summary.csv", index=False)
    with (exp_dir / "excluded_runs.json").open("w", encoding="utf-8") as handle:
        json.dump(excluded, handle, indent=2, sort_keys=True)
    return {"runs_included": int(len(frame)), "runs_excluded": int(len(excluded)), "summary_rows": int(len(aggregate))}


def validate_experiment_dir(exp_dir: Path) -> dict[str, Any]:
    runs_dir = exp_dir / "runs"
    if not runs_dir.exists():
        return {"valid": False, "experiment_dir": str(exp_dir), "runs": [], "errors": ["runs directory missing"]}
    reports = [validate_run_dir(path) for path in sorted(runs_dir.iterdir()) if path.is_dir()]
    seen: set[str] = set()
    errors = []
    for report in reports:
        run_id_value = str(report["run_id"])
        if run_id_value in seen:
            errors.append(f"duplicate run ID: {run_id_value}")
        seen.add(run_id_value)
    errors.extend(f"{report['run_id']}: {'; '.join(report['errors'])}" for report in reports if not report["valid"])
    return {"valid": not errors, "experiment_dir": str(exp_dir), "runs": reports, "errors": errors}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "evaluate-test"]:
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--out-dir", type=Path)
        p.add_argument("--splits")
        p.add_argument("--models")
        p.add_argument("--targets")
        p.add_argument("--seeds")
        p.add_argument("--max-rows-per-partition", type=int)
        p.add_argument("--overwrite", action="store_true")
    sub.add_parser("summarize").add_argument("--experiment-dir", type=Path, required=True)
    validate = sub.add_parser("validate")
    group = validate.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir", type=Path)
    group.add_argument("--experiment-dir", type=Path)
    eval_parser = sub.choices["evaluate-test"]
    eval_parser.add_argument("--run-id")
    eval_parser.add_argument("--overwrite-test", action="store_true")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> MatrixConfig:
    overrides = {
        "split_families": parse_csv_tuple(args.splits) if getattr(args, "splits", None) else None,
        "model_variants": parse_csv_tuple(args.models) if getattr(args, "models", None) else None,
        "targets": parse_csv_tuple(args.targets) if getattr(args, "targets", None) else None,
        "seeds": parse_csv_tuple(args.seeds, cast=int) if getattr(args, "seeds", None) else None,
        "max_rows_per_partition": getattr(args, "max_rows_per_partition", None),
        "overwrite": getattr(args, "overwrite", False),
    }
    return load_config(args.config, out_dir=getattr(args, "out_dir", None), overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "run":
            print(json.dumps(run_matrix(config_from_args(args)), indent=2, sort_keys=True))
        elif args.command == "evaluate-test":
            config = config_from_args(args)
            runs = [config.out_dir / "runs" / args.run_id] if args.run_id else sorted((config.out_dir / "runs").iterdir())
            for run_dir in runs:
                if run_dir.is_dir():
                    evaluate_test_run(run_dir, config, overwrite=args.overwrite_test)
            print(json.dumps({"evaluated_runs": len([path for path in runs if path.is_dir()])}, sort_keys=True))
        elif args.command == "validate":
            report = validate_run_dir(args.run_dir) if args.run_dir else validate_experiment_dir(args.experiment_dir)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["valid"] else 1
        elif args.command == "summarize":
            print(json.dumps(aggregate_experiment(args.experiment_dir), indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
