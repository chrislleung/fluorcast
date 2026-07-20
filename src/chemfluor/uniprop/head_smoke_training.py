"""Deterministic head-only smoke training for FluorCast UniProp exports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .lmdb_export import DEFAULT_TARGET_COLUMNS, file_sha256, read_lmdb_records


TRAINING_SCHEMA_VERSION = "fluorcast_uniprop_head_smoke_v1"
DEFAULT_TARGETS = ("absorption_nm", "emission_nm", "quantum_yield")


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for UniProp head-only smoke training.") from exc
    return torch


@dataclass(frozen=True)
class HeadSmokeConfig:
    data_dir: Path
    output_dir: Path
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seed: int = 42
    subset_size: int = 24
    train_batch_size: int = 8
    valid_batch_size: int = 32
    max_updates: int = 20
    learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    hidden_dim: int = 32
    molecule_feature_dim: int = 32
    solvent_feature_dim: int = 24
    solvent_adapter_dim: int = 16
    validation_partition: str = "valid"
    device: str = "auto"
    resume: bool = False
    stop_after_updates: int | None = None


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def load_config(path: Path, output_dir: Path | None = None, overrides: dict[str, Any] | None = None) -> HeadSmokeConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {None, TRAINING_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported training config schema: {payload.get('schema_version')}")
    values = dict(payload)
    values.pop("schema_version", None)
    values["data_dir"] = Path(values["data_dir"])
    values["output_dir"] = Path(output_dir) if output_dir is not None else Path(values["output_dir"])
    if "targets" in values:
        values["targets"] = tuple(values["targets"])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
    return HeadSmokeConfig(**values)


def resolved_config(config: HeadSmokeConfig) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "targets": list(config.targets),
        "seed": config.seed,
        "subset_size": config.subset_size,
        "train_batch_size": config.train_batch_size,
        "valid_batch_size": config.valid_batch_size,
        "max_updates": config.max_updates,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "hidden_dim": config.hidden_dim,
        "molecule_feature_dim": config.molecule_feature_dim,
        "solvent_feature_dim": config.solvent_feature_dim,
        "solvent_adapter_dim": config.solvent_adapter_dim,
        "validation_partition": config.validation_partition,
        "device": config.device,
    }


def environment_report(torch: Any) -> dict[str, Any]:
    cuda_available = bool(torch.cuda.is_available())
    return {
        "created_at": _utc_now(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_runtime": getattr(torch.version, "cuda", None),
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def _stable_float_features(text: str, dim: int) -> np.ndarray:
    values = []
    for index in range(dim):
        digest = hashlib.sha256(f"{index}|{text}".encode("utf-8")).digest()
        integer = int.from_bytes(digest[:4], "big", signed=False)
        values.append((integer / 2**32) * 2.0 - 1.0)
    return np.asarray(values, dtype=np.float32)


def record_to_sample(record: dict[str, Any], targets: tuple[str, ...]) -> dict[str, Any]:
    target_columns = [str(item) for item in np.asarray(record.get("target_columns", DEFAULT_TARGET_COLUMNS)).tolist()]
    target_values = np.asarray(record["target"], dtype=np.float32)
    target_mask = np.asarray(record["target_mask"], dtype=np.bool_)
    indices = [target_columns.index(target) for target in targets]
    selected_target = target_values[indices].astype(np.float32, copy=False)
    selected_mask = target_mask[indices].astype(np.bool_, copy=False)
    molecule_text = "|".join(
        [
            str(record.get("smi", "")),
            ",".join(np.asarray(record.get("atoms", [])).astype(str).tolist()),
            np.asarray(record.get("label_pos", []), dtype=np.float32).round(4).tobytes().hex(),
        ]
    )
    solvent_text = str(record.get("solvent_smi", ""))
    return {
        "row_id": str(record["row_id"]),
        "molecule_id": str(record.get("molecule_id", "")),
        "solvent_id": str(record.get("solvent_id", "")),
        "molecule_text": molecule_text,
        "solvent_text": solvent_text,
        "target": selected_target,
        "target_mask": selected_mask,
    }


def load_partition(data_dir: Path, partition: str, targets: tuple[str, ...]) -> list[dict[str, Any]]:
    path = data_dir / f"{partition}.lmdb"
    return [record_to_sample(record, targets) for _, record in read_lmdb_records(path)]


def select_deterministic_subset(samples: list[dict[str, Any]], subset_size: int, targets: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    if not samples:
        raise ValueError("Training partition is empty.")
    scored = sorted(
        samples,
        key=lambda sample: hashlib.sha256(f"{seed}|{sample['row_id']}".encode("utf-8")).hexdigest(),
    )
    selected = scored[: min(subset_size, len(scored))]
    coverage = np.stack([sample["target_mask"] for sample in selected]).sum(axis=0)
    if np.any(coverage == 0):
        missing = [targets[index] for index, count in enumerate(coverage) if count == 0]
        raise ValueError(f"Subset has no non-missing examples for target(s): {missing}")
    return selected


def dataset_hash(samples: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for sample in sorted(samples, key=lambda item: item["row_id"]):
        digest.update(str(sample["row_id"]).encode("utf-8"))
        digest.update(np.asarray(sample["target"], dtype=np.float32).tobytes())
        digest.update(np.asarray(sample["target_mask"], dtype=np.bool_).tobytes())
    return digest.hexdigest()


def fit_target_scaler(samples: list[dict[str, Any]], targets: tuple[str, ...]) -> dict[str, Any]:
    stacked_y = np.stack([sample["target"] for sample in samples]).astype(np.float32)
    stacked_mask = np.stack([sample["target_mask"] for sample in samples]).astype(bool)
    means = []
    scales = []
    for index, target in enumerate(targets):
        values = stacked_y[stacked_mask[:, index], index]
        if values.size == 0:
            raise ValueError(f"Target {target} has no training examples.")
        mean = float(values.mean())
        std = float(values.std())
        means.append(mean)
        scales.append(std if std > 1.0e-8 else 1.0)
    return {"targets": list(targets), "mean": means, "scale": scales}


def apply_target_scaler(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return (values - mean) / scale


def inverse_target_scaler(values: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    return values * scale + mean


class HeadOnlySmokeModel:
    @staticmethod
    def build(torch: Any, config: HeadSmokeConfig) -> Any:
        nn = torch.nn

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(config.molecule_feature_dim, config.hidden_dim)
                self.solvent_adapter = nn.Sequential(
                    nn.Linear(config.solvent_feature_dim, config.solvent_adapter_dim),
                    nn.ReLU(),
                    nn.Linear(config.solvent_adapter_dim, config.solvent_adapter_dim),
                )
                self.fusion = nn.Sequential(
                    nn.Linear(config.hidden_dim + config.solvent_adapter_dim, config.hidden_dim),
                    nn.ReLU(),
                )
                self.heads = nn.Linear(config.hidden_dim, len(config.targets))
                with torch.no_grad():
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(config.seed)
                    self.backbone.weight.copy_(torch.randn(self.backbone.weight.shape, generator=generator) * 0.05)
                    self.backbone.bias.zero_()
                for parameter in self.backbone.parameters():
                    parameter.requires_grad = False

            def forward(self, molecule: Any, solvent: Any) -> Any:
                mol = self.backbone(molecule)
                sol = self.solvent_adapter(solvent)
                fused = self.fusion(torch.cat([mol, sol], dim=1))
                return self.heads(fused)

        return Model()


def trainable_parameter_names(model: Any) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def assert_only_intended_trainable(model: Any) -> None:
    allowed = ("solvent_adapter.", "fusion.", "heads.")
    bad = [name for name in trainable_parameter_names(model) if not name.startswith(allowed)]
    if bad:
        raise ValueError(f"Unexpected trainable parameter(s): {bad}")
    frozen_bad = [name for name, parameter in model.named_parameters() if name.startswith("backbone.") and parameter.requires_grad]
    if frozen_bad:
        raise ValueError(f"Backbone parameter(s) are not frozen: {frozen_bad}")


def _features_for_samples(samples: list[dict[str, Any]], config: HeadSmokeConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    molecule = np.stack([_stable_float_features(sample["molecule_text"], config.molecule_feature_dim) for sample in samples])
    solvent = np.stack([_stable_float_features(sample["solvent_text"], config.solvent_feature_dim) for sample in samples])
    target = np.stack([sample["target"] for sample in samples]).astype(np.float32)
    mask = np.stack([sample["target_mask"] for sample in samples]).astype(np.bool_)
    return molecule, solvent, target, mask


def _batch_indices(n_samples: int, batch_size: int, seed: int, update_index: int) -> np.ndarray:
    rng = np.random.default_rng(seed + update_index)
    order = rng.permutation(n_samples)
    return order[: min(batch_size, n_samples)]


def _masked_mse(torch: Any, pred: Any, target: Any, mask: Any) -> Any:
    if int(mask.sum().item()) == 0:
        raise ValueError("Empty target batch detected.")
    diff = pred[mask] - target[mask]
    loss = (diff * diff).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Training loss is NaN or infinite.")
    return loss


def _optimizer_state_is_finite(torch: Any, model: Any) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError(f"NaN or infinite gradient detected in {name}.")


def evaluate(torch: Any, model: Any, samples: list[dict[str, Any]], config: HeadSmokeConfig, scaler: dict[str, Any], device: Any) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    molecule, solvent, target, mask = _features_for_samples(samples, config)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(samples), config.valid_batch_size):
            end = start + config.valid_batch_size
            pred_scaled = model(
                torch.tensor(molecule[start:end], dtype=torch.float32, device=device),
                torch.tensor(solvent[start:end], dtype=torch.float32, device=device),
            )
            predictions.append(pred_scaled.detach().cpu().numpy())
    pred = inverse_target_scaler(np.concatenate(predictions, axis=0), scaler)
    metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for target_index, target_name in enumerate(config.targets):
        target_mask = mask[:, target_index]
        if not target_mask.any():
            continue
        errors = pred[target_mask, target_index] - target[target_mask, target_index]
        metrics[f"{target_name}_mae"] = float(np.abs(errors).mean())
        metrics[f"{target_name}_mse"] = float((errors * errors).mean())
    metrics["mean_mse"] = float(np.mean([value for key, value in metrics.items() if key.endswith("_mse")]))
    for sample_index, sample in enumerate(samples):
        row = {"row_id": sample["row_id"], "molecule_id": sample["molecule_id"], "solvent_id": sample["solvent_id"]}
        for target_index, target_name in enumerate(config.targets):
            is_available = bool(mask[sample_index, target_index])
            row[f"{target_name}_true"] = float(target[sample_index, target_index]) if is_available else ""
            row[f"{target_name}_pred"] = float(pred[sample_index, target_index])
            row[f"{target_name}_available"] = is_available
        rows.append(row)
    return metrics, rows


def save_checkpoint(
    torch: Any,
    path: Path,
    *,
    kind: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    config: HeadSmokeConfig,
    scaler: dict[str, Any],
    update_index: int,
    best_metric: float,
    metrics_history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "checkpoint_kind": kind,
            "update_index": update_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "resolved_config": resolved_config(config),
            "scaler": scaler,
            "best_metric": best_metric,
            "metrics_history": metrics_history,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def write_predictions(path: Path, rows: list[dict[str, Any]], targets: tuple[str, ...]) -> None:
    fieldnames = ["row_id", "molecule_id", "solvent_id"]
    for target in targets:
        fieldnames.extend([f"{target}_true", f"{target}_pred", f"{target}_available"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recompute_metrics_from_predictions(path: Path, targets: tuple[str, ...]) -> dict[str, float]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics: dict[str, float] = {}
    for target in targets:
        errors = []
        for row in rows:
            if str(row[f"{target}_available"]).lower() == "true":
                errors.append(float(row[f"{target}_pred"]) - float(row[f"{target}_true"]))
        if errors:
            arr = np.asarray(errors, dtype=np.float64)
            metrics[f"{target}_mae"] = float(np.abs(arr).mean())
            metrics[f"{target}_mse"] = float((arr * arr).mean())
    metrics["mean_mse"] = float(np.mean([value for key, value in metrics.items() if key.endswith("_mse")]))
    return metrics


def train_head_smoke(config: HeadSmokeConfig) -> dict[str, Any]:
    if config.validation_partition == "test":
        raise ValueError("Refusing accidental test-set evaluation; use valid for smoke training.")
    torch = _require_torch()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))

    train_samples = select_deterministic_subset(load_partition(config.data_dir, "train", config.targets), config.subset_size, config.targets, config.seed)
    valid_samples = load_partition(config.data_dir, config.validation_partition, config.targets)
    if not valid_samples:
        raise ValueError("Validation partition is empty.")
    scaler = fit_target_scaler(train_samples, config.targets)
    model = HeadOnlySmokeModel.build(torch, config).to(device)
    assert_only_intended_trainable(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    start_update = 0
    best_metric = math.inf
    metrics_history: list[dict[str, Any]] = []
    last_checkpoint = config.output_dir / "last_checkpoint.pt"
    if config.resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume because last checkpoint is missing: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler = checkpoint["scaler"]
        start_update = int(checkpoint["update_index"]) + 1
        best_metric = float(checkpoint["best_metric"])
        metrics_history = list(checkpoint["metrics_history"])

    train_molecule, train_solvent, train_target, train_mask = _features_for_samples(train_samples, config)
    train_target_scaled = apply_target_scaler(train_target, scaler)
    stop_update = config.max_updates
    if config.stop_after_updates is not None:
        stop_update = min(stop_update, int(config.stop_after_updates))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(json.dumps(resolved_config(config), indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "environment_report.json").write_text(json.dumps(environment_report(torch), indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "scalers.json").write_text(json.dumps(scaler, indent=2, sort_keys=True), encoding="utf-8")
    split_report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "train_rows": len(train_samples),
        "valid_rows": len(valid_samples),
        "train_hash": dataset_hash(train_samples),
        "valid_hash": dataset_hash(valid_samples),
        "source_files": {
            "train_lmdb_sha256": file_sha256(config.data_dir / "train.lmdb"),
            "valid_lmdb_sha256": file_sha256(config.data_dir / f"{config.validation_partition}.lmdb"),
        },
    }
    (config.output_dir / "dataset_split_hashes.json").write_text(json.dumps(split_report, indent=2, sort_keys=True), encoding="utf-8")

    for update_index in range(start_update, stop_update):
        model.train()
        batch = _batch_indices(len(train_samples), config.train_batch_size, config.seed, update_index)
        batch_mask = train_mask[batch]
        if not batch_mask.any():
            raise ValueError("Empty target batch detected.")
        pred = model(
            torch.tensor(train_molecule[batch], dtype=torch.float32, device=device),
            torch.tensor(train_solvent[batch], dtype=torch.float32, device=device),
        )
        target = torch.tensor(train_target_scaled[batch], dtype=torch.float32, device=device)
        mask = torch.tensor(batch_mask, dtype=torch.bool, device=device)
        loss = _masked_mse(torch, pred, target, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        _optimizer_state_is_finite(torch, model)
        optimizer.step()
        scheduler.step()
        metrics, rows = evaluate(torch, model, valid_samples, config, scaler, device)
        entry = {"update_index": update_index, "train_loss": float(loss.detach().cpu().item()), **metrics}
        metrics_history.append(entry)
        write_predictions(config.output_dir / "validation_predictions.csv", rows, config.targets)
        current_metric = float(metrics["mean_mse"])
        if current_metric < best_metric:
            best_metric = current_metric
            save_checkpoint(
                torch,
                config.output_dir / "best_checkpoint.pt",
                kind="best",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                scaler=scaler,
                update_index=update_index,
                best_metric=best_metric,
                metrics_history=metrics_history,
            )
        save_checkpoint(
            torch,
            last_checkpoint,
            kind="last",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            scaler=scaler,
            update_index=update_index,
            best_metric=best_metric,
            metrics_history=metrics_history,
        )
        (config.output_dir / "metrics_history.json").write_text(json.dumps(metrics_history, indent=2, sort_keys=True), encoding="utf-8")

    final_metrics = metrics_history[-1] if metrics_history else {}
    recomputed = recompute_metrics_from_predictions(config.output_dir / "validation_predictions.csv", config.targets)
    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "updates_completed": len(metrics_history),
        "next_update_index": len(metrics_history),
        "best_metric": best_metric,
        "final_metrics": final_metrics,
        "recomputed_prediction_metrics": recomputed,
        "trainable_parameters": trainable_parameter_names(model),
        "frozen_backbone": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
        "artifacts": {
            "resolved_config": str(config.output_dir / "resolved_config.json"),
            "environment_report": str(config.output_dir / "environment_report.json"),
            "dataset_split_hashes": str(config.output_dir / "dataset_split_hashes.json"),
            "best_checkpoint": str(config.output_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(last_checkpoint),
            "scalers": str(config.output_dir / "scalers.json"),
            "metrics_history": str(config.output_dir / "metrics_history.json"),
            "validation_predictions": str(config.output_dir / "validation_predictions.csv"),
        },
    }
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "data_dir": args.data_dir,
        "max_updates": args.max_updates,
        "stop_after_updates": args.stop_after_updates,
        "resume": args.resume,
        "device": args.device,
    }
    try:
        config = load_config(args.config, output_dir=args.output_dir, overrides=overrides)
        if not config.resume and config.output_dir.exists() and any(config.output_dir.iterdir()):
            shutil.rmtree(config.output_dir)
        summary = train_head_smoke(config)
    except (FileNotFoundError, ImportError, ValueError, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.print_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(json.dumps({"updates_completed": summary["updates_completed"], "best_metric": summary["best_metric"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
