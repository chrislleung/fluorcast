"""Second-stage UniProp smoke training with the backbone unfrozen."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .head_smoke_training import (
    DEFAULT_TARGETS,
    TRAINING_SCHEMA_VERSION,
    HeadOnlySmokeModel,
    _batch_indices,
    _features_for_samples,
    _masked_mse,
    _optimizer_state_is_finite,
    _require_torch,
    _utc_now,
    apply_target_scaler,
    dataset_hash,
    environment_report,
    evaluate,
    fit_target_scaler,
    load_partition,
    recompute_metrics_from_predictions,
    select_deterministic_subset,
    trainable_parameter_names,
    write_predictions,
)
from .lmdb_export import file_sha256


FINETUNE_SCHEMA_VERSION = "fluorcast_uniprop_backbone_finetune_v1"


@dataclass(frozen=True)
class BackboneFinetuneConfig:
    data_dir: Path
    output_dir: Path
    head_checkpoint: Path
    targets: tuple[str, ...] = DEFAULT_TARGETS
    seed: int = 42
    subset_size: int = 24
    train_batch_size: int = 8
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    valid_batch_size: int = 32
    max_updates: int = 20
    backbone_learning_rate: float = 1.0e-4
    head_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 5
    hidden_dim: int = 32
    molecule_feature_dim: int = 32
    solvent_feature_dim: int = 24
    solvent_adapter_dim: int = 16
    validation_partition: str = "valid"
    device: str = "auto"
    use_amp: bool = True
    use_ema: bool = False
    ema_decay: float = 0.99
    resume: bool = False
    stop_after_updates: int | None = None


def load_config(path: Path, output_dir: Path | None = None, overrides: dict[str, Any] | None = None) -> BackboneFinetuneConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {None, FINETUNE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported fine-tune config schema: {payload.get('schema_version')}")
    values = dict(payload)
    values.pop("schema_version", None)
    values["data_dir"] = Path(values["data_dir"])
    values["head_checkpoint"] = Path(values["head_checkpoint"])
    values["output_dir"] = Path(output_dir) if output_dir is not None else Path(values["output_dir"])
    if "targets" in values:
        values["targets"] = tuple(values["targets"])
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = value
    return BackboneFinetuneConfig(**values)


def resolved_config(config: BackboneFinetuneConfig) -> dict[str, Any]:
    return {
        "schema_version": FINETUNE_SCHEMA_VERSION,
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "head_checkpoint": str(config.head_checkpoint),
        "targets": list(config.targets),
        "seed": config.seed,
        "subset_size": config.subset_size,
        "train_batch_size": config.train_batch_size,
        "micro_batch_size": config.micro_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "valid_batch_size": config.valid_batch_size,
        "max_updates": config.max_updates,
        "backbone_learning_rate": config.backbone_learning_rate,
        "head_learning_rate": config.head_learning_rate,
        "weight_decay": config.weight_decay,
        "max_grad_norm": config.max_grad_norm,
        "early_stopping_patience": config.early_stopping_patience,
        "hidden_dim": config.hidden_dim,
        "molecule_feature_dim": config.molecule_feature_dim,
        "solvent_feature_dim": config.solvent_feature_dim,
        "solvent_adapter_dim": config.solvent_adapter_dim,
        "validation_partition": config.validation_partition,
        "device": config.device,
        "use_amp": config.use_amp,
        "use_ema": config.use_ema,
        "ema_decay": config.ema_decay,
    }


def build_model_from_head_checkpoint(torch: Any, config: BackboneFinetuneConfig, device: Any) -> Any:
    head_config = type(
        "HeadConfigForTransition",
        (),
        {
            "molecule_feature_dim": config.molecule_feature_dim,
            "hidden_dim": config.hidden_dim,
            "solvent_feature_dim": config.solvent_feature_dim,
            "solvent_adapter_dim": config.solvent_adapter_dim,
            "targets": config.targets,
            "seed": config.seed,
        },
    )()
    model = HeadOnlySmokeModel.build(torch, head_config)
    checkpoint = torch.load(config.head_checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != TRAINING_SCHEMA_VERSION:
        raise ValueError("Fine-tuning must initialize from a head-only checkpoint.")
    model.load_state_dict(checkpoint["model_state_dict"])
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True
    return model.to(device)


def parameter_counts_by_component(model: Any) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        component = name.split(".", 1)[0]
        counts.setdefault(component, {"total": 0, "trainable": 0})
        amount = int(parameter.numel())
        counts[component]["total"] += amount
        if parameter.requires_grad:
            counts[component]["trainable"] += amount
    return counts


def optimizer_parameter_groups(model: Any, config: BackboneFinetuneConfig) -> list[dict[str, Any]]:
    return [
        {
            "name": "backbone",
            "params": [parameter for name, parameter in model.named_parameters() if name.startswith("backbone.")],
            "lr": config.backbone_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "prediction_layers",
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if name.startswith(("solvent_adapter.", "fusion.", "heads."))
            ],
            "lr": config.head_learning_rate,
            "weight_decay": config.weight_decay,
        },
    ]


def optimizer_group_report(optimizer: Any) -> list[dict[str, Any]]:
    return [
        {"name": group.get("name"), "lr": float(group["lr"]), "weight_decay": float(group.get("weight_decay", 0.0))}
        for group in optimizer.param_groups
    ]


def amp_enabled(torch: Any, config: BackboneFinetuneConfig, device: Any) -> bool:
    return bool(config.use_amp and device.type == "cuda" and torch.cuda.is_available())


def autocast_context(torch: Any, enabled: bool) -> Any:
    return torch.amp.autocast("cuda", enabled=enabled)


def make_grad_scaler(torch: Any, enabled: bool) -> Any:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def memory_report(torch: Any, device: Any) -> dict[str, Any]:
    report: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda" and torch.cuda.is_available():
        report.update(
            {
                "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return report


class ExponentialMovingAverage:
    def __init__(self, model: Any, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    def update(self, model: Any) -> None:
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {name: value.detach().clone() for name, value in state["shadow"].items()}

    def copy_to(self, model: Any) -> dict[str, Any]:
        backup = {}
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                backup[name] = parameter.detach().clone()
                parameter.data.copy_(self.shadow[name].data)
        return backup

    @staticmethod
    def restore(model: Any, backup: dict[str, Any]) -> None:
        for name, parameter in model.named_parameters():
            if name in backup:
                parameter.data.copy_(backup[name].data)


def evaluate_deterministic(
    torch: Any,
    model: Any,
    samples: list[dict[str, Any]],
    config: BackboneFinetuneConfig,
    scaler: dict[str, Any],
    device: Any,
    ema: ExponentialMovingAverage | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    backup = ema.copy_to(model) if ema is not None else None
    try:
        metrics, rows = evaluate(torch, model, samples, config, scaler, device)
        metrics_again, rows_again = evaluate(torch, model, samples, config, scaler, device)
    finally:
        if backup is not None:
            ExponentialMovingAverage.restore(model, backup)
    if metrics != metrics_again or rows != rows_again:
        raise FloatingPointError("Evaluation mode is not deterministic.")
    return metrics, rows


def save_checkpoint(
    torch: Any,
    path: Path,
    *,
    kind: str,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    grad_scaler: Any,
    config: BackboneFinetuneConfig,
    scaler: dict[str, Any],
    update_index: int,
    best_metric: float,
    best_update_index: int | None,
    stale_updates: int,
    metrics_history: list[dict[str, Any]],
    ema: ExponentialMovingAverage | None,
) -> None:
    torch.save(
        {
            "schema_version": FINETUNE_SCHEMA_VERSION,
            "checkpoint_kind": kind,
            "update_index": update_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "amp_scaler_state_dict": grad_scaler.state_dict(),
            "ema_state_dict": ema.state_dict() if ema is not None else None,
            "resolved_config": resolved_config(config),
            "scaler": scaler,
            "best_metric": best_metric,
            "best_update_index": best_update_index,
            "stale_updates": stale_updates,
            "metrics_history": metrics_history,
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def _oom_message(exc: RuntimeError, torch: Any, device: Any) -> str:
    report = memory_report(torch, device)
    return (
        "CUDA out of memory during UniProp backbone fine-tuning. "
        "Reduce micro_batch_size or gradient_accumulation_steps, enable AMP, "
        f"or request more GPU memory. Memory report: {json.dumps(report, sort_keys=True)}"
    )


def train_backbone_finetune(config: BackboneFinetuneConfig) -> dict[str, Any]:
    if config.validation_partition == "test":
        raise ValueError("Refusing accidental test-set evaluation; use valid for fine-tuning.")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    torch = _require_torch()
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    amp = amp_enabled(torch, config, device)

    train_samples = select_deterministic_subset(load_partition(config.data_dir, "train", config.targets), config.subset_size, config.targets, config.seed)
    valid_samples = load_partition(config.data_dir, config.validation_partition, config.targets)
    if not valid_samples:
        raise ValueError("Validation partition is empty.")
    scaler = fit_target_scaler(train_samples, config.targets)
    model = build_model_from_head_checkpoint(torch, config, device)
    counts = parameter_counts_by_component(model)
    if counts.get("backbone", {}).get("trainable", 0) == 0:
        raise ValueError("Backbone did not transition to trainable parameters.")
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, config))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
    grad_scaler = make_grad_scaler(torch, amp)
    ema = ExponentialMovingAverage(model, config.ema_decay) if config.use_ema else None

    start_update = 0
    best_metric = math.inf
    best_update_index: int | None = None
    stale_updates = 0
    metrics_history: list[dict[str, Any]] = []
    last_checkpoint = config.output_dir / "last_checkpoint.pt"
    if config.resume:
        if not last_checkpoint.exists():
            raise FileNotFoundError(f"Cannot resume because last checkpoint is missing: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        grad_scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        if ema is not None and checkpoint.get("ema_state_dict") is not None:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        scaler = checkpoint["scaler"]
        start_update = int(checkpoint["update_index"]) + 1
        best_metric = float(checkpoint["best_metric"])
        best_update_index = checkpoint.get("best_update_index")
        stale_updates = int(checkpoint.get("stale_updates", 0))
        metrics_history = list(checkpoint["metrics_history"])

    train_molecule, train_solvent, train_target, train_mask = _features_for_samples(train_samples, config)
    train_target_scaled = apply_target_scaler(train_target, scaler)
    stop_update = config.max_updates
    if config.stop_after_updates is not None:
        stop_update = min(stop_update, int(config.stop_after_updates))

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(json.dumps(resolved_config(config), indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "environment_report.json").write_text(json.dumps(environment_report(torch), indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "parameter_counts.json").write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    (config.output_dir / "optimizer_groups.json").write_text(json.dumps(optimizer_group_report(optimizer), indent=2, sort_keys=True), encoding="utf-8")
    split_report = {
        "schema_version": FINETUNE_SCHEMA_VERSION,
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

    try:
        for update_index in range(start_update, stop_update):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = _batch_indices(len(train_samples), config.train_batch_size, config.seed, update_index)
            micro_losses = []
            for micro_index in range(config.gradient_accumulation_steps):
                start = micro_index * config.micro_batch_size
                end = min(start + config.micro_batch_size, len(batch))
                if start >= end:
                    continue
                micro = batch[start:end]
                batch_mask = train_mask[micro]
                if not batch_mask.any():
                    raise ValueError("Empty target batch detected.")
                with autocast_context(torch, amp):
                    pred = model(
                        torch.tensor(train_molecule[micro], dtype=torch.float32, device=device),
                        torch.tensor(train_solvent[micro], dtype=torch.float32, device=device),
                    )
                    target = torch.tensor(train_target_scaled[micro], dtype=torch.float32, device=device)
                    mask = torch.tensor(batch_mask, dtype=torch.bool, device=device)
                    loss = _masked_mse(torch, pred, target, mask) / float(config.gradient_accumulation_steps)
                grad_scaler.scale(loss).backward()
                micro_losses.append(float(loss.detach().cpu().item()))
            grad_scaler.unscale_(optimizer)
            _optimizer_state_is_finite(torch, model)
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update(model)

            metrics, rows = evaluate_deterministic(torch, model, valid_samples, config, scaler, device, ema)
            write_predictions(config.output_dir / "validation_predictions.csv", rows, config.targets)
            current_metric = float(metrics["mean_mse"])
            improved = current_metric < best_metric
            if improved:
                best_metric = current_metric
                best_update_index = update_index
                stale_updates = 0
            else:
                stale_updates += 1
            entry = {
                "update_index": update_index,
                "train_loss": float(np.sum(micro_losses)),
                "best_metric": best_metric,
                "stale_updates": stale_updates,
                **metrics,
                "memory": memory_report(torch, device),
            }
            metrics_history.append(entry)
            if improved:
                save_checkpoint(
                    torch,
                    config.output_dir / "best_checkpoint.pt",
                    kind="best",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_scaler=grad_scaler,
                    config=config,
                    scaler=scaler,
                    update_index=update_index,
                    best_metric=best_metric,
                    best_update_index=best_update_index,
                    stale_updates=stale_updates,
                    metrics_history=metrics_history,
                    ema=ema,
                )
            save_checkpoint(
                torch,
                last_checkpoint,
                kind="last",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=grad_scaler,
                config=config,
                scaler=scaler,
                update_index=update_index,
                best_metric=best_metric,
                best_update_index=best_update_index,
                stale_updates=stale_updates,
                metrics_history=metrics_history,
                ema=ema,
            )
            (config.output_dir / "metrics_history.json").write_text(json.dumps(metrics_history, indent=2, sort_keys=True), encoding="utf-8")
            if stale_updates >= config.early_stopping_patience:
                break
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(_oom_message(exc, torch, device)) from exc
        raise

    final_metrics = metrics_history[-1] if metrics_history else {}
    recomputed = recompute_metrics_from_predictions(config.output_dir / "validation_predictions.csv", config.targets)
    summary = {
        "schema_version": FINETUNE_SCHEMA_VERSION,
        "completed_at": _utc_now(),
        "updates_completed": len(metrics_history),
        "next_update_index": len(metrics_history),
        "best_metric": best_metric,
        "best_update_index": best_update_index,
        "final_metrics": final_metrics,
        "recomputed_prediction_metrics": recomputed,
        "trainable_parameters": trainable_parameter_names(model),
        "parameter_counts": counts,
        "optimizer_groups": optimizer_group_report(optimizer),
        "amp_enabled": amp,
        "ema_enabled": ema is not None,
        "memory": memory_report(torch, device),
        "artifacts": {
            "resolved_config": str(config.output_dir / "resolved_config.json"),
            "environment_report": str(config.output_dir / "environment_report.json"),
            "dataset_split_hashes": str(config.output_dir / "dataset_split_hashes.json"),
            "parameter_counts": str(config.output_dir / "parameter_counts.json"),
            "optimizer_groups": str(config.output_dir / "optimizer_groups.json"),
            "best_checkpoint": str(config.output_dir / "best_checkpoint.pt"),
            "last_checkpoint": str(last_checkpoint),
            "metrics_history": str(config.output_dir / "metrics_history.json"),
            "validation_predictions": str(config.output_dir / "validation_predictions.csv"),
        },
    }
    (config.output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def forward_numpy(torch: Any, model: Any, molecule: np.ndarray, solvent: np.ndarray, device: Any, *, use_amp: bool) -> np.ndarray:
    model.eval()
    with torch.no_grad(), autocast_context(torch, use_amp):
        pred = model(
            torch.tensor(molecule, dtype=torch.float32, device=device),
            torch.tensor(solvent, dtype=torch.float32, device=device),
        )
    return pred.detach().cpu().float().numpy()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--head-checkpoint", type=Path)
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "data_dir": args.data_dir,
        "head_checkpoint": args.head_checkpoint,
        "max_updates": args.max_updates,
        "stop_after_updates": args.stop_after_updates,
        "resume": args.resume,
        "device": args.device,
        "use_amp": False if args.no_amp else None,
    }
    try:
        config = load_config(args.config, output_dir=args.output_dir, overrides=overrides)
        if not config.resume and config.output_dir.exists() and any(config.output_dir.iterdir()):
            shutil.rmtree(config.output_dir)
        summary = train_backbone_finetune(config)
    except (FileNotFoundError, ImportError, ValueError, FloatingPointError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary if args.print_summary else {"updates_completed": summary["updates_completed"], "best_metric": summary["best_metric"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
