from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.backbone_finetune import (  # noqa: E402
    BackboneFinetuneConfig,
    amp_enabled,
    build_model_from_head_checkpoint,
    evaluate_deterministic,
    forward_numpy,
    optimizer_parameter_groups,
    train_backbone_finetune,
)
from chemfluor.uniprop.head_smoke_training import (  # noqa: E402
    DEFAULT_TARGETS,
    HeadOnlySmokeModel,
    HeadSmokeConfig,
    _features_for_samples,
    fit_target_scaler,
    load_partition,
    train_head_smoke,
)
from chemfluor.uniprop.lmdb_export import encode_int_key  # noqa: E402


pytest.importorskip("lmdb")
torch = pytest.importorskip("torch")


def _write_lmdb(path: Path, records: list[dict[str, object]]) -> None:
    import lmdb

    path.parent.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=64 * 1024 * 1024,
    )
    try:
        with env.begin(write=True) as txn:
            for index, record in enumerate(records):
                txn.put(encode_int_key(index), gzip.compress(pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)))
        env.sync()
    finally:
        env.close()
    path.with_suffix(path.suffix + ".complete").write_text("{}", encoding="utf-8")


def _record(index: int, partition: str) -> dict[str, object]:
    solvent_values = ["O", "CCO", "CC#N", "CS(C)=O"]
    molecule_values = ["CCO", "CCN", "c1ccccc1", "CCCl"]
    targets = np.asarray([320.0 + index, 450.0 + 2.0 * index, 0.1 + 0.02 * index], dtype=np.float32)
    mask = np.asarray([True, index % 5 != 1, index % 4 != 2], dtype=np.bool_)
    targets[~mask] = np.nan
    return {
        "atoms": np.asarray(["C", "C", "O"]),
        "input_pos": [np.zeros((3, 3), dtype=np.float32) + index],
        "label_pos": np.zeros((3, 3), dtype=np.float32) + index,
        "smi": molecule_values[index % len(molecule_values)],
        "solvent_smi": solvent_values[index % len(solvent_values)],
        "node_attr": np.zeros((3, 9), dtype=np.int32),
        "edge_index": np.zeros((2, 0), dtype=np.int32),
        "edge_attr": np.zeros((0, 3), dtype=np.int32),
        "target": targets,
        "target_mask": mask,
        "target_columns": np.asarray(DEFAULT_TARGETS),
        "row_id": f"{partition}_{index:03d}",
        "molecule_id": f"mol_{index % 4}",
        "solvent_id": f"solv_{index % 4}",
    }


def _fixture_lmdb(tmp_path: Path) -> Path:
    data_dir = tmp_path / "lmdb"
    _write_lmdb(data_dir / "train.lmdb", [_record(index, "train") for index in range(12)])
    _write_lmdb(data_dir / "valid.lmdb", [_record(index, "valid") for index in range(10)])
    _write_lmdb(data_dir / "test.lmdb", [_record(index, "test") for index in range(3)])
    return data_dir


def _head_checkpoint(tmp_path: Path, data_dir: Path) -> Path:
    config = HeadSmokeConfig(
        data_dir=data_dir,
        output_dir=tmp_path / "head",
        seed=13,
        subset_size=12,
        train_batch_size=12,
        valid_batch_size=12,
        max_updates=3,
        learning_rate=0.01,
        hidden_dim=48,
        solvent_adapter_dim=24,
        device="cpu",
    )
    summary = train_head_smoke(config)
    return Path(summary["artifacts"]["best_checkpoint"])


def _finetune_config(
    tmp_path: Path,
    data_dir: Path,
    head_checkpoint: Path,
    output_name: str,
    *,
    max_updates: int = 4,
    resume: bool = False,
    stop_after_updates: int | None = None,
    use_ema: bool = False,
) -> BackboneFinetuneConfig:
    return BackboneFinetuneConfig(
        data_dir=data_dir,
        output_dir=tmp_path / output_name,
        head_checkpoint=head_checkpoint,
        seed=13,
        subset_size=12,
        train_batch_size=8,
        micro_batch_size=4,
        gradient_accumulation_steps=2,
        valid_batch_size=10,
        max_updates=max_updates,
        backbone_learning_rate=0.0003,
        head_learning_rate=0.003,
        early_stopping_patience=20,
        hidden_dim=48,
        solvent_adapter_dim=24,
        device="cpu",
        use_amp=True,
        use_ema=use_ema,
        resume=resume,
        stop_after_updates=stop_after_updates,
    )


def test_backbone_frozen_before_transition_and_trainable_after(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    head_config = HeadSmokeConfig(data_dir=data_dir, output_dir=tmp_path / "unused", hidden_dim=48, solvent_adapter_dim=24)
    head_model = HeadOnlySmokeModel.build(torch, head_config)
    assert all(not parameter.requires_grad for parameter in head_model.backbone.parameters())

    config = _finetune_config(tmp_path, data_dir, checkpoint, "transition")
    model = build_model_from_head_checkpoint(torch, config, torch.device("cpu"))

    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_backbone_parameters_change_after_optimization(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config = _finetune_config(tmp_path, data_dir, checkpoint, "change", max_updates=2)
    before = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]

    summary = train_backbone_finetune(config)
    after = torch.load(Path(summary["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)["model_state_dict"]

    assert not torch.equal(before["backbone.weight"], after["backbone.weight"])


def test_separate_learning_rates_are_correct(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config = _finetune_config(tmp_path, data_dir, checkpoint, "lr")
    model = build_model_from_head_checkpoint(torch, config, torch.device("cpu"))
    groups = optimizer_parameter_groups(model, config)

    assert groups[0]["name"] == "backbone"
    assert groups[0]["lr"] == pytest.approx(config.backbone_learning_rate)
    assert groups[1]["name"] == "prediction_layers"
    assert groups[1]["lr"] == pytest.approx(config.head_learning_rate)


def test_amp_and_non_amp_forward_passes_are_numerically_compatible(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config = _finetune_config(tmp_path, data_dir, checkpoint, "amp")
    model = build_model_from_head_checkpoint(torch, config, torch.device("cpu"))
    samples = load_partition(data_dir, "valid", DEFAULT_TARGETS)
    molecule, solvent, _, _ = _features_for_samples(samples[:4], config)

    plain = forward_numpy(torch, model, molecule, solvent, torch.device("cpu"), use_amp=False)
    amp = forward_numpy(torch, model, molecule, solvent, torch.device("cpu"), use_amp=amp_enabled(torch, config, torch.device("cpu")))

    np.testing.assert_allclose(plain, amp, rtol=1e-5, atol=1e-5)


def test_checkpoint_resume_includes_optimizer_scaler_scheduler_and_ema(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    train_backbone_finetune(_finetune_config(tmp_path, data_dir, checkpoint, "ema", max_updates=5, stop_after_updates=2, use_ema=True))
    partial = torch.load(tmp_path / "ema" / "last_checkpoint.pt", map_location="cpu", weights_only=False)

    assert partial["optimizer_state_dict"]
    assert "amp_scaler_state_dict" in partial
    assert partial["scheduler_state_dict"]
    assert partial["ema_state_dict"] is not None

    summary = train_backbone_finetune(_finetune_config(tmp_path, data_dir, checkpoint, "ema", max_updates=5, resume=True, use_ema=True))
    resumed = torch.load(Path(summary["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    assert resumed["update_index"] == 4
    assert resumed["ema_state_dict"] is not None


def test_resumed_and_uninterrupted_finetuning_predictions_are_reproducible(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    uninterrupted = train_backbone_finetune(_finetune_config(tmp_path, data_dir, checkpoint, "uninterrupted", max_updates=6))

    train_backbone_finetune(_finetune_config(tmp_path, data_dir, checkpoint, "resumed", max_updates=6, stop_after_updates=3))
    resumed = train_backbone_finetune(_finetune_config(tmp_path, data_dir, checkpoint, "resumed", max_updates=6, resume=True))

    a = torch.load(Path(uninterrupted["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    b = torch.load(Path(resumed["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    for name, tensor in a["model_state_dict"].items():
        torch.testing.assert_close(tensor, b["model_state_dict"][name], rtol=1e-6, atol=1e-6)
    assert a["metrics_history"] == b["metrics_history"]
    assert Path(uninterrupted["artifacts"]["validation_predictions"]).read_text(encoding="utf-8") == Path(
        resumed["artifacts"]["validation_predictions"]
    ).read_text(encoding="utf-8")


def test_evaluation_mode_is_deterministic(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config = _finetune_config(tmp_path, data_dir, checkpoint, "deterministic")
    model = build_model_from_head_checkpoint(torch, config, torch.device("cpu"))
    samples = load_partition(data_dir, "valid", DEFAULT_TARGETS)
    scaler = fit_target_scaler(load_partition(data_dir, "train", DEFAULT_TARGETS), DEFAULT_TARGETS)

    first, rows = evaluate_deterministic(torch, model, samples, config, scaler, torch.device("cpu"))
    second, rows_again = evaluate_deterministic(torch, model, samples, config, scaler, torch.device("cpu"))

    assert first == second
    assert rows == rows_again


def test_small_end_to_end_finetuning_smoke_run_finishes(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config = _finetune_config(tmp_path, data_dir, checkpoint, "smoke", max_updates=3)
    summary = train_backbone_finetune(config)

    assert Path(summary["artifacts"]["validation_predictions"]).exists()
    assert Path(summary["artifacts"]["best_checkpoint"]).exists()
    assert summary["parameter_counts"]["backbone"]["trainable"] > 0
    assert summary["recomputed_prediction_metrics"]["mean_mse"] == pytest.approx(summary["final_metrics"]["mean_mse"])


def test_backbone_finetune_cli_smoke(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    checkpoint = _head_checkpoint(tmp_path, data_dir)
    config_path = tmp_path / "finetune.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fluorcast_uniprop_backbone_finetune_v1",
                "data_dir": str(data_dir),
                "output_dir": str(tmp_path / "cli"),
                "head_checkpoint": str(checkpoint),
                "targets": list(DEFAULT_TARGETS),
                "seed": 13,
                "subset_size": 12,
                "train_batch_size": 8,
                "micro_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "valid_batch_size": 10,
                "max_updates": 2,
                "backbone_learning_rate": 0.0003,
                "head_learning_rate": 0.003,
                "early_stopping_patience": 10,
                "hidden_dim": 48,
                "solvent_adapter_dim": 24,
                "device": "cpu",
                "use_amp": True,
                "use_ema": False
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_uniprop_backbone_finetune.py",
            "--config",
            str(config_path),
            "--print-summary",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli" / "validation_predictions.csv").exists()


def test_backbone_finetune_slurm_script_is_cuda_gpu_compatible() -> None:
    script = (PROJECT_ROOT / "slurm/uniprop/run_uniprop_backbone_finetune.sbatch").read_text(encoding="utf-8")

    assert "--gpus-per-node" in script
    assert "any CUDA GPU" in script
    assert "scripts/train_uniprop_backbone_finetune.py" in script
