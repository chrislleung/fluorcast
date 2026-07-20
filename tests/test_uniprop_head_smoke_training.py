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

from chemfluor.uniprop.head_smoke_training import (  # noqa: E402
    DEFAULT_TARGETS,
    HeadOnlySmokeModel,
    HeadSmokeConfig,
    assert_only_intended_trainable,
    recompute_metrics_from_predictions,
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


def _record(index: int, partition: str, *, all_missing: bool = False) -> dict[str, object]:
    solvent_values = ["O", "CCO", "CC#N", "CS(C)=O"]
    molecule_values = ["CCO", "CCN", "c1ccccc1", "CCCl"]
    solvent = solvent_values[index % len(solvent_values)]
    molecule = molecule_values[index % len(molecule_values)]
    base = float(index)
    targets = np.asarray([300.0 + 2.0 * base, 430.0 + 3.0 * base, 0.05 + 0.01 * base], dtype=np.float32)
    mask = np.asarray([True, index % 5 != 1, index % 4 != 2], dtype=np.bool_)
    if all_missing:
        targets[:] = np.nan
        mask[:] = False
    else:
        targets[~mask] = np.nan
    return {
        "atoms": np.asarray(["C", "C", "O"]),
        "input_pos": [np.zeros((3, 3), dtype=np.float32) + index],
        "label_pos": np.zeros((3, 3), dtype=np.float32) + index,
        "smi": molecule,
        "solvent_smi": solvent,
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


def _fixture_lmdb(tmp_path: Path, *, n_train: int = 12, n_valid: int = 12) -> Path:
    data_dir = tmp_path / "lmdb"
    _write_lmdb(data_dir / "train.lmdb", [_record(index, "train") for index in range(n_train)])
    _write_lmdb(data_dir / "valid.lmdb", [_record(index, "valid") for index in range(n_valid)])
    _write_lmdb(data_dir / "test.lmdb", [_record(index, "test") for index in range(3)])
    return data_dir


def _config(tmp_path: Path, data_dir: Path, output_name: str, *, max_updates: int = 12, resume: bool = False, stop_after_updates: int | None = None) -> HeadSmokeConfig:
    return HeadSmokeConfig(
        data_dir=data_dir,
        output_dir=tmp_path / output_name,
        seed=11,
        subset_size=12,
        train_batch_size=12,
        valid_batch_size=12,
        max_updates=max_updates,
        learning_rate=0.01,
        hidden_dim=48,
        solvent_adapter_dim=24,
        resume=resume,
        stop_after_updates=stop_after_updates,
        device="cpu",
    )


def test_training_loss_is_finite_and_artifacts_are_written(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    summary = train_head_smoke(_config(tmp_path, data_dir, "run", max_updates=4))
    out_dir = Path(summary["artifacts"]["metrics_history"]).parent
    history = json.loads((out_dir / "metrics_history.json").read_text(encoding="utf-8"))

    assert np.isfinite(history[-1]["train_loss"])
    for name in [
        "resolved_config.json",
        "environment_report.json",
        "dataset_split_hashes.json",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "scalers.json",
        "metrics_history.json",
        "validation_predictions.csv",
    ]:
        assert (out_dir / name).exists()


def test_at_least_one_relevant_loss_decreases_on_overfit_fixture(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    config = _config(tmp_path, data_dir, "overfit", max_updates=35)
    summary = train_head_smoke(config)
    history = json.loads((Path(summary["artifacts"]["metrics_history"])).read_text(encoding="utf-8"))

    assert history[-1]["train_loss"] < history[0]["train_loss"]


def test_only_intended_parameters_are_trainable_and_change(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    config = _config(tmp_path, data_dir, "params", max_updates=2)
    torch.manual_seed(config.seed)
    before_model = HeadOnlySmokeModel.build(torch, config)
    before = {name: value.detach().clone() for name, value in before_model.state_dict().items()}
    assert_only_intended_trainable(before_model)

    summary = train_head_smoke(config)
    checkpoint = torch.load(Path(summary["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    after = checkpoint["model_state_dict"]

    assert torch.equal(before["backbone.weight"], after["backbone.weight"])
    assert torch.equal(before["backbone.bias"], after["backbone.bias"])
    changed = [name for name, value in after.items() if not name.startswith("backbone.") and not torch.equal(before[name], value)]
    assert any(name.startswith("solvent_adapter.") for name in changed)
    assert any(name.startswith("fusion.") for name in changed)
    assert any(name.startswith("heads.") for name in changed)


def test_interrupted_training_resumes_at_exact_next_update(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    train_head_smoke(_config(tmp_path, data_dir, "resume", max_updates=8, stop_after_updates=3))

    partial = torch.load(tmp_path / "resume" / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert partial["update_index"] == 2

    train_head_smoke(_config(tmp_path, data_dir, "resume", max_updates=8, resume=True))
    resumed = torch.load(tmp_path / "resume" / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert resumed["update_index"] == 7
    assert [entry["update_index"] for entry in resumed["metrics_history"]] == list(range(8))


def test_resumed_and_uninterrupted_deterministic_runs_agree(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    uninterrupted = train_head_smoke(_config(tmp_path, data_dir, "uninterrupted", max_updates=10))

    train_head_smoke(_config(tmp_path, data_dir, "resumed", max_updates=10, stop_after_updates=4))
    resumed = train_head_smoke(_config(tmp_path, data_dir, "resumed", max_updates=10, resume=True))

    a = torch.load(Path(uninterrupted["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    b = torch.load(Path(resumed["artifacts"]["last_checkpoint"]), map_location="cpu", weights_only=False)
    for name, tensor in a["model_state_dict"].items():
        torch.testing.assert_close(tensor, b["model_state_dict"][name], rtol=1e-6, atol=1e-6)
    assert a["metrics_history"] == b["metrics_history"]


def test_best_and_last_checkpoints_are_distinct_and_valid(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    summary = train_head_smoke(_config(tmp_path, data_dir, "checkpoints", max_updates=5))
    best_path = Path(summary["artifacts"]["best_checkpoint"])
    last_path = Path(summary["artifacts"]["last_checkpoint"])

    assert best_path != last_path
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    assert best["checkpoint_kind"] == "best"
    assert last["checkpoint_kind"] == "last"
    assert "optimizer_state_dict" in last
    assert "scheduler_state_dict" in last


def test_metrics_recomputed_from_predictions_match_logged_metrics(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    summary = train_head_smoke(_config(tmp_path, data_dir, "metrics", max_updates=4))
    recomputed = recompute_metrics_from_predictions(Path(summary["artifacts"]["validation_predictions"]), DEFAULT_TARGETS)

    for key, value in recomputed.items():
        assert value == pytest.approx(summary["final_metrics"][key], rel=1e-6, abs=1e-6)


def test_empty_target_batches_and_test_evaluation_are_rejected(tmp_path: Path) -> None:
    data_dir = tmp_path / "bad_lmdb"
    _write_lmdb(data_dir / "train.lmdb", [_record(index, "train", all_missing=True) for index in range(3)])
    _write_lmdb(data_dir / "valid.lmdb", [_record(index, "valid") for index in range(3)])
    with pytest.raises(ValueError, match="no non-missing"):
        train_head_smoke(_config(tmp_path, data_dir, "bad", max_updates=1))

    good_dir = _fixture_lmdb(tmp_path / "good")
    config = _config(tmp_path, good_dir, "test_eval", max_updates=1)
    config = HeadSmokeConfig(**{**config.__dict__, "validation_partition": "test"})
    with pytest.raises(ValueError, match="test-set evaluation"):
        train_head_smoke(config)


def test_cli_smoke(tmp_path: Path) -> None:
    data_dir = _fixture_lmdb(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "fluorcast_uniprop_head_smoke_v1",
                "data_dir": str(data_dir),
                "output_dir": str(tmp_path / "cli"),
                "targets": list(DEFAULT_TARGETS),
                "seed": 11,
                "subset_size": 12,
                "train_batch_size": 12,
                "valid_batch_size": 12,
                "max_updates": 3,
                "learning_rate": 0.01,
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/train_uniprop_head_smoke.py",
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


def test_head_smoke_slurm_script_has_minimal_gpu_job() -> None:
    script = (PROJECT_ROOT / "slurm/uniprop/run_uniprop_head_smoke.sbatch").read_text(encoding="utf-8")

    assert "--gpus-per-node" in script
    assert "scripts/train_uniprop_head_smoke.py" in script
    assert "validation_predictions.csv" in script
