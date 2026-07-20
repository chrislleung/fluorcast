from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.experiment_matrix import (  # noqa: E402
    MatrixConfig,
    aggregate_experiment,
    evaluate_test_run,
    metrics_from_arrays,
    run_matrix,
    validate_run_dir,
)
from chemfluor.uniprop.manifests import MANIFEST_SCHEMA_VERSION, make_split_assignments, stable_hash  # noqa: E402


def _write_matrix_fixture(tmp_path: Path) -> MatrixConfig:
    molecules = []
    smiles = ["CCO", "CCN", "CCC", "CCCl", "c1ccccc1", "CC(C)O"]
    for smi in smiles:
        molecule_id = stable_hash("mol", MANIFEST_SCHEMA_VERSION, smi)
        molecules.append(
            {
                "molecule_id": molecule_id,
                "canonical_isomeric_smiles": smi,
                "canonical_nonisomeric_smiles": smi,
                "source_row_count": 6,
            }
        )
    solvents = ["O", "CCO", "CC#N", "CS(C)=O", "CCl4", "CO"]
    rows = []
    index = 0
    for mol_index, molecule in enumerate(molecules):
        for solvent_index, solvent in enumerate(solvents):
            solvent_id = stable_hash("solv", MANIFEST_SCHEMA_VERSION, solvent)
            rows.append(
                {
                    "row_id": f"row_{index:04d}",
                    "molecule_id": molecule["molecule_id"],
                    "solvent_id": solvent_id,
                    "canonical_solvent_smiles": solvent,
                    "source_dataset": "fixture",
                    "absorption_nm": 300.0 + 2.0 * mol_index + solvent_index,
                    "absorption_nm_available": True,
                    "emission_nm": 430.0 + 3.0 * mol_index + solvent_index,
                    "emission_nm_available": True,
                    "quantum_yield": 0.05 + 0.03 * mol_index + 0.01 * solvent_index,
                    "quantum_yield_available": True,
                }
            )
            index += 1
    root = tmp_path / "manifest"
    root.mkdir()
    row_manifest = pd.DataFrame(rows)
    molecule_manifest = pd.DataFrame(molecules)
    split_assignments = make_split_assignments(row_manifest, molecule_manifest, test_size=0.25, seed=5)
    row_manifest.to_csv(root / "row_manifest.csv", index=False)
    molecule_manifest.to_csv(root / "molecule_manifest.csv", index=False)
    split_assignments.to_csv(root / "split_assignments.csv", index=False)
    return MatrixConfig(
        row_manifest=root / "row_manifest.csv",
        molecule_manifest=root / "molecule_manifest.csv",
        split_assignments=root / "split_assignments.csv",
        out_dir=tmp_path / "experiment",
        split_families=("random", "molecule"),
        model_variants=("morgan_rdkit_baseline", "uniprop_frozen_backbone"),
        targets=("emission_nm", "quantum_yield"),
        seeds=(1, 2, 3),
        max_rows_per_partition=8,
        bootstrap_samples=25,
        overwrite=True,
    )


def test_split_leakage_audit_runs_before_training(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    splits = pd.read_csv(config.split_assignments)
    splits["molecule"] = splits["random"]
    splits.to_csv(config.split_assignments, index=False)
    bad = MatrixConfig(**{**config.__dict__, "split_families": ("molecule",)})

    with pytest.raises(ValueError, match="leakage audit failed"):
        run_matrix(bad)

    assert not (config.out_dir / "runs").exists()


def test_one_complete_small_scale_matrix_runs_and_summarizes(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    status = run_matrix(config)

    assert status["expected_runs"] == 2 * 2 * 2 * 3
    assert status["completed_runs"] == status["expected_runs"]
    for run_dir in sorted((config.out_dir / "runs").iterdir()):
        evaluate_test_run(run_dir, config)
        assert validate_run_dir(run_dir)["valid"]
    aggregate = aggregate_experiment(config.out_dir, bootstrap_samples=25)

    assert aggregate["runs_included"] == status["expected_runs"]
    assert aggregate["summary_rows"] == 2 * 2 * 2
    summary = pd.read_csv(config.out_dir / "aggregate_summary.csv")
    assert set(["mae_mean", "mae_std", "mae_ci_low", "mae_ci_high", "train_rows", "valid_rows", "test_rows"]).issubset(summary.columns)


def test_metrics_independently_recompute_from_predictions(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    run_dir = sorted((config.out_dir / "runs").iterdir())[0]
    evaluate_test_run(run_dir, config)
    report = validate_run_dir(run_dir)

    assert report["valid"]
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    recomputed = metrics_from_arrays(predictions["y_true"].to_numpy(), predictions["y_pred"].to_numpy(), payload["target"], 0.25)
    assert recomputed["mae"] == pytest.approx(payload["metrics"]["test"]["overall"]["mae"])


def test_models_use_identical_target_rows_for_same_comparison(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    left = config.out_dir / "runs" / "random__emission_nm__morgan_rdkit_baseline__seed1" / "valid_predictions.csv"
    right = config.out_dir / "runs" / "random__emission_nm__uniprop_frozen_backbone__seed1" / "valid_predictions.csv"

    assert pd.read_csv(left)["row_id"].tolist() == pd.read_csv(right)["row_id"].tolist()


def test_failed_or_partial_runs_are_excluded_and_reported(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    run_dirs = sorted((config.out_dir / "runs").iterdir())
    evaluate_test_run(run_dirs[0], config)
    aggregate = aggregate_experiment(config.out_dir, bootstrap_samples=10)
    excluded = json.loads((config.out_dir / "excluded_runs.json").read_text(encoding="utf-8"))

    assert aggregate["runs_included"] == 1
    assert aggregate["runs_excluded"] == len(run_dirs) - 1
    assert any("test metrics missing" in " ".join(item["errors"]) for item in excluded)


def test_aggregation_is_invariant_to_file_order(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    for run_dir in sorted((config.out_dir / "runs").iterdir()):
        evaluate_test_run(run_dir, config)
    first = aggregate_experiment(config.out_dir, bootstrap_samples=10)
    first_csv = (config.out_dir / "aggregate_summary.csv").read_text(encoding="utf-8")
    aggregate_summary = config.out_dir / "aggregate_summary.csv"
    aggregate_summary.unlink()
    second = aggregate_experiment(config.out_dir, bootstrap_samples=10)
    second_csv = aggregate_summary.read_text(encoding="utf-8")

    assert first == second
    assert first_csv == second_csv


def test_duplicate_run_ids_are_rejected(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    for run_dir in sorted((config.out_dir / "runs").iterdir())[:1]:
        evaluate_test_run(run_dir, config)
        clone = config.out_dir / "runs" / "duplicate_dir"
        shutil.copytree(run_dir, clone)

    with pytest.raises(ValueError, match="Duplicate run IDs"):
        aggregate_experiment(config.out_dir)


def test_test_predictions_cannot_be_overwritten_accidentally(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    run_matrix(config)
    run_dir = sorted((config.out_dir / "runs").iterdir())[0]
    evaluate_test_run(run_dir, config)

    with pytest.raises(FileExistsError, match="already exist"):
        evaluate_test_run(run_dir, config)


def test_summary_row_counts_match_experiment_matrix(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    status = run_matrix(config)
    for run_dir in sorted((config.out_dir / "runs").iterdir()):
        evaluate_test_run(run_dir, config)
    aggregate_experiment(config.out_dir)
    per_run = pd.read_csv(config.out_dir / "per_run_summary.csv")

    assert len(per_run) == status["expected_runs"]


def test_experiment_matrix_cli_smoke(tmp_path: Path) -> None:
    config = _write_matrix_fixture(tmp_path)
    config_path = tmp_path / "matrix.json"
    payload = {
        "schema_version": "fluorcast_uniprop_experiment_matrix_v1",
        "row_manifest": str(config.row_manifest),
        "molecule_manifest": str(config.molecule_manifest),
        "split_assignments": str(config.split_assignments),
        "out_dir": str(tmp_path / "cli_matrix"),
        "split_families": ["random"],
        "model_variants": ["morgan_rdkit_baseline"],
        "targets": ["emission_nm"],
        "seeds": [1],
        "max_rows_per_partition": 8,
        "bootstrap_samples": 10,
    }
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    for command in [
        [sys.executable, "scripts/run_uniprop_experiment_matrix.py", "run", "--config", str(config_path), "--overwrite"],
        [sys.executable, "scripts/run_uniprop_experiment_matrix.py", "evaluate-test", "--config", str(config_path)],
        [sys.executable, "scripts/run_uniprop_experiment_matrix.py", "validate", "--experiment-dir", str(tmp_path / "cli_matrix")],
        [sys.executable, "scripts/run_uniprop_experiment_matrix.py", "summarize", "--experiment-dir", str(tmp_path / "cli_matrix")],
    ]:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr

    assert (tmp_path / "cli_matrix" / "aggregate_summary.csv").exists()


def test_matrix_slurm_script_submits_without_source_edits() -> None:
    script = (PROJECT_ROOT / "slurm/uniprop/run_uniprop_experiment_matrix.sbatch").read_text(encoding="utf-8")

    assert "FLUORCAST_UNIPROP_MATRIX_CONFIG" in script
    assert "run_uniprop_experiment_matrix.py run" in script
    assert "run_uniprop_experiment_matrix.py evaluate-test" in script
    assert "run_uniprop_experiment_matrix.py summarize" in script
