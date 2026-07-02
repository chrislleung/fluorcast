"""Tests for manuscript classification metrics from regression predictions."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "manuscript" / "compute_prediction_f1_scores.py"
SPEC = importlib.util.spec_from_file_location("compute_prediction_f1_scores", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quantum_yield_binning() -> None:
    assert MODULE.bin_quantum_yield([0.0, 0.25, 0.25001]).tolist() == [
        "dim",
        "dim",
        "bright",
    ]


def test_wavelength_binning_boundaries() -> None:
    values = [399.9, 400, 499.9, 500, 559.9, 560, 619.9, 620]
    assert MODULE.bin_wavelength(values).tolist() == [
        "UV",
        "blue",
        "blue",
        "green",
        "green",
        "yellow/orange",
        "yellow/orange",
        "red/NIR",
    ]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (
            "emission_nm__rf__molecule__seed0.csv",
            {"target": "emission_nm", "model": "rf", "split": "molecule", "seed": 0},
        ),
        (
            "quantum_yield__extratrees__random__seed2.csv",
            {
                "target": "quantum_yield",
                "model": "extratrees",
                "split": "random",
                "seed": 2,
            },
        ),
    ],
)
def test_filename_parsing(filename: str, expected: dict[str, object]) -> None:
    assert MODULE.parse_prediction_filename(filename) == expected


def test_metric_calculation_from_artificial_prediction_csv(tmp_path: Path) -> None:
    path = tmp_path / "quantum_yield__rf__random__seed0.csv"
    pd.DataFrame(
        {
            "actual_quantum_yield": [0.1, 0.2, 0.4, None],
            "predicted_quantum_yield": [0.2, 0.3, 0.5, 0.1],
        }
    ).to_csv(path, index=False)
    metadata = MODULE.parse_prediction_filename(path)
    overall, per_class, confusion = MODULE.process_prediction_file(path, metadata, 0.25)

    assert overall["n_rows"] == 3
    assert overall["accuracy"] == pytest.approx(2 / 3)
    assert overall["macro_f1"] == pytest.approx(2 / 3)
    dim = per_class[per_class["class_label"] == "dim"].iloc[0]
    assert dim["support"] == 2
    assert confusion["count"].sum() == 3


def test_cli_smoke_run_creates_all_required_outputs(tmp_path: Path) -> None:
    results_dir = tmp_path / "paper_results"
    predictions_dir = results_dir / "predictions"
    predictions_dir.mkdir(parents=True)
    pd.DataFrame(
        {"y_true": [390, 410, 530, 580, 650], "y_pred": [395, 405, 540, 610, 640]}
    ).to_csv(predictions_dir / "emission_nm__rf__molecule__seed0.csv", index=False)
    pd.DataFrame(
        {"actual": [0.1, 0.25, 0.5], "predicted": [0.2, 0.3, 0.4]}
    ).to_csv(predictions_dir / "quantum_yield__rf__molecule__seed0.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--paper-results-dir",
            str(results_dir),
            "--targets",
            "emission_nm,quantum_yield",
            "--splits",
            "molecule",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    out_dir = results_dir / "classification_metrics"
    expected = [
        "f1_metrics_by_target_split_model_seed.csv",
        "f1_metrics_aggregated_by_seed.csv",
        "per_class_f1_by_target_split_model_seed.csv",
        "per_class_f1_aggregated_by_seed.csv",
        "confusion_matrices.csv",
        "best_f1_by_target_split.csv",
        "f1_summary.md",
    ]
    for filename in expected:
        assert (out_dir / filename).exists()
    metrics = pd.read_csv(out_dir / expected[0])
    assert len(metrics) == 2
    summary = (out_dir / "f1_summary.md").read_text(encoding="utf-8")
    assert "interpreted separately from regression MAE" in summary
    assert "QY <= 0.25" in summary
