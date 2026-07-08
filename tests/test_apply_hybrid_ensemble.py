from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from chemfluor.hybrid.ensemble import save_hybrid_ensemble, train_hybrid_ensemble
from chemfluor.hybrid.uncertainty import calibration_residuals


SCRIPT = PROJECT_ROOT / "scripts" / "apply_hybrid_ensemble.py"


def _run_apply(tmp_path: Path, target: str, prediction_column: str) -> dict:
    prediction_csv = tmp_path / f"{target}_predictions.csv"
    pd.DataFrame(
        {
            "model": ["a", "b"],
            prediction_column: [400.0, 420.0],
            "overall_confidence_score": [0.8, 0.8],
        }
    ).to_csv(prediction_csv, index=False)
    training_features = pd.DataFrame(
        {
            f"{target}_mean": np.linspace(390, 430, 12),
            "overall_confidence_score": np.linspace(0.4, 0.9, 12),
        }
    )
    model = train_hybrid_ensemble(
        training_features, pd.Series(np.linspace(395, 435, 12)), target
    )
    model_dir = tmp_path / f"{target}_model"
    save_hybrid_ensemble(
        model, list(training_features.columns), model_dir, {"conformal_coverage": 0.9}
    )
    calibration_residuals([400, 410], [398, 407], [0.8, 0.8]).to_csv(
        model_dir / "calibration_residuals.csv", index=False
    )
    output = tmp_path / f"{target}_report.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--prediction-csv",
            str(prediction_csv),
            "--model-dir",
            str(model_dir),
            "--out-json",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_absorption_target_writes_absorption_field(tmp_path: Path) -> None:
    report = _run_apply(tmp_path, "absorption_nm", "predicted_absorption_nm")
    assert report["final_absorption_prediction_nm"] is not None
    assert report.get("final_emission_prediction_nm") is None
    assert report.get("final_quantum_yield_prediction") is None


def test_emission_target_writes_emission_field(tmp_path: Path) -> None:
    report = _run_apply(tmp_path, "emission_nm", "predicted_emission_nm")
    assert report["final_emission_prediction_nm"] is not None
    assert report.get("final_absorption_prediction_nm") is None
    assert report.get("final_quantum_yield_prediction") is None


def test_quantum_yield_target_writes_quantum_yield_field(tmp_path: Path) -> None:
    report = _run_apply(tmp_path, "quantum_yield", "predicted_quantum_yield")
    assert report["final_quantum_yield_prediction"] is not None
    assert report.get("final_absorption_prediction_nm") is None
    assert report.get("final_emission_prediction_nm") is None
