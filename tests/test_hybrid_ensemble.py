from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.ensemble import (
    align_features, load_hybrid_ensemble, predict_hybrid_ensemble,
    save_hybrid_ensemble, train_hybrid_ensemble,
)
from chemfluor.hybrid.uncertainty import calibration_residuals


def test_trained_model_saves_and_reloads(tmp_path: Path) -> None:
    features = pd.DataFrame({"emission_nm_mean": np.arange(10.0), "pair_seen_score": [1, 0] * 5})
    labels = pd.Series(400 + 2 * np.arange(10.0))
    model = train_hybrid_ensemble(features, labels, "emission_nm")
    save_hybrid_ensemble(model, list(features.columns), tmp_path)
    loaded, columns, metadata = load_hybrid_ensemble(tmp_path)
    result = predict_hybrid_ensemble(loaded, align_features(features.iloc[[0]], columns))
    assert isinstance(result["prediction"], float)
    assert metadata["target_name"] == "emission_nm"
    assert (tmp_path / "feature_columns.json").exists()


def test_quantum_yield_trains_calibrated_bright_classifier() -> None:
    features = pd.DataFrame({"quantum_yield_mean": np.linspace(0, 1, 12)})
    model = train_hybrid_ensemble(features, pd.Series(np.linspace(0, 0.5, 12)), "quantum_yield")
    assert model["classifier"] is not None
    assert "bright_probability" in predict_hybrid_ensemble(model, features.iloc[[0]])


def test_apply_script_produces_valid_json(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    prediction_csv = tmp_path / "predictions.csv"
    table = pd.DataFrame({
        "model": ["a", "b"],
        "predicted_emission_nm": [500.0, 510.0],
        "overall_confidence_score": [0.8, 0.8],
    })
    table.to_csv(prediction_csv, index=False)
    training_features = pd.DataFrame({
        "emission_nm_mean": np.arange(10.0) + 500,
        "overall_confidence_score": np.linspace(0.4, 0.9, 10),
    })
    model = train_hybrid_ensemble(
        training_features, pd.Series(np.arange(10.0) + 501), "emission_nm"
    )
    model_dir = tmp_path / "model"
    save_hybrid_ensemble(model, list(training_features.columns), model_dir, {"conformal_coverage": 0.9})
    calibration_residuals([500, 510], [498, 507], [0.8, 0.8]).to_csv(
        model_dir / "calibration_residuals.csv", index=False
    )
    output = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable, str(project / "scripts" / "apply_hybrid_ensemble.py"),
            "--prediction-csv", str(prediction_csv), "--model-dir", str(model_dir),
            "--out-json", str(output), "--out-md", str(tmp_path / "report.md"),
        ],
        cwd=project, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["hybrid_ensemble"]["prediction_interval"]["lower"] <= report["hybrid_ensemble"]["prediction"]
    assert "Prediction interval" in (tmp_path / "report.md").read_text(encoding="utf-8")
