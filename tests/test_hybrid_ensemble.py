from pathlib import Path
import json
import importlib.util
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.ensemble import (
    align_features, load_hybrid_ensemble, predict_hybrid_ensemble,
    save_hybrid_ensemble, train_hybrid_ensemble,
)
from chemfluor.hybrid.uncertainty import calibration_residuals

EVALUATOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_hybrid_ensemble.py"
EVALUATOR_SPEC = importlib.util.spec_from_file_location("evaluate_hybrid_ensemble", EVALUATOR_PATH)
assert EVALUATOR_SPEC and EVALUATOR_SPEC.loader
evaluator = importlib.util.module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


class StubPredictor:
    def predict(self, features):
        return np.zeros(len(features))


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


def _run_wide_training(
    tmp_path: Path, target_name: str, target_column: str, labels: list[float]
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    project = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / f"{target_name}_wide.csv"
    prediction_suffix = target_name
    rows = len(labels)
    table = pd.DataFrame({
        target_column: labels,
        "solvent": ["water"] * rows,
        "canonical_smiles": [f"C{'C' * (index % 4)}" for index in range(rows)],
        f"rf_{prediction_suffix}": np.linspace(0.1, 1.0, rows),
        f"gbdt_{prediction_suffix}": np.linspace(0.2, 1.1, rows),
        "prediction_mean": np.linspace(0.15, 1.05, rows),
        "prediction_std": np.linspace(0.01, 0.1, rows),
        "prediction_count": [2] * rows,
        "overall_confidence_score": np.linspace(0.2, 0.9, rows),
    })
    table.loc[0, target_column] = np.nan
    table.to_csv(csv_path, index=False)
    out_dir = tmp_path / f"model_{target_name}"
    completed = subprocess.run(
        [
            sys.executable, str(project / "scripts" / "train_hybrid_ensemble.py"),
            "--prediction-csv", str(csv_path), "--target-column", target_column,
            "--target-name", target_name, "--out-dir", str(out_dir),
        ],
        cwd=project, capture_output=True, text=True,
    )
    return out_dir, completed


def test_wide_multirow_emission_table_trains_and_ignores_identifiers(tmp_path: Path) -> None:
    labels = list(np.linspace(400, 600, 16))
    out_dir, completed = _run_wide_training(
        tmp_path, "emission_nm", "true_emission_nm", labels
    )
    assert completed.returncode == 0, completed.stderr
    assert all((out_dir / name).exists() for name in (
        "model.joblib", "feature_columns.json", "metadata.json", "calibration_residuals.csv"
    ))
    columns = json.loads((out_dir / "feature_columns.json").read_text(encoding="utf-8"))
    assert "true_emission_nm" not in columns
    assert "solvent" not in columns
    assert "canonical_smiles" not in columns
    assert "rf_emission_nm" in columns
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_format"] == "wide"
    assert metadata["n_dropped_missing_targets"] == 1


def test_wide_multirow_quantum_yield_trains_regression_and_classifier(tmp_path: Path) -> None:
    labels = [0.02, 0.30] * 10
    out_dir, completed = _run_wide_training(
        tmp_path, "quantum_yield", "true_quantum_yield", labels
    )
    assert completed.returncode == 0, completed.stderr
    model, columns, _ = load_hybrid_ensemble(out_dir)
    assert "true_quantum_yield" not in columns
    assert "rf_quantum_yield" in columns
    assert model["regressor"] is not None
    assert model["classifier"] is not None


def test_extract_predictor_accepts_raw_estimator() -> None:
    predictor = StubPredictor()
    assert evaluator.extract_predictor(predictor) is predictor


def test_extract_predictor_accepts_likely_dictionary_keys() -> None:
    model_predictor = StubPredictor()
    estimator_predictor = StubPredictor()
    assert evaluator.extract_predictor({"model": model_predictor}) is model_predictor
    assert evaluator.extract_predictor({"estimator": estimator_predictor}) is estimator_predictor


def test_extract_predictor_finds_one_unexpected_dictionary_value() -> None:
    predictor = StubPredictor()
    assert evaluator.extract_predictor({"metadata": {}, "fitted_object": predictor}) is predictor


def test_extract_predictor_rejects_dictionary_without_predictor() -> None:
    with pytest.raises(ValueError, match="no object.*Available keys"):
        evaluator.extract_predictor({"metadata": {}, "target": "emission_nm"})


def test_extract_predictor_rejects_multiple_unexpected_predictors() -> None:
    with pytest.raises(ValueError, match="multiple predictors.*Available keys"):
        evaluator.extract_predictor({"first": StubPredictor(), "second": StubPredictor()})
