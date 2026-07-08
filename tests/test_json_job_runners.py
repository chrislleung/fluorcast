from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str) -> Any:
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


prediction_runner = load_script("run_prediction_job")
duplicate_runner = load_script("run_duplicate_check_job")


class ConstantRegressor:
    def __init__(self, value: float, n_features: int = 33) -> None:
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=float)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def prediction_input() -> dict[str, Any]:
    return {
        "job_id": "job-1",
        "user_id": "user-1",
        "molecule_smiles": "CCO",
        "solvent_smiles": "O",
        "model_choice": "all",
        "requested_at": "2026-01-01T00:00:00Z",
    }


def duplicate_input() -> dict[str, Any]:
    return {
        "submission_id": "submission-1",
        "user_id": "user-1",
        "molecule_smiles": "OCC",
        "solvent_smiles": "O",
        "submitted_at": "2026-01-01T00:00:00Z",
    }


def test_prediction_backend_failure_is_written(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "nested" / "output.json"
    write_json(input_path, prediction_input())

    def disconnected(_: dict[str, Any]) -> Any:
        raise prediction_runner.JobError(
            "PREDICTION_BACKEND_NOT_CONNECTED", "No artifacts are configured."
        )

    assert prediction_runner.run_job(input_path, output_path, disconnected) == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["job_id"] == "job-1"
    assert result["error_code"] == "PREDICTION_BACKEND_NOT_CONNECTED"
    assert result["error_message"]
    assert "Traceback" in result["traceback"]
    assert result["warnings"] == []


def test_prediction_invalid_input_is_written(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, {"job_id": "job-2"})

    assert prediction_runner.run_job(input_path, output_path) == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["error_code"] == "INVALID_INPUT"


def test_prediction_invalid_smiles_is_written(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    payload = prediction_input()
    payload["molecule_smiles"] = "not smiles"
    write_json(input_path, payload)

    assert prediction_runner.run_job(input_path, output_path) == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_message"]
    assert result["traceback"]


def test_duplicate_invalid_input_is_written(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, {"submission_id": "submission-2"})

    assert duplicate_runner.run_job(input_path, output_path, None) == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_code"] == "INVALID_INPUT"
    assert result["error_message"]
    assert result["traceback"]


def test_duplicate_checker_finds_canonical_exact_pair(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "record_id": ["record-7", "record-8"],
            "molecule_smiles": ["CCO", "c1ccccc1"],
            "solvent_smiles": ["[OH2]", "CCO"],
            "absorption_nm": [350.0, 390.0],
            "emission_nm": [510.0, 420.0],
            "quantum_yield": [0.25, None],
            "lifetime_ns": [3.2, None],
            "source_doi": ["10.1234/example", None],
        }
    ).to_csv(dataset, index=False)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, duplicate_input())

    assert duplicate_runner.run_job(input_path, output_path, dataset, 2) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["exact_duplicate_found"] is True
    assert result["exact_duplicate_record_id"] == "record-7"
    assert result["canonical_molecule_smiles"] == "CCO"
    assert result["canonical_solvent_smiles"] == "O"
    assert result["nearest_matches"][0]["record_id"] == "record-7"
    assert result["nearest_matches"][0]["similarity"] == 1.0
    assert "emission_nm" in result["nearest_matches"][0]
    assert "quantum_yield" in result["nearest_matches"][0]
    assert "absorption_nm" not in result["nearest_matches"][0]
    assert "lifetime_ns" not in result["nearest_matches"][0]


def test_duplicate_nearest_match_missing_nullable_fields_are_null(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "record_id": ["record-missing"],
            "molecule_smiles": ["CCO"],
            "solvent_smiles": [None],
            "emission_nm": [None],
            "quantum_yield": [None],
            "source_doi": [None],
        }
    ).to_csv(dataset, index=False)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, duplicate_input())

    assert duplicate_runner.run_job(input_path, output_path, dataset, 1) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    nearest = result["nearest_matches"][0]
    assert nearest["record_id"] == "record-missing"
    assert nearest["solvent_smiles"] is None
    assert nearest["emission_nm"] is None
    assert nearest["quantum_yield"] is None
    assert nearest["source_doi"] is None


def test_prediction_success_contract_with_injected_backend(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, prediction_input())

    def backend(_: dict[str, Any]) -> Any:
        return ([{"model_name": "test"}], ["test warning"], "CCO", "O")

    assert prediction_runner.run_job(input_path, output_path, backend) == 0
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "success"
    assert result["predictions"] == [{"model_name": "test"}]
    assert result["warnings"] == ["test warning"]


def prediction_table(model_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "predicted_absorption_nm": 350.0,
                "predicted_emission_nm": 500.0,
                "predicted_quantum_yield": 0.3,
                "nearest_training_similarity": 0.75,
                "nearest_training_smiles": "CCO",
                "confidence_label": "high",
                "outside_applicability_domain": False,
            }
        ]
    )


def install_model_availability_fixture(monkeypatch: Any) -> None:
    def collect(_: dict[str, Any], model_name: str) -> Any:
        if model_name in {"rf", "extratrees"}:
            return prediction_table(model_name), [], "CCO", "O"
        if model_name == "graph_model_later":
            raise prediction_runner.JobError("MODEL_UNAVAILABLE", "Graph unavailable")
        return (
            prediction_table(model_name).iloc[0:0],
            [f"Failed to predict with {model_name}; skipping: No module named '_loss'"],
            "CCO",
            "O",
        )

    monkeypatch.setattr(prediction_runner, "_collect_model", collect)

    def collect_all(_: dict[str, Any]) -> Any:
        warnings = [
            prediction_runner._unavailable_message(name)
            for name in ("gbdt", "histgb", "graph_model_later")
        ]
        return (
            pd.concat([prediction_table("rf"), prediction_table("extratrees")]),
            warnings,
            "CCO",
            "O",
        )

    monkeypatch.setattr(prediction_runner, "_collect_all_models", collect_all)


def test_model_choice_all_respects_model_directory_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tree_dir = tmp_path / "custom_tree"
    neural_dir = tmp_path / "custom_neural"
    captured: dict[str, Any] = {}

    def collect_predictions(args: Any) -> Any:
        captured["tree_model_dir"] = args.tree_model_dir
        captured["neural_model_dir"] = args.neural_model_dir
        return prediction_table("rf"), [], "CCO", "O", None

    monkeypatch.setenv("FLUORCAST_TREE_MODEL_DIR", str(tree_dir))
    monkeypatch.setenv("FLUORCAST_NEURAL_MODEL_DIR", str(neural_dir))
    monkeypatch.setattr(
        prediction_runner.predict_all_models, "collect_predictions", collect_predictions
    )

    predictions, _, _, _ = prediction_runner.fluorcast_prediction_backend(
        prediction_input()
    )

    assert captured == {
        "tree_model_dir": tree_dir,
        "neural_model_dir": neural_dir,
    }
    assert [prediction["model_name"] for prediction in predictions] == ["rf"]


def test_model_choice_all_skips_unavailable_experimental_models(
    monkeypatch: Any,
) -> None:
    install_model_availability_fixture(monkeypatch)
    payload = prediction_input()

    predictions, warnings, _, _ = prediction_runner.fluorcast_prediction_backend(payload)

    assert [prediction["model_name"] for prediction in predictions] == [
        "rf",
        "extratrees",
    ]
    for prediction in predictions:
        assert prediction["predicted_absorption_nm"] == 350.0
        assert prediction["predicted_emission_nm"] == 500.0
        assert prediction["predicted_quantum_yield"] == 0.3
        assert prediction["confidence_label"] == "high"
        assert prediction["outside_applicability_domain"] is False
        assert prediction["predicted_stokes_shift_nm"] == 150.0
        assert prediction["predicted_stokes_shift_cm^-1"] == pytest.approx(
            1e7 / 350.0 - 1e7 / 500.0
        )
        assert prediction["physically_valid_stokes"] is True
    assert any("Skipped model gbdt" in warning for warning in warnings)
    assert any("Skipped model histgb" in warning for warning in warnings)
    assert any("Skipped model graph_model_later" in warning for warning in warnings)


def test_model_choice_all_json_preserves_absorption_and_stokes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    tree_root = tmp_path / "absorption_capable_models"
    model_dir = tree_root / "rf"
    model_dir.mkdir(parents=True)
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "median_values_used_for_imputation": {
            target: {"dielectric_constant": 1.0}
            for target in ("absorption_nm", "emission_nm", "quantum_yield")
        },
        "model_type": "rf",
        "target_columns": ["absorption_nm", "emission_nm", "quantum_yield"],
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    joblib.dump(ConstantRegressor(350.0), model_dir / "absorption_nm_rf.joblib")
    joblib.dump(ConstantRegressor(500.0), model_dir / "emission_nm_rf.joblib")
    joblib.dump(ConstantRegressor(0.3), model_dir / "quantum_yield_rf.joblib")
    monkeypatch.setenv("FLUORCAST_TREE_MODEL_DIR", str(tree_root))
    monkeypatch.setenv("FLUORCAST_NEURAL_MODEL_DIR", str(tmp_path / "missing_neural"))
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    write_json(input_path, prediction_input())

    assert prediction_runner.run_job(input_path, output_path) == 0

    prediction = json.loads(output_path.read_text(encoding="utf-8"))["predictions"][0]
    assert prediction["predicted_absorption_nm"] == 350.0
    assert prediction["predicted_stokes_shift_nm"] == 150.0
    assert prediction["predicted_stokes_shift_cm^-1"] == pytest.approx(
        1e7 / 350.0 - 1e7 / 500.0
    )


def test_prediction_records_remain_compatible_without_absorption() -> None:
    table = prediction_table("rf").drop(columns=["predicted_absorption_nm"])

    prediction = prediction_runner._prediction_records(table)[0]

    assert prediction["predicted_absorption_nm"] is None
    assert prediction["predicted_emission_nm"] == 500.0
    assert "predicted_stokes_shift_nm" not in prediction
    assert "predicted_stokes_shift_cm^-1" not in prediction


def test_model_choice_rf_returns_only_rf(monkeypatch: Any) -> None:
    install_model_availability_fixture(monkeypatch)
    payload = prediction_input()
    payload["model_choice"] = "rf"

    predictions, _, _, _ = prediction_runner.fluorcast_prediction_backend(payload)

    assert [prediction["model_name"] for prediction in predictions] == ["rf"]


def test_model_choice_extratrees_returns_only_extratrees(monkeypatch: Any) -> None:
    install_model_availability_fixture(monkeypatch)
    payload = prediction_input()
    payload["model_choice"] = "extratrees"

    predictions, _, _, _ = prediction_runner.fluorcast_prediction_backend(payload)

    assert [prediction["model_name"] for prediction in predictions] == ["extratrees"]


def test_model_choice_histgb_writes_model_unavailable(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_model_availability_fixture(monkeypatch)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    payload = prediction_input()
    payload["model_choice"] = "histgb"
    write_json(input_path, payload)

    assert prediction_runner.run_job(input_path, output_path) == 1
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_code"] == "MODEL_UNAVAILABLE"
    assert "could not be loaded in the current environment" in result["error_message"]
    assert any("_loss" in warning for warning in result["warnings"])
