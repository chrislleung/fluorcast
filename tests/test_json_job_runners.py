from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd


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
        assert "predicted_absorption_nm" not in prediction
        assert prediction["predicted_emission_nm"] == 500.0
        assert prediction["predicted_quantum_yield"] == 0.3
    assert any("Skipped model gbdt" in warning for warning in warnings)
    assert any("Skipped model histgb" in warning for warning in warnings)
    assert any("Skipped model graph_model_later" in warning for warning in warnings)


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
