from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_worker() -> Any:
    path = PROJECT_ROOT / "scripts" / "nibi_supabase_worker.py"
    spec = importlib.util.spec_from_file_location("nibi_supabase_worker", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["nibi_supabase_worker"] = module
    spec.loader.exec_module(module)
    return module


worker = load_worker()


def test_create_prediction_input_payload() -> None:
    payload = worker.create_prediction_input_payload(
        {
            "id": "job-123",
            "user_id": "user-7",
            "molecule_smiles": "CCO",
            "solvent_smiles": "O",
            "model_choice": "rf",
        },
        requested_at="2026-06-23T12:00:00+00:00",
    )

    assert payload == {
        "job_id": "job-123",
        "user_id": "user-7",
        "molecule_smiles": "CCO",
        "solvent_smiles": "O",
        "model_choice": "rf",
        "requested_at": "2026-06-23T12:00:00+00:00",
    }


def test_read_prediction_output_success(tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps(
            {
                "status": "success",
                "job_id": "job-123",
                "predictions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    assert worker.read_prediction_output(output_path)["status"] == "success"


def test_read_prediction_output_rejects_invalid_status(tmp_path: Path) -> None:
    output_path = tmp_path / "output.json"
    output_path.write_text(
        json.dumps({"status": "pending", "job_id": "job-123"}),
        encoding="utf-8",
    )

    try:
        worker.read_prediction_output(output_path)
    except worker.WorkerError as exc:
        assert "invalid status" in str(exc)
    else:
        raise AssertionError("Expected invalid status to raise WorkerError.")


def test_prediction_output_converts_to_prediction_results_rows() -> None:
    rows = worker.prediction_results_rows(
        {
            "status": "success",
            "job_id": "job-123",
            "predictions": [
                {
                    "model_name": "rf",
                    "predicted_absorption_nm": 350.0,
                    "predicted_emission_nm": 512.3,
                    "predicted_quantum_yield": 0.41,
                    "nearest_training_similarity": 0.82,
                    "nearest_training_smiles": "CCO",
                    "warnings": [],
                }
            ],
        }
    )

    assert rows == [
        {
            "prediction_job_id": "job-123",
            "model_name": "rf",
            "predicted_emission_nm": 512.3,
            "predicted_quantum_yield": 0.41,
            "nearest_training_similarity": 0.82,
            "nearest_training_smiles": "CCO",
            "warnings": [],
        }
    ]
    assert "predicted_absorption_nm" not in rows[0]


def test_capture_slurm_job_id() -> None:
    assert worker.capture_slurm_job_id("Submitted batch job 123456\n") == "123456"
    assert worker.capture_slurm_job_id("no id here") is None
