from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_prediction_job.py"
SPEC = importlib.util.spec_from_file_location("run_prediction_job", SCRIPT)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def payload(tmp_path: Path) -> dict[str, Any]:
    return {
        "job_id": "job-hybrid",
        "user_id": "user-1",
        "molecule_smiles": "CCO",
        "solvent_smiles": "O",
        "model_choice": "hybrid",
        "requested_at": "2026-01-01T00:00:00Z",
        "temp_dir": str(tmp_path / "job-output"),
    }


def test_model_choice_hybrid_is_accepted() -> None:
    runner.validate_input(
        {
            "job_id": "job-hybrid",
            "user_id": "user-1",
            "molecule_smiles": "CCO",
            "solvent_smiles": "O",
            "model_choice": "hybrid",
            "requested_at": "2026-01-01T00:00:00Z",
        }
    )


def test_hybrid_backend_maps_full_workflow_report(tmp_path: Path, monkeypatch: Any) -> None:
    def fake_run_workflow(args: Any) -> dict[str, Any]:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "base_predictions.csv").write_text(
            "nearest_training_similarity,nearest_training_smiles\n0.91,CCO\n",
            encoding="utf-8",
        )
        return {
            "predicted_absorption_nm": 410.0,
            "predicted_emission_nm": 520.0,
            "stokes_shift_nm": 110.0,
            "stokes_shift_cm^-1": 5159.0,
            "predicted_quantum_yield": 0.31,
            "brightness_class": "bright",
            "physically_valid_stokes": True,
            "absorption_uncertainty_interval": {"lower": 400, "upper": 420},
            "emission_uncertainty_interval": {"lower": 500, "upper": 540},
            "quantum_yield_uncertainty_interval": {"lower": 0.2, "upper": 0.4},
            "absorption_outside_applicability_domain": False,
            "emission_outside_applicability_domain": True,
            "quantum_yield_outside_applicability_domain": False,
            "outside_applicability_domain": True,
            "warnings": ["Outside applicability domain: emission."],
        }

    monkeypatch.setattr(runner.predict_full_fluorcast, "run_workflow", fake_run_workflow)
    predictions, warnings, molecule, solvent = runner.fluorcast_prediction_backend(
        payload(tmp_path)
    )

    assert molecule == "CCO"
    assert solvent == "O"
    assert warnings == ["Outside applicability domain: emission."]
    assert len(predictions) == 1
    record = predictions[0]
    assert record["model_name"] == "hybrid"
    assert record["predicted_absorption_nm"] == pytest.approx(410.0)
    assert record["predicted_emission_nm"] == pytest.approx(520.0)
    assert record["predicted_stokes_shift_nm"] == pytest.approx(110.0)
    assert record["predicted_stokes_shift_cm^-1"] == pytest.approx(5159.0)
    assert record["predicted_quantum_yield"] == pytest.approx(0.31)
    assert record["brightness_class"] == "bright"
    assert record["physically_valid_stokes"] is True
    assert set(record["prediction_intervals"]) == {
        "absorption_nm",
        "emission_nm",
        "quantum_yield",
    }
    assert record["applicability_domain"]["outside_applicability_domain"] is True
    assert record["nearest_training_similarity"] == pytest.approx(0.91)
    assert record["nearest_training_smiles"] == "CCO"
    assert record["warnings"] == warnings


def test_hybrid_backend_preserves_nonpositive_stokes_warning(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def fake_run_workflow(args: Any) -> dict[str, Any]:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        return {
            "predicted_absorption_nm": 520.0,
            "predicted_emission_nm": 510.0,
            "stokes_shift_nm": -10.0,
            "stokes_shift_cm^-1": -377.1,
            "physically_valid_stokes": False,
            "warnings": [
                "Predicted emission is less than or equal to predicted absorption; "
                "the calculated Stokes shift is nonpositive."
            ],
        }

    monkeypatch.setattr(runner.predict_full_fluorcast, "run_workflow", fake_run_workflow)
    predictions, warnings, _, _ = runner.fluorcast_prediction_backend(payload(tmp_path))

    assert predictions[0]["physically_valid_stokes"] is False
    assert any("nonpositive" in warning for warning in warnings)
