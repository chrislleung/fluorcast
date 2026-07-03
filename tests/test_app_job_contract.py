"""Dependency-free checks for the desktop application job fixtures."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"
INPUT_FIELDS = {
    "job_id", "user_id", "molecule_smiles", "solvent_smiles",
    "model_choice", "requested_at",
}
OUTPUT_FIELDS = {
    "job_id", "status", "canonical_molecule_smiles",
    "canonical_solvent_smiles", "predictions",
    "applicability_domain", "warnings",
}


def load_fixture(name: str) -> dict:
    payload = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_prediction_input_fixture_has_required_nonempty_strings() -> None:
    payload = load_fixture("app_prediction_input.example.json")
    assert INPUT_FIELDS <= payload.keys()
    assert all(isinstance(payload[field], str) and payload[field] for field in INPUT_FIELDS)


def test_success_output_fixture_has_stable_envelope() -> None:
    payload = load_fixture("app_prediction_output.success.example.json")
    assert OUTPUT_FIELDS <= payload.keys()
    assert payload["status"] == "success"
    assert isinstance(payload["predictions"], list) and payload["predictions"]
    assert isinstance(payload["applicability_domain"], dict)
    assert isinstance(payload["warnings"], list)
    assert "error" not in payload


def test_failure_output_fixture_has_stable_envelope_and_error() -> None:
    payload = load_fixture("app_prediction_output.failure.example.json")
    assert OUTPUT_FIELDS <= payload.keys()
    assert payload["status"] == "failed"
    assert payload["predictions"] == []
    assert payload["applicability_domain"] is None
    assert isinstance(payload["warnings"], list)
    assert set(payload["error"]) >= {"code", "message"}
    assert all(isinstance(payload["error"][field], str) and payload["error"][field] for field in ("code", "message"))
