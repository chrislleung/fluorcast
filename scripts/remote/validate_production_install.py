"""Validate production artifacts with a deterministic FluorCast prediction."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


REQUIRED_DIRS = (
    "tree",
    "neural",
    "hybrid/absorption_nm",
    "hybrid/emission_nm",
    "hybrid/quantum_yield",
)


def _error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _finite_prediction(record: dict[str, Any]) -> bool:
    numeric_keys = (
        "predicted_absorption_nm",
        "predicted_emission_nm",
        "predicted_quantum_yield",
        "predicted_stokes_shift_nm",
        "predicted_stokes_shift_cm^-1",
    )
    values = [record.get(key) for key in numeric_keys if record.get(key) is not None]
    return bool(values) and all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repo_dir = args.repo_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    errors: list[dict[str, str]] = []
    for rel in REQUIRED_DIRS:
        if not (artifact_dir / rel).is_dir():
            errors.append(_error("ARTIFACT_DIR_MISSING", f"Required artifact directory is missing: {rel}"))
    if errors:
        return {"schema_version": 1, "status": "failed", "errors": errors}, 1

    input_payload = {
        "job_id": "production-fixture",
        "user_id": "fixture",
        "molecule_smiles": "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2",
        "solvent_smiles": "CS(=O)C",
        "model_choice": "hybrid",
        "requested_at": "2026-01-01T00:00:00Z",
    }
    with tempfile.TemporaryDirectory(prefix="fluorcast-production-validation-") as tmp:
        input_path = Path(tmp) / "input.json"
        output_path = Path(tmp) / "output.json"
        input_path.write_text(json.dumps(input_payload), encoding="utf-8")
        env = os.environ.copy()
        env.update(
            {
                "FLUORCAST_TREE_MODEL_DIR": str(artifact_dir / "tree"),
                "FLUORCAST_NEURAL_MODEL_DIR": str(artifact_dir / "neural"),
                "FLUORCAST_ABS_HYBRID_DIR": str(artifact_dir / "hybrid" / "absorption_nm"),
                "FLUORCAST_EM_HYBRID_DIR": str(artifact_dir / "hybrid" / "emission_nm"),
                "FLUORCAST_QY_HYBRID_DIR": str(artifact_dir / "hybrid" / "quantum_yield"),
            }
        )
        completed = subprocess.run(
            [sys.executable, str(repo_dir / "scripts" / "run_prediction_job.py"), "--input", str(input_path), "--output", str(output_path)],
            cwd=repo_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return {"schema_version": 1, "status": "failed", "errors": [_error("FIXTURE_PREDICTION_FAILED", "Fixture prediction failed.")]}, 1
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "status": "failed", "errors": [_error("FIXTURE_OUTPUT_INVALID_JSON", "Fixture output is not valid JSON.")]}, 1
    predictions = output.get("predictions")
    if output.get("status") != "success" or not isinstance(predictions, list):
        return {"schema_version": 1, "status": "failed", "errors": [_error("FIXTURE_OUTPUT_SCHEMA_INVALID", "Fixture output does not match the success schema.")]}, 1
    if not all(isinstance(row, dict) and _finite_prediction(row) for row in predictions):
        return {"schema_version": 1, "status": "failed", "errors": [_error("FIXTURE_PREDICTION_NONFINITE", "Fixture predictions must contain finite numeric values.")]}, 1

    state = {
        "schema_version": 1,
        "status": "ready",
        "artifact_dir": artifact_dir.name,
        "state_file": str(args.state_file.resolve()),
        "validated_at": "slurm-validation",
    }
    _write_state(args.state_file, state)
    return {"schema_version": 1, "status": "success", "errors": [], "state": state}, 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--state-file", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    result, code = validate(parse_args(argv))
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
