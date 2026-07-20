from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch = pytest.importorskip("torch")

from chemfluor.uniprop.geometry_cache import GEOMETRY_SCHEMA_VERSION, atomic_write_json, cache_path, generate_geometry_entry  # noqa: E402
from chemfluor.uniprop.physics_constraints import PHYSICS_SCHEMA_VERSION, PhysicsConstrainedOutputHead  # noqa: E402
from chemfluor.uniprop.production_inference import (  # noqa: E402
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    PREDICTION_SCHEMA_VERSION,
    file_sha256,
    load_bundle,
    molecule_id_for_canonical_smiles,
    predict_json,
    predict_one,
    to_backend_prediction_contract,
    validate_prediction_output_schema,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _bundle_fixture(tmp_path: Path, *, version: str = "2026.07.20-test") -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    architecture = {
        "architecture_name": "physics_head_fixture",
        "input_dim": 24,
        "molecule_feature_dim": 16,
        "solvent_feature_dim": 8,
        "physics_variant": "complete",
    }
    targets = {"targets": ["absorption_nm", "emission_nm", "quantum_yield", "lifetime_ns", "log_extinction"]}
    scalers = {"targets": targets["targets"], "policy": "already_physical_units"}
    solvent_assets = {
        "supported_solvent_smiles": ["O", "CCO"],
        "solvent_name_to_smiles": {"water": "O", "ethanol": "CCO"},
        "solvent_vectors": {
            "O": [0.1] * 8,
            "CCO": [0.2] * 8,
        },
    }
    _write_json(bundle_dir / "architecture_config.json", architecture)
    _write_json(bundle_dir / "target_definitions.json", targets)
    _write_json(bundle_dir / "scalers.json", scalers)
    _write_json(bundle_dir / "solvent_encoder_assets.json", solvent_assets)
    model = PhysicsConstrainedOutputHead.build(torch, input_dim=architecture["input_dim"], variant="complete")
    torch.manual_seed(7)
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) * 0.02)
    torch.save({"model_state_dict": model.state_dict()}, bundle_dir / "model_weights.pt")
    assets = {
        "architecture_config": "architecture_config.json",
        "target_definitions": "target_definitions.json",
        "scalers": "scalers.json",
        "solvent_encoder_assets": "solvent_encoder_assets.json",
        "model_weights": "model_weights.pt",
    }
    metadata = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model_name": "uniprop_complete_physics_constrained",
        "model_version": version,
        "physics_schema_version": PHYSICS_SCHEMA_VERSION,
        "upstream_revision": {"repo": "AI4DD/nablaColors", "commit": "39095389c0a4ecb47872ef74d00b8d13597939c8"},
        "checkpoint_hashes": {"source_checkpoint": "sha256:test"},
        "supported_geometry_schema": GEOMETRY_SCHEMA_VERSION,
        "training_data_fingerprint": {"fingerprint_sha256": "train-fp", "reference_smiles": ["CCO", "c1ccccc1"]},
        "metrics_summary": {"validation": {"absorption_nm_mae": 1.2, "emission_nm_mae": 2.3}},
        "applicability_domain": {"similarity_threshold": 0.2},
        "asset_sha256": {key: file_sha256(bundle_dir / filename) for key, filename in assets.items()},
    }
    _write_json(bundle_dir / "metadata.json", metadata)
    return bundle_dir


def _request(**overrides: Any) -> dict[str, Any]:
    payload = {
        "request_id": "req-1",
        "chromophore_smiles": "OCC",
        "solvent": "water",
    }
    payload.update(overrides)
    return payload


def test_known_cached_molecule_reuses_valid_geometry(tmp_path: Path) -> None:
    bundle = load_bundle(_bundle_fixture(tmp_path))
    cache_dir = tmp_path / "geometry"
    canonical = "CCO"
    molecule_id = molecule_id_for_canonical_smiles(canonical)
    entry = generate_geometry_entry(molecule_id, canonical)
    atomic_write_json(cache_path(cache_dir, molecule_id), entry)

    result = predict_one(_request(), bundle, cache_dir=cache_dir)

    assert result["status"] == "success"
    assert result["geometry_source"] == "cache_hit"
    assert result["canonical_molecule_smiles"] == canonical
    assert result["predictions"][0]["model_version"] == "2026.07.20-test"
    validate_prediction_output_schema(result)


def test_new_valid_molecule_generates_one_geometry(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    result = predict_json(_request(chromophore_smiles="CCN"), bundle_dir=bundle_dir, cache_dir=cache_dir)

    assert result["status"] == "success"
    assert result["geometry_source"] == "generated"
    assert (cache_dir / f"{result['molecule_id']}.json").exists()
    assert result["predictions"][0]["predicted_absorption_nm"] is not None


def test_invalid_smiles_returns_structured_failure(tmp_path: Path) -> None:
    result = predict_json(_request(chromophore_smiles="not smiles"), bundle_dir=_bundle_fixture(tmp_path))

    assert result["status"] == "failed"
    assert result["error"]["code"] == "PredictionInputError"


def test_unsupported_solvent_returns_structured_failure(tmp_path: Path) -> None:
    result = predict_json(_request(solvent="hexane"), bundle_dir=_bundle_fixture(tmp_path))

    assert result["status"] == "failed"
    assert "Unsupported solvent" in result["error"]["message"]


def test_batch_prediction_mixed_successes_and_failures(tmp_path: Path) -> None:
    result = predict_json(
        {"batch": [_request(request_id="ok"), _request(request_id="bad", chromophore_smiles="???")]},
        bundle_dir=_bundle_fixture(tmp_path),
        cache_dir=tmp_path / "cache",
    )

    assert result["status"] == "success"
    assert [item["status"] for item in result["results"]] == ["success", "failed"]
    assert result["results"][0]["request_id"] == "ok"
    assert result["results"][1]["request_id"] == "bad"


def test_deterministic_repeated_prediction(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    cache_dir = tmp_path / "cache"
    first = predict_json(_request(), bundle_dir=bundle_dir, cache_dir=cache_dir)
    second = predict_json(_request(), bundle_dir=bundle_dir, cache_dir=cache_dir)

    assert first["predictions"] == second["predictions"]
    assert first["applicability_domain"] == second["applicability_domain"]


def test_cpu_and_cuda_request_tolerance(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    cpu = predict_json(_request(), bundle_dir=bundle_dir, device="cpu")
    cuda_or_cpu = predict_json(_request(), bundle_dir=bundle_dir, device="cuda")

    assert cuda_or_cpu["status"] == "success"
    assert cuda_or_cpu["predictions"][0]["predicted_emission_nm"] == pytest.approx(cpu["predictions"][0]["predicted_emission_nm"], rel=1e-5)


def test_bundle_missing_or_corrupted_asset_is_rejected(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    (bundle_dir / "scalers.json").write_text("{corrupt", encoding="utf-8")

    with pytest.raises(BundleError, match="checksum mismatch"):
        load_bundle(bundle_dir)


def test_model_version_mismatch_is_rejected(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["schema_version"] = "future_bundle_v999"
    _write_json(bundle_dir / "metadata.json", metadata)

    with pytest.raises(BundleError, match="Unsupported model bundle schema"):
        load_bundle(bundle_dir)


def test_json_schema_validation_requires_provenance(tmp_path: Path) -> None:
    result = predict_json(_request(), bundle_dir=_bundle_fixture(tmp_path))
    validate_prediction_output_schema(result)
    del result["predictions"][0]["model_version"]

    with pytest.raises(ValueError, match="model provenance"):
        validate_prediction_output_schema(result)


def test_backend_contract_adapter_is_versioned_and_compatible(tmp_path: Path) -> None:
    result = predict_json(_request(request_id="job-123"), bundle_dir=_bundle_fixture(tmp_path))

    adapted = to_backend_prediction_contract(result)

    assert adapted["schema_version"] == "fluorcast_backend_prediction_adapter_v1"
    assert adapted["job_id"] == "job-123"
    assert adapted["status"] == "success"
    assert adapted["canonical_molecule_smiles"] == "CCO"
    assert adapted["predictions"][0]["predicted_absorption_nm"] is not None
    assert "applicability_domain" in adapted


def test_end_to_end_command_line_prediction(tmp_path: Path) -> None:
    bundle_dir = _bundle_fixture(tmp_path)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    _write_json(input_path, _request(solvent_smiles="O"))
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/predict_uniprop_bundle.py",
            "--bundle-dir",
            str(bundle_dir),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--geometry-cache-dir",
            str(tmp_path / "geometry"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == PREDICTION_SCHEMA_VERSION
    assert result["status"] == "success"
    assert result["provenance"]["model_version"] == "2026.07.20-test"


def test_inference_module_does_not_import_training_code() -> None:
    source = (PROJECT_ROOT / "src/chemfluor/uniprop/production_inference.py").read_text(encoding="utf-8")

    assert "head_smoke_training" not in source
    assert "backbone_finetune" not in source
    assert "experiment_matrix" not in source
