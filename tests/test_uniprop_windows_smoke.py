from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chemfluor.uniprop.geometry_cache import GEOMETRY_SCHEMA_VERSION, generate_geometry_entry
from chemfluor.uniprop.lmdb_export import read_lmdb_records
from chemfluor.uniprop.physics_constraints import PHYSICS_SCHEMA_VERSION
from chemfluor.uniprop.production_inference import (
    BUNDLE_SCHEMA_VERSION,
    BundleError,
    file_sha256,
    load_bundle,
)
from chemfluor.uniprop.windows_smoke import (
    TINY_3D_SMOKE_MODEL_KIND,
    WINDOWS_SMOKE_PREDICTION_SCHEMA_VERSION,
    WINDOWS_SMOKE_PROFILE,
    WINDOWS_SMOKE_SCHEMA_VERSION,
    run_windows_smoke,
    validate_windows_smoke_prediction_schema,
)

pytest.importorskip("lmdb")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.windows_smoke


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    output_dir = tmp_path_factory.mktemp("uniprop_windows_smoke")
    summary = run_windows_smoke(output_dir, seed=123, overwrite=True)
    return output_dir, summary


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_complete_smoke_summary_declares_non_real_profile(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    _, summary = smoke_run

    assert summary["profile"] == WINDOWS_SMOKE_PROFILE
    assert summary["model_kind"] == TINY_3D_SMOKE_MODEL_KIND
    assert summary["real_uniprop_used"] is False
    assert summary["real_checkpoint_loaded"] is False
    assert summary["tiny_backbone_used"] is True
    assert summary["all_stages_passed"] is True


def test_deterministic_rdkit_geometry_coordinates() -> None:
    first = generate_geometry_entry("mol_smoke_deterministic", "CCO")
    second = generate_geometry_entry("mol_smoke_deterministic", "CCO")

    assert first["schema_version"] == GEOMETRY_SCHEMA_VERSION
    assert first["atom_symbols"] == second["atom_symbols"]
    np.testing.assert_allclose(first["coordinates"], second["coordinates"], rtol=0, atol=0)


def test_one_geometry_per_unique_chromophore_and_reuse(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, _ = smoke_run
    report = _read_json(output_dir / "geometry_validation_report.json")

    assert report["geometry_count"] == report["unique_chromophores"]
    assert report["one_geometry_per_unique_chromophore"] is True
    assert report["repeated_chromophores_reuse_geometry"] is True
    assert any(len(group["row_ids"]) > 1 for group in report["repeated_chromophore_groups"])


def test_lmdb_round_trip_and_missing_label_masks(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, _ = smoke_run
    report = _read_json(output_dir / "lmdb_validation_report.json")
    records = []
    for partition in ["train", "valid", "test"]:
        assert report["partitions"][partition]["valid"] is True
        records.extend(record for _, record in read_lmdb_records(output_dir / "lmdb" / f"{partition}.lmdb"))

    assert records
    assert any((~np.asarray(record["target_mask"], dtype=bool)).any() for record in records)
    for record in records:
        target = np.asarray(record["target"], dtype=float)
        mask = np.asarray(record["target_mask"], dtype=bool)
        assert np.isnan(target[~mask]).all()


def test_forward_backward_gradients_and_optimizer_change(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, summary = smoke_run
    shapes = _read_json(output_dir / "tensor_shape_report.json")
    gradients = _read_json(output_dir / "gradient_statistics.json")
    changed = _read_json(output_dir / "changed_parameter_names.json")

    assert summary["stages"]["forward"]["status"] == "passed"
    assert shapes["shapes"]["target"][-1] == 3
    assert gradients["all_finite"] is True
    assert changed["changed_parameter_names"]


def test_checkpoint_save_load_identity_and_model_kind(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, summary = smoke_run
    checkpoint = torch.load(output_dir / "checkpoints" / "checkpoint.pt", map_location="cpu", weights_only=False)

    assert checkpoint["schema_version"] == WINDOWS_SMOKE_SCHEMA_VERSION
    assert checkpoint["model_kind"] == TINY_3D_SMOKE_MODEL_KIND
    assert checkpoint["real_uniprop_used"] is False
    assert summary["stages"]["checkpoint_reload_identity"]["status"] == "passed"


def test_exact_resume_report(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, summary = smoke_run
    report = _read_json(output_dir / "exact_resume_report.json")

    assert report["exact"] is True
    assert report["parameter_match_count"] == report["parameter_total"]
    assert summary["stages"]["exact_resume"]["status"] == "passed"


def test_windows_smoke_prediction_json_schema(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, _ = smoke_run
    prediction = _read_json(output_dir / "predictions.json")

    validate_windows_smoke_prediction_schema(prediction)
    assert prediction["schema_version"] == WINDOWS_SMOKE_PREDICTION_SCHEMA_VERSION
    assert prediction["predictions"][0]["model_kind"] == TINY_3D_SMOKE_MODEL_KIND


def test_smoke_outputs_cannot_be_loaded_as_real_production_bundle(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    _, summary = smoke_run

    with pytest.raises(BundleError, match="Unsupported model bundle schema"):
        load_bundle(Path(summary["artifacts"]["smoke_bundle"]))


def test_real_model_mode_refuses_tiny_smoke_checkpoint(tmp_path: Path, smoke_run: tuple[Path, dict[str, Any]]) -> None:
    _, summary = smoke_run
    bundle_dir = tmp_path / "real_shaped_bundle"
    bundle_dir.mkdir()
    (bundle_dir / "architecture_config.json").write_text(
        json.dumps(
            {
                "architecture_name": "physics_head_fixture",
                "input_dim": 24,
                "molecule_feature_dim": 16,
                "solvent_feature_dim": 8,
                "physics_variant": "complete",
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "target_definitions.json").write_text(
        json.dumps({"targets": ["absorption_nm", "emission_nm", "quantum_yield", "lifetime_ns", "log_extinction"]}),
        encoding="utf-8",
    )
    (bundle_dir / "scalers.json").write_text(json.dumps({"targets": ["absorption_nm"], "policy": "test"}), encoding="utf-8")
    (bundle_dir / "solvent_encoder_assets.json").write_text(
        json.dumps({"supported_solvent_smiles": ["O"], "solvent_vectors": {"O": [0.1] * 8}}),
        encoding="utf-8",
    )
    source_checkpoint = Path(summary["artifacts"]["resumed_checkpoint"])
    (bundle_dir / "model_weights.pt").write_bytes(source_checkpoint.read_bytes())
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
        "model_version": "tiny-refusal-test",
        "physics_schema_version": PHYSICS_SCHEMA_VERSION,
        "checkpoint_hashes": {"source_checkpoint": "sha256:test"},
        "supported_geometry_schema": GEOMETRY_SCHEMA_VERSION,
        "asset_sha256": {key: file_sha256(bundle_dir / filename) for key, filename in assets.items()},
    }
    (bundle_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(BundleError, match="Tiny smoke checkpoint"):
        load_bundle(bundle_dir)


def test_training_normalization_excludes_test_labels(smoke_run: tuple[Path, dict[str, Any]]) -> None:
    output_dir, _ = smoke_run
    row_manifest = pd.read_csv(output_dir / "manifests" / "row_manifest.csv")
    splits = pd.read_csv(output_dir / "manifests" / "split_assignments.csv")
    normalizer = _read_json(output_dir / "training_normalization.json")
    test_row_ids = set(splits.loc[splits["random"] == "test", "row_id"].astype(str))

    assert normalizer["no_test_labels_used"] is True
    assert not set(normalizer["fit_row_ids"]).intersection(test_row_ids)
    test_absorption = row_manifest[row_manifest["row_id"].isin(test_row_ids)]["absorption_nm"].dropna()
    assert test_absorption.min() > 9000
    assert max(normalizer["mean"][:2]) < 1000
