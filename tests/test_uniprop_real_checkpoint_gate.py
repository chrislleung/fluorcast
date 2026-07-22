from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from chemfluor.uniprop import real_checkpoint_gate as gate
from chemfluor.uniprop.real_checkpoint_gate import (
    DEFAULT_FEATURE_SCHEMA,
    GateConfig,
    GateFailure,
    changed_parameters,
    feature_schema_report,
    gradient_report,
    load_feature_schema,
    load_pretrained_backbone,
    masked_mse,
    resolve_checkpoint,
    validate_feature_indices,
    validate_real_gate_report,
    verify_upstream_sources,
)


torch = pytest.importorskip("torch")


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def _write_manifest(
    tmp_path: Path,
    content: bytes,
    checksum: str | None = None,
    size: int | None = None,
    *,
    size_is_exact: bool = True,
    fixture: bool = False,
) -> tuple[Path, Path]:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "real.pt").write_bytes(content)
    manifest = {
        "schema_version": 1,
        "checkpoint_dir_env": "IGNORED",
        "default_checkpoint_dir": str(checkpoint_dir),
        "checksum_type": "md5",
        "checkpoints": [
            {
                "filename": "real.pt",
                "expected_size_bytes": len(content) if size is None else size,
                "size_is_exact": size_is_exact,
                "checksum_type": "md5",
                "checksum": _md5(content) if checksum is None else checksum,
                "fixture": fixture,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, checkpoint_dir


def _minimal_schema(tmp_path: Path, *, fixture: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    schema = {
        "schema_version": "fluorcast_uniprop_feature_schema_v1",
        "schema_kind": "categorical_feature_schema",
        "upstream_commit": "abc",
        "fixture": fixture,
        "upstream_source_files": [{"path": "source.py", "sha256": hashlib.sha256(b"source").hexdigest()}],
        "atom_channels": [{"name": f"atom_{i}", "cardinality": size} for i, size in enumerate([128, 16, 16, 16, 16, 16, 16, 16, 16])],
        "edge_channels": [{"name": f"edge_{i}", "cardinality": 16} for i in range(3)],
    }
    path = tmp_path / ("fixture_feature_schema.json" if fixture else "feature_schema.json")
    path.write_text(json.dumps(schema), encoding="utf-8")
    return path


def test_checkpoint_hash_enforcement_and_approximate_size(tmp_path: Path) -> None:
    manifest, checkpoint_dir = _write_manifest(tmp_path, b"checkpoint")
    report = resolve_checkpoint(GateConfig(checkpoint_manifest=manifest, checkpoint_dir=checkpoint_dir, checkpoint_id="real.pt"))
    assert report["hash_matches"] is True
    assert report["actual_sha256"] == hashlib.sha256(b"checkpoint").hexdigest()

    approximate_manifest, approximate_dir = _write_manifest(tmp_path / "approx", b"checkpoint", size=459500000, size_is_exact=False)
    approximate = resolve_checkpoint(GateConfig(checkpoint_manifest=approximate_manifest, checkpoint_dir=approximate_dir, checkpoint_id="real.pt"))
    assert approximate["size_matches"] is False
    assert approximate["size_is_exact"] is False

    exact_manifest, exact_dir = _write_manifest(tmp_path / "exact", b"checkpoint", size=459500000, size_is_exact=True)
    with pytest.raises(GateFailure, match="size mismatch"):
        resolve_checkpoint(GateConfig(checkpoint_manifest=exact_manifest, checkpoint_dir=exact_dir, checkpoint_id="real.pt"))

    bad_manifest, bad_dir = _write_manifest(tmp_path / "bad", b"checkpoint", checksum="0" * 32, size_is_exact=False)
    with pytest.raises(GateFailure, match="mismatch") as exc:
        resolve_checkpoint(GateConfig(checkpoint_manifest=bad_manifest, checkpoint_dir=bad_dir, checkpoint_id="real.pt"))
    assert exc.value.category == "checkpoint_hash_mismatch"


def test_feature_schema_hash_and_ordered_channels() -> None:
    report = feature_schema_report(DEFAULT_FEATURE_SCHEMA, gate.DEFAULT_FEATURE_SCHEMA_SHA256)
    assert report["hash_matches"] is True
    schema = load_feature_schema(DEFAULT_FEATURE_SCHEMA, gate.DEFAULT_FEATURE_SCHEMA_SHA256)
    assert schema["schema_kind"] == "categorical_feature_schema"
    assert [channel["name"] for channel in schema["atom_channels"]] == [
        "atomic_number",
        "chirality",
        "total_degree",
        "formal_charge",
        "total_num_h",
        "radical_electrons",
        "hybridization",
        "is_aromatic",
        "is_in_ring",
    ]
    assert [channel["cardinality"] for channel in schema["atom_channels"]] == [128, 16, 16, 16, 16, 16, 16, 16, 16]
    assert [channel["name"] for channel in schema["edge_channels"]] == ["bond_type", "bond_stereo", "is_conjugated"]
    assert [channel["cardinality"] for channel in schema["edge_channels"]] == [16, 16, 16]


def test_feature_schema_source_hashes() -> None:
    schema = load_feature_schema(DEFAULT_FEATURE_SCHEMA, gate.DEFAULT_FEATURE_SCHEMA_SHA256)
    report = verify_upstream_sources(Path("third_party/nablacolors"), schema)
    assert report["all_sources_match"] is True
    assert {row["path"] for row in report["files"]} >= {
        "unimol_plus/scripts/get_3d_lmdb.py",
        "unimol_plus/unimol_plus/data/pcq_dataset.py",
        "unimol_plus/unimol_plus/models/uniprop.py",
        "Uni-Core/unicore/checkpoint_utils.py",
    }


def test_feature_index_bounds_for_all_channels(tmp_path: Path) -> None:
    schema = load_feature_schema(_minimal_schema(tmp_path), None)
    node = np.asarray([[0, 1, 2, 3, 4, 5, 6, 1, 0]], dtype=np.int32)
    edge = np.asarray([[0, 1, 1]], dtype=np.int32)
    report = validate_feature_indices(node, edge, schema)
    assert report["atom_channel_cardinalities"] == [128, 16, 16, 16, 16, 16, 16, 16, 16]
    assert report["edge_channel_cardinalities"] == [16, 16, 16]

    bad_node = node.copy()
    bad_node[0, 4] = 16
    with pytest.raises(GateFailure) as exc:
        validate_feature_indices(bad_node, edge, schema)
    assert exc.value.category == "unsupported_feature_schema"

    bad_edge = edge.copy()
    bad_edge[0, 2] = -1
    with pytest.raises(GateFailure):
        validate_feature_indices(node, bad_edge, schema)


def test_checkpoint_key_policy_reports_and_rejects_mismatches(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    report = load_pretrained_backbone(torch, model, checkpoint)
    assert report["missing_required_backbone_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["shape_mismatches"] == []
    assert report["loaded_backbone_parameter_count"] > 0
    assert report["loaded_backbone_parameter_fraction"] == 1.0

    missing = tmp_path / "missing.pt"
    torch.save({"model": {"weight": model.weight.detach().clone()}}, missing)
    with pytest.raises(GateFailure) as exc:
        load_pretrained_backbone(torch, model, missing)
    assert exc.value.category == "checkpoint_key_incompatibility"

    shape = tmp_path / "shape.pt"
    torch.save({"model": {"weight": torch.zeros(1, 3), "bias": model.bias.detach().clone()}}, shape)
    with pytest.raises(GateFailure, match="Shape"):
        load_pretrained_backbone(torch, model, shape)

    unexpected = tmp_path / "unexpected.pt"
    payload = model.state_dict()
    payload["extra.weight"] = torch.zeros(1)
    torch.save({"model": payload}, unexpected)
    with pytest.raises(GateFailure, match="Unexpected"):
        load_pretrained_backbone(torch, model, unexpected)


def test_zero_loaded_backbone_parameters_rejected(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    checkpoint = tmp_path / "zero.pt"
    torch.save({"model": {"other.weight": torch.zeros(1)}}, checkpoint)
    with pytest.raises(GateFailure, match="Zero real backbone"):
        load_pretrained_backbone(torch, model, checkpoint)


def test_output_finiteness_loss_gradient_and_parameter_change_checks() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.ones(1, 2)
    pred = model(x)
    target = torch.zeros(1, 2)
    mask = torch.tensor([[True, False]])
    loss = masked_mse(torch, pred, target, mask)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grads = gradient_report(torch, model)
    optimizer.step()
    changed = changed_parameters(torch, before, model)
    assert grads["finite_gradient_count"] > 0
    assert grads["nonzero_gradient_count"] > 0
    assert changed

    with pytest.raises(GateFailure) as exc:
        masked_mse(torch, torch.tensor([[float("inf"), 0.0]]), target, mask)
    assert exc.value.category == "nonfinite_loss"


def test_reload_tolerance_json_schema_and_tiny_guard() -> None:
    report = {
        "real_uniprop_used": True,
        "real_checkpoint_loaded": True,
        "upstream_commits": {"nablacolors": "abc"},
        "upstream_sources": {"all_sources_match": True},
        "imported_real_model": {"module": "unimol_plus.models.uniprop", "class_name": "UniPropModel"},
        "environment_ready": True,
        "device": "cpu",
        "checkpoint": {},
        "feature_schema_hash": "abc",
        "selected_molecule_id": "mol",
        "selected_solvent_id": "solv",
        "geometry": {},
        "feature_schema_compatibility": {"compatible": True},
        "preprocessing_tensor_shapes": {},
        "checkpoint_key_policy": {"loaded_backbone_parameter_count": 1},
        "forward_output_shape": [1, 6],
        "finite_forward_outputs": True,
        "available_target_mask": [[True]],
        "finite_loss": True,
        "parameter_count_with_gradients": 1,
        "finite_gradient_count": 1,
        "nonzero_gradient_count": 1,
        "changed_parameter_names": ["fluorcast_task_head.0.weight"],
        "reload_agreement": {"passed": True},
        "final_gate_status": "passed",
        "all_stages_passed": True,
    }
    validate_real_gate_report(report)
    report["model_kind"] = "tiny_3d_smoke_backbone"
    with pytest.raises(ValueError, match="Tiny3DSmokeBackbone"):
        validate_real_gate_report(report)


def test_full_gate_stops_after_non_ready_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "environment_audit", lambda config: {"environment_ready": False, "pytorch": {"cuda_available": False}})
    monkeypatch.setattr(gate, "resolve_checkpoint", lambda *args, **kwargs: pytest.fail("full gate continued after non-ready audit"))
    with pytest.raises(GateFailure) as exc:
        gate.run_real_checkpoint_gate(GateConfig(output_dir=tmp_path, overwrite=True))
    assert exc.value.category == "audit_not_ready"


def test_audit_only_writes_json_and_exits_zero_when_not_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "environment_audit", lambda config: {"environment_ready": False})
    assert gate.main(["--audit-only", "--output-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "environment_audit.json").read_text(encoding="utf-8"))["environment_ready"] is False


def test_wrong_upstream_commit_rejected_after_ready_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    revision = tmp_path / "REVISION"
    revision.write_text("commit=expected\n", encoding="utf-8")
    monkeypatch.setattr(gate, "environment_audit", lambda config: {"environment_ready": True, "pytorch": {"cuda_available": False}})
    monkeypatch.setattr(gate, "git_commit", lambda path: "actual")
    with pytest.raises(GateFailure) as exc:
        gate.run_real_checkpoint_gate(GateConfig(output_dir=tmp_path / "out", revision_file=revision, overwrite=True))
    assert exc.value.category == "wrong_upstream_commit"


def test_incorrect_source_hash_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / "source.py").write_text("changed", encoding="utf-8")
    schema = _minimal_schema(tmp_path)
    revision = tmp_path / "REVISION"
    revision.write_text("commit=abc\n", encoding="utf-8")
    monkeypatch.setattr(gate, "environment_audit", lambda config: {"environment_ready": True, "pytorch": {"cuda_available": False}})
    monkeypatch.setattr(gate, "git_commit", lambda path: "abc")
    monkeypatch.setattr(gate, "resolve_checkpoint", lambda config: {"path": str(tmp_path / "real.pt"), "fixture": False})
    with pytest.raises(GateFailure) as exc:
        gate.run_real_checkpoint_gate(
            GateConfig(output_dir=tmp_path / "out", upstream_dir=upstream, revision_file=revision, feature_schema=schema, expected_feature_schema_hash=None, overwrite=True)
        )
    assert exc.value.category == "upstream_source_hash_mismatch"


def test_fixture_schema_and_checkpoint_rejected_in_real_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = _minimal_schema(tmp_path, fixture=True)
    revision = tmp_path / "REVISION"
    revision.write_text("commit=abc\n", encoding="utf-8")
    monkeypatch.setattr(gate, "environment_audit", lambda config: {"environment_ready": True, "pytorch": {"cuda_available": False}})
    monkeypatch.setattr(gate, "git_commit", lambda path: "abc")
    monkeypatch.setattr(gate, "resolve_checkpoint", lambda config: {"path": str(tmp_path / "real.pt"), "fixture": False})
    with pytest.raises(GateFailure) as exc:
        gate.run_real_checkpoint_gate(GateConfig(output_dir=tmp_path / "schema_out", revision_file=revision, feature_schema=schema, expected_feature_schema_hash=None, overwrite=True))
    assert exc.value.category == "feature_schema_fixture"

    schema = _minimal_schema(tmp_path / "schema2")
    monkeypatch.setattr(gate, "resolve_checkpoint", lambda config: {"path": str(tmp_path / "fixture.pt"), "fixture": True})
    with pytest.raises(GateFailure) as exc:
        gate.run_real_checkpoint_gate(GateConfig(output_dir=tmp_path / "checkpoint_out", revision_file=revision, feature_schema=schema, expected_feature_schema_hash=None, overwrite=True))
    assert exc.value.category == "checkpoint_fixture"


def test_gate_files_do_not_trigger_full_cache_or_training() -> None:
    script = Path("slurm/uniprop/run_uniprop_real_checkpoint_gate.sbatch").read_text(encoding="utf-8")
    cli = Path("scripts/run_uniprop_real_checkpoint_gate.py").read_text(encoding="utf-8")
    assert "build_uniprop_geometry_cache.py" not in script
    assert "train_uniprop" not in script
    assert "run_uniprop_real_checkpoint_gate" in script
    assert "--feature-schema" in script
    assert "module load rdkit" not in script
    assert "--gpus-per-node=1" in script
    assert "train_uniprop" not in cli
