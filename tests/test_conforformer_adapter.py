from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from chemfluor.conforformer.adapter import (
    AssetUnavailableError,
    CompatibilityError,
    InferenceError,
    InputValidationError,
    MissingDependencyError,
    dependency_report,
    embeddings_close,
    extract_cls_embedding,
    inspect_assets,
    inspect_checkpoint,
    require_torch,
    sha256_file,
    tensors_from_preprocessed,
    validate_dictionary_checkpoint_compatibility,
    validate_input_arrays,
    validate_embedding_tensor,
)
from chemfluor.conforformer.dictionary import load_conforformer_dictionary
from chemfluor.conforformer.preprocess import preprocess_conformer
from chemfluor.conforformer.schemas import (
    ConformerRecord,
    GenerationStatus,
    MoleculeConformerCacheRecord,
    MoleculeStatus,
)


try:
    import torch
except ImportError:  # pragma: no cover - depends on local optional environment
    torch = None

requires_torch = pytest.mark.skipif(torch is None, reason="PyTorch-specific adapter test")


def _dict(tmp_path: Path, tokens: list[str] | None = None) -> Path:
    path = tmp_path / "dict.txt"
    path.write_text("\n".join(tokens or ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "O"]) + "\n", encoding="utf-8")
    return path


def _checkpoint(tmp_path: Path, *, vocab_size: int = 6, embed_dim: int = 4, heads: int = 2) -> Path:
    if torch is None:
        pytest.skip("PyTorch-specific adapter test")
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": {
                "embed_tokens.weight": torch.zeros(vocab_size, embed_dim),
                "gbf.mul.weight": torch.zeros(vocab_size * vocab_size, 1),
                "gbf.means.weight": torch.zeros(1, 128),
                "gbf_proj.linear2.weight": torch.zeros(heads, 128),
                "encoder.layers.0.self_attn.k_proj.weight": torch.zeros(embed_dim, embed_dim),
            },
            "args": {
                "arch": "contrast",
                "encoder_embed_dim": embed_dim,
                "encoder_attention_heads": heads,
                "encoder_layers": 1,
                "max_seq_len": 8,
            },
        },
        path,
    )
    return path


def _record(tmp_path: Path):
    dictionary = load_conforformer_dictionary(_dict(tmp_path))
    conformer = ConformerRecord(
        conformer_id="conf-1",
        atom_symbols=["C", "H", "O"],
        atomic_numbers=[6, 1, 8],
        coordinates=[[0, 0, 0], [9, 9, 9], [1, 0, 0]],
        energy=None,
        energy_units=None,
        optimizer="MMFF94",
        optimization_convergence_status="converged",
        generation_status=GenerationStatus.OK,
    )
    molecule = MoleculeConformerCacheRecord(
        chromophore_id="mol-1",
        input_smiles="CO",
        canonical_smiles="CO",
        isomeric_canonical_smiles="CO",
        conformer_cache_key="cache-key",
        requested_conformer_count=1,
        successful_conformer_count=1,
        status=MoleculeStatus.OK,
        failure_reason=None,
        conformer_records=[conformer],
        rdkit_version="test",
        configuration_payload={},
    )
    return preprocess_conformer(molecule, conformer, dictionary)


def test_module_imports_and_reports_optional_dependencies() -> None:
    module = importlib.import_module("chemfluor.conforformer")
    assert hasattr(module, "ConforFormerEncoderAdapter")
    report = dependency_report()
    assert isinstance(report.pytorch_available, bool)
    assert isinstance(report.unicore_available, bool)


def test_missing_torch_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    import chemfluor.conforformer.adapter as adapter

    def fail_import(name: str):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return importlib.import_module(name)

    monkeypatch.setattr(adapter.importlib, "import_module", fail_import)
    with pytest.raises(MissingDependencyError, match="PyTorch is required"):
        adapter.require_torch()


def test_missing_checkpoint_and_dictionary_errors(tmp_path: Path) -> None:
    with pytest.raises(AssetUnavailableError, match="checkpoint unavailable"):
        inspect_checkpoint(tmp_path / "missing.pt")
    checkpoint = _checkpoint(tmp_path)
    with pytest.raises(AssetUnavailableError, match="dictionary unavailable"):
        inspect_assets(tmp_path / "missing-dict.txt", checkpoint)


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "payload.txt"
    path.write_text("abc", encoding="utf-8")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@requires_torch
def test_checkpoint_metadata_parsing(tmp_path: Path) -> None:
    checkpoint = inspect_checkpoint(_checkpoint(tmp_path, vocab_size=6, embed_dim=4, heads=2))
    assert checkpoint.has_model
    assert checkpoint.has_args
    assert checkpoint.inferred_vocab_size == 6
    assert checkpoint.inferred_embedding_dim == 4
    assert checkpoint.file_size_bytes > 0
    assert checkpoint.architecture.encoder_layers == 1
    assert checkpoint.architecture.encoder_attention_heads == 2
    assert checkpoint.tensor_shapes["embed_tokens.weight"] == (6, 4)


@requires_torch
def test_dictionary_checkpoint_vocab_mismatch_detection(tmp_path: Path) -> None:
    dictionary = load_conforformer_dictionary(_dict(tmp_path))
    checkpoint = inspect_checkpoint(_checkpoint(tmp_path, vocab_size=7))
    with pytest.raises(CompatibilityError, match="vocabulary size"):
        validate_dictionary_checkpoint_compatibility(dictionary, checkpoint)


@requires_torch
def test_shape_conversion_to_tensors_and_no_input_mutation(tmp_path: Path) -> None:
    record = _record(tmp_path)
    original = record.src_coord.copy()
    tensors = tensors_from_preprocessed(record, device="cpu", max_sequence_length=8)
    assert tensors["src_tokens"].dtype == torch.int64
    assert tensors["src_coord"].dtype == torch.float32
    assert tensors["src_distance"].dtype == torch.float32
    assert tensors["src_edge_type"].dtype == torch.int64
    assert tensors["src_tokens"].shape[0] == 1
    assert np.array_equal(record.src_coord, original)


def test_invalid_input_shapes_and_nonfinite_values_fail() -> None:
    arrays = {
        "src_tokens": np.zeros((1, 3), dtype=np.int64),
        "src_coord": np.zeros((1, 3, 3), dtype=np.float32),
        "src_distance": np.zeros((1, 3, 3), dtype=np.float32),
        "src_edge_type": np.zeros((1, 2, 2), dtype=np.int64),
    }
    with pytest.raises(InputValidationError, match="src_edge_type"):
        validate_input_arrays(arrays)
    arrays["src_edge_type"] = np.zeros((1, 3, 3), dtype=np.int64)
    arrays["src_coord"][0, 0, 0] = np.nan
    with pytest.raises(InputValidationError, match="finite"):
        validate_input_arrays(arrays)


@requires_torch
def test_cls_extraction_and_output_validation() -> None:
    encoder_rep = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    cls = extract_cls_embedding((encoder_rep, None, None), expected_batch_size=2, expected_dim=4)
    assert cls.shape == (2, 4)
    assert torch.equal(cls, encoder_rep[:, 0, :])
    with pytest.raises(InferenceError, match="batch size"):
        validate_embedding_tensor(cls, expected_batch_size=3, expected_dim=4)
    with pytest.raises(InferenceError, match="dimension"):
        validate_embedding_tensor(cls, expected_batch_size=2, expected_dim=5)
    bad = cls.clone()
    bad[0, 0] = float("nan")
    with pytest.raises(InferenceError, match="non-finite"):
        validate_embedding_tensor(bad, expected_batch_size=2, expected_dim=4)


@requires_torch
def test_repeat_comparison_and_no_grad_behavior() -> None:
    first = np.asarray([[1.0, 2.0]], dtype=np.float32)
    second = np.asarray([[1.0, 2.0 + 1e-7]], dtype=np.float32)
    assert embeddings_close(first, second)
    with torch.inference_mode():
        output = torch.ones(1, 2, 3)
        cls = extract_cls_embedding(output, expected_batch_size=1, expected_dim=3)
    assert not cls.requires_grad


@requires_torch
def test_device_selection_cpu(tmp_path: Path) -> None:
    tensors = tensors_from_preprocessed(_record(tmp_path), device="cpu", max_sequence_length=8)
    assert tensors["src_tokens"].device.type == "cpu"


@requires_torch
def test_inspect_only_cli_does_not_construct_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.smoke_conforformer_encoder as smoke

    def fail_construct(*args: object, **kwargs: object) -> object:
        raise AssertionError("inspect-only must not construct the adapter")

    monkeypatch.setattr(smoke, "ConforFormerEncoderAdapter", fail_construct)
    exit_code = smoke.main(
        [
            "--inspect-only",
            "--dictionary",
            str(_dict(tmp_path)),
            "--checkpoint",
            str(_checkpoint(tmp_path)),
        ]
    )
    assert exit_code == 0


def test_env_report_cli_does_not_require_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.smoke_conforformer_encoder as smoke

    def fail_inspect(*args: object, **kwargs: object) -> object:
        raise AssertionError("env-report must not inspect checkpoint assets")

    monkeypatch.setattr(smoke, "inspect_assets", fail_inspect)
    assert smoke.main(["--env-report"]) == 0


@requires_torch
def test_inspect_assets_compatibility_success(tmp_path: Path) -> None:
    dictionary, checkpoint, compatibility = inspect_assets(_dict(tmp_path), _checkpoint(tmp_path))
    assert dictionary.sha256 == compatibility.dictionary_sha256
    assert checkpoint.checkpoint_sha256 == compatibility.checkpoint_sha256
    assert compatibility.compatible
