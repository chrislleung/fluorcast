from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.embed_conforformer_shard as embed_conforformer_shard
import chemfluor.conforformer.downstream as downstream
from chemfluor.conforformer.adapter import ArchitectureMetadata
from chemfluor.conforformer.cache import save_conformer_cache_record
from chemfluor.conforformer.config import ConformerGenerationConfig
from chemfluor.conforformer.downstream import build_feature_bundle, join_embeddings, make_split_assignments, metrics, train_downstream
from chemfluor.conforformer.embedding_store import (
    EmbeddingStoreError,
    expected_identity,
    finalize_embeddings,
    load_valid_embedding_shard,
    write_embedding_shard,
)
from chemfluor.conforformer.inventory import build_inventory, build_inventory_frame, load_inventory, sha256_file
from chemfluor.conforformer.pooling import boltzmann_pool, lowest_energy_pool, mean_pool, pool_all
from chemfluor.conforformer.preprocess import ConforFormerPreprocessingConfig
from chemfluor.conforformer.conformers import generate_conformer_cache_record
from chemfluor.conforformer.dictionary import load_conforformer_dictionary


def _dataset(path: Path) -> Path:
    rows = pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["CCO", "C", "", None, "C", "N"],
            "canonical_solvent_smiles": ["O", "O", "O", "O", "CCO", "N"],
            "solvent_original": ["water", "water", "water", "water", "ethanol", "amine"],
            "absorption_nm": [300, 310, 320, 330, 311, 340],
            "emission_nm": [360, 370, 380, 390, 371, 410],
            "quantum_yield": [0.1, 0.2, 0.3, 0.4, 0.21, 0.5],
        }
    )
    rows.to_csv(path, index=False)
    return path


def _identity(manifest: dict, checkpoint: Path, dictionary: Path) -> dict:
    return expected_identity(
        inventory_manifest=manifest,
        checkpoint_sha256=sha256_file(checkpoint),
        dictionary_sha256=sha256_file(dictionary),
        upstream_commit="fake-upstream",
        architecture_payload={"encoder_embed_dim": 512, "source": "fake_adapter"},
        preprocessing_payload=ConforFormerPreprocessingConfig().to_payload(),
        conformer_config_payload=ConformerGenerationConfig().to_payload(),
    )


class ConstantRegressor:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "ConstantRegressor":
        self.value = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x),), self.value, dtype=float)


def test_inventory_is_deterministic_deduped_and_sharded(tmp_path: Path) -> None:
    rows = pd.DataFrame({"canonical_chromophore_smiles": ["N", "C", "C", "", None, "CCO"]})
    inventory = build_inventory_frame(rows, shard_size=2)
    assert inventory["canonical_chromophore_smiles"].tolist() == ["C", "CCO", "N"]
    assert inventory["source_row_count"].tolist() == [2, 1, 1]
    assert inventory["molecule_index"].tolist() == [0, 1, 2]
    assert inventory["shard_index"].tolist() == [0, 0, 1]
    assert inventory["molecule_id"].is_unique

    capped = build_inventory_frame(rows, shard_size=2, max_molecules=2)
    assert capped["canonical_chromophore_smiles"].tolist() == ["C", "CCO"]


def test_inventory_manifest_hash_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "combined.csv")
    result = build_inventory(source_csv=dataset, output_dir=tmp_path / "run" / "inventory", shard_size=2)
    inventory, manifest = load_inventory(tmp_path / "run")
    assert len(inventory) == 3
    assert manifest["source_csv_sha256"] == sha256_file(dataset)
    result.inventory_path.write_text(result.inventory_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_inventory(tmp_path / "run")


def test_inventory_csv_hash_is_stable_and_created_at_is_not_shard_identity(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "combined.csv")
    first = build_inventory(source_csv=dataset, output_dir=tmp_path / "run_a" / "inventory", shard_size=2, git_root=Path.cwd())
    second = build_inventory(source_csv=dataset, output_dir=tmp_path / "run_b" / "inventory", shard_size=2, git_root=Path.cwd())
    assert sha256_file(first.inventory_path) == sha256_file(second.inventory_path)

    manifest_a = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    manifest_b = json.loads(second.manifest_path.read_text(encoding="utf-8"))
    manifest_b["created_at"] = "different"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nN 1\nO 1\n", encoding="utf-8")
    assert _identity(manifest_a, checkpoint, dictionary) == _identity(manifest_b, checkpoint, dictionary)


def test_pooling_methods_and_boltzmann_fallbacks() -> None:
    embeddings = np.asarray([[1, 1], [3, 5], [5, 9]], dtype=np.float32)
    energies = np.asarray([0.0, 1.0, 2.0], dtype=float)
    assert np.allclose(mean_pool(embeddings), [3, 5])
    lowest, idx, reason = lowest_energy_pool(embeddings, energies)
    assert idx == 0
    assert reason is None
    assert np.allclose(lowest, [1, 1])
    boltz, used, reason, weights = boltzmann_pool(embeddings, energies)
    manual = np.exp(-(energies - energies.min()) / (0.00198720425864083 * 298.15))
    manual = manual / manual.sum()
    assert used is True
    assert reason is None
    assert np.allclose(weights, manual)
    assert np.allclose(boltz, manual @ embeddings)

    fallback, used, reason, weights = boltzmann_pool(embeddings, np.asarray([0.0, np.nan, 1.0]))
    assert used is False
    assert reason == "nonfinite_or_missing_energies"
    assert np.allclose(fallback, mean_pool(embeddings))
    assert np.isnan(weights).all()
    all_pooled = pool_all(embeddings, energies)
    assert all_pooled.lowest_energy_index == 0


def test_embedding_shard_atomic_write_resume_and_validation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "combined.csv")
    run_root = tmp_path / "run"
    inv = build_inventory(source_csv=dataset, output_dir=run_root / "inventory", shard_size=2)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nN 1\nO 1\n", encoding="utf-8")
    identity = _identity(manifest, checkpoint, dictionary)
    shard_rows = inventory[inventory["shard_index"] == 0].reset_index(drop=True)
    embeddings = [np.ones((2, 512), dtype=np.float32), np.full((1, 512), 2, dtype=np.float32)]
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=shard_rows,
        conformer_ids_by_molecule=[["a", "b"], ["c"]],
        embeddings_by_molecule=embeddings,
        energies_by_molecule=[np.asarray([1.0, 0.0]), np.asarray([np.nan])],
        failure_codes=[None, None],
        failure_messages=[None, None],
        identity=identity,
    )
    data, done = load_valid_embedding_shard(run_root, 0, expected_molecule_count=2, identity=identity)
    assert data["molecule_offsets"].tolist() == [0, 2, 3]
    assert done["success_count"] == 2
    data.close()
    (run_root / "embeddings" / "shard_00000.npz").write_text("corrupt", encoding="utf-8")
    with pytest.raises(EmbeddingStoreError, match="SHA-256"):
        load_valid_embedding_shard(run_root, 0, expected_molecule_count=2, identity=identity)


def test_embedding_shard_rejects_stale_identity_dimensions(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "combined.csv")
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset, output_dir=run_root / "inventory", shard_size=10)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nN 1\nO 1\n", encoding="utf-8")
    identity = _identity(manifest, checkpoint, dictionary)
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[[f"c{i}"] for i in range(len(inventory))],
        embeddings_by_molecule=[np.ones((1, 512), dtype=np.float32) for _ in range(len(inventory))],
        energies_by_molecule=[np.asarray([0.0]) for _ in range(len(inventory))],
        failure_codes=[None] * len(inventory),
        failure_messages=[None] * len(inventory),
        identity=identity,
    )
    stale_variants = {
        "dataset": {**identity, "dataset_sha256": "changed"},
        "shard_size_or_inventory": {**identity, "inventory_sha256": "changed"},
        "checkpoint": {**identity, "checkpoint_sha256": "changed"},
        "dictionary": {**identity, "dictionary_sha256": "changed"},
        "preprocessing": {**identity, "preprocessing_version": "changed"},
        "architecture": {**identity, "architecture_configuration_identity": {"encoder_embed_dim": 256}},
        "conformer_configuration": {**identity, "conformer_configuration_hash": "changed"},
        "pooling_identity": {**identity, "pooling_configuration": {"implementation_version": "changed"}},
    }
    for stale in stale_variants.values():
        with pytest.raises(EmbeddingStoreError, match="provenance mismatch"):
            load_valid_embedding_shard(run_root, 0, expected_molecule_count=len(inventory), identity=stale)


def test_embed_shard_skips_complete_real_adapter_before_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = pd.DataFrame({"canonical_chromophore_smiles": ["C"], "canonical_solvent_smiles": ["O"]})
    dataset_path = tmp_path / "combined.csv"
    dataset.to_csv(dataset_path, index=False)
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset_path, output_dir=run_root / "inventory", shard_size=1)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    dictionary_path = tmp_path / "dict.txt"
    dictionary_path.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\n", encoding="utf-8")
    dictionary = load_conforformer_dictionary(dictionary_path)

    def inspect_assets(_dictionary_path: Path, _checkpoint_path: Path) -> tuple[object, object, object]:
        checkpoint_info = SimpleNamespace(checkpoint_sha256=sha256_file(checkpoint))
        compatibility = SimpleNamespace(architecture=ArchitectureMetadata(encoder_embed_dim=512, source="fake_adapter"))
        return dictionary, checkpoint_info, compatibility

    monkeypatch.setattr(embed_conforformer_shard, "inspect_assets", inspect_assets)
    identity, _dictionary = embed_conforformer_shard.build_embedding_identity_and_dictionary(
        inventory_manifest=manifest,
        checkpoint_path=checkpoint,
        dictionary_path=dictionary_path,
        fake_adapter=False,
        preprocess_config=ConforFormerPreprocessingConfig(),
        conformer_config=ConformerGenerationConfig(),
    )
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[["c0"]],
        embeddings_by_molecule=[np.ones((1, 512), dtype=np.float32)],
        energies_by_molecule=[np.asarray([0.0])],
        failure_codes=[None],
        failure_messages=[None],
        identity=identity,
    )

    def fail_construct(*args: object, **kwargs: object) -> object:
        raise AssertionError("complete resume must not construct the encoder")

    monkeypatch.setattr(embed_conforformer_shard, "ConforFormerEncoderAdapter", fail_construct)
    monkeypatch.setattr(
        embed_conforformer_shard,
        "parse_args",
        lambda: SimpleNamespace(
            run_root=run_root,
            shard_index=0,
            checkpoint=checkpoint,
            dictionary=dictionary_path,
            conformer_cache_dir=None,
            batch_size=8,
            device="cpu",
            fake_adapter=False,
        ),
    )
    assert embed_conforformer_shard.main() == 0
    assert "shard 0 already complete" in capsys.readouterr().out


def test_embed_shard_constructs_adapter_for_stale_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = pd.DataFrame({"canonical_chromophore_smiles": ["C"], "canonical_solvent_smiles": ["O"]})
    dataset_path = tmp_path / "combined.csv"
    dataset.to_csv(dataset_path, index=False)
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset_path, output_dir=run_root / "inventory", shard_size=1)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    dictionary_path = tmp_path / "dict.txt"
    dictionary_path.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\n", encoding="utf-8")
    dictionary = load_conforformer_dictionary(dictionary_path)
    record = generate_conformer_cache_record("C", chromophore_id=str(inventory.loc[0, "molecule_id"]), config=ConformerGenerationConfig())
    save_conformer_cache_record(record, run_root / "conformer_cache")

    def inspect_assets(_dictionary_path: Path, _checkpoint_path: Path) -> tuple[object, object, object]:
        checkpoint_info = SimpleNamespace(checkpoint_sha256=sha256_file(checkpoint))
        compatibility = SimpleNamespace(architecture=ArchitectureMetadata(encoder_embed_dim=512, source="fake_adapter"))
        return dictionary, checkpoint_info, compatibility

    monkeypatch.setattr(embed_conforformer_shard, "inspect_assets", inspect_assets)
    stale_identity, _dictionary = embed_conforformer_shard.build_embedding_identity_and_dictionary(
        inventory_manifest=manifest,
        checkpoint_path=checkpoint,
        dictionary_path=dictionary_path,
        fake_adapter=False,
        preprocess_config=ConforFormerPreprocessingConfig(),
        conformer_config=ConformerGenerationConfig(),
    )
    stale_identity = {**stale_identity, "checkpoint_sha256": "stale"}
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[["old"]],
        embeddings_by_molecule=[np.ones((1, 512), dtype=np.float32)],
        energies_by_molecule=[np.asarray([0.0])],
        failure_codes=[None],
        failure_messages=[None],
        identity=stale_identity,
    )
    constructed = {"count": 0}

    class DummyAdapter:
        def __init__(self, *args: object, **kwargs: object) -> None:
            constructed["count"] += 1

        def encode(self, batch: object) -> object:
            return SimpleNamespace(embedding_array=np.full((len(batch.conformer_ids), 512), 3, dtype=np.float32))

    monkeypatch.setattr(embed_conforformer_shard, "ConforFormerEncoderAdapter", DummyAdapter)
    monkeypatch.setattr(
        embed_conforformer_shard,
        "parse_args",
        lambda: SimpleNamespace(
            run_root=run_root,
            shard_index=0,
            checkpoint=checkpoint,
            dictionary=dictionary_path,
            conformer_cache_dir=None,
            batch_size=8,
            device="cpu",
            fake_adapter=False,
        ),
    )
    assert embed_conforformer_shard.main() == 0
    assert constructed["count"] == 1
    data, _done = load_valid_embedding_shard(run_root, 0, expected_molecule_count=1, identity={**stale_identity, "checkpoint_sha256": sha256_file(checkpoint)})
    assert np.allclose(data["mean_embeddings"][0], 3)
    data.close()


def test_finalize_rejects_stale_provenance_and_nonfinite_embeddings(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "combined.csv")
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset, output_dir=run_root / "inventory", shard_size=10)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nN 1\nO 1\n", encoding="utf-8")
    identity = _identity(manifest, checkpoint, dictionary)
    embeddings = [np.ones((1, 512), dtype=np.float32) for _ in range(len(inventory))]
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[[f"c{i}"] for i in range(len(inventory))],
        embeddings_by_molecule=embeddings,
        energies_by_molecule=[np.asarray([0.0]) for _ in range(len(inventory))],
        failure_codes=[None] * len(inventory),
        failure_messages=[None] * len(inventory),
        identity=identity,
    )
    bad = dict(identity)
    bad["checkpoint_sha256"] = "wrong"
    with pytest.raises(EmbeddingStoreError, match="checkpoint"):
        finalize_embeddings(run_root=run_root, identity=bad)
    manifest_out = finalize_embeddings(run_root=run_root, identity=identity)
    assert manifest_out["success_count"] == len(inventory)


def test_join_embeddings_reports_missing_without_dropping_silently() -> None:
    dataset = pd.DataFrame({"canonical_chromophore_smiles": ["C", "C", "N"], "value": [1, 2, 3]})
    embeddings = pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["C", "N"],
            "embedding_status": ["success", "terminal_failure"],
            "mean": [np.zeros(512, dtype=np.float32), np.zeros(512, dtype=np.float32)],
        }
    )
    included, excluded = join_embeddings(dataset, embeddings)
    assert len(included) == 2
    assert excluded["canonical_chromophore_smiles"].tolist() == ["N"]


def test_feature_bundle_keeps_conforformer_rows_when_morgan_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["C", "bad", "N"],
            "canonical_solvent_smiles": ["O", "O", "O"],
            "solvent_original": ["water", "water", "water"],
            "absorption_nm": [300, 310, 320],
            "emission_nm": [360, 370, 390],
            "quantum_yield": [0.1, 0.2, 0.3],
        }
    )
    dataset_path = tmp_path / "combined.csv"
    dataset.to_csv(dataset_path, index=False)
    solvent_path = tmp_path / "solvent.csv"
    pd.DataFrame({"canonical_solvent_smiles": ["O"], "solvent_original": ["water"], "polarity": [1.0]}).to_csv(solvent_path, index=False)
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset_path, output_dir=run_root / "inventory", shard_size=10)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nN 1\nO 1\n", encoding="utf-8")
    identity = _identity(manifest, checkpoint, dictionary)
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[[f"c{i}"] for i in range(len(inventory))],
        embeddings_by_molecule=[np.full((1, 512), i + 1, dtype=np.float32) for i in range(len(inventory))],
        energies_by_molecule=[np.asarray([0.0]) for _ in range(len(inventory))],
        failure_codes=[None] * len(inventory),
        failure_messages=[None] * len(inventory),
        identity=identity,
    )
    finalize_embeddings(run_root=run_root, identity=identity)

    def fake_morgan(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
        if smiles == "bad":
            return None
        return np.ones((n_bits,), dtype=np.float32)

    monkeypatch.setattr(downstream, "morgan_fingerprint", fake_morgan)
    bundle, embedding_excluded, morgan_excluded = build_feature_bundle(
        dataset_csv=dataset_path,
        embedding_run_root=run_root,
        solvent_descriptors=solvent_path,
        n_bits=16,
    )
    assert embedding_excluded.empty
    assert len(bundle.rows) == 3
    assert downstream._feature_set_mask(bundle, "conforformer_solvent").tolist() == [True, True, True]
    assert downstream._feature_set_mask(bundle, "morgan_solvent").tolist() == [True, False, True]
    assert morgan_excluded["canonical_chromophore_smiles"].tolist() == ["bad"]


def test_morgan_pooling_grid_is_not_applicable_once() -> None:
    poolings = ["mean", "lowest_energy", "boltzmann_298k"]
    assert downstream._poolings_for_feature_set(poolings, "conforformer_solvent") == poolings
    assert downstream._poolings_for_feature_set(poolings, "conforformer_morgan_solvent") == poolings
    assert downstream._poolings_for_feature_set(poolings, "morgan_solvent") == ["not_applicable"]


def test_train_downstream_writes_morgan_not_applicable_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    smiles = ["C" * length for length in range(1, 13)]
    dataset = pd.DataFrame(
        {
            "canonical_chromophore_smiles": smiles,
            "canonical_solvent_smiles": ["O"] * len(smiles),
            "solvent_original": ["water"] * len(smiles),
            "absorption_nm": np.arange(300, 312, dtype=float),
            "emission_nm": np.arange(360, 372, dtype=float),
            "quantum_yield": np.linspace(0.1, 0.9, len(smiles)),
        }
    )
    dataset_path = tmp_path / "combined.csv"
    dataset.to_csv(dataset_path, index=False)
    solvent_path = tmp_path / "solvent.csv"
    pd.DataFrame({"canonical_solvent_smiles": ["O"], "solvent_original": ["water"], "polarity": [1.0]}).to_csv(solvent_path, index=False)
    run_root = tmp_path / "run"
    build_inventory(source_csv=dataset_path, output_dir=run_root / "inventory", shard_size=20)
    inventory, manifest = load_inventory(run_root)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\nO 1\n", encoding="utf-8")
    identity = _identity(manifest, checkpoint, dictionary)
    write_embedding_shard(
        run_root=run_root,
        shard_index=0,
        rows=inventory,
        conformer_ids_by_molecule=[[f"c{i}"] for i in range(len(inventory))],
        embeddings_by_molecule=[np.full((1, 512), i + 1, dtype=np.float32) for i in range(len(inventory))],
        energies_by_molecule=[np.asarray([0.0]) for _ in range(len(inventory))],
        failure_codes=[None] * len(inventory),
        failure_messages=[None] * len(inventory),
        identity=identity,
    )
    finalize_embeddings(run_root=run_root, identity=identity)
    monkeypatch.setattr(downstream, "morgan_fingerprint", lambda smiles, *, radius=2, n_bits=2048: np.ones((n_bits,), dtype=np.float32))
    monkeypatch.setattr(downstream, "make_candidates", lambda seed, n_jobs: {"constant": ConstantRegressor()})

    manifest_out = train_downstream(
        dataset_csv=dataset_path,
        embedding_run_root=run_root,
        solvent_descriptors=solvent_path,
        out_dir=tmp_path / "downstream",
        model_out_dir=tmp_path / "models",
        n_bits=8,
        n_jobs=1,
        feature_sets=["morgan_solvent"],
        pooling_methods=["mean", "lowest_energy", "boltzmann_298k"],
    )
    metrics_df = pd.read_csv(tmp_path / "downstream" / "metrics.csv")
    assert set(metrics_df["pooling_method"]) == {"not_applicable"}
    assert manifest_out["effective_pooling_methods_by_feature_set"]["morgan_solvent"] == ["not_applicable"]
    assert (tmp_path / "downstream" / "predictions" / "absorption_nm__not_applicable__morgan_solvent.csv").exists()
    assert not list((tmp_path / "downstream" / "predictions").glob("*__mean__morgan_solvent.csv"))


def test_fixed_split_has_no_molecule_or_scaffold_leakage() -> None:
    rows = pd.DataFrame(
        {
            "row_id": range(12),
            "canonical_chromophore_smiles": ["C", "C", "CC", "CC", "CCC", "CCC", "N", "N", "O", "O", "CO", "CO"],
        }
    )
    assignments, leakage = make_split_assignments(rows, split_type="molecule", seed=7)
    assert leakage["leakage_group_count"] == 0
    for _mol, part in assignments.groupby("canonical_chromophore_smiles"):
        assert part["split"].nunique() == 1
    scaffold_assignments, scaffold_leakage = make_split_assignments(rows, split_type="scaffold", seed=7)
    assert scaffold_leakage["leakage_group_count"] == 0
    assert set(scaffold_assignments["split"]).issubset({"base_train", "model_selection", "final_test"})


def test_quantum_yield_clipping_metric_fields() -> None:
    y = np.asarray([0.0, 0.5, 1.0])
    pred = np.asarray([-0.2, 0.4, 1.2])
    raw = metrics(y, pred)
    assert raw["count"] == 3
    assert np.mean(pred < 0) == pytest.approx(1 / 3)
    assert np.mean(pred > 1) == pytest.approx(1 / 3)


def test_cli_fake_adapter_canary(tmp_path: Path) -> None:
    dataset = pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["C", "CC"],
            "canonical_solvent_smiles": ["O", "O"],
        }
    )
    dataset_path = tmp_path / "combined.csv"
    dataset.to_csv(dataset_path, index=False)
    run_root = tmp_path / "run"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("fake checkpoint", encoding="utf-8")
    dictionary = tmp_path / "dict.txt"
    dictionary.write_text("[PAD] 1\n[CLS] 1\n[SEP] 1\n[UNK] 1\nC 1\n", encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "scripts/build_conforformer_inventory.py",
            "--dataset",
            str(dataset_path),
            "--run-root",
            str(run_root),
            "--shard-size",
            "2",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "scripts/build_conformer_cache_shard.py", "--run-root", str(run_root), "--shard-index", "0", "--num-conformers", "1"], check=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/embed_conforformer_shard.py",
            "--run-root",
            str(run_root),
            "--shard-index",
            "0",
            "--checkpoint",
            str(checkpoint),
            "--dictionary",
            str(dictionary),
            "--fake-adapter",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/finalize_conforformer_embeddings.py",
            "--run-root",
            str(run_root),
            "--checkpoint",
            str(checkpoint),
            "--dictionary",
            str(dictionary),
            "--fake-architecture",
        ],
        check=True,
    )
    assert (run_root / "embedding_manifest.json").exists()
