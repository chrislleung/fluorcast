from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chemfluor.conforformer.config import ConformerGenerationConfig
from chemfluor.conforformer.downstream import join_embeddings, make_split_assignments, metrics, train_downstream
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

