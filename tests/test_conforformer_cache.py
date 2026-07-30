from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from rdkit import rdBase

import scripts.build_conformer_cache as build_conformer_cache
import scripts.build_conformer_cache_shard as build_conformer_cache_shard
from chemfluor.conforformer.cache import (
    CONFORMER_CACHE_SCHEMA_VERSION,
    CacheError,
    EmbeddingCacheKeyPayload,
    PoolingCacheKeyPayload,
    build_conformer_cache_key,
    build_embedding_cache_key,
    build_pooling_cache_key,
    load_conformer_cache_record,
    save_conformer_cache_record,
    sha256_payload,
    stable_json_dumps,
)
from chemfluor.conforformer.config import ConformerGenerationConfig
from chemfluor.conforformer.conformers import generate_conformer_cache_record


def _record():
    config = ConformerGenerationConfig(num_conformers=2, retry_conformer_counts=(1,))
    return generate_conformer_cache_record("CCO", chromophore_id="ethanol", config=config)


def test_identical_payloads_and_dictionary_order_have_identical_keys() -> None:
    assert sha256_payload({"b": 1, "a": None}) == sha256_payload({"a": None, "b": 1})
    assert stable_json_dumps({"a": None}) == '{"a":null}'


def test_conformer_key_changes_with_seed_and_rdkit_version() -> None:
    base = ConformerGenerationConfig(num_conformers=2, retry_conformer_counts=(1,), random_seed=1)
    changed_seed = ConformerGenerationConfig(num_conformers=2, retry_conformer_counts=(1,), random_seed=2)
    key = build_conformer_cache_key(
        canonical_smiles="CCO",
        isomeric_canonical_smiles="CCO",
        config=base,
        rdkit_version=rdBase.rdkitVersion,
    )
    assert key != build_conformer_cache_key(
        canonical_smiles="CCO",
        isomeric_canonical_smiles="CCO",
        config=changed_seed,
        rdkit_version=rdBase.rdkitVersion,
    )
    assert key != build_conformer_cache_key(
        canonical_smiles="CCO",
        isomeric_canonical_smiles="CCO",
        config=base,
        rdkit_version="different",
    )


def test_embedding_key_changes_with_checkpoint_dictionary_and_hydrogen_policy() -> None:
    payload = EmbeddingCacheKeyPayload(
        conformer_cache_key="conf",
        upstream_commit="commit",
        checkpoint_sha256="checkpoint-a",
        dictionary_sha256="dict-a",
        architecture_payload={"embed_dim": 512},
        hydrogen_removal_policy="remove_all",
    )
    key = build_embedding_cache_key(payload)
    assert key != build_embedding_cache_key(
        EmbeddingCacheKeyPayload(**{**payload.__dict__, "checkpoint_sha256": "checkpoint-b"})
    )
    assert key != build_embedding_cache_key(
        EmbeddingCacheKeyPayload(**{**payload.__dict__, "dictionary_sha256": "dict-b"})
    )
    assert key != build_embedding_cache_key(
        EmbeddingCacheKeyPayload(**{**payload.__dict__, "hydrogen_removal_policy": "keep_all"})
    )


def test_pooling_method_changes_only_pooling_key() -> None:
    embedding_payload = EmbeddingCacheKeyPayload(
        conformer_cache_key="conf",
        upstream_commit="commit",
        checkpoint_sha256="checkpoint",
        dictionary_sha256="dict",
        architecture_payload={"embed_dim": 512},
        hydrogen_removal_policy="remove_all",
    )
    embedding_key = build_embedding_cache_key(embedding_payload)
    mean_key = build_pooling_cache_key(PoolingCacheKeyPayload(embedding_cache_key=embedding_key, pooling_method="mean"))
    boltzmann_key = build_pooling_cache_key(
        PoolingCacheKeyPayload(
            embedding_cache_key=embedding_key,
            pooling_method="boltzmann",
            temperature_kelvin=298.15,
        )
    )
    hotter_key = build_pooling_cache_key(
        PoolingCacheKeyPayload(
            embedding_cache_key=embedding_key,
            pooling_method="boltzmann",
            temperature_kelvin=350.0,
        )
    )
    assert mean_key != boltzmann_key
    assert boltzmann_key != hotter_key
    assert build_embedding_cache_key(embedding_payload) == embedding_key


def test_cache_round_trip_preserves_record(tmp_path: Path) -> None:
    record = _record()
    path = save_conformer_cache_record(record, tmp_path)
    loaded = load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key)
    assert loaded.to_payload() == record.to_payload()
    assert path.name == f"{record.conformer_cache_key}.json"


def test_corrupted_metadata_fails(tmp_path: Path) -> None:
    record = _record()
    path = save_conformer_cache_record(record, tmp_path)
    wrapped = json.loads(path.read_text(encoding="utf-8"))
    wrapped["record"]["chromophore_id"] = "changed"
    path.write_text(json.dumps(wrapped), encoding="utf-8")
    with pytest.raises(CacheError, match="hash mismatch"):
        load_conformer_cache_record(path)


def test_wrong_cache_key_and_schema_version_fail(tmp_path: Path) -> None:
    record = _record()
    path = save_conformer_cache_record(record, tmp_path)
    with pytest.raises(CacheError, match="expected key"):
        load_conformer_cache_record(path, expected_cache_key="wrong")
    wrapped = json.loads(path.read_text(encoding="utf-8"))
    wrapped["schema_version"] = CONFORMER_CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(wrapped), encoding="utf-8")
    with pytest.raises(CacheError, match="unsupported"):
        load_conformer_cache_record(path)


def test_valid_records_are_not_overwritten_without_permission(tmp_path: Path) -> None:
    record = _record()
    save_conformer_cache_record(record, tmp_path)
    with pytest.raises(FileExistsError):
        save_conformer_cache_record(record, tmp_path)
    save_conformer_cache_record(record, tmp_path, overwrite=True)


def test_atomic_write_failure_leaves_no_final_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    record = _record()

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        save_conformer_cache_record(record, tmp_path)
    assert not (tmp_path / f"{record.conformer_cache_key}.json").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_records_preserve_smiles(tmp_path: Path) -> None:
    record = generate_conformer_cache_record("not a smiles", chromophore_id="bad")
    path = save_conformer_cache_record(record, tmp_path)
    loaded = load_conformer_cache_record(path)
    assert loaded.input_smiles == "not a smiles"
    assert loaded.canonical_smiles is None
    assert loaded.failure_reason == "invalid_smiles"


def test_cli_dry_run_smiles_creates_no_files(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_conformer_cache.py",
            "--smiles",
            "CCO",
            "CCCC",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "requested molecules: 2" in result.stdout
    assert not list(tmp_path.glob("*.json"))


def test_cli_dry_run_does_not_generate_conformers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_generate(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not generate conformers")

    monkeypatch.setattr(build_conformer_cache, "generate_conformer_cache_record", fail_generate)
    exit_code = build_conformer_cache.main(
        [
            "--smiles",
            "CCO",
            "--output-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert not list(tmp_path.glob("*.json"))


def test_shard_helper_loads_valid_cache_before_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ConformerGenerationConfig(num_conformers=1, retry_conformer_counts=())
    record = generate_conformer_cache_record("CCO", chromophore_id="ethanol", config=config)
    save_conformer_cache_record(record, tmp_path)

    def fail_generate(*args: object, **kwargs: object) -> object:
        raise AssertionError("valid resume must not generate conformers")

    monkeypatch.setattr(build_conformer_cache_shard, "generate_conformer_cache_record", fail_generate)
    loaded, path, cache_hit = build_conformer_cache_shard.load_or_generate_cache_record(
        smiles="OCC",
        chromophore_id="ethanol",
        cache_dir=tmp_path,
        config=config,
    )
    assert loaded.to_payload() == record.to_payload()
    assert path.name == f"{record.conformer_cache_key}.json"
    assert cache_hit is True


def test_shard_helper_regenerates_corrupt_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = ConformerGenerationConfig(num_conformers=1, retry_conformer_counts=())
    record = generate_conformer_cache_record("CCO", chromophore_id="ethanol", config=config)
    path = save_conformer_cache_record(record, tmp_path)
    path.write_text("corrupt", encoding="utf-8")
    calls = {"count": 0}

    def generate(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        return record

    monkeypatch.setattr(build_conformer_cache_shard, "generate_conformer_cache_record", generate)
    loaded, _path, cache_hit = build_conformer_cache_shard.load_or_generate_cache_record(
        smiles="CCO",
        chromophore_id="ethanol",
        cache_dir=tmp_path,
        config=config,
    )
    assert loaded.to_payload() == record.to_payload()
    assert cache_hit is False
    assert calls["count"] == 1
    assert load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key).to_payload() == record.to_payload()


def test_cli_csv_dry_run_deduplicates_and_counts_invalid(tmp_path: Path) -> None:
    csv_path = tmp_path / "smiles.csv"
    csv_path.write_text("id,smiles\none,CCO\ntwo,OCC\nbad,not a smiles\nbutane,CCCC\n", encoding="utf-8")
    output_dir = tmp_path / "cache"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_conformer_cache.py",
            "--input-csv",
            str(csv_path),
            "--smiles-column",
            "smiles",
            "--id-column",
            "id",
            "--output-dir",
            str(output_dir),
            "--max-molecules",
            "3",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "requested molecules: 3" in result.stdout
    assert "unique canonical molecules: 2" in result.stdout
    assert "invalid SMILES: 1" in result.stdout
    assert not output_dir.exists()
