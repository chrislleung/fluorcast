from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.geometry_cache import (
    GEOMETRY_SCHEMA_VERSION,
    atomic_write_json,
    build_geometry_cache,
    cache_path,
    generate_geometry_entry,
    load_molecule_manifest,
    molecule_seed,
    payload_checksum,
    read_valid_cache,
    validate_geometry_entry,
)
from chemfluor.uniprop.manifests import MANIFEST_SCHEMA_VERSION, stable_hash


def molecule_id(smiles: str) -> str:
    return stable_hash("mol", MANIFEST_SCHEMA_VERSION, smiles)


def write_manifest(tmp_path: Path, smiles: list[str]) -> Path:
    rows = pd.DataFrame(
        {
            "molecule_id": [molecule_id(smi) for smi in smiles],
            "canonical_isomeric_smiles": smiles,
            "canonical_nonisomeric_smiles": smiles,
            "source_row_count": [1] * len(smiles),
        }
    )
    path = tmp_path / "molecule_manifest.csv"
    rows.to_csv(path, index=False)
    return path


def test_deterministic_coordinates_for_same_molecule_and_seed() -> None:
    mid = molecule_id("CCO")
    first = generate_geometry_entry(mid, "CCO")
    second = generate_geometry_entry(mid, "CCO")

    assert first["seed"] == molecule_seed(mid)
    np.testing.assert_allclose(first["coordinates"], second["coordinates"], atol=1e-8)
    assert first["atom_symbols"] == ["C", "C", "O"]


def test_identical_molecule_used_multiple_times_maps_to_one_cache_file(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["CCO"])
    cache_dir = tmp_path / "cache"
    results = build_geometry_cache(path, cache_dir)

    assert [result.status for result in results] == ["generated"]
    assert len(list(cache_dir.glob("*.json"))) == 1
    assert list(cache_dir.glob("*.json"))[0].name == f"{molecule_id('CCO')}.json"


def test_valid_cache_hit_performs_no_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = write_manifest(tmp_path, ["CCO"])
    cache_dir = tmp_path / "cache"
    assert build_geometry_cache(path, cache_dir)[0].status == "generated"

    def fail_generate(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("cache hit should not regenerate")

    monkeypatch.setattr("chemfluor.uniprop.geometry_cache.generate_geometry_entry", fail_generate)
    assert build_geometry_cache(path, cache_dir, resume=True)[0].status == "hit"


def test_corrupt_and_partial_files_are_detected(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["CCO"])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path(cache_dir, molecule_id("CCO")).write_text("{not json", encoding="utf-8")

    result = build_geometry_cache(path, cache_dir, resume=True, overwrite_invalid=False)[0]

    assert result.status == "invalid_cache"
    assert result.failure_reason == "invalid_cache"


def test_invalid_smiles_produces_structured_failure(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["not_a_smiles"])
    result = build_geometry_cache(path, tmp_path / "cache")[0]

    assert result.status == "failed"
    assert result.failure_reason == "generation_failed"
    assert "Invalid canonical SMILES" in str(result.detail)


def test_atom_and_coordinate_shapes_match() -> None:
    entry = generate_geometry_entry(molecule_id("c1ccccc1"), "c1ccccc1")

    assert len(entry["atom_symbols"]) == len(entry["atomic_numbers"])
    assert len(entry["atom_symbols"]) == len(entry["coordinates"])
    assert all(len(row) == 3 for row in entry["coordinates"])


def test_topology_changes_are_rejected() -> None:
    entry = generate_geometry_entry(molecule_id("CCO"), "CCO")
    entry["topology_signature"]["atomic_numbers"][0] = 8
    entry["checksum"] = payload_checksum(entry)

    with pytest.raises(ValueError, match="topology"):
        validate_geometry_entry(entry, molecule_id=molecule_id("CCO"), canonical_smiles="CCO")


def test_mmff_and_fallback_provenance_are_correct() -> None:
    mmff = generate_geometry_entry(molecule_id("CCO"), "CCO", mmff_variant="MMFF94")
    fallback = generate_geometry_entry(molecule_id("B(O)O"), "B(O)O")

    assert mmff["optimization_method"] == "MMFF94"
    assert fallback["optimization_method"] == "UFF"
    assert fallback["energy"] is not None


def test_atomic_write_does_not_leave_valid_looking_partial_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = generate_geometry_entry(molecule_id("CCO"), "CCO")
    output = tmp_path / "cache" / "entry.json"
    original_replace = os.replace

    def fail_replace(src: str, dst: str) -> None:
        raise OSError("simulated interrupted replace")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError):
        atomic_write_json(output, entry)
    monkeypatch.setattr(os, "replace", original_replace)

    assert not output.exists()
    assert not list(output.parent.glob("*.tmp"))


def test_overwrite_invalid_regenerates_after_intentional_corruption(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["CCO"])
    cache_dir = tmp_path / "cache"
    generated = build_geometry_cache(path, cache_dir)[0]
    assert generated.status == "generated"
    cache_path(cache_dir, molecule_id("CCO")).write_text("{}", encoding="utf-8")

    invalid = build_geometry_cache(path, cache_dir, overwrite_invalid=False)[0]
    regenerated = build_geometry_cache(path, cache_dir, overwrite_invalid=True)[0]

    assert invalid.status == "invalid_cache"
    assert regenerated.status == "generated"
    read_valid_cache(cache_path(cache_dir, molecule_id("CCO")), molecule_id("CCO"), "CCO")


def test_three_molecule_cli_smoke_reconciles(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["CCO", "CCN", "c1ccccc1"])
    cache_dir = tmp_path / "cache"
    status_json = tmp_path / "status.json"
    failure_json = tmp_path / "failures.json"
    failure_csv = tmp_path / "failures.csv"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_uniprop_geometry_cache.py",
            "--molecule-manifest",
            str(path),
            "--cache-dir",
            str(cache_dir),
            "--workers",
            "2",
            "--status-json",
            str(status_json),
            "--failure-json",
            str(failure_json),
            "--failure-csv",
            str(failure_csv),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    summary = json.loads(status_json.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert summary["reconciles"]
    assert summary["status_counts"] == {"generated": 3}
    assert json.loads(failure_json.read_text(encoding="utf-8")) == []
    assert len(list(cache_dir.glob("*.json"))) == 3


def test_twenty_molecule_cli_smoke(tmp_path: Path) -> None:
    smiles = [
        "CCO", "CCN", "CCC", "CCCl", "CCBr",
        "c1ccccc1", "Cc1ccccc1", "c1ccncc1", "C1CCCCC1", "CC(=O)O",
        "COC", "CCS", "CC(C)O", "CC(C)N", "CC#N",
        "C=CC", "CC=O", "CC(=O)N", "CNC", "CCF",
    ]
    path = write_manifest(tmp_path, smiles)
    cache_dir = tmp_path / "cache"
    status_json = tmp_path / "status.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_uniprop_geometry_cache.py",
            "--molecule-manifest",
            str(path),
            "--cache-dir",
            str(cache_dir),
            "--limit",
            "20",
            "--workers",
            "2",
            "--status-json",
            str(status_json),
            "--failure-json",
            str(tmp_path / "failures.json"),
            "--failure-csv",
            str(tmp_path / "failures.csv"),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    summary = json.loads(status_json.read_text(encoding="utf-8"))

    assert completed.returncode == 0, completed.stderr
    assert summary["processed_total"] == 20
    assert summary["status_counts"] == {"generated": 20}
    assert len(list(cache_dir.glob("*.json"))) == 20


def test_molecule_manifest_loader_rejects_target_columns_requirement(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, ["CCO"])
    loaded = load_molecule_manifest(path)

    assert "absorption_nm" not in loaded.columns
    assert loaded.loc[0, "molecule_id"] == molecule_id("CCO")
    assert GEOMETRY_SCHEMA_VERSION.startswith("uniprop_geometry_cache")
