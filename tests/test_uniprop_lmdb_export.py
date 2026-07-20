from __future__ import annotations

import gzip
import builtins
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.geometry_cache import atomic_write_json, generate_geometry_entry
from chemfluor.uniprop.lmdb_export import (
    DEFAULT_TARGET_COLUMNS,
    ExportConfig,
    build_lmdb_record,
    decode_int_key,
    encode_int_key,
    export_uniprop_lmdb,
    get_graph,
    load_export_inputs,
    read_lmdb_records,
    validate_lmdb,
    validate_record,
)
from chemfluor.uniprop.manifests import MANIFEST_SCHEMA_VERSION, stable_hash
from chemfluor.uniprop.upstream_compat import TargetMaskDataset


TARGETS = ("absorption_nm", "emission_nm", "quantum_yield")


def molecule_id(smiles: str) -> str:
    return stable_hash("mol", MANIFEST_SCHEMA_VERSION, smiles)


def solvent_id(smiles: str) -> str:
    return stable_hash("solv", MANIFEST_SCHEMA_VERSION, smiles)


def write_fixture_manifests(tmp_path: Path, n_rows: int = 6) -> dict[str, Path]:
    smiles_cycle = ["CCO", "CCO", "CCN", "c1ccccc1", "CCO", "CCN"]
    solvent_cycle = ["O", "CCO", "O", "CC#N", "CS(C)=O", "CCO"]
    rows = []
    for index in range(n_rows):
        smiles = smiles_cycle[index % len(smiles_cycle)]
        solvent = solvent_cycle[index % len(solvent_cycle)]
        rows.append(
            {
                "row_id": f"row_{index:04d}",
                "molecule_id": molecule_id(smiles),
                "solvent_id": solvent_id(solvent),
                "canonical_solvent_smiles": solvent,
                "source_dataset": "fixture",
                "absorption_nm": 350.0 + index if index % 3 != 1 else np.nan,
                "absorption_nm_available": index % 3 != 1,
                "emission_nm": 450.0 + index,
                "emission_nm_available": True,
                "quantum_yield": 0.1 * index if index % 4 != 2 else np.nan,
                "quantum_yield_available": index % 4 != 2,
            }
        )
    row_manifest = pd.DataFrame(rows)
    molecules = (
        row_manifest[["molecule_id"]]
        .drop_duplicates()
        .assign(
            canonical_isomeric_smiles=lambda df: df["molecule_id"].map(
                {
                    molecule_id("CCO"): "CCO",
                    molecule_id("CCN"): "CCN",
                    molecule_id("c1ccccc1"): "c1ccccc1",
                }
            )
        )
    )
    molecules["canonical_nonisomeric_smiles"] = molecules["canonical_isomeric_smiles"]
    molecules["source_row_count"] = molecules["molecule_id"].map(row_manifest["molecule_id"].value_counts())
    splits = pd.DataFrame(
        {
            "row_id": row_manifest["row_id"],
            "random": ["train" if index % 5 else "test" for index in range(n_rows)],
        }
    )
    geometry_dir = tmp_path / "geometry"
    for _, molecule in molecules.iterrows():
        entry = generate_geometry_entry(
            str(molecule["molecule_id"]),
            str(molecule["canonical_isomeric_smiles"]),
        )
        atomic_write_json(geometry_dir / f"{molecule['molecule_id']}.json", entry)

    row_path = tmp_path / "row_manifest.csv"
    molecule_path = tmp_path / "molecule_manifest.csv"
    split_path = tmp_path / "split_assignments.csv"
    row_manifest.to_csv(row_path, index=False)
    molecules.to_csv(molecule_path, index=False)
    splits.to_csv(split_path, index=False)
    return {
        "row_manifest": row_path,
        "molecule_manifest": molecule_path,
        "split_assignments": split_path,
        "geometry_dir": geometry_dir,
    }


def config(tmp_path: Path, n_rows: int = 6) -> ExportConfig:
    paths = write_fixture_manifests(tmp_path, n_rows=n_rows)
    return ExportConfig(
        row_manifest_path=paths["row_manifest"],
        molecule_manifest_path=paths["molecule_manifest"],
        split_assignments_path=paths["split_assignments"],
        geometry_cache_dir=paths["geometry_dir"],
        output_dir=tmp_path / "lmdb",
        split_family="random",
        seed=7,
        target_columns=TARGETS,
        map_size=64 * 1024 * 1024,
        batch_size=2,
        overwrite=True,
        resume=True,
        valid_size=0.25,
    )


def test_exact_required_keys_shapes_and_dtypes(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    rows = load_export_inputs(cfg)
    row = rows.iloc[0]
    geometry = json.loads((cfg.geometry_cache_dir / f"{row['molecule_id']}.json").read_text())
    record = build_lmdb_record(row, geometry, TARGETS, integer_id=0)

    required = {
        "atoms",
        "input_pos",
        "label_pos",
        "smi",
        "solvent_smi",
        "node_attr",
        "edge_index",
        "edge_attr",
        "target",
        "target_mask",
        "row_id",
        "molecule_id",
    }
    assert required.issubset(record)
    assert record["node_attr"].dtype == np.int32
    assert record["edge_index"].dtype == np.int32
    assert record["edge_attr"].dtype == np.int32
    assert np.asarray(record["input_pos"][0]).shape == np.asarray(record["label_pos"]).shape
    assert validate_record(record, TARGETS) == []


def test_graph_features_agree_with_direct_rdkit_calculation(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    rows = load_export_inputs(cfg)
    row = rows.iloc[0]
    geometry = json.loads((cfg.geometry_cache_dir / f"{row['molecule_id']}.json").read_text())
    record = build_lmdb_record(row, geometry, TARGETS, integer_id=0)

    expected = get_graph(Chem.MolFromSmiles(str(row["canonical_isomeric_smiles"])))

    np.testing.assert_array_equal(record["node_attr"], expected[0])
    np.testing.assert_array_equal(record["edge_index"], expected[1])
    np.testing.assert_array_equal(record["edge_attr"], expected[2])


def test_repeated_chromophore_rows_share_coordinates_but_keep_row_solvent_targets(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    rows = load_export_inputs(cfg)
    cco_rows = rows[rows["molecule_id"] == molecule_id("CCO")]
    records = []
    for integer_id, (_, row) in enumerate(cco_rows.iterrows()):
        geometry = json.loads((cfg.geometry_cache_dir / f"{row['molecule_id']}.json").read_text())
        records.append(build_lmdb_record(row, geometry, TARGETS, integer_id=integer_id))

    np.testing.assert_array_equal(records[0]["input_pos"][0], records[1]["input_pos"][0])
    assert records[0]["solvent_smi"] != records[1]["solvent_smi"]
    assert records[0]["row_id"] != records[1]["row_id"]
    assert not np.array_equal(records[0]["target"], records[1]["target"], equal_nan=True)


def test_masks_match_missingness_and_missing_targets_are_nan(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    rows = load_export_inputs(cfg)
    row = rows[rows["absorption_nm"].isna()].iloc[0]
    geometry = json.loads((cfg.geometry_cache_dir / f"{row['molecule_id']}.json").read_text())
    record = build_lmdb_record(row, geometry, TARGETS, integer_id=0)

    assert record["target_mask"].tolist() == [False, True, True]
    assert np.isnan(record["target"][0])
    assert record["target"][0] != 0.0


class _NumpyCollaterDataset:
    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"target_mask": np.asarray([index == 0, index == 1, True])}

    def collater(self, items: list[dict[str, object]]) -> dict[str, object]:
        return {"target": np.zeros((len(items), 3), dtype=np.float32)}


def test_target_mask_dataset_preserves_masks_as_numpy_for_numpy_collater() -> None:
    wrapped = TargetMaskDataset(_NumpyCollaterDataset())
    batch = wrapped.collater([wrapped[0], wrapped[1]])

    assert isinstance(batch["target_mask"], np.ndarray)
    assert batch["target_mask"].shape == (2, 3)
    assert batch["target_mask"].dtype == np.bool_
    np.testing.assert_array_equal(
        batch["target_mask"],
        np.asarray([[True, False, True], [False, True, True]], dtype=np.bool_),
    )


def test_target_mask_dataset_operates_without_torch_for_numpy_collater(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def import_without_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_torch)

    wrapped = TargetMaskDataset(_NumpyCollaterDataset())
    batch = wrapped.collater([wrapped[0], wrapped[1]])

    assert isinstance(batch["target_mask"], np.ndarray)
    assert batch["target_mask"].dtype == np.bool_


def test_target_mask_dataset_preserves_empty_items() -> None:
    wrapped = TargetMaskDataset(_NumpyCollaterDataset())

    assert wrapped.collater([])["target"].shape == (0, 3)


def test_target_mask_dataset_preserves_items_without_masks() -> None:
    class DummyDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, object]:
            return {"target": np.asarray([float(index)], dtype=np.float32)}

        def collater(self, items: list[dict[str, object]]) -> dict[str, object]:
            return {"target": np.zeros((len(items), 1), dtype=np.float32)}

    wrapped = TargetMaskDataset(DummyDataset())
    batch = wrapped.collater([wrapped[0], wrapped[1]])

    assert "target_mask" not in batch


def test_target_mask_dataset_returns_torch_mask_when_target_is_torch_tensor() -> None:
    torch = pytest.importorskip("torch")

    class TorchCollaterDataset:
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int) -> dict[str, object]:
            return {"target_mask": np.asarray([index == 0, True])}

        def collater(self, items: list[dict[str, object]]) -> dict[str, object]:
            return {"target": torch.zeros((len(items), 2), dtype=torch.float32)}

    wrapped = TargetMaskDataset(TorchCollaterDataset())
    batch = wrapped.collater([wrapped[0], wrapped[1]])

    assert torch.is_tensor(batch["target_mask"])
    assert batch["target_mask"].dtype == torch.bool
    assert batch["target_mask"].device == batch["target"].device
    assert tuple(batch["target_mask"].shape) == (2, 2)


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_lmdb_key_order_is_deterministic_and_row_ids_once(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    metadata = export_uniprop_lmdb(cfg)
    train_records = read_lmdb_records(cfg.output_dir / "train.lmdb")

    assert metadata["row_counts"]["total_selected"] == 6
    assert [decode_int_key(key) for key, _ in train_records] == list(range(len(train_records)))
    row_ids = [record["row_id"] for _, record in train_records]
    assert len(row_ids) == len(set(row_ids))


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_corrupt_or_incomplete_databases_are_rejected(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    export_uniprop_lmdb(cfg)
    marker = cfg.output_dir / "train.lmdb.complete"
    marker.unlink()

    report = validate_lmdb(cfg.output_dir / "train.lmdb", target_columns=TARGETS)

    assert not report["valid"]
    assert "completion marker is missing" in report["errors"]


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_validation_reports_reconcile_with_row_manifest(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    metadata = export_uniprop_lmdb(cfg)
    row_ids = set(metadata["partition_reports"]["test"]["row_ids"])

    report = validate_lmdb(
        cfg.output_dir / "test.lmdb",
        expected_row_ids=row_ids,
        target_columns=TARGETS,
    )

    assert report["valid"]
    assert set(report["row_ids"]) == row_ids


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_all_row_ids_occur_exactly_once_across_partitions(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    export_uniprop_lmdb(cfg)
    exported = []
    for partition in ["train", "valid", "test"]:
        exported.extend(record["row_id"] for _, record in read_lmdb_records(cfg.output_dir / f"{partition}.lmdb"))
    expected = set(pd.read_csv(cfg.row_manifest_path)["row_id"].astype(str))

    assert len(exported) == len(set(exported))
    assert set(exported) == expected


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_export_does_not_generate_geometries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = config(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("LMDB export must not generate conformers")

    monkeypatch.setattr("chemfluor.uniprop.geometry_cache.generate_geometry_entry", forbidden)

    metadata = export_uniprop_lmdb(cfg)

    assert metadata["row_counts"]["total_selected"] == 6


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_export_cli_and_validate_cli_smoke(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/export_uniprop_lmdb.py",
            "--row-manifest",
            str(cfg.row_manifest_path),
            "--molecule-manifest",
            str(cfg.molecule_manifest_path),
            "--split-assignments",
            str(cfg.split_assignments_path),
            "--geometry-cache-dir",
            str(cfg.geometry_cache_dir),
            "--out-dir",
            str(cfg.output_dir),
            "--split-family",
            "random",
            "--targets",
            ",".join(TARGETS),
            "--map-size",
            str(cfg.map_size),
            "--batch-size",
            "2",
            "--overwrite",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    validation = subprocess.run(
        [
            sys.executable,
            "scripts/validate_uniprop_lmdb.py",
            str(cfg.output_dir / "test.lmdb"),
            "--targets",
            ",".join(TARGETS),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr


@pytest.mark.skipif(importlib.util.find_spec("lmdb") is None, reason="lmdb is not installed")
def test_hundred_row_lmdb_loads_through_upstream_lmdb_dataset(tmp_path: Path) -> None:
    cfg = config(tmp_path, n_rows=100)
    export_uniprop_lmdb(cfg)
    upstream_lmdb = PROJECT_ROOT / "third_party/nablacolors/unimol_plus/unimol_plus/data/lmdb_dataset.py"
    spec = importlib.util.spec_from_file_location("pinned_uniprop_lmdb_dataset", upstream_lmdb)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dataset = module.LMDBDataset(str(cfg.output_dir / "train.lmdb"))
    first = dataset[0]

    assert len(dataset) > 0
    assert "row_id" in first
    assert "molecule_id" in first
    assert "target_mask" in first
    assert np.asarray(first["target"]).shape == (len(TARGETS),)
