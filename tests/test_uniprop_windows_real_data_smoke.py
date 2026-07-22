from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from chemfluor.uniprop.lmdb_export import read_lmdb_records
from chemfluor.uniprop.manifests import audit_split_leakage, build_manifests
from chemfluor.uniprop.windows_real_data_smoke import (
    WINDOWS_REAL_DATA_SMOKE_PROFILE,
    run_windows_real_data_smoke,
    select_real_data_subset,
    source_dataset_sha256,
)
from chemfluor.uniprop.windows_smoke import TINY_3D_SMOKE_MODEL_KIND

pytest.importorskip("lmdb")
pytest.importorskip("torch")

pytestmark = pytest.mark.windows_smoke

TARGETS = [
    "absorption_nm",
    "emission_nm",
    "lifetime_ns",
    "quantum_yield",
    "log_extinction",
    "stokes_shift_nm",
]


def processed_fixture_rows(include_invalid: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        ("CCO", "water", "CCO", "O", 350.0, 450.0, 1.2, 0.10, 4.1, 100.0),
        ("C(C)O", "ethanol", "CCO", "CCO", 351.0, None, None, 0.11, 4.2, None),
        ("OCC", "acetonitrile", "CCO", "CC#N", None, 452.0, 1.3, None, 4.3, None),
        ("CCN", "water", "CCN", "O", 390.0, 500.0, None, 0.20, None, 110.0),
        ("NCC", "ethanol", "CCN", "CCO", 392.0, 502.0, 2.0, None, 4.5, 110.0),
        ("c1ccccc1", "acetonitrile", "c1ccccc1", "CC#N", 360.0, 430.0, 1.1, 0.05, 3.9, 70.0),
        ("c1ccccc1", "dmso", "c1ccccc1", "CS(C)=O", None, 431.0, None, 0.06, None, None),
        ("Oc1ccccc1", "water", "Oc1ccccc1", "O", 370.0, 460.0, 1.5, 0.30, 4.0, 90.0),
        ("Cc1ccccc1", "ethanol", "Cc1ccccc1", "CCO", 380.0, 480.0, None, 0.25, 4.4, 100.0),
        ("CCCl", "dmso", "CCCl", "CS(C)=O", 365.0, 470.0, 1.7, 0.15, None, 105.0),
        ("CC(=O)O", "unknown", "CC(=O)O", None, 310.0, 390.0, None, None, 3.1, 80.0),
        ("C1CCCCC1", "ethanol", "C1CCCCC1", "CCO", 300.0, 350.0, 0.9, 0.02, 2.8, 50.0),
    ]
    if include_invalid:
        rows.append(("bad", "water", "not_a_smiles", "O", 333.0, 444.0, None, 0.2, 3.0, 111.0))
    return pd.DataFrame(
        [
            {
                "chromophore_smiles": chromophore,
                "solvent_original": solvent_original,
                "canonical_chromophore_smiles": canonical_chromophore,
                "canonical_solvent_smiles": canonical_solvent,
                "absorption_nm": absorption,
                "emission_nm": emission,
                "lifetime_ns": lifetime,
                "quantum_yield": quantum_yield,
                "log_extinction": log_extinction,
                "source_dataset": "real_smoke_fixture",
                "stokes_shift_nm": stokes,
            }
            for (
                chromophore,
                solvent_original,
                canonical_chromophore,
                canonical_solvent,
                absorption,
                emission,
                lifetime,
                quantum_yield,
                log_extinction,
                stokes,
            ) in rows
        ]
    )


def write_processed_fixture(tmp_path: Path, *, include_invalid: bool = True) -> Path:
    path = tmp_path / "processed_fluorcast_fixture.csv"
    processed_fixture_rows(include_invalid=include_invalid).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def real_smoke_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, dict[str, Any]]:
    tmp_path = tmp_path_factory.mktemp("uniprop_windows_real_data_smoke")
    dataset = write_processed_fixture(tmp_path, include_invalid=True)
    output_dir = tmp_path / "artifacts" / "real_smoke"
    summary = run_windows_real_data_smoke(
        output_dir,
        dataset=dataset,
        max_molecules=20,
        seed=123,
        workers=1,
        overwrite=True,
    )
    return dataset, output_dir, summary


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_deterministic_subset_selection(tmp_path: Path) -> None:
    dataset = write_processed_fixture(tmp_path, include_invalid=False)
    bundle = build_manifests(dataset, TARGETS)

    first_rows, first_molecules, first_report = select_real_data_subset(
        bundle.row_manifest,
        bundle.molecule_manifest,
        max_molecules=5,
        max_rows=None,
        seed=7,
    )
    second_rows, second_molecules, second_report = select_real_data_subset(
        bundle.row_manifest,
        bundle.molecule_manifest,
        max_molecules=5,
        max_rows=None,
        seed=7,
    )

    assert first_report == second_report
    assert first_rows["row_id"].tolist() == second_rows["row_id"].tolist()
    assert first_molecules["molecule_id"].tolist() == second_molecules["molecule_id"].tolist()
    assert first_report["selection_method"].startswith("molecules ordered by SHA-256")


def test_source_data_hashing(tmp_path: Path) -> None:
    dataset = write_processed_fixture(tmp_path, include_invalid=False)
    resolved, digest = source_dataset_sha256(dataset)

    assert resolved == dataset
    assert digest == hashlib.sha256(dataset.read_bytes()).hexdigest()


def test_real_smoke_summary_declares_tiny_non_real_profile(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, _, summary = real_smoke_run

    assert summary["profile"] == WINDOWS_REAL_DATA_SMOKE_PROFILE
    assert summary["model_kind"] == TINY_3D_SMOKE_MODEL_KIND
    assert summary["real_uniprop_used"] is False
    assert summary["real_checkpoint_loaded"] is False
    assert summary["all_stages_passed"] is True


def test_cache_resume(tmp_path: Path) -> None:
    dataset = write_processed_fixture(tmp_path, include_invalid=True)
    output_dir = tmp_path / "artifacts" / "resume_smoke"
    first = run_windows_real_data_smoke(output_dir, dataset=dataset, max_molecules=20, seed=321, overwrite=True)
    second = run_windows_real_data_smoke(output_dir, dataset=dataset, max_molecules=20, seed=321, resume=True)

    assert first["first_run_writes"] >= first["successful_geometry_count"]
    assert second["first_run_writes"] == 0
    assert second["stages"]["geometry"]["second_run_cache_hits"] == second["successful_geometry_count"]


def test_repeated_molecule_coordinate_identity(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, summary = real_smoke_run
    report = read_json(output_dir / "source_row_reconciliation.json")

    assert summary["source_row_reconciliation"]["row_identity_passed"] is True
    assert report["repeated_molecule_groups"]
    assert report["repeated_molecule_coordinate_identity"] is True
    assert all(len(group["unique_coordinate_hashes"]) == 1 for group in report["repeated_molecule_groups"])


def test_structured_failure_reporting(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, summary = real_smoke_run
    geometry_failures = read_json(output_dir / "geometry_failures.json")
    failed_rows = read_json(output_dir / "failed_rows.json")

    assert summary["failed_geometry_count"] == 1
    assert geometry_failures["failures"][0]["failure_category"] == "invalid_smiles"
    assert failed_rows["failed_rows"][0]["failure_category"] == "invalid_smiles"
    assert (output_dir / "geometry_failures.csv").read_text(encoding="utf-8").startswith("molecule_id,")
    assert (output_dir / "failed_rows.csv").read_text(encoding="utf-8").startswith("row_id,")


def test_target_and_mask_preservation(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, summary = real_smoke_run
    report = read_json(output_dir / "source_row_reconciliation.json")

    assert report["target_values_unchanged"] is True
    assert report["masks_match_missingness"] is True
    assert report["target_value_mismatches"] == 0
    assert report["mask_mismatches"] == 0
    assert summary["source_row_reconciliation"]["target_values_unchanged"] is True


def test_lmdb_row_identity(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, summary = real_smoke_run
    row_manifest = pd.read_csv(output_dir / "manifests" / "row_manifest.csv")
    exported = []
    for partition in ["train", "valid", "test"]:
        exported.extend(record for _, record in read_lmdb_records(output_dir / "lmdb" / f"{partition}.lmdb"))

    assert len(exported) == len(row_manifest)
    assert {record["row_id"] for record in exported} == set(row_manifest["row_id"])
    assert all("source_row_number" in record for record in exported)
    assert summary["source_row_reconciliation"]["identity_holds"] is True


def test_leakage_detection(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, _ = real_smoke_run
    rows = pd.read_csv(output_dir / "manifests" / "row_manifest.csv")
    molecules = pd.read_csv(output_dir / "manifests" / "molecule_manifest.csv")
    splits = pd.read_csv(output_dir / "manifests" / "split_assignments.csv")

    assert pd.read_csv(output_dir / "manifests" / "split_leakage_audit.csv")["passed"].all()
    repeated_molecule = rows["molecule_id"].value_counts().index[0]
    repeated_rows = rows[rows["molecule_id"] == repeated_molecule]["row_id"].tolist()
    if len(repeated_rows) < 2:
        pytest.skip("fixture selection did not retain repeated rows")
    contaminated = splits.copy()
    contaminated.loc[contaminated["row_id"] == repeated_rows[0], "molecule"] = "train"
    contaminated.loc[contaminated["row_id"] == repeated_rows[1], "molecule"] = "test"

    audit = audit_split_leakage(rows, molecules, contaminated)

    assert not bool(audit[audit["split_family"] == "molecule"]["passed"].iloc[0])


def test_prediction_to_source_joining(real_smoke_run: tuple[Path, Path, dict[str, Any]]) -> None:
    _, output_dir, summary = real_smoke_run
    report = read_json(output_dir / "prediction_join_report.json")
    predictions = pd.read_csv(output_dir / "predictions.csv")
    row_manifest = pd.read_csv(output_dir / "manifests" / "row_manifest.csv")

    assert report["join_passed"] is True
    assert report["prediction_rows"] == len(row_manifest)
    assert set(predictions["source_row_number"].astype(int)) == set(row_manifest["source_row_number"].astype(int))
    assert summary["source_row_reconciliation"]["predictions_join_back_to_source"] is True
