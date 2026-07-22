from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.manifests import (
    audit_split_leakage,
    build_manifests,
    make_split_assignments,
    split_statistics,
    training_normalization_statistics,
    validate_manifest_reconciliation,
)


TARGETS = ["absorption_nm", "emission_nm", "quantum_yield"]


def fixture_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chromophore_smiles": [
                "C(C)O",
                "CCO",
                "CCN",
                "NCC",
                "c1ccccc1",
                "Cc1ccccc1",
                "c1ccncc1",
                "C1CCCCC1",
                "CCCl",
                "ClCC",
                "C[C@H](O)F",
                "C[C@@H](O)F",
            ],
            "solvent_original": ["water", "ethanol"] * 6,
            "canonical_chromophore_smiles": [
                "CCO",
                "CCO",
                "CCN",
                "CCN",
                "c1ccccc1",
                "Cc1ccccc1",
                "c1ccncc1",
                "C1CCCCC1",
                "CCCl",
                "CCCl",
                "C[C@H](O)F",
                "C[C@@H](O)F",
            ],
            "canonical_solvent_smiles": ["O", "CCO", "O", "CCO", "CC#N", "CS(C)=O"] * 2,
            "source_dataset": ["fixture"] * 12,
            "absorption_nm": [350.0, 351.0, 410.0, pd.NA, 390.0, 420.0, 430.0, 440.0, 450.0, 451.0, 360.0, 361.0],
            "emission_nm": [450.0, pd.NA, 500.0, 501.0, 470.0, 520.0, 530.0, 540.0, 550.0, 551.0, 460.0, 461.0],
            "quantum_yield": [0.1, 0.2, pd.NA, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.3, 0.31],
        }
    )


def write_fixture(tmp_path: Path, rows: pd.DataFrame) -> Path:
    path = tmp_path / "processed.csv"
    rows.to_csv(path, index=False)
    return path


def test_equivalent_smiles_map_to_one_stable_molecule_id(tmp_path: Path) -> None:
    bundle = build_manifests(write_fixture(tmp_path, fixture_rows()), TARGETS)
    ethanol_ids = set(
        bundle.row_manifest.loc[[0, 1], "molecule_id"]
    )
    ethylamine_ids = set(bundle.row_manifest.loc[[2, 3], "molecule_id"])

    assert len(ethanol_ids) == 1
    assert len(ethylamine_ids) == 1
    assert len(bundle.molecule_manifest) == 9


def test_stereochemical_policy_is_deterministic_and_documented(tmp_path: Path) -> None:
    bundle = build_manifests(
        write_fixture(tmp_path, fixture_rows()), TARGETS, compute_nonisomeric=True
    )
    stereo = bundle.molecule_manifest[
        bundle.molecule_manifest["canonical_nonisomeric_smiles"] == "CC(O)F"
    ]

    assert len(stereo) == 2
    assert "isomeric SMILES" in bundle.metadata["stereochemical_policy"]


def test_row_order_does_not_change_ids_or_split_assignments(tmp_path: Path) -> None:
    rows = fixture_rows()
    first = build_manifests(write_fixture(tmp_path, rows), TARGETS)
    shuffled = rows.sample(frac=1, random_state=99).reset_index(drop=True)
    second = build_manifests(write_fixture(tmp_path, shuffled), TARGETS)

    first_splits = make_split_assignments(first.row_manifest, first.molecule_manifest, seed=7)
    second_splits = make_split_assignments(second.row_manifest, second.molecule_manifest, seed=7)

    first_keyed = first.row_manifest.merge(first_splits, on="row_id").sort_values("row_id").reset_index(drop=True)
    second_keyed = second.row_manifest.merge(second_splits, on="row_id").sort_values("row_id").reset_index(drop=True)
    assert first_keyed[["row_id", "molecule_id", "random", "molecule", "scaffold", "solvent", "double_cold_start"]].equals(
        second_keyed[["row_id", "molecule_id", "random", "molecule", "scaffold", "solvent", "double_cold_start"]]
    )


def test_split_regeneration_with_same_seed_is_identical(tmp_path: Path) -> None:
    bundle = build_manifests(write_fixture(tmp_path, fixture_rows()), TARGETS)
    first = make_split_assignments(bundle.row_manifest, bundle.molecule_manifest, seed=123)
    second = make_split_assignments(bundle.row_manifest, bundle.molecule_manifest, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_every_source_row_appears_once_and_counts_reconcile(tmp_path: Path) -> None:
    bundle = build_manifests(write_fixture(tmp_path, fixture_rows()), TARGETS)

    validate_manifest_reconciliation(bundle)
    assert len(bundle.row_manifest) == len(fixture_rows())
    assert bundle.row_manifest["source_row_number"].nunique() == len(fixture_rows())


def test_leakage_tests_fail_when_contaminated_fixtures_are_supplied(tmp_path: Path) -> None:
    bundle = build_manifests(write_fixture(tmp_path, fixture_rows()), TARGETS)
    splits = make_split_assignments(bundle.row_manifest, bundle.molecule_manifest, seed=4)
    molecule_id = bundle.row_manifest["molecule_id"].iloc[0]
    contaminated_rows = bundle.row_manifest[bundle.row_manifest["molecule_id"] == molecule_id]
    splits.loc[splits["row_id"] == contaminated_rows["row_id"].iloc[0], "molecule"] = "train"
    splits.loc[splits["row_id"] == contaminated_rows["row_id"].iloc[-1], "molecule"] = "test"

    audit = audit_split_leakage(bundle.row_manifest, bundle.molecule_manifest, splits)

    molecule_audit = audit[audit["split_family"] == "molecule"].iloc[0]
    assert not bool(molecule_audit["passed"])
    assert molecule_audit["overlapping_molecule_ids"] == 1


def test_missing_target_masks_are_correct_and_values_unchanged(tmp_path: Path) -> None:
    rows = fixture_rows()
    bundle = build_manifests(write_fixture(tmp_path, rows), TARGETS)

    for target in TARGETS:
        assert bundle.row_manifest[target].isna().tolist() == rows[target].isna().tolist()
        pd.testing.assert_series_equal(
            bundle.row_manifest.loc[bundle.row_manifest[target].notna(), target].astype(float),
            rows.loc[rows[target].notna(), target].astype(float),
            check_names=False,
            check_dtype=False,
        )
        assert bundle.row_manifest[f"{target}_available"].tolist() == rows[target].notna().tolist()


def test_all_split_families_pass_leakage_and_emit_reports(tmp_path: Path) -> None:
    bundle = build_manifests(write_fixture(tmp_path, fixture_rows()), TARGETS)
    splits = make_split_assignments(bundle.row_manifest, bundle.molecule_manifest, seed=8)
    audit = audit_split_leakage(bundle.row_manifest, bundle.molecule_manifest, splits)
    stats = split_statistics(bundle.row_manifest, bundle.molecule_manifest, splits, TARGETS)
    normalization = training_normalization_statistics(bundle.row_manifest, splits, TARGETS)

    assert set(splits.columns) == {"row_id", "random", "molecule", "scaffold", "solvent", "double_cold_start"}
    assert audit["passed"].all()
    assert set(audit["split_family"]) == {"random", "molecule", "scaffold", "solvent", "double_cold_start"}
    assert set(stats["split_family"]) == {"random", "molecule", "scaffold", "solvent", "double_cold_start"}
    assert set(normalization["split_family"]) == {"random", "molecule", "scaffold", "solvent", "double_cold_start"}


def test_manifest_requires_authoritative_processed_columns(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"smiles": ["CCO"]}).to_csv(bad, index=False)

    with pytest.raises(ValueError, match="missing required"):
        build_manifests(bad, TARGETS)


def test_uniprop_solvent_overlay_repairs_blank_aliases_without_rewriting_source(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "chromophore_smiles": ["CCO", "CCN", "CCC"],
            "solvent_original": ["EtOH", "CCO", "gas"],
            "canonical_chromophore_smiles": ["CCO", "CCN", "CCC"],
            "canonical_solvent_smiles": [pd.NA, "CCO", pd.NA],
            "source_dataset": ["fixture"] * 3,
            "absorption_nm": [350.0, 360.0, 370.0],
            "emission_nm": [450.0, 460.0, 470.0],
            "quantum_yield": [0.1, 0.2, 0.3],
        }
    )

    bundle = build_manifests(write_fixture(tmp_path, rows), TARGETS)
    manifest = bundle.row_manifest

    assert pd.isna(manifest.loc[0, "source_canonical_solvent_smiles"])
    assert manifest.loc[0, "uniprop_canonical_solvent_smiles"] == "CCO"
    assert manifest.loc[0, "canonical_solvent_smiles"] == "CCO"
    assert manifest.loc[0, "uniprop_solvent_mapping_status"] == "resolved_alias"
    assert manifest.loc[0, "solvent_id"] == manifest.loc[1, "solvent_id"]
    assert pd.isna(manifest.loc[2, "uniprop_canonical_solvent_smiles"])
    assert manifest.loc[2, "environment_type"] == "gas_phase"
    assert bundle.metadata["uniprop_solvent_overlay"]["uniprop_alias_repaired_rows"] == 1
