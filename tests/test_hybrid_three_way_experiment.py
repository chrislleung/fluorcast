"""Tests for the leakage-safe three-way hybrid experiment."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "run_hybrid_three_way_experiment.py"
SPEC = importlib.util.spec_from_file_location("hybrid_three_way", SCRIPT)
assert SPEC and SPEC.loader
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def sample_rows(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": range(n),
        "canonical_chromophore_smiles": [f"mol-{index // 5}" for index in range(n)],
        "scaffold": [f"scaffold-{index // 10}" for index in range(n)],
    })


def test_random_split_is_disjoint_and_approximately_matches_fractions() -> None:
    rows = sample_rows()
    rows["split"] = experiment.assign_three_way_splits(rows, "random", (.6, .2, .2), 0)
    assert set(rows["split"]) == set(experiment.SPLITS)
    assert rows.groupby("row_id")["split"].nunique().max() == 1
    assert rows["split"].value_counts().to_dict() == {
        "base_model_train": 60, "hybrid_meta_train": 20, "final_test": 20
    }


def test_molecule_split_has_no_overlap() -> None:
    rows = sample_rows()
    rows["split"] = experiment.assign_three_way_splits(rows, "molecule", (.6, .2, .2), 1)
    report = experiment.leakage_report(rows, "molecule")
    assert not report["leakage_detected"]
    assert all(value == 0 for value in report["overlap_counts"].values())


def test_scaffold_split_has_no_overlap() -> None:
    rows = sample_rows()
    rows["split"] = experiment.assign_three_way_splits(rows, "scaffold", (.6, .2, .2), 2)
    assert not experiment.leakage_report(rows, "scaffold")["leakage_detected"]
    counts = rows["split"].value_counts(normalize=True)
    assert abs(counts["base_model_train"] - .6) <= .1


def test_prediction_table_has_expected_columns() -> None:
    rows = sample_rows(4)
    rows["emission_nm"] = [400., 450., 500., 550.]
    table = experiment.prediction_table(
        rows, {"rf": [401., 451., 501., 551.], "extratrees": [399., 449., 499., 549.]},
        "emission_nm",
    )
    expected = {"rf_emission_nm", "extratrees_emission_nm", "prediction_mean",
                "prediction_std", "prediction_range", "prediction_count"}
    assert expected <= set(table.columns)


def test_help_works() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=PROJECT,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--final-test-fraction" in result.stdout
    assert "--invalid-smiles-policy" in result.stdout


def chemical_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "row_id": range(7),
        "canonical_chromophore_smiles": [
            "c1ccccc1", "c1ccncc1", "C1CCCCC1", "CCO", "CCN", "CC(=O)O",
            "not a smiles",  # deliberately invalid
        ],
    })


def test_safe_scaffold_does_not_raise_for_bad_stereo() -> None:
    # Conflicting directional bonds have triggered RDKit Canon preconditions in
    # affected releases. The result may be repaired or rejected, but never raised.
    result = experiment.safe_murcko_scaffold("C/C=C(/C)(/F)")
    assert result is None or isinstance(result, str)


def test_invalid_scaffold_rows_are_dropped_and_saved(tmp_path: Path) -> None:
    prepared, counts = experiment.prepare_split_identifiers(
        chemical_rows(), "scaffold", "drop", tmp_path
    )
    assert len(prepared) == 6
    assert counts["invalid_scaffold_rows"] == 1
    assert (tmp_path / "invalid_scaffold_rows.csv").exists()
    prepared["split"] = experiment.assign_three_way_splits(
        prepared, "scaffold", (.6, .2, .2), 0
    )
    assert not experiment.leakage_report(prepared, "scaffold")["leakage_detected"]


def test_invalid_scaffold_rows_stay_in_one_group(tmp_path: Path) -> None:
    rows = chemical_rows()
    rows.loc[len(rows)] = [len(rows), "still not smiles"]
    prepared, counts = experiment.prepare_split_identifiers(
        rows, "scaffold", "keep-invalid-group", tmp_path
    )
    invalid = prepared[prepared["scaffold"] == "INVALID_SCAFFOLD"]
    assert len(invalid) == 2
    assert counts["invalid_scaffold_rows"] == 2
    prepared["split"] = experiment.assign_three_way_splits(
        prepared, "scaffold", (.6, .2, .2), 0
    )
    assert prepared.loc[invalid.index, "split"].nunique() == 1


def test_molecule_split_safely_drops_invalid_smiles(tmp_path: Path) -> None:
    prepared, counts = experiment.prepare_split_identifiers(
        chemical_rows(), "molecule", "drop", tmp_path
    )
    assert counts["invalid_molecule_rows"] == 1
    assert (tmp_path / "invalid_molecule_rows.csv").exists()
    assert prepared["canonical_chromophore_smiles"].notna().all()


def test_all_invalid_rows_raise_clear_error(tmp_path: Path) -> None:
    rows = pd.DataFrame({
        "row_id": [0, 1],
        "canonical_chromophore_smiles": ["not smiles", "C1=CC"],
    })
    with pytest.raises(ValueError, match="All rows have invalid"):
        experiment.prepare_split_identifiers(rows, "scaffold", "drop", tmp_path)
