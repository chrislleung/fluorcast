"""Contract tests for the paired absorption/emission Stokes workflow."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = PROJECT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_script("build_paired_stokes_dataset")
experiment = load_script("run_paired_spectral_three_way_experiment")
reporter = load_script("render_combined_prediction_report")


def test_stokes_calculations_are_correct() -> None:
    paired, _ = builder.build_paired_dataset(pd.DataFrame({"absorption_nm": [400.], "emission_nm": [500.]}))
    assert paired.loc[0, "stokes_shift_nm"] == pytest.approx(100.)
    assert paired.loc[0, "stokes_shift_cm^-1"] == pytest.approx(1e7 / 400 - 1e7 / 500)


def test_missing_and_nonpositive_wavelengths_are_dropped() -> None:
    rows = pd.DataFrame({"absorption_nm": [400, None, 0, 300], "emission_nm": [500, 500, 500, None]})
    paired, summary = builder.build_paired_dataset(rows)
    assert len(paired) == 1
    assert summary["rows_dropped_for_missing_absorption"] == 1
    assert summary["rows_dropped_for_missing_emission"] == 1
    assert summary["rows_dropped_for_nonpositive_wavelengths"] == 1


def test_negative_stokes_is_flagged_not_removed() -> None:
    paired, summary = builder.build_paired_dataset(pd.DataFrame({"absorption_nm": [500], "emission_nm": [400]}))
    assert len(paired) == 1
    assert not bool(paired.loc[0, "physically_valid_stokes"])
    assert summary["negative_or_zero_stokes_shift_count"] == 1


def split_rows() -> pd.DataFrame:
    return pd.DataFrame({"row_id": range(60), "canonical_chromophore_smiles": [f"m{i // 3}" for i in range(60)], "scaffold": [f"s{i // 6}" for i in range(60)]})


@pytest.mark.parametrize("split_type", ["random", "molecule", "scaffold"])
def test_three_way_splits_are_disjoint(split_type: str) -> None:
    rows = split_rows()
    rows["split"] = experiment.shared.assign_three_way_splits(rows, split_type, (.6, .2, .2), 0)
    assert set(rows["split"]) == set(experiment.SPLITS)
    assert rows.groupby("row_id")["split"].nunique().max() == 1
    if split_type != "random":
        assert not experiment.shared.leakage_report(rows, split_type)["leakage_detected"]


def test_invalid_scaffold_smiles_do_not_crash(tmp_path: Path) -> None:
    rows = pd.DataFrame({"row_id": [1, 2], "canonical_chromophore_smiles": ["c1ccccc1", "not smiles"]})
    prepared, counts = experiment.shared.prepare_split_identifiers(rows, "scaffold", "drop", tmp_path)
    assert len(prepared) == 1
    assert counts["invalid_scaffold_rows"] == 1


def test_final_predictions_join_by_row_id() -> None:
    absorption = pd.DataFrame({"row_id": [1, 2], "true_absorption_nm": [400, 450], "hybrid_prediction": [405, 455]})
    emission = pd.DataFrame({"row_id": [2, 1], "true_emission_nm": [550, 500], "hybrid_prediction": [545, 505]})
    result = experiment.join_final_predictions(absorption, emission).set_index("row_id")
    assert result.loc[1, "true_stokes_shift_nm"] == 100
    assert result.loc[2, "predicted_stokes_shift_nm"] == 90


def test_stokes_metrics_use_only_supplied_final_rows() -> None:
    final = experiment.join_final_predictions(
        pd.DataFrame({"row_id": [1], "true_absorption_nm": [400], "hybrid_prediction": [410]}),
        pd.DataFrame({"row_id": [1], "true_emission_nm": [500], "hybrid_prediction": [510]}),
    )
    metrics = experiment.stokes_metrics(final)
    assert set(metrics["N"]) == {1}
    assert metrics.loc[metrics["unit"] == "nm", "MAE"].iloc[0] == 0


def test_combined_report_warns_for_nonpositive_stokes() -> None:
    report = reporter.combine_reports({"final_absorption_prediction_nm": 500}, {"final_emission_prediction_nm": 490})
    assert not report["physically_valid_stokes"]
    assert any("less than or equal" in warning for warning in report["warnings"])


@pytest.mark.parametrize("script", ["build_paired_stokes_dataset.py", "run_paired_spectral_three_way_experiment.py", "render_combined_prediction_report.py"])
def test_cli_help(script: str) -> None:
    result = subprocess.run([sys.executable, str(PROJECT / "scripts" / script), "--help"], cwd=PROJECT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
