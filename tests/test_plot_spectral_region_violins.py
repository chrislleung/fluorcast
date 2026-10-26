from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_spectral_region_violins_for_tests",
    PROJECT_ROOT / "scripts" / "plot_spectral_region_violins.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_five_region_boundaries_are_exact() -> None:
    definition = MODULE.region_definition("five")
    values = [399.999, 400, 479.999, 480, 559.999, 560, 620, 620.001]
    assert [MODULE.assign_spectral_region(value, definition) for value in values] == [
        "UV", "Blue", "Blue", "Green", "Green", "Yellow/Orange", "Yellow/Orange", "Red/NIR",
    ]


def test_three_region_boundaries_are_exact() -> None:
    definition = MODULE.region_definition("three")
    assert [MODULE.assign_spectral_region(value, definition) for value in (399.9, 400, 699.9, 700)] == [
        "UV", "Visible", "Visible", "Near-infrared",
    ]


def test_errors_summary_counts_mae_and_rmse() -> None:
    frame = pd.DataFrame({
        "y_true": [400.0, 410.0], "y_pred": [390.0, 430.0],
        "target": ["absorption_nm"] * 2, "model": ["ridge"] * 2,
        "representation": ["Morgan fingerprint"] * 2, "split_type": ["molecule"] * 2,
        "seed": [0] * 2,
    })
    errors = MODULE.calculate_errors(frame, MODULE.region_definition("five"))
    summary = MODULE.summarize_errors(errors, MODULE.region_definition("five")).iloc[0]
    assert errors["absolute_error"].tolist() == [10.0, 20.0]
    assert summary["n"] == 2
    assert summary["mae"] == pytest.approx(15.0)
    assert summary["rmse"] == pytest.approx(np.sqrt(250.0))


def test_invalid_wavelength_values_are_removed() -> None:
    frame = pd.DataFrame({"y_true": [500, None, "bad", np.inf], "y_pred": [510, 3, 4, 5]})
    result = MODULE.calculate_errors(frame, MODULE.region_definition("five"))
    assert len(result) == 1
    assert result.iloc[0]["spectral_region"] == "Green"


def test_conforformer_schema_uses_only_final_test_and_selected_estimator(tmp_path: Path) -> None:
    prediction_dir = tmp_path / "experiment" / "predictions"
    prediction_dir.mkdir(parents=True)
    path = prediction_dir / "absorption_nm__mean__conforformer_morgan_solvent.csv"
    pd.DataFrame({
        "row_id": ["a", "b", "c"], "split": ["final_test", "base_train", "model_selection"],
        "target": ["absorption_nm"] * 3, "y_true": [400, 500, 600], "y_pred": [410, 510, 610],
    }).to_csv(path, index=False)
    pd.DataFrame([{
        "target": "absorption_nm", "pooling_method": "mean",
        "feature_set": "conforformer_morgan_solvent", "model": "ridge_alpha_1",
        "split": "final_test", "mae": 10.0,
    }]).to_csv(tmp_path / "experiment" / "metrics.csv", index=False)
    (tmp_path / "experiment" / "run_manifest.json").write_text(
        '{"split_type": "molecule", "split_seed": 7}', encoding="utf-8"
    )
    loaded, notes = MODULE.load_prediction_file(path, MODULE.region_definition("five"))
    assert loaded["row_id"].tolist() == ["a"]
    assert loaded.iloc[0]["model"] == "ridge_alpha_1"
    assert loaded.iloc[0]["representation"] == "Morgan + ConforFormer (mean pooling)"
    assert loaded.iloc[0]["split_type"] == "molecule"
    assert loaded.iloc[0]["seed"] == 7
    assert notes


def test_file_without_split_provenance_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "arbitrary.csv"
    pd.DataFrame({"target": ["emission_nm"], "y_true": [500], "y_pred": [510]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="split provenance"):
        MODULE.load_prediction_file(path, MODULE.region_definition("five"))


def test_legacy_predictions_schema_is_accepted_as_test_output(tmp_path: Path) -> None:
    path = tmp_path / "predictions_emission_nm.csv"
    pd.DataFrame({"y_true": [500, 650], "y_pred": [510, 620]}).to_csv(path, index=False)
    loaded, _ = MODULE.load_prediction_file(
        path, MODULE.region_definition("five"), fallback_model="rf",
        fallback_representation="morgan_solvent", fallback_split_type="molecule", fallback_seed=42,
    )
    assert loaded["absolute_error"].tolist() == [10, 30]
    assert loaded["spectral_region"].tolist() == ["Green", "Red/NIR"]


def test_headless_plot_writes_high_resolution_png_and_pdf(tmp_path: Path) -> None:
    rows = []
    for region_base in (360, 420, 510, 590, 660):
        for offset in range(6):
            rows.append({"y_true": region_base + offset, "y_pred": region_base + offset + (-1) ** offset * (5 + offset)})
    path = tmp_path / "predictions_absorption_nm.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    args = MODULE.parse_args([
        "--input", str(path), "--out-dir", str(tmp_path / "plots"),
        "--model", "ridge", "--representation", "Morgan fingerprint",
        "--split-type", "molecule", "--seed", "42", "--dpi", "300",
    ])
    generated = MODULE.run(args)
    assert any(item.suffix == ".png" for item in generated)
    assert any(item.suffix == ".pdf" for item in generated)
    assert (tmp_path / "plots" / "spectral_region_error_summary.csv").exists()
