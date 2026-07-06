from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.predict_full_fluorcast import run_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "predict_full_fluorcast.py"


def args(tmp_path: Path, *, skip_hybrid: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        smiles="CCO",
        solvent_smiles="O",
        out_dir=tmp_path,
        tree_model_dir=tmp_path / "tree",
        neural_model_dir=tmp_path / "neural",
        graph_model_dirs=[],
        absorption_hybrid_model_dir=None,
        emission_hybrid_model_dir=None,
        quantum_yield_hybrid_model_dir=None,
        skip_hybrid=skip_hybrid,
        known_absorption_nm=None,
        known_emission_nm=None,
        known_quantum_yield=None,
    )


def collector(table: pd.DataFrame):
    def collect(_):
        return table, [], "CCO", "O", "O"
    return collect


def predictions(include_absorption=True, include_emission=True, include_qy=True) -> pd.DataFrame:
    rows = {"model": ["a", "b"], "model_family": ["tree", "neural"]}
    if include_absorption:
        rows["predicted_absorption_nm"] = [400.0, 420.0]
    if include_emission:
        rows["predicted_emission_nm"] = [500.0, 520.0]
    if include_qy:
        rows["predicted_quantum_yield"] = [0.2, 0.4]
    rows["confidence_label"] = ["high", "medium"]
    rows["outside_applicability_domain"] = [False, False]
    return pd.DataFrame(rows)


def test_creates_all_expected_files_and_full_report(tmp_path: Path) -> None:
    full = run_workflow(args(tmp_path), collector(predictions()))
    expected = {
        "base_predictions.csv",
        "absorption_predictions.csv", "emission_predictions.csv", "quantum_yield_predictions.csv",
        "absorption_report.json", "absorption_report.md",
        "emission_report.json", "emission_report.md",
        "quantum_yield_report.json", "quantum_yield_report.md",
        "full_fluorcast_report.json", "full_fluorcast_report.md",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert full["predicted_absorption_nm"] == pytest.approx(410)
    assert full["predicted_emission_nm"] == pytest.approx(510)
    assert full["stokes_shift_nm"] == pytest.approx(100)
    assert full["predicted_quantum_yield"] == pytest.approx(0.3)
    assert full["brightness_class"] == "bright"


def test_works_without_qy_report(tmp_path: Path) -> None:
    full = run_workflow(args(tmp_path), collector(predictions(include_qy=False)))
    assert "predicted_quantum_yield" not in full
    assert (tmp_path / "quantum_yield_predictions.csv").exists()
    assert not (tmp_path / "quantum_yield_report.json").exists()


@pytest.mark.parametrize("missing", ["absorption", "emission"])
def test_required_spectral_prediction_fails_clearly(tmp_path: Path, missing: str) -> None:
    table = predictions(
        include_absorption=missing != "absorption",
        include_emission=missing != "emission",
    )
    with pytest.raises(ValueError, match=f"no finite {missing}_nm prediction"):
        run_workflow(args(tmp_path), collector(table))


def test_skip_hybrid_uses_base_prediction_mean(tmp_path: Path) -> None:
    settings = args(tmp_path, skip_hybrid=True)
    settings.absorption_hybrid_model_dir = tmp_path / "unused-hybrid"
    full = run_workflow(settings, collector(predictions()))
    target = json.loads((tmp_path / "absorption_report.json").read_text(encoding="utf-8"))
    assert target["prediction_method"] == "base_prediction_mean"
    assert full["predicted_absorption_nm"] == pytest.approx(410)


def test_cli_help() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "--solvent-smiles" in completed.stdout
