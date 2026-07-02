from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.report import build_hybrid_report, load_prediction_table, render_report_markdown


def test_high_confidence_report(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    pd.DataFrame({
        "model": ["rf", "gnn"],
        "predicted_emission_nm": [500.0, 510.0],
        "predicted_quantum_yield": [0.2, 0.3],
        "overall_confidence_score": [0.9, 0.9],
        "nearest_training_similarity": [0.85, 0.85],
    }).to_csv(path, index=False)
    report = build_hybrid_report(load_prediction_table(path), "C(C)O", "O")
    assert report["final_emission_prediction_nm"] == pytest.approx(505)
    assert report["final_quantum_yield_prediction"] == pytest.approx(0.25)
    assert report["confidence_label"] == "high"
    assert report["canonical_molecule_smiles"] == "CCO"
    assert "505 nm" in render_report_markdown(report)


def test_missing_values_and_confidence_columns() -> None:
    table = pd.DataFrame({"model": ["a", "b"], "predicted_emission_nm": [400, 500]})
    report = build_hybrid_report(table)
    assert report["final_quantum_yield_prediction"] is None
    assert report["confidence_label"] == "low"
    assert "No quantum-yield" in " ".join(report["warnings"])
    assert "Caution" in render_report_markdown(report)
