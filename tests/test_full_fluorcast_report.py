import json
from pathlib import Path

import pytest

from scripts.render_full_fluorcast_report import build_report, main, render_markdown


def test_stokes_calculations_and_diagnostics() -> None:
    report = build_report(
        {"final_absorption_prediction_nm": 400, "confidence_label": "high"},
        {
            "final_emission_prediction_nm": 500,
            "hybrid_ensemble": {"prediction_interval": {"lower": 480, "upper": 520}},
            "emission_model_standard_deviation": 12,
        },
    )
    assert report["stokes_shift_nm"] == pytest.approx(100)
    assert report["stokes_shift_cm^-1"] == pytest.approx(1e7 / 400 - 1e7 / 500)
    assert report["emission_uncertainty_interval"] == {"lower": 480, "upper": 520}
    assert report["emission_model_disagreement"]["emission_model_standard_deviation"] == 12


def test_nonpositive_stokes_warning() -> None:
    report = build_report({"predicted_absorption_nm": 500}, {"predicted_emission_nm": 490})
    assert report["physically_valid_stokes"] is False
    assert "less than or equal" in " ".join(report["warnings"])
    assert "less than or equal" in render_markdown(report)


@pytest.mark.parametrize(("qy", "expected"), [(0.25, "dim"), (0.25001, "bright")])
def test_qy_brightness_classification(qy: float, expected: str) -> None:
    report = build_report(
        {"predicted_absorption_nm": 400},
        {"predicted_emission_nm": 500},
        {"predicted_quantum_yield": qy},
    )
    assert report["brightness_class"] == expected


def test_missing_qy_report_still_works() -> None:
    report = build_report({"absorption_nm": 400}, {"emission_nm": 500})
    assert "predicted_quantum_yield" not in report
    assert "Brightness class" not in render_markdown(report)


def test_cli_creates_json_and_markdown(tmp_path: Path) -> None:
    absorption = tmp_path / "absorption.json"
    emission = tmp_path / "emission.json"
    out_json = tmp_path / "nested" / "report.json"
    out_md = tmp_path / "nested" / "report.md"
    absorption.write_text(json.dumps({"predicted_absorption_nm": 410}), encoding="utf-8")
    emission.write_text(json.dumps({"predicted_emission_nm": 510}), encoding="utf-8")

    assert main([
        "--absorption-report-json", str(absorption),
        "--emission-report-json", str(emission),
        "--out-json", str(out_json),
        "--out-md", str(out_md),
    ]) == 0
    assert out_json.is_file()
    assert out_md.is_file()
    assert json.loads(out_json.read_text(encoding="utf-8"))["stokes_shift_nm"] == 100
    assert "not experimental facts" in out_md.read_text(encoding="utf-8")
