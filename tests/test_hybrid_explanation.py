from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.explanation import collect_confidence_reasons, confidence_label, summarize_prediction


def test_confidence_labels() -> None:
    assert [confidence_label(value) for value in (0.9, 0.7, 0.5, 0.2, None)] == [
        "high", "medium", "low-medium", "low", "low"
    ]


def test_low_confidence_summary_names_diagnostics() -> None:
    report = {
        "final_emission_prediction_nm": 450,
        "confidence_label": "low",
        "nearest_training_similarity": 0.3,
        "emission_model_standard_deviation": 40,
        "emission_prediction_range": 100,
        "pair_seen_score": 0,
        "outside_applicability_domain": True,
    }
    reasons = collect_confidence_reasons(report)
    summary = summarize_prediction(report)
    assert len(reasons) == 4
    assert "450.0 nm" in summary
    assert "models disagree" in summary
    assert "not seen in training" in summary
    assert "outside the applicability domain" in summary


def test_quantum_yield_omitted_when_missing() -> None:
    summary = summarize_prediction({"final_emission_prediction_nm": 500, "confidence_label": "medium"})
    assert "quantum yield" not in summary.lower()
