from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.uncertainty import calibration_residuals, confidence_bin, prediction_interval


def test_confidence_bins() -> None:
    assert [confidence_bin(value) for value in [0.8, 0.6, 0.4, 0.1]] == [
        "high", "medium", "low_medium", "low"
    ]


def test_intervals_contain_prediction_and_widen_at_low_confidence() -> None:
    residuals = calibration_residuals(
        [101, 99, 105, 95, 120, 80, 130, 70],
        [100] * 8,
        [0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.1, 0.1],
    )
    high = prediction_interval(100, residuals, 0.8)
    low = prediction_interval(100, residuals, 0.1)
    assert high[0] <= 100 <= high[1]
    assert low[0] <= 100 <= low[1]
    assert low[1] - low[0] > high[1] - high[0]
