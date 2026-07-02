"""Rule-based, chemistry-conservative explanations of prediction diagnostics."""

from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float | None:
    """Return a finite float, or None for unavailable values."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def confidence_label(score: float | None) -> str:
    """Map a normalized confidence score to a stable qualitative label."""
    value = _number(score)
    if value is None:
        return "low"
    if value >= 0.80:
        return "high"
    if value >= 0.60:
        return "medium"
    if value >= 0.40:
        return "low-medium"
    return "low"


def collect_confidence_reasons(report: dict) -> list[str]:
    """Collect diagnostic-only reasons that warrant caution."""
    reasons: list[str] = []
    similarity = _number(report.get("nearest_training_similarity"))
    if similarity is not None and similarity < 0.5:
        reasons.append("Low nearest-training similarity (below 0.5).")

    emission_std = _number(report.get("emission_model_standard_deviation"))
    emission_range = _number(report.get("emission_prediction_range"))
    if (emission_std is not None and emission_std > 25) or (
        emission_range is not None and emission_range > 50
    ):
        reasons.append("Emission models disagree substantially.")

    pair_seen = report.get("pair_seen_score")
    if pair_seen is False or _number(pair_seen) == 0:
        reasons.append("The molecule-solvent pair was not seen in training.")
    if bool(report.get("outside_applicability_domain", False)):
        reasons.append("The prediction is outside the applicability domain.")
    return reasons


def summarize_prediction(report: dict) -> str:
    """Generate a concise explanation without mechanistic inference."""
    emission = _number(report.get("final_emission_prediction_nm"))
    quantum_yield = _number(report.get("final_quantum_yield_prediction"))
    label = str(report.get("confidence_label") or "low")

    parts = [
        "Predicted emission is unavailable."
        if emission is None
        else f"Predicted emission is {emission:.1f} nm."
    ]
    if quantum_yield is not None:
        parts.append(f"Predicted quantum yield is {quantum_yield:.3g}.")
    parts.append(f"Diagnostic confidence is {label}.")
    reasons = collect_confidence_reasons(report)
    if reasons:
        parts.append("Caution: " + " ".join(reasons))
    return " ".join(parts)
