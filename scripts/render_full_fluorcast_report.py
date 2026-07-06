"""Render a deterministic, chemist-facing FluorCast prediction report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PREDICTION_KEYS = {
    "absorption": ("final_absorption_prediction_nm", "predicted_absorption_nm", "absorption_nm"),
    "emission": ("final_emission_prediction_nm", "predicted_emission_nm", "emission_nm"),
    "quantum_yield": ("final_quantum_yield_prediction", "predicted_quantum_yield", "quantum_yield"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absorption-report-json", required=True, type=Path)
    parser.add_argument("--emission-report-json", required=True, type=Path)
    parser.add_argument("--quantum-yield-report-json", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prediction(report: dict[str, Any], target: str) -> float:
    final_prediction = report.get("final_prediction")
    if isinstance(final_prediction, dict):
        nested_key = "quantum_yield" if target == "quantum_yield" else f"{target}_nm"
        number = _finite(final_prediction.get(nested_key))
        if number is not None:
            return number
    for key in PREDICTION_KEYS[target]:
        number = _finite(report.get(key))
        if number is not None:
            return number
    hybrid = report.get("hybrid_ensemble")
    if isinstance(hybrid, dict):
        number = _finite(hybrid.get("prediction"))
        if number is not None:
            return number
    raise ValueError(f"{target.replace('_', ' ').title()} report has no finite prediction")


def _interval(report: dict[str, Any]) -> Any:
    confidence = report.get("confidence")
    if isinstance(confidence, dict):
        for key in ("prediction_interval", "uncertainty_interval", "confidence_interval"):
            if confidence.get(key) is not None:
                return confidence[key]
    for key in ("prediction_interval", "uncertainty_interval", "confidence_interval"):
        if report.get(key) is not None:
            return report[key]
    hybrid = report.get("hybrid_ensemble")
    if isinstance(hybrid, dict):
        for key in ("prediction_interval", "uncertainty_interval", "confidence_interval"):
            if hybrid.get(key) is not None:
                return hybrid[key]
    return None


def _disagreement(report: dict[str, Any], target: str) -> dict[str, Any] | Any | None:
    diagnostics = report.get("diagnostics")
    if isinstance(diagnostics, dict):
        for key in ("model_disagreement", f"{target}_model_disagreement"):
            if diagnostics.get(key) is not None:
                return diagnostics[key]
    for key in ("model_disagreement", f"{target}_model_disagreement"):
        if report.get(key) is not None:
            return report[key]
    fields = {
        key: value
        for key, value in report.items()
        if value is not None
        and ("disagreement" in key or "model_standard_deviation" in key or "prediction_range" in key)
    }
    return fields or None


def _add_diagnostics(result: dict[str, Any], report: dict[str, Any], target: str) -> None:
    confidence_block = report.get("confidence")
    confidence = report.get("confidence_label")
    if confidence is None and isinstance(confidence_block, dict):
        confidence = confidence_block.get("label") or confidence_block.get("confidence_label")
    if confidence is not None:
        result[f"{target}_confidence_label"] = confidence
    interval = _interval(report)
    if interval is not None:
        result[f"{target}_uncertainty_interval"] = interval
    diagnostics = report.get("diagnostics")
    outside_value = report.get("outside_applicability_domain", False)
    if isinstance(diagnostics, dict):
        outside_value = diagnostics.get("outside_applicability_domain", outside_value)
    outside = bool(outside_value)
    result[f"{target}_outside_applicability_domain"] = outside
    disagreement = _disagreement(report, target)
    if disagreement is not None:
        result[f"{target}_model_disagreement"] = disagreement


def build_report(
    absorption: dict[str, Any],
    emission: dict[str, Any],
    quantum_yield: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine model reports without treating their predictions as observations."""
    absorption_nm = _prediction(absorption, "absorption")
    emission_nm = _prediction(emission, "emission")
    if absorption_nm <= 0 or emission_nm <= 0:
        raise ValueError("Predicted absorption and emission wavelengths must be positive")

    report: dict[str, Any] = {
        "report_type": "FluorCast prediction report",
        "values_are_predictions": True,
        "predicted_absorption_nm": absorption_nm,
        "predicted_emission_nm": emission_nm,
        "stokes_shift_nm": emission_nm - absorption_nm,
        "stokes_shift_cm^-1": 1e7 / absorption_nm - 1e7 / emission_nm,
        "physically_valid_stokes": emission_nm > absorption_nm,
        "stokes_shift_method": "calculated from predicted emission and absorption; not directly modeled",
        "warnings": [],
    }
    _add_diagnostics(report, absorption, "absorption")
    _add_diagnostics(report, emission, "emission")

    if quantum_yield is not None:
        qy = _prediction(quantum_yield, "quantum_yield")
        report["predicted_quantum_yield"] = qy
        report["brightness_class"] = "dim" if qy <= 0.25 else "bright"
        _add_diagnostics(report, quantum_yield, "quantum_yield")

    outside_targets = [
        target.replace("_", " ")
        for target in ("absorption", "emission", "quantum_yield")
        if report.get(f"{target}_outside_applicability_domain")
    ]
    report["outside_applicability_domain"] = bool(outside_targets)
    if outside_targets:
        report["warnings"].append(
            "Outside applicability domain: " + ", ".join(outside_targets) + "."
        )
    if not report["physically_valid_stokes"]:
        report["warnings"].append(
            "Predicted emission is less than or equal to predicted absorption; "
            "the calculated Stokes shift is nonpositive."
        )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# FluorCast full prediction report",
        "",
        "> All numerical values below are model predictions, not experimental facts.",
        "",
        "## Predicted properties",
        "",
        f"- Predicted absorption maximum: {report['predicted_absorption_nm']:.2f} nm",
        f"- Predicted emission maximum: {report['predicted_emission_nm']:.2f} nm",
        f"- Predicted Stokes shift: {report['stokes_shift_nm']:.2f} nm",
        f"- Predicted Stokes shift: {report['stokes_shift_cm^-1']:.2f} cm^-1",
    ]
    if "predicted_quantum_yield" in report:
        lines.extend(
            [
                f"- Predicted quantum yield: {report['predicted_quantum_yield']:.4f}",
                f"- Brightness class: {report['brightness_class']}",
            ]
        )

    labels = [
        f"{target.replace('_', ' ').title()}: {report[f'{target}_confidence_label']}"
        for target in ("absorption", "emission", "quantum_yield")
        if report.get(f"{target}_confidence_label") is not None
    ]
    lines.extend(["", "## Confidence summary", ""])
    lines.append("- " + "; ".join(labels) if labels else "- Confidence labels were not available.")

    lines.extend(["", "## Applicability and cautions", ""])
    if report["warnings"]:
        lines.extend(f"- {warning}" for warning in report["warnings"])
    else:
        lines.append("- No applicability-domain warnings were reported by the input reports.")
    lines.extend(
        [
            "",
            "The Stokes shift is calculated from the predicted absorption and emission maxima; "
            "it is not directly modeled.",
        ]
    )
    return "\n".join(lines) + "\n"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(
        _load(args.absorption_report_json),
        _load(args.emission_report_json),
        _load(args.quantum_yield_report_json) if args.quantum_yield_report_json else None,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
