"""Combine absorption, emission, and optional quantum-yield prediction reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--absorption-report-json", required=True, type=Path)
    parser.add_argument("--emission-report-json", required=True, type=Path)
    parser.add_argument("--quantum-yield-report-json", type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    return parser.parse_args(argv)


def _number(report: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = report.get(key)
        if value is not None:
            try:
                result = float(value)
                if math.isfinite(result):
                    return result
            except (TypeError, ValueError):
                pass
    hybrid = report.get("hybrid_ensemble", {})
    value = hybrid.get("prediction") if isinstance(hybrid, dict) else None
    if value is not None and math.isfinite(float(value)):
        return float(value)
    raise ValueError(f"Report has no finite prediction in {keys}")


def combine_reports(absorption: dict, emission: dict, quantum_yield: dict | None = None) -> dict:
    absorption_nm = _number(absorption, ("final_absorption_prediction_nm", "predicted_absorption_nm", "absorption_nm"))
    emission_nm = _number(emission, ("final_emission_prediction_nm", "predicted_emission_nm", "emission_nm"))
    result = {
        "predicted_absorption_nm": absorption_nm,
        "predicted_emission_nm": emission_nm,
        "predicted_stokes_shift_nm": emission_nm - absorption_nm,
        "predicted_stokes_shift_cm^-1": 1e7 / absorption_nm - 1e7 / emission_nm,
        "physically_valid_stokes": emission_nm > absorption_nm,
        "absorption_confidence_label": absorption.get("confidence_label"),
        "emission_confidence_label": emission.get("confidence_label"),
        "outside_applicability_domain": bool(absorption.get("outside_applicability_domain", False) or emission.get("outside_applicability_domain", False)),
        "warnings": [],
    }
    if not result["physically_valid_stokes"]:
        result["warnings"].append("Predicted emission is less than or equal to predicted absorption; the calculated Stokes shift is nonpositive.")
    if result["outside_applicability_domain"]:
        result["warnings"].append("At least one spectral prediction is outside its applicability domain.")
    if quantum_yield is not None:
        result["predicted_quantum_yield"] = _number(quantum_yield, ("final_quantum_yield_prediction", "predicted_quantum_yield", "quantum_yield"))
        result["quantum_yield_confidence_label"] = quantum_yield.get("confidence_label")
        result["brightness_class"] = quantum_yield.get("brightness_class") or quantum_yield.get("predicted_brightness_class")
    return result


def render_markdown(report: dict) -> str:
    lines = ["# Combined Spectral Prediction", "", f"- Predicted absorption maximum: {report['predicted_absorption_nm']:.2f} nm", f"- Predicted emission maximum: {report['predicted_emission_nm']:.2f} nm", f"- Predicted Stokes shift: {report['predicted_stokes_shift_nm']:.2f} nm", f"- Predicted Stokes shift: {report['predicted_stokes_shift_cm^-1']:.2f} cm^-1"]
    for label, key in (("Absorption confidence", "absorption_confidence_label"), ("Emission confidence", "emission_confidence_label")):
        if report.get(key):
            lines.append(f"- {label}: {report[key]}")
    if "predicted_quantum_yield" in report:
        lines.append(f"- Predicted quantum yield: {report['predicted_quantum_yield']:.4f}")
        if report.get("brightness_class"):
            lines.append(f"- Brightness class: {report['brightness_class']}")
    if report["warnings"]:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in report["warnings"]]])
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load = lambda path: json.loads(path.read_text(encoding="utf-8"))
    report = combine_reports(load(args.absorption_report_json), load(args.emission_report_json), load(args.quantum_yield_report_json) if args.quantum_yield_report_json else None)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.out_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
