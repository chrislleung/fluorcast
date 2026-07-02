"""Build JSON-safe hybrid reports from all-model prediction tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .explanation import confidence_label, summarize_prediction

_DIAGNOSTICS = (
    "nearest_training_similarity",
    "molecule_seen_score",
    "solvent_seen_score",
    "pair_seen_score",
    "model_agreement_score",
    "overall_confidence_score",
)


def load_prediction_table(path: str | Path) -> pd.DataFrame:
    """Load a prediction CSV, raising a useful error for an empty file."""
    source = Path(path)
    try:
        return pd.read_csv(source)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Prediction CSV is empty: {source}") from exc


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _values(table: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in table:
            return pd.to_numeric(table[column], errors="coerce").dropna()
    return pd.Series(dtype=float)


def _first(table: pd.DataFrame, column: str) -> Any:
    if column not in table:
        return None
    values = table[column].dropna()
    if values.empty:
        return None
    value = values.iloc[0]
    if isinstance(value, (bool, str)):
        return value
    return _finite(value)


def _canonicalize(smiles: str | None) -> str | None:
    if not smiles:
        return None
    try:
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(molecule) if molecule is not None else smiles
    except ImportError:
        return smiles


def _json_value(value: Any) -> Any:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def build_hybrid_report(
    predictions: pd.DataFrame,
    molecule_smiles: str | None = None,
    solvent_smiles: str | None = None,
) -> dict:
    """Aggregate available model outputs and diagnostics deterministically."""
    emission = _values(predictions, "predicted_emission_nm", "emission_nm")
    qy = _values(predictions, "predicted_quantum_yield", "quantum_yield")
    if molecule_smiles is None:
        for column in ("canonical_molecule_smiles", "molecule_smiles", "smiles"):
            candidate = _first(predictions, column)
            if candidate:
                molecule_smiles = str(candidate)
                break
    if solvent_smiles is None:
        for column in ("canonical_solvent_smiles", "solvent_smiles"):
            candidate = _first(predictions, column)
            if candidate:
                solvent_smiles = str(candidate)
                break

    report: dict[str, Any] = {
        "canonical_molecule_smiles": _canonicalize(molecule_smiles),
        "canonical_solvent_smiles": _canonicalize(solvent_smiles),
        "model_predictions": [
            {str(key): _json_value(value) for key, value in row.items()}
            for row in predictions.to_dict(orient="records")
        ],
        "final_emission_prediction_nm": None if emission.empty else float(emission.mean()),
        "final_quantum_yield_prediction": None if qy.empty else float(qy.mean()),
        "emission_model_standard_deviation": (
            None if emission.empty else float(emission.std(ddof=0))
        ),
        "emission_prediction_range": (
            None if emission.empty else float(emission.max() - emission.min())
        ),
        "quantum_yield_model_standard_deviation": (
            None if qy.empty else float(qy.std(ddof=0))
        ),
    }
    for diagnostic in _DIAGNOSTICS:
        value = _first(predictions, diagnostic)
        if value is not None:
            report[diagnostic] = value

    if "outside_applicability_domain" in predictions:
        outside_values = predictions["outside_applicability_domain"].dropna()
        outside = any(
            value is True
            or str(value).strip().lower() in {"true", "1", "yes"}
            for value in outside_values
        )
    else:
        outside = False
    report["outside_applicability_domain"] = outside

    score = _finite(report.get("overall_confidence_score"))
    if score is not None:
        label = confidence_label(score)
    else:
        supplied = _first(predictions, "confidence_label")
        label = str(supplied).lower() if supplied in {"high", "medium", "low-medium", "low"} else "low"
    report["confidence_label"] = label
    report["warnings"] = []
    if emission.empty:
        report["warnings"].append("No emission predictions were available.")
    if qy.empty:
        report["warnings"].append("No quantum-yield predictions were available.")
    from .explanation import collect_confidence_reasons

    report["warnings"].extend(collect_confidence_reasons(report))
    report["chemist_summary"] = summarize_prediction(report)
    return report


def write_report_json(report: dict, path: str | Path) -> None:
    """Write a report as standards-compliant, readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _format(value: Any, suffix: str = "") -> str:
    number = _finite(value)
    return "unavailable" if number is None else f"{number:.4g}{suffix}"


def render_report_markdown(report: dict) -> str:
    """Render the structured report as a compact Markdown document."""
    lines = [
        "# FluorCast hybrid prediction report",
        "",
        f"- Molecule: `{report.get('canonical_molecule_smiles') or 'not provided'}`",
        f"- Solvent: `{report.get('canonical_solvent_smiles') or 'not provided'}`",
        f"- Final emission prediction: {_format(report.get('final_emission_prediction_nm'), ' nm')}",
        f"- Final quantum yield prediction: {_format(report.get('final_quantum_yield_prediction'))}",
        f"- Confidence: {report.get('confidence_label', 'low')}",
        "",
        "## Diagnostics",
        "",
        f"- Emission model standard deviation: {_format(report.get('emission_model_standard_deviation'), ' nm')}",
        f"- Emission prediction range: {_format(report.get('emission_prediction_range'), ' nm')}",
        f"- Quantum-yield model standard deviation: {_format(report.get('quantum_yield_model_standard_deviation'))}",
        f"- Outside applicability domain: {str(bool(report.get('outside_applicability_domain'))).lower()}",
        "",
        "## Chemist summary",
        "",
        str(report.get("chemist_summary", "")),
    ]
    hybrid = report.get("hybrid_ensemble")
    if isinstance(hybrid, dict):
        interval = hybrid.get("prediction_interval") or {}
        coverage = _finite(interval.get("coverage"))
        coverage_text = "" if coverage is None else f" ({coverage:.0%} coverage)"
        lines.extend(
            [
                "",
                "## Trained hybrid ensemble",
                "",
                f"- Target: {hybrid.get('target', 'unavailable')}",
                f"- Calibrated prediction: {_format(hybrid.get('prediction'))}",
                "- Prediction interval"
                f"{coverage_text}: {_format(interval.get('lower'))} to {_format(interval.get('upper'))}",
            ]
        )
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Cautions", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"
