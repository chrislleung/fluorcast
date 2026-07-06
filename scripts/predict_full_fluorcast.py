"""Run the end-to-end FluorCast prediction and reporting workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemfluor.hybrid.ensemble import (  # noqa: E402
    align_features,
    load_hybrid_ensemble,
    predict_hybrid_ensemble,
)
from chemfluor.hybrid.meta_features import add_wide_feature_aliases, build_meta_features  # noqa: E402
from chemfluor.hybrid.uncertainty import prediction_interval  # noqa: E402
from scripts import predict_all_models  # noqa: E402
from scripts.render_full_fluorcast_report import build_report, render_markdown  # noqa: E402


TARGETS = ("absorption_nm", "emission_nm", "quantum_yield")
PREDICTION_COLUMNS = {target: f"predicted_{target}" for target in TARGETS}
REPORT_STEMS = {
    "absorption_nm": "absorption",
    "emission_nm": "emission",
    "quantum_yield": "quantum_yield",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--solvent-smiles", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tree-model-dir", type=Path, default=predict_all_models.DEFAULT_TREE_MODEL_DIR)
    parser.add_argument("--neural-model-dir", type=Path, default=predict_all_models.DEFAULT_NEURAL_MODEL_DIR)
    parser.add_argument("--graph-model-dirs", nargs="*", type=Path)
    parser.add_argument("--absorption-hybrid-model-dir", type=Path)
    parser.add_argument("--emission-hybrid-model-dir", type=Path)
    parser.add_argument("--quantum-yield-hybrid-model-dir", type=Path)
    parser.add_argument("--skip-hybrid", action="store_true")
    parser.add_argument("--known-absorption-nm", type=float)
    parser.add_argument("--known-emission-nm", type=float)
    parser.add_argument("--known-quantum-yield", type=float)
    return parser.parse_args(argv)


def _finite_values(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table:
        return pd.Series(dtype=float)
    values = pd.to_numeric(table[column], errors="coerce")
    return values[values.map(lambda value: math.isfinite(value) if pd.notna(value) else False)]


def _confidence(table: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "confidence_label" in table:
        labels = table["confidence_label"].dropna().astype(str)
        if not labels.empty:
            # Lowest confidence wins, making the summary conservative and deterministic.
            order = {"low": 0, "low-medium": 1, "medium": 2, "high": 3}
            result["label"] = min(labels, key=lambda label: order.get(label.lower(), -1))
    if "overall_confidence_score" in table:
        scores = _finite_values(table, "overall_confidence_score")
        if not scores.empty:
            result["overall_confidence_score"] = float(scores.mean())
    return result


def _diagnostics(table: pd.DataFrame, values: pd.Series) -> dict[str, Any]:
    disagreement = {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
        "count": int(values.count()),
    }
    outside = False
    if "outside_applicability_domain" in table:
        outside = table["outside_applicability_domain"].map(
            lambda value: value is True or str(value).strip().lower() in {"true", "1", "yes"}
        ).any()
    result: dict[str, Any] = {
        "model_disagreement": disagreement,
        "outside_applicability_domain": bool(outside),
    }
    if "nearest_training_similarity" in table:
        similarities = _finite_values(table, "nearest_training_similarity")
        if not similarities.empty:
            result["nearest_training_similarity"] = float(similarities.max())
    return result


def _apply_hybrid(
    base_predictions: pd.DataFrame,
    target: str,
    model_dir: Path,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    model, columns, metadata = load_hybrid_ensemble(model_dir)
    model_target = str(model.get("target_name"))
    if model_target != target:
        raise ValueError(f"Hybrid model target is {model_target!r}, expected {target!r}: {model_dir}")
    raw = add_wide_feature_aliases(build_meta_features(base_predictions), target)
    features = align_features(raw, columns)
    hybrid = predict_hybrid_ensemble(model, features)
    confidence: dict[str, Any] = {}
    score = raw.get("overall_confidence_score")
    confidence_value = None if score is None or pd.isna(score.iloc[0]) else float(score.iloc[0])
    residual_path = model_dir / "calibration_residuals.csv"
    if residual_path.exists():
        coverage = float(metadata.get("conformal_coverage", 0.90))
        lower, upper = prediction_interval(
            float(hybrid["prediction"]), pd.read_csv(residual_path), confidence_value, coverage
        )
        confidence["prediction_interval"] = {
            "lower": lower,
            "upper": upper,
            "coverage": coverage,
        }
    diagnostics = {key: value for key, value in hybrid.items() if key != "prediction"}
    diagnostics["hybrid_model_dir"] = str(model_dir)
    return float(hybrid["prediction"]), confidence, diagnostics


def build_target_report(
    target: str,
    target_table: pd.DataFrame,
    base_predictions: pd.DataFrame,
    hybrid_model_dir: Path | None = None,
    skip_hybrid: bool = False,
) -> dict[str, Any]:
    column = PREDICTION_COLUMNS[target]
    values = _finite_values(target_table, column)
    if values.empty:
        raise ValueError(f"No finite {target} prediction is available")
    prediction = float(values.mean())
    confidence = _confidence(target_table)
    diagnostics = _diagnostics(target_table, values)
    warnings: list[str] = []
    method = "base_prediction_mean"
    if not skip_hybrid and hybrid_model_dir is not None:
        if hybrid_model_dir.exists():
            prediction, hybrid_confidence, hybrid_diagnostics = _apply_hybrid(
                base_predictions, target, hybrid_model_dir
            )
            confidence.update(hybrid_confidence)
            diagnostics["hybrid"] = hybrid_diagnostics
            method = "hybrid_ensemble"
        else:
            warnings.append(f"Hybrid model directory does not exist; used base prediction mean: {hybrid_model_dir}")
    if diagnostics["outside_applicability_domain"]:
        warnings.append("At least one base prediction is outside its applicability domain.")
    return {
        "target": target,
        "final_prediction": {target: prediction},
        "prediction_method": method,
        "confidence": confidence,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def render_target_markdown(report: dict[str, Any]) -> str:
    target = str(report["target"])
    value = float(report["final_prediction"][target])
    unit = " nm" if target.endswith("_nm") else ""
    lines = [
        f"# FluorCast {target.replace('_', ' ')} prediction",
        "",
        "> This is a model prediction, not an experimental fact.",
        "",
        f"- Final prediction: {value:.4g}{unit}",
        f"- Prediction method: {report['prediction_method']}",
    ]
    label = report.get("confidence", {}).get("label")
    if label:
        lines.append(f"- Confidence: {label}")
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", "", *[f"- {warning}" for warning in warnings]])
    return "\n".join(lines) + "\n"


def _write_json(value: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _prediction_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        smiles=args.smiles,
        solvent=None,
        solvent_smiles=args.solvent_smiles,
        solvent_descriptors=predict_all_models.DEFAULT_SOLVENT_DESCRIPTORS,
        standardized_combined=predict_all_models.DEFAULT_STANDARDIZED_COMBINED,
        tree_model_dir=args.tree_model_dir,
        neural_model_dir=args.neural_model_dir,
        graph_model_dirs=args.graph_model_dirs,
        known_absorption_nm=args.known_absorption_nm,
        known_emission_nm=args.known_emission_nm,
        known_quantum_yield=args.known_quantum_yield,
        applicability_threshold=predict_all_models.DEFAULT_APPLICABILITY_THRESHOLD,
    )


def run_workflow(
    args: argparse.Namespace,
    collector: Callable[[argparse.Namespace], tuple[pd.DataFrame, list[str], str, str | None, str]] | None = None,
) -> dict[str, Any]:
    """Run prediction once, then create target and combined artifacts."""
    collect = collector or predict_all_models.collect_predictions
    base, collection_warnings, _, _, _ = collect(_prediction_namespace(args))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base.to_csv(args.out_dir / "base_predictions.csv", index=False)

    reports: dict[str, dict[str, Any]] = {}
    hybrid_dirs = {
        "absorption_nm": args.absorption_hybrid_model_dir,
        "emission_nm": args.emission_hybrid_model_dir,
        "quantum_yield": args.quantum_yield_hybrid_model_dir,
    }
    for target in TARGETS:
        column = PREDICTION_COLUMNS[target]
        values = _finite_values(base, column)
        target_table = base.loc[values.index].copy()
        stem = REPORT_STEMS[target]
        target_table.to_csv(args.out_dir / f"{stem}_predictions.csv", index=False)
        if target_table.empty:
            if target in {"absorption_nm", "emission_nm"}:
                raise ValueError(
                    f"Cannot create a full FluorCast report: no finite {target} prediction is available"
                )
            continue
        report = build_target_report(
            target, target_table, base, hybrid_dirs[target], args.skip_hybrid
        )
        if collection_warnings:
            report["warnings"].extend(collection_warnings)
        reports[target] = report
        _write_json(report, args.out_dir / f"{stem}_report.json")
        (args.out_dir / f"{stem}_report.md").write_text(
            render_target_markdown(report), encoding="utf-8"
        )

    full = build_report(
        reports["absorption_nm"],
        reports["emission_nm"],
        reports.get("quantum_yield"),
    )
    _write_json(full, args.out_dir / "full_fluorcast_report.json")
    (args.out_dir / "full_fluorcast_report.md").write_text(
        render_markdown(full), encoding="utf-8"
    )
    return full


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_workflow(args)
    except (FileNotFoundError, ImportError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
