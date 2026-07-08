"""Apply a trained hybrid ensemble and render an updated prediction report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.hybrid.ensemble import (  # noqa: E402
    align_features,
    load_hybrid_ensemble,
    predict_hybrid_ensemble,
)
from chemfluor.hybrid.explanation import summarize_prediction  # noqa: E402
from chemfluor.hybrid.meta_features import add_wide_feature_aliases, build_meta_features  # noqa: E402
from chemfluor.hybrid.report import (  # noqa: E402
    build_hybrid_report,
    load_prediction_table,
    render_report_markdown,
    write_report_json,
)
from chemfluor.hybrid.uncertainty import prediction_interval  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained FluorCast hybrid ensemble.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predictions = load_prediction_table(args.prediction_csv)
    model, columns, metadata = load_hybrid_ensemble(args.model_dir)
    raw_features = add_wide_feature_aliases(
        build_meta_features(predictions), str(model["target_name"])
    )
    features = align_features(raw_features, columns)
    hybrid = predict_hybrid_ensemble(model, features)
    confidence_value = features.get("overall_confidence_score")
    confidence = None if confidence_value is None else float(confidence_value.iloc[0])
    residuals = pd.read_csv(args.model_dir / "calibration_residuals.csv")
    coverage = float(metadata.get("conformal_coverage", 0.90))
    lower, upper = prediction_interval(hybrid["prediction"], residuals, confidence, coverage)

    report = build_hybrid_report(predictions)
    target = str(model["target_name"])
    target_fields = {
        "absorption_nm": "final_absorption_prediction_nm",
        "emission_nm": "final_emission_prediction_nm",
        "quantum_yield": "final_quantum_yield_prediction",
    }
    if target not in target_fields:
        raise ValueError(f"Unsupported hybrid target: {target}")
    report[target_fields[target]] = hybrid["prediction"]
    report["hybrid_ensemble"] = {
        "target": target,
        "prediction": hybrid["prediction"],
        "prediction_interval": {"lower": lower, "upper": upper, "coverage": coverage},
        **{key: value for key, value in hybrid.items() if key != "prediction"},
    }
    report["chemist_summary"] = summarize_prediction(report)
    write_report_json(report, args.out_json)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_report_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
