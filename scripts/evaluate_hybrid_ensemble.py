from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


PREDICTOR_KEYS = (
    "model",
    "pipeline",
    "estimator",
    "regressor",
    "classifier",
    "calibrated_classifier",
    "regression_model",
    "classification_model",
    "predictor",
)


def extract_predictor(payload: Any) -> Any:
    """Extract a sklearn-like predictor from a raw object or saved payload."""
    if callable(getattr(payload, "predict", None)):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(
            "Loaded model payload is neither a predictor with .predict() nor a dictionary; "
            f"got {type(payload).__name__}."
        )

    for key in PREDICTOR_KEYS:
        candidate = payload.get(key)
        if callable(getattr(candidate, "predict", None)):
            return candidate

    predictors = [
        value for value in payload.values() if callable(getattr(value, "predict", None))
    ]
    available = list(payload.keys())
    if len(predictors) == 1:
        return predictors[0]
    if not predictors:
        raise ValueError(
            "Model dictionary contains no object with .predict(). "
            f"Available keys: {available}"
        )
    raise ValueError(
        "Model dictionary contains multiple predictors under unrecognized keys; "
        f"cannot choose unambiguously. Available keys: {available}"
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    valid = y_true.notna() & y_pred.notna()
    yt = y_true.loc[valid].astype(float).to_numpy()
    yp = y_pred.loc[valid].astype(float).to_numpy()

    if len(yt) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    return {
        "n": int(len(yt)),
        "mae": float(mean_absolute_error(yt, yp)),
        "rmse": rmse(yt, yp),
        "r2": float(r2_score(yt, yp)) if len(yt) > 1 else np.nan,
    }


def load_feature_columns(model_dir: Path) -> list[str]:
    feature_path = model_dir / "feature_columns.json"
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature column file: {feature_path}")

    data = load_json(feature_path)

    if isinstance(data, list):
        return [str(col) for col in data]

    if isinstance(data, dict):
        for key in ["feature_columns", "features", "columns"]:
            if key in data and isinstance(data[key], list):
                return [str(col) for col in data[key]]

    raise ValueError(
        f"Could not parse feature columns from {feature_path}. "
        "Expected a JSON list or a dict containing feature_columns."
    )


def prepare_features(table: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=table.index)

    missing_columns: list[str] = []

    for col in feature_columns:
        if col in table.columns:
            features[col] = pd.to_numeric(table[col], errors="coerce")
        else:
            features[col] = np.nan
            missing_columns.append(col)

    if missing_columns:
        print("Warning: validation CSV is missing these saved feature columns:")
        for col in missing_columns:
            print(f"  - {col}")
        print("They were filled with NaN and should be handled by the trained pipeline/imputer.")

    return features[feature_columns]


def base_prediction_columns(table: pd.DataFrame, target_name: str, target_column: str) -> list[str]:
    suffix = f"_{target_name}"

    exclude = {
        target_column,
        f"true_{target_name}",
        f"actual_{target_name}",
    }

    skip_prefixes = (
        "true_",
        "actual_",
        "lower_",
        "upper_",
        "interval_",
    )

    skip_exact = {
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "prediction_range",
        "prediction_count",
    }

    cols: list[str] = []

    for col in table.columns:
        if col in exclude or col in skip_exact:
            continue

        if col.startswith(skip_prefixes):
            continue

        if col == "prediction_mean":
            cols.append(col)
            continue

        if col.endswith(suffix):
            cols.append(col)

    # Keep prediction_mean last-ish, after base models, for readability.
    cols = sorted([c for c in cols if c != "prediction_mean"])
    if "prediction_mean" in table.columns:
        cols.append("prediction_mean")

    return cols


def find_interval_columns(table: pd.DataFrame) -> tuple[str | None, str | None]:
    candidates = [
        ("prediction_lower", "prediction_upper"),
        ("lower", "upper"),
        ("interval_lower", "interval_upper"),
        ("lower_bound", "upper_bound"),
        ("hybrid_lower", "hybrid_upper"),
        ("hybrid_prediction_lower", "hybrid_prediction_upper"),
    ]

    for lo, hi in candidates:
        if lo in table.columns and hi in table.columns:
            return lo, hi

    return None, None


def interval_metrics(
    y_true: pd.Series,
    lower: pd.Series | None,
    upper: pd.Series | None,
) -> dict[str, float | int | None]:
    if lower is None or upper is None:
        return {
            "interval_n": 0,
            "interval_coverage": None,
            "mean_interval_width": None,
        }

    valid = y_true.notna() & lower.notna() & upper.notna()
    if valid.sum() == 0:
        return {
            "interval_n": 0,
            "interval_coverage": None,
            "mean_interval_width": None,
        }

    yt = y_true.loc[valid].astype(float)
    lo = lower.loc[valid].astype(float)
    hi = upper.loc[valid].astype(float)

    covered = ((yt >= lo) & (yt <= hi)).mean()
    width = (hi - lo).mean()

    return {
        "interval_n": int(valid.sum()),
        "interval_coverage": float(covered),
        "mean_interval_width": float(width),
    }


def markdown_summary(
    metrics: pd.DataFrame,
    target_name: str,
    target_column: str,
    out_dir: Path,
) -> str:
    sorted_metrics = metrics.dropna(subset=["mae"]).sort_values("mae", ascending=True)

    hybrid_row = metrics.loc[metrics["model"] == "hybrid_ensemble"]
    base_rows = metrics.loc[metrics["model"] != "hybrid_ensemble"].dropna(subset=["mae"])

    lines: list[str] = []
    lines.append(f"# Hybrid ensemble evaluation: `{target_name}`")
    lines.append("")
    lines.append(f"Target column: `{target_column}`")
    lines.append("")

    if not sorted_metrics.empty:
        best_overall = sorted_metrics.iloc[0]
        lines.append(
            f"Best overall model: **{best_overall['model']}** "
            f"with MAE = **{best_overall['mae']:.4f}**."
        )
        lines.append("")

    if not hybrid_row.empty:
        h = hybrid_row.iloc[0]
        lines.append("## Hybrid ensemble")
        lines.append("")
        lines.append(f"- N: {int(h['n'])}")
        lines.append(f"- MAE: {h['mae']:.4f}")
        lines.append(f"- RMSE: {h['rmse']:.4f}")
        lines.append(f"- R²: {h['r2']:.4f}")
        lines.append("")

    if not base_rows.empty and not hybrid_row.empty:
        best_base = base_rows.sort_values("mae", ascending=True).iloc[0]
        hybrid = hybrid_row.iloc[0]

        lines.append("## Comparison to best base model")
        lines.append("")
        lines.append(
            f"Best base model: **{best_base['model']}** "
            f"with MAE = **{best_base['mae']:.4f}**."
        )
        lines.append(
            f"Hybrid ensemble MAE = **{hybrid['mae']:.4f}**."
        )

        delta = float(hybrid["mae"] - best_base["mae"])
        if delta < 0:
            pct = abs(delta) / float(best_base["mae"]) * 100
            lines.append(
                f"The hybrid ensemble improved over the best base model by "
                f"**{abs(delta):.4f}** MAE units (**{pct:.2f}%** relative improvement)."
            )
        elif delta > 0:
            pct = delta / float(best_base["mae"]) * 100
            lines.append(
                f"The hybrid ensemble did **not** improve over the best base model. "
                f"It was worse by **{delta:.4f}** MAE units (**{pct:.2f}%** relative difference)."
            )
        else:
            lines.append("The hybrid ensemble tied the best base model by MAE.")

        lines.append("")

    lines.append("## Metrics table")
    lines.append("")
    lines.append("| Model | N | MAE | RMSE | R² |")
    lines.append("|---|---:|---:|---:|---:|")

    for _, row in metrics.sort_values("mae", ascending=True, na_position="last").iterrows():
        n = int(row["n"]) if not pd.isna(row["n"]) else 0
        mae = "NA" if pd.isna(row["mae"]) else f"{row['mae']:.4f}"
        row_rmse = "NA" if pd.isna(row["rmse"]) else f"{row['rmse']:.4f}"
        r2 = "NA" if pd.isna(row["r2"]) else f"{row['r2']:.4f}"
        lines.append(f"| {row['model']} | {n} | {mae} | {row_rmse} | {r2} |")

    lines.append("")
    lines.append("## Output files")
    lines.append("")
    lines.append(f"- `{out_dir / 'evaluated_predictions.csv'}`")
    lines.append(f"- `{out_dir / 'metrics_table.csv'}`")
    lines.append(f"- `{out_dir / 'metrics_summary.md'}`")
    lines.append("")
    lines.append(
        "Note: this evaluation is only valid as a performance claim if the validation CSV "
        "was not used to train the hybrid model."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained FluorCast hybrid ensemble against base model predictions."
    )
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-column", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--out-dir", required=True)

    args = parser.parse_args()

    prediction_csv = Path(args.prediction_csv)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)

    model_path = model_dir / "model.joblib"
    if not prediction_csv.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {prediction_csv}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    table = pd.read_csv(prediction_csv)

    if args.target_column not in table.columns:
        raise ValueError(
            f"Target column '{args.target_column}' not found in {prediction_csv}. "
            f"Available columns: {list(table.columns)}"
        )

    y_true = pd.to_numeric(table[args.target_column], errors="coerce")
    valid_target = y_true.notna()

    if valid_target.sum() == 0:
        raise ValueError(f"No finite labels found in target column: {args.target_column}")

    table = table.loc[valid_target].reset_index(drop=True)
    y_true = y_true.loc[valid_target].reset_index(drop=True)

    model_payload = joblib.load(model_path)
    model = extract_predictor(model_payload)
    feature_columns = load_feature_columns(model_dir)
    features = prepare_features(table, feature_columns)

    hybrid_pred = pd.Series(model.predict(features), name=f"hybrid_{args.target_name}")

    evaluated = table.copy()
    evaluated[f"hybrid_{args.target_name}"] = hybrid_pred

    rows: list[dict[str, Any]] = []

    hybrid_metrics = regression_metrics(y_true, hybrid_pred)
    rows.append(
        {
            "model": "hybrid_ensemble",
            **hybrid_metrics,
        }
    )

    for col in base_prediction_columns(table, args.target_name, args.target_column):
        pred = pd.to_numeric(table[col], errors="coerce")
        rows.append(
            {
                "model": col,
                **regression_metrics(y_true, pred),
            }
        )

    lo_col, hi_col = find_interval_columns(evaluated)
    if lo_col and hi_col:
        interval_info = interval_metrics(
            y_true,
            pd.to_numeric(evaluated[lo_col], errors="coerce"),
            pd.to_numeric(evaluated[hi_col], errors="coerce"),
        )
        for row in rows:
            if row["model"] == "hybrid_ensemble":
                row.update(interval_info)

    metrics = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(evaluated, out_dir / "evaluated_predictions.csv")
    write_csv(metrics, out_dir / "metrics_table.csv")
    write_text(
        out_dir / "metrics_summary.md",
        markdown_summary(
            metrics=metrics,
            target_name=args.target_name,
            target_column=args.target_column,
            out_dir=out_dir,
        ),
    )

    print(f"Wrote evaluated predictions: {out_dir / 'evaluated_predictions.csv'}")
    print(f"Wrote metrics table: {out_dir / 'metrics_table.csv'}")
    print(f"Wrote summary: {out_dir / 'metrics_summary.md'}")

    sorted_metrics = metrics.dropna(subset=["mae"]).sort_values("mae", ascending=True)
    if not sorted_metrics.empty:
        print("")
        print("Best models by MAE:")
        print(sorted_metrics[["model", "n", "mae", "rmse", "r2"]].head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
