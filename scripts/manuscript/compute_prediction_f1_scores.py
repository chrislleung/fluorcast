"""Compute manuscript classification metrics from regression prediction CSVs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


DEFAULT_TARGETS = ["absorption_nm", "emission_nm", "quantum_yield"]
DEFAULT_SPLITS = ["random", "molecule", "scaffold"]
QY_CLASSES = ["dim", "bright"]
WAVELENGTH_CLASSES = ["UV", "blue", "green", "yellow/orange", "red/NIR"]
FILENAME_RE = re.compile(
    r"^(?P<target>.+?)__(?P<model>.+?)__(?P<split>.+?)__seed(?P<seed>\d+)\.csv$"
)

METRIC_COLUMNS = [
    "target",
    "split",
    "model",
    "seed",
    "classification_task",
    "n_rows",
    "classes",
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
    "micro_f1",
    "prediction_path",
]
AGGREGATE_COLUMNS = [
    "target",
    "split",
    "model",
    "n_seeds",
    "accuracy_mean",
    "accuracy_std",
    "balanced_accuracy_mean",
    "balanced_accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "weighted_f1_mean",
    "weighted_f1_std",
    "micro_f1_mean",
    "micro_f1_std",
]
PER_CLASS_COLUMNS = [
    "target",
    "split",
    "model",
    "seed",
    "class_label",
    "precision",
    "recall",
    "f1",
    "support",
]
PER_CLASS_AGGREGATE_COLUMNS = [
    "target",
    "split",
    "model",
    "class_label",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "support_mean",
]
CONFUSION_COLUMNS = [
    "target",
    "split",
    "model",
    "seed",
    "actual_class",
    "predicted_class",
    "count",
]
BEST_COLUMNS = [
    "target",
    "split",
    "best_model_by_macro_f1",
    "macro_f1_mean",
    "macro_f1_std",
    "best_model_by_weighted_f1",
    "weighted_f1_mean",
    "weighted_f1_std",
]


def warn(message: str) -> None:
    """Print a consistent warning."""
    print(f"WARNING: {message}", file=sys.stderr)


def comma_list(value: str) -> list[str]:
    """Parse a comma-separated CLI value."""
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-results-dir",
        type=Path,
        default=Path("outputs/paper_comparison"),
    )
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--targets", type=comma_list, default=DEFAULT_TARGETS, metavar="TARGETS"
    )
    parser.add_argument(
        "--splits", type=comma_list, default=DEFAULT_SPLITS, metavar="SPLITS"
    )
    parser.add_argument(
        "--models",
        type=comma_list,
        default=None,
        metavar="MODELS",
        help="Comma-separated models (default: discover from filenames).",
    )
    parser.add_argument(
        "--seeds",
        type=comma_list,
        default=None,
        metavar="SEEDS",
        help="Comma-separated integer seeds (default: discover from filenames).",
    )
    parser.add_argument("--qy-threshold", type=float, default=0.25)
    parser.add_argument("--on-missing", choices=["warn", "error"], default="warn")
    args = parser.parse_args(argv)
    if args.seeds is not None:
        try:
            args.seeds = [int(seed) for seed in args.seeds]
        except ValueError as exc:
            parser.error(f"--seeds must contain integers: {exc}")
    if args.predictions_dir is None:
        args.predictions_dir = args.paper_results_dir / "predictions"
    if args.out_dir is None:
        args.out_dir = args.paper_results_dir / "classification_metrics"
    return args


def parse_prediction_filename(path: str | Path) -> dict[str, object] | None:
    """Extract target, model, split, and seed from a prediction filename."""
    match = FILENAME_RE.match(Path(path).name)
    if match is None:
        return None
    parsed: dict[str, object] = match.groupdict()
    parsed["seed"] = int(parsed["seed"])
    return parsed


def bin_quantum_yield(values: Sequence[float], threshold: float = 0.25) -> np.ndarray:
    """Map quantum yields to dim/bright labels."""
    array = np.asarray(values, dtype=float)
    return np.where(array <= threshold, "dim", "bright")


def bin_wavelength(values: Sequence[float]) -> np.ndarray:
    """Map wavelengths to manuscript spectral-region labels."""
    array = np.asarray(values, dtype=float)
    return np.select(
        [array < 400, array < 500, array < 560, array < 620],
        WAVELENGTH_CLASSES[:-1],
        default=WAVELENGTH_CLASSES[-1],
    )


def class_labels(target: str) -> list[str]:
    """Return the ordered labels for a supported target."""
    if target == "quantum_yield":
        return QY_CLASSES
    if target in {"absorption_nm", "emission_nm"}:
        return WAVELENGTH_CLASSES
    raise ValueError(f"Unsupported target: {target}")


def classify(values: Sequence[float], target: str, qy_threshold: float) -> np.ndarray:
    """Convert continuous target values to classification labels."""
    if target == "quantum_yield":
        return bin_quantum_yield(values, qy_threshold)
    if target in {"absorption_nm", "emission_nm"}:
        return bin_wavelength(values)
    raise ValueError(f"Unsupported target: {target}")


def detect_prediction_columns(frame: pd.DataFrame, target: str) -> tuple[str, str]:
    """Detect actual and predicted numeric columns, preferring target-specific names."""
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    actual_specific = [
        f"actual_{target}",
        f"true_{target}",
        f"observed_{target}",
        target,
    ]
    predicted_specific = [
        f"predicted_{target}",
        f"prediction_{target}",
        f"pred_{target}",
        f"y_pred_{target}",
    ]
    actual_generic = ["actual", "y_true", "true", "observed"]
    predicted_generic = ["predicted", "prediction", "y_pred", "pred"]

    def first(candidates: list[str]) -> str | None:
        return next((normalized[name] for name in candidates if name in normalized), None)

    actual = first(actual_specific) or first(actual_generic)
    predicted = first(predicted_specific) or first(predicted_generic)
    if actual is None or predicted is None or actual == predicted:
        raise ValueError(
            "could not detect distinct actual and predicted columns "
            f"(columns: {', '.join(map(str, frame.columns))})"
        )
    return actual, predicted


def calculate_metrics(
    actual_values: Sequence[float],
    predicted_values: Sequence[float],
    target: str,
    qy_threshold: float = 0.25,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    """Calculate overall, per-class, and confusion-matrix metrics."""
    actual_class = classify(actual_values, target, qy_threshold)
    predicted_class = classify(predicted_values, target, qy_threshold)
    labels = class_labels(target)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(actual_class, predicted_class)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual_class, predicted_class)
        ),
    }
    for average in ["macro", "weighted", "micro"]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_class,
            predicted_class,
            labels=labels,
            average=average,
            zero_division=0,
        )
        if average != "micro":
            metrics[f"{average}_precision"] = float(precision)
            metrics[f"{average}_recall"] = float(recall)
        metrics[f"{average}_f1"] = float(f1)

    precision, recall, f1, support = precision_recall_fscore_support(
        actual_class, predicted_class, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class_label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        }
    )
    matrix = confusion_matrix(actual_class, predicted_class, labels=labels)
    confusion = pd.DataFrame(
        [
            {
                "actual_class": actual_label,
                "predicted_class": predicted_label,
                "count": int(matrix[i, j]),
            }
            for i, actual_label in enumerate(labels)
            for j, predicted_label in enumerate(labels)
        ]
    )
    return metrics, per_class, confusion


def process_prediction_file(
    path: Path,
    metadata: dict[str, object],
    qy_threshold: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Read and score one prediction CSV."""
    frame = pd.read_csv(path)
    target = str(metadata["target"])
    actual_column, predicted_column = detect_prediction_columns(frame, target)
    numeric = pd.DataFrame(
        {
            "actual": pd.to_numeric(frame[actual_column], errors="coerce"),
            "predicted": pd.to_numeric(frame[predicted_column], errors="coerce"),
        }
    ).dropna()
    if numeric.empty:
        raise ValueError("no rows have numeric actual and predicted values")
    scores, per_class, confusion = calculate_metrics(
        numeric["actual"], numeric["predicted"], target, qy_threshold
    )
    task = (
        f"quantum yield bright/dim at threshold {qy_threshold:g}"
        if target == "quantum_yield"
        else "wavelength spectral region"
    )
    overall: dict[str, object] = {
        **metadata,
        "classification_task": task,
        "n_rows": len(numeric),
        "classes": ",".join(class_labels(target)),
        **scores,
        "prediction_path": str(path),
    }
    identity = {key: metadata[key] for key in ["target", "split", "model", "seed"]}
    per_class = per_class.assign(**identity)
    confusion = confusion.assign(**identity)
    return overall, per_class[PER_CLASS_COLUMNS], confusion[CONFUSION_COLUMNS]


def aggregate_over_seeds(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate selected overall metrics over seeds."""
    if metrics.empty:
        return pd.DataFrame(columns=AGGREGATE_COLUMNS)
    keys = ["target", "split", "model"]
    values = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "micro_f1"]
    grouped = metrics.groupby(keys, as_index=False).agg(
        n_seeds=("seed", "nunique"),
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in values
            for stat in ["mean", "std"]
        },
    )
    return grouped[AGGREGATE_COLUMNS]


def aggregate_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-class metrics over seeds."""
    if per_class.empty:
        return pd.DataFrame(columns=PER_CLASS_AGGREGATE_COLUMNS)
    keys = ["target", "split", "model", "class_label"]
    grouped = per_class.groupby(keys, as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        support_mean=("support", "mean"),
    )
    return grouped[PER_CLASS_AGGREGATE_COLUMNS]


def select_best_models(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Select the best macro- and weighted-F1 model for every target/split."""
    if aggregated.empty:
        return pd.DataFrame(columns=BEST_COLUMNS)
    rows: list[dict[str, object]] = []
    for (target, split), subset in aggregated.groupby(["target", "split"], sort=True):
        macro = subset.sort_values(
            ["macro_f1_mean", "model"], ascending=[False, True], na_position="last"
        ).iloc[0]
        weighted = subset.sort_values(
            ["weighted_f1_mean", "model"], ascending=[False, True], na_position="last"
        ).iloc[0]
        rows.append(
            {
                "target": target,
                "split": split,
                "best_model_by_macro_f1": macro["model"],
                "macro_f1_mean": macro["macro_f1_mean"],
                "macro_f1_std": macro["macro_f1_std"],
                "best_model_by_weighted_f1": weighted["model"],
                "weighted_f1_mean": weighted["weighted_f1_mean"],
                "weighted_f1_std": weighted["weighted_f1_std"],
            }
        )
    return pd.DataFrame(rows, columns=BEST_COLUMNS)


def markdown_summary(best: pd.DataFrame, qy_threshold: float) -> str:
    """Create the human-readable manuscript classification summary."""
    lines = [
        "# Classification F1 summary",
        "",
        (
            "F1 is a classification metric. Wavelength F1 values were computed by "
            "binning continuous absorption and emission predictions into UV, blue, "
            "green, yellow/orange, and red/NIR spectral regions. Absorption results "
            "are therefore wavelength-region classification metrics derived from "
            "continuous absorption predictions."
        ),
        "",
        f"Quantum yield was classified as dim at QY <= {qy_threshold:g} and bright at QY > {qy_threshold:g}.",
        "",
        (
            "Wavelength-region F1 describes correct region assignment and should be "
            "interpreted separately from regression MAE, which measures the magnitude "
            "of wavelength error."
        ),
        "",
        "## Best models by target and split",
        "",
    ]
    if best.empty:
        lines.append("_No parseable prediction files matched the requested filters._")
    else:
        lines.extend(
            [
                "| Target | Split | Best macro-F1 model | Macro F1 (mean +/- SD) | Best weighted-F1 model | Weighted F1 (mean +/- SD) |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in best.itertuples(index=False):
            macro_std = "NA" if pd.isna(row.macro_f1_std) else f"{row.macro_f1_std:.4f}"
            weighted_std = "NA" if pd.isna(row.weighted_f1_std) else f"{row.weighted_f1_std:.4f}"
            lines.append(
                f"| {row.target} | {row.split} | {row.best_model_by_macro_f1} | "
                f"{row.macro_f1_mean:.4f} +/- {macro_std} | "
                f"{row.best_model_by_weighted_f1} | "
                f"{row.weighted_f1_mean:.4f} +/- {weighted_std} |"
            )
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    confusion: pd.DataFrame,
    qy_threshold: float,
) -> None:
    """Write all required CSV and Markdown outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = metrics.reindex(columns=METRIC_COLUMNS)
    per_class = per_class.reindex(columns=PER_CLASS_COLUMNS)
    confusion = confusion.reindex(columns=CONFUSION_COLUMNS)
    aggregated = aggregate_over_seeds(metrics)
    per_class_aggregated = aggregate_per_class(per_class)
    best = select_best_models(aggregated)
    outputs = {
        "f1_metrics_by_target_split_model_seed.csv": metrics,
        "f1_metrics_aggregated_by_seed.csv": aggregated,
        "per_class_f1_by_target_split_model_seed.csv": per_class,
        "per_class_f1_aggregated_by_seed.csv": per_class_aggregated,
        "confusion_matrices.csv": confusion,
        "best_f1_by_target_split.csv": best,
    }
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False)
    (out_dir / "f1_summary.md").write_text(
        markdown_summary(best, qy_threshold), encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Discover, score, aggregate, and report prediction classifications."""
    args = parse_args(argv)
    if not args.predictions_dir.is_dir():
        message = f"Predictions directory not found: {args.predictions_dir}"
        if args.on_missing == "error":
            raise FileNotFoundError(message)
        warn(message)

    discovered: list[tuple[Path, dict[str, object]]] = []
    if args.predictions_dir.is_dir():
        for path in sorted(args.predictions_dir.glob("*.csv")):
            metadata = parse_prediction_filename(path)
            if metadata is None:
                warn(f"Ignoring prediction file with unrecognized name: {path}")
            else:
                discovered.append((path, metadata))

    models = args.models or sorted({str(item[1]["model"]) for item in discovered})
    seeds = args.seeds or sorted({int(item[1]["seed"]) for item in discovered})
    selected = [
        (path, metadata)
        for path, metadata in discovered
        if metadata["target"] in args.targets
        and metadata["split"] in args.splits
        and metadata["model"] in models
        and metadata["seed"] in seeds
    ]
    available = {
        (str(meta["target"]), str(meta["model"]), str(meta["split"]), int(meta["seed"]))
        for _, meta in selected
    }
    expected = {
        (target, model, split, seed)
        for target in args.targets
        for model in models
        for split in args.splits
        for seed in seeds
    }
    missing = sorted(expected - available)
    if missing:
        preview = ", ".join(
            f"{target}/{model}/{split}/seed{seed}"
            for target, model, split, seed in missing[:10]
        )
        suffix = " ..." if len(missing) > 10 else ""
        message = f"Missing {len(missing)} requested prediction file(s): {preview}{suffix}"
        if args.on_missing == "error":
            raise FileNotFoundError(message)
        warn(message)

    overall_rows: list[dict[str, object]] = []
    per_class_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    parse_failures: list[str] = []
    for path, metadata in selected:
        try:
            overall, per_class, confusion = process_prediction_file(
                path, metadata, args.qy_threshold
            )
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            message = f"Could not parse prediction file {path}: {exc}"
            warn(message)
            parse_failures.append(message)
            continue
        overall_rows.append(overall)
        per_class_frames.append(per_class)
        confusion_frames.append(confusion)

    if parse_failures and args.on_missing == "error":
        raise ValueError(parse_failures[0])
    metrics = pd.DataFrame(overall_rows, columns=METRIC_COLUMNS)
    per_class = (
        pd.concat(per_class_frames, ignore_index=True)
        if per_class_frames
        else pd.DataFrame(columns=PER_CLASS_COLUMNS)
    )
    confusion = (
        pd.concat(confusion_frames, ignore_index=True)
        if confusion_frames
        else pd.DataFrame(columns=CONFUSION_COLUMNS)
    )
    write_outputs(args.out_dir, metrics, per_class, confusion, args.qy_threshold)
    print(f"Processed {len(metrics)} prediction file(s).")
    print(f"Saved classification metrics: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
