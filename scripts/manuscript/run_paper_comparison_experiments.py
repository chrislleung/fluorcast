"""Run reproducible experiments for the FluorCast manuscript comparison.

Smoke-test example::

    python scripts/manuscript/run_paper_comparison_experiments.py --max-rows 200 --models rf --targets stokes_shift_nm --splits random --seeds 0 --out-dir outputs/paper_comparison_stokes_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
SCRIPTS_PATH = PROJECT_ROOT / "scripts"
MANUSCRIPT_PATH = Path(__file__).resolve().parent
for import_path in (SRC_PATH, SCRIPTS_PATH, MANUSCRIPT_PATH):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import train_combined_predictors as trainer  # noqa: E402
from manuscript_metrics import (  # noqa: E402
    bootstrap_regression_metrics,
    classification_metrics,
    region_metrics,
    regression_metrics,
)
from manuscript_plots import dataset_figures, result_figures  # noqa: E402
from manuscript_splits import add_scaffold_column, make_split  # noqa: E402


DEFAULT_COMBINED = Path("data/processed/fluodb_lite/combined_deduplicated.csv")
DEFAULT_SOLVENTS = Path("data/solvent_descriptors_expanded_deep4chem.csv")
DEFAULT_OUT_DIR = Path("outputs/paper_comparison")
VALID_MODELS = {"rf", "extratrees", "histgb", "gbdt", "mlp"}
VALID_TARGETS = {
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "stokes_shift_nm",
}
VALID_SPLITS = {"random", "molecule", "scaffold"}
IDENTITY_COLUMNS = [
    "canonical_chromophore_smiles",
    "solvent_original",
    "canonical_solvent_smiles",
    "source_dataset",
    "bemis_murcko_scaffold",
]


def parse_csv_list(text: str, cast=str) -> list[Any]:
    """Parse a non-empty comma-separated CLI value."""
    values = [cast(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("Comma-separated options cannot be empty.")
    return values


def parse_args() -> argparse.Namespace:
    """Parse manuscript pipeline command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--solvent-descriptors", type=Path, default=DEFAULT_SOLVENTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--models", default="rf,extratrees,histgb,gbdt,mlp")
    parser.add_argument(
        "--targets",
        default="absorption_nm,emission_nm,quantum_yield,stokes_shift_nm",
    )
    parser.add_argument("--splits", default="random,molecule,scaffold")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-training", action="store_true")
    return parser.parse_args()


def validate_options(
    models: list[str], targets: list[str], splits: list[str]
) -> None:
    """Validate requested experiment names."""
    invalid = {
        "models": sorted(set(models) - VALID_MODELS),
        "targets": sorted(set(targets) - VALID_TARGETS),
        "splits": sorted(set(splits) - VALID_SPLITS),
    }
    messages = [f"{key}: {values}" for key, values in invalid.items() if values]
    if messages:
        raise ValueError("Unknown options: " + "; ".join(messages))


def add_stokes_shift_target(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive the non-negative wavelength Stokes shift when both inputs exist."""
    derived = rows.copy()
    if {"absorption_nm", "emission_nm"}.issubset(derived.columns):
        absorption = pd.to_numeric(derived["absorption_nm"], errors="coerce")
        emission = pd.to_numeric(derived["emission_nm"], errors="coerce")
        stokes_shift = emission - absorption
        derived["stokes_shift_nm"] = stokes_shift.where(
            absorption.notna() & emission.notna() & stokes_shift.ge(0)
        )
    else:
        derived["stokes_shift_nm"] = np.nan
    return derived


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame without requiring the optional tabulate package."""
    if frame.empty:
        return "_No rows available._"
    printable = frame.copy()
    for column in printable.select_dtypes(include=[np.number]).columns:
        printable[column] = printable[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4g}"
        )
    headers = [str(column) for column in printable.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in printable.to_numpy()
    )
    return "\n".join(lines)


def write_dataset_audit(
    rows: pd.DataFrame, combined_path: Path, out_dir: Path
) -> None:
    """Write long-form CSV and readable Markdown dataset audits."""
    audit_rows: list[dict[str, Any]] = []

    def add(section: str, metric: str, value: Any, source: str = "all") -> None:
        audit_rows.append(
            {"section": section, "source_dataset": source, "metric": metric, "value": value}
        )

    add("overall", "total_rows", len(rows))
    add("overall", "unique_canonical_chromophores", rows["canonical_chromophore_smiles"].nunique())
    add("overall", "unique_canonical_solvents", rows["canonical_solvent_smiles"].nunique())
    pairs = rows[["canonical_chromophore_smiles", "canonical_solvent_smiles"]].drop_duplicates()
    add("overall", "unique_chromophore_solvent_pairs", len(pairs))
    for source, count in rows["source_dataset"].astype(str).value_counts().items():
        add("rows_by_source", "rows", int(count), source)
    for target in sorted(VALID_TARGETS):
        add("target_coverage", target, int(rows[target].notna().sum()))
    for source, subset in rows.groupby("source_dataset", dropna=False):
        source_name = str(source)
        for target in sorted(VALID_TARGETS):
            add("target_coverage_by_source", target, int(subset[target].notna().sum()), source_name)
        emission = subset["emission_nm"].dropna()
        for region, count in emission.map(
            lambda value: (
                "UV" if value < 400 else "blue" if value < 500 else
                "green" if value < 560 else "yellow/orange" if value < 620 else "red/NIR"
            )
        ).value_counts().items():
            add("emission_region_by_source", str(region), int(count), source_name)

    before_path = combined_path.with_name("combined_before_dedup.csv")
    if before_path.exists():
        before_rows = len(pd.read_csv(before_path, usecols=[0]))
        add("deduplication", "rows_before_deduplication", before_rows)
        add("deduplication", "rows_after_deduplication", len(rows))
        add("deduplication", "duplicates_removed", before_rows - len(rows))
    else:
        add("deduplication", "rows_before_deduplication", "not_recoverable")
        add("deduplication", "rows_after_deduplication", len(rows))
        add("deduplication", "duplicates_removed", "not_recoverable")

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(out_dir / "dataset_audit.csv", index=False)
    sections = ["# Dataset audit", ""]
    for section, frame in audit.groupby("section", sort=False):
        sections.extend([f"## {section.replace('_', ' ').title()}", "", markdown_table(frame.drop(columns="section")), ""])
    (out_dir / "dataset_audit.md").write_text("\n".join(sections), encoding="utf-8")


def make_classifier(model_name: str, seed: int, n_jobs: int) -> Any:
    """Construct the classifier analogue of a requested regression family."""
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs
        )
    if model_name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, random_state=seed
        )
    if model_name == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=3, random_state=seed
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256),
                        early_stopping=True,
                        max_iter=300,
                        random_state=seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown classifier model: {model_name}")


def feature_matrices(
    target_rows: pd.DataFrame,
    target_fingerprints: np.ndarray,
    descriptor_columns: list[str],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train/test features with train-only descriptor imputation."""
    descriptor_values = target_rows[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    medians = descriptor_values.iloc[train_idx].median(numeric_only=True)
    x_train = trainer.build_feature_matrix(
        target_fingerprints[train_idx], descriptor_values.iloc[train_idx], medians
    )
    x_test = trainer.build_feature_matrix(
        target_fingerprints[test_idx], descriptor_values.iloc[test_idx], medians
    )
    return x_train, x_test


def split_counts(rows: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> dict[str, int]:
    """Return manuscript sample and diversity counts."""
    train = rows.iloc[train_idx]
    test = rows.iloc[test_idx]
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "unique_train_molecules": train["canonical_chromophore_smiles"].nunique(),
        "unique_test_molecules": test["canonical_chromophore_smiles"].nunique(),
        "unique_train_scaffolds": train["bemis_murcko_scaffold"].nunique(),
        "unique_test_scaffolds": test["bemis_murcko_scaffold"].nunique(),
    }


def run_training(
    rows: pd.DataFrame,
    fingerprints: np.ndarray,
    descriptor_columns: list[str],
    models: list[str],
    targets: list[str],
    splits: list[str],
    seeds: list[int],
    test_size: float,
    n_jobs: int,
    out_dir: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[pd.DataFrame],
]:
    """Train all requested experiments and write per-run predictions."""
    metric_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    region_tables: list[pd.DataFrame] = []
    classifier_rows: list[dict[str, Any]] = []
    bootstrap_tables: list[pd.DataFrame] = []
    prediction_tables: list[pd.DataFrame] = []
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        target_mask = rows[target].notna().to_numpy()
        target_rows = rows.loc[target_mask].reset_index(drop=True)
        target_fingerprints = fingerprints[target_mask]
        if len(target_rows) < 2:
            print(f"WARNING: skipping {target}; fewer than two usable rows.")
            continue
        for split_name in splits:
            for seed in seeds:
                split = make_split(target_rows, split_name, test_size, seed)
                leakage_rows.append(
                    {
                        "target": target,
                        "split": split_name,
                        "seed": seed,
                        "group_column": split.group_column or "",
                        "train_groups": split.train_groups,
                        "test_groups": split.test_groups,
                        "overlapping_groups": split.overlapping_groups,
                        "leakage_detected": bool(split.overlapping_groups),
                    }
                )
                counts = split_counts(target_rows, split.train_indices, split.test_indices)
                x_train, x_test = feature_matrices(
                    target_rows,
                    target_fingerprints,
                    descriptor_columns,
                    split.train_indices,
                    split.test_indices,
                )
                y_train = target_rows[target].iloc[split.train_indices].to_numpy(float)
                y_test = target_rows[target].iloc[split.test_indices].to_numpy(float)

                for model_name in models:
                    model = trainer.make_model(model_name, random_state=seed, n_jobs=n_jobs)
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    prediction = target_rows.iloc[split.test_indices][IDENTITY_COLUMNS].copy()
                    prediction.insert(0, "row_index", target_rows.iloc[split.test_indices].index)
                    prediction["target"] = target
                    prediction["model"] = model_name
                    prediction["split"] = split_name
                    prediction["seed"] = seed
                    prediction["y_true"] = y_test
                    prediction["y_pred"] = y_pred
                    prediction["residual"] = y_test - y_pred
                    path = prediction_dir / f"{target}__{model_name}__{split_name}__seed{seed}.csv"
                    prediction.to_csv(path, index=False)
                    prediction_tables.append(prediction)

                    metric_rows.append(
                        {
                            "target": target,
                            "model": model_name,
                            "split": split_name,
                            "seed": seed,
                            **regression_metrics(y_test, y_pred, target),
                            **counts,
                            "prediction_path": str(path),
                        }
                    )
                    boot = bootstrap_regression_metrics(prediction, target, seed)
                    boot.insert(0, "seed", seed)
                    boot.insert(0, "split", split_name)
                    boot.insert(0, "model", model_name)
                    boot.insert(0, "target", target)
                    bootstrap_tables.append(boot)

                    if target == "emission_nm":
                        by_region = region_metrics(prediction)
                        by_region.insert(0, "seed", seed)
                        by_region.insert(0, "split", split_name)
                        by_region.insert(0, "model", model_name)
                        region_tables.append(by_region)

                    if target == "quantum_yield":
                        y_train_class = (y_train > 0.25).astype(int)
                        y_test_class = (y_test > 0.25).astype(int)
                        if len(np.unique(y_train_class)) < 2:
                            print(
                                f"WARNING: skipping QY classifier for {model_name}/{split_name}/"
                                f"{seed}; training data has one class."
                            )
                            continue
                        classifier = make_classifier(model_name, seed, n_jobs)
                        classifier.fit(x_train, y_train_class)
                        class_prediction = classifier.predict(x_test)
                        probability = (
                            classifier.predict_proba(x_test)[:, 1]
                            if hasattr(classifier, "predict_proba")
                            else None
                        )
                        classifier_rows.append(
                            {
                                "model": model_name,
                                "split": split_name,
                                "seed": seed,
                                "threshold": 0.25,
                                **counts,
                                **classification_metrics(
                                    y_test_class, class_prediction, probability
                                ),
                            }
                        )

    metrics = pd.DataFrame(metric_rows)
    leakage = pd.DataFrame(leakage_rows)
    regions = pd.concat(region_tables, ignore_index=True) if region_tables else pd.DataFrame()
    classifiers = pd.DataFrame(classifier_rows)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame()
    return metrics, leakage, regions, classifiers, bootstrap, prediction_tables


def load_existing_predictions(out_dir: Path) -> list[pd.DataFrame]:
    """Load prediction CSVs produced by an earlier training run."""
    paths = sorted((out_dir / "predictions").glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No existing prediction CSVs found under {out_dir / 'predictions'}"
        )
    return [pd.read_csv(path) for path in paths]


def summarize_existing_predictions(
    predictions: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute regression, regional, and bootstrap summaries."""
    metric_rows = []
    region_tables = []
    bootstrap_tables = []
    for prediction in predictions:
        first = prediction.iloc[0]
        target = str(first["target"])
        metadata = {
            "target": target,
            "model": first["model"],
            "split": first["split"],
            "seed": int(first["seed"]),
        }
        metric_rows.append(
            {
                **metadata,
                **regression_metrics(prediction["y_true"], prediction["y_pred"], target),
                "train_rows": np.nan,
                "test_rows": len(prediction),
                "unique_train_molecules": np.nan,
                "unique_test_molecules": prediction["canonical_chromophore_smiles"].nunique(),
                "unique_train_scaffolds": np.nan,
                "unique_test_scaffolds": prediction["bemis_murcko_scaffold"].nunique(),
            }
        )
        boot = bootstrap_regression_metrics(
            prediction, target, int(first["seed"])
        )
        for key, value in reversed(list(metadata.items())):
            boot.insert(0, key, value)
        bootstrap_tables.append(boot)
        if target == "emission_nm":
            region = region_metrics(prediction)
            for key in ["seed", "split", "model"]:
                region.insert(0, key, metadata[key])
            region_tables.append(region)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(region_tables, ignore_index=True) if region_tables else pd.DataFrame(),
        pd.concat(bootstrap_tables, ignore_index=True),
    )


def write_paper_tables(
    metrics: pd.DataFrame, regions: pd.DataFrame, classifiers: pd.DataFrame, path: Path
) -> None:
    """Write compact manuscript-ready Markdown summary tables."""
    sections = ["# Paper comparison tables", ""]
    if not metrics.empty:
        regression = (
            metrics.groupby(["target", "split", "model"], as_index=False)
            .agg(
                mae=("mae", "mean"),
                rmse=("rmse", "mean"),
                r2=("r2", "mean"),
                median_absolute_error=("median_absolute_error", "mean"),
            )
            .sort_values(["target", "split", "mae"])
        )
        sections.extend(["## Regression performance", "", markdown_table(regression), ""])
    if not regions.empty:
        regional = (
            regions.groupby(["split", "model", "region"], as_index=False)
            .agg(rows=("rows", "sum"), mae=("mae", "mean"), rmse=("rmse", "mean"))
        )
        sections.extend(["## Emission performance by region", "", markdown_table(regional), ""])
    if not classifiers.empty:
        qy = (
            classifiers.groupby(["split", "model"], as_index=False)
            .agg(
                accuracy=("accuracy", "mean"),
                balanced_accuracy=("balanced_accuracy", "mean"),
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                f1=("f1", "mean"),
                roc_auc=("roc_auc", "mean"),
            )
        )
        sections.extend(["## Bright quantum-yield classification", "", markdown_table(qy), ""])
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> int:
    """Run the complete manuscript-results workflow."""
    args = parse_args()
    try:
        models = parse_csv_list(args.models)
        targets = parse_csv_list(args.targets)
        splits = parse_csv_list(args.splits)
        seeds = parse_csv_list(args.seeds, int)
        validate_options(models, targets, splits)
        args.out_dir.mkdir(parents=True, exist_ok=True)

        full_rows = trainer.load_standardized_combined(args.standardized_combined)
        full_rows = add_stokes_shift_target(full_rows)
        write_dataset_audit(full_rows, args.standardized_combined, args.out_dir)
        dataset_figures(full_rows, args.out_dir / "figures")

        if args.skip_training:
            prediction_tables = load_existing_predictions(args.out_dir)
            metrics, regions, bootstrap = summarize_existing_predictions(prediction_tables)
            leakage_path = args.out_dir / "split_leakage_report.csv"
            leakage = pd.read_csv(leakage_path) if leakage_path.exists() else pd.DataFrame()
            classifier_path = args.out_dir / "qy_classifier_metrics.csv"
            classifiers = pd.read_csv(classifier_path) if classifier_path.exists() else pd.DataFrame()
        else:
            training_rows = full_rows
            if args.max_rows is not None and len(training_rows) > args.max_rows:
                training_rows = training_rows.sample(
                    n=args.max_rows, random_state=seeds[0]
                ).reset_index(drop=True)
                print(f"Using {len(training_rows)} sampled rows due to --max-rows.")
            training_rows = add_scaffold_column(training_rows)
            descriptors = trainer.load_solvent_descriptors(args.solvent_descriptors)
            modeling_rows, descriptor_columns = trainer.merge_solvent_descriptors(
                training_rows, descriptors
            )
            modeling_rows, fingerprints = trainer.add_fingerprints(
                modeling_rows, radius=2, n_bits=2048
            )
            modeling_rows = add_scaffold_column(modeling_rows)
            (
                metrics,
                leakage,
                regions,
                classifiers,
                bootstrap,
                prediction_tables,
            ) = run_training(
                modeling_rows,
                fingerprints,
                descriptor_columns,
                models,
                targets,
                splits,
                seeds,
                args.test_size,
                args.n_jobs,
                args.out_dir,
            )

        metrics.to_csv(args.out_dir / "metrics_by_split_model_target.csv", index=False)
        bootstrap.to_csv(args.out_dir / "metrics_with_bootstrap_ci.csv", index=False)
        regions.to_csv(args.out_dir / "region_metrics_by_split_model.csv", index=False)
        classifiers.to_csv(args.out_dir / "qy_classifier_metrics.csv", index=False)
        leakage.to_csv(args.out_dir / "split_leakage_report.csv", index=False)
        write_paper_tables(
            metrics, regions, classifiers, args.out_dir / "paper_tables.md"
        )
        result_figures(metrics, regions, prediction_tables, args.out_dir / "figures")
        print(f"Manuscript comparison outputs saved to: {args.out_dir}")
        return 0
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
