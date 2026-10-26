"""Create publication-quality spectral-region violin plots from held-out predictions.

The script deliberately consumes prediction-level records only.  Aggregate metric or
region-summary tables cannot be used to reconstruct a distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402


TARGETS = ("absorption_nm", "emission_nm")
FINAL_SPLIT_LABELS = {"final_test", "test", "held_out", "heldout", "holdout"}
SUMMARY_COLUMNS = [
    "target", "model", "representation", "split_type", "seed",
    "spectral_region", "n", "mae", "median_ae", "std_ae", "rmse", "q25", "q75",
]
ERROR_COLUMNS = [
    "row_id", "row_id_source", "target", "model", "representation", "split_type", "seed",
    "spectral_region", "y_true", "y_pred", "absolute_error", "source_prediction_file",
    "experiment",
]
FIVE_LABELS = ("UV", "Blue", "Green", "Yellow/Orange", "Red/NIR")
THREE_LABELS = ("UV", "Visible", "Near-infrared")
REGION_COLORS = {
    "UV": "#7651A1",
    "Blue": "#3B82C4",
    "Green": "#3C9D70",
    "Yellow/Orange": "#E3A62F",
    "Red/NIR": "#C94C4C",
    "Visible": "#39978A",
    "Near-infrared": "#9D4A68",
}
REPRESENTATION_LABELS = {
    "morgan_solvent": "Morgan fingerprint",
    "conforformer_solvent": "ConforFormer",
    "conforformer_morgan_solvent": "Morgan + ConforFormer",
}


class RegionDefinition(NamedTuple):
    name: str
    boundaries: tuple[float, ...]
    labels: tuple[str, ...]


def region_definition(scheme: str, boundaries: Sequence[float] | None = None) -> RegionDefinition:
    """Return validated region boundaries; intervals are left-closed except the first."""
    defaults = (400.0, 480.0, 560.0, 620.0) if scheme == "five" else (400.0, 700.0)
    labels = FIVE_LABELS if scheme == "five" else THREE_LABELS
    values = tuple(float(value) for value in (boundaries or defaults))
    if len(values) != len(labels) - 1 or any(not math.isfinite(v) for v in values):
        raise ValueError(f"The {scheme}-region scheme requires {len(labels) - 1} finite boundaries")
    if list(values) != sorted(values) or len(set(values)) != len(values):
        raise ValueError("Region boundaries must be strictly increasing")
    return RegionDefinition(scheme, values, labels)


def assign_spectral_region(value: Any, definition: RegionDefinition) -> str | None:
    """Assign a finite experimental wavelength using the configured exact boundaries."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    # The established five-region convention uniquely includes 620 in
    # Yellow/Orange, while 400, 480, and 560 begin their next interval.
    if definition.name == "five" and number == definition.boundaries[-1]:
        return definition.labels[-2]
    index = int(np.searchsorted(np.asarray(definition.boundaries), number, side="right"))
    return definition.labels[index]


def calculate_errors(frame: pd.DataFrame, definition: RegionDefinition) -> pd.DataFrame:
    """Coerce wavelengths, remove invalid rows, and calculate record-level absolute errors."""
    required = {"y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction table is missing column(s): {sorted(missing)}")
    result = frame.copy()
    result["y_true"] = pd.to_numeric(result["y_true"], errors="coerce")
    result["y_pred"] = pd.to_numeric(result["y_pred"], errors="coerce")
    result = result[np.isfinite(result["y_true"]) & np.isfinite(result["y_pred"])].copy()
    result["absolute_error"] = (result["y_true"] - result["y_pred"]).abs()
    result["spectral_region"] = result["y_true"].map(lambda value: assign_spectral_region(value, definition))
    return result.dropna(subset=["spectral_region"])


def summarize_errors(frame: pd.DataFrame, definition: RegionDefinition) -> pd.DataFrame:
    """Calculate region statistics from individual absolute errors."""
    rows: list[dict[str, Any]] = []
    group_columns = ["target", "model", "representation", "split_type", "seed"]
    for keys, group in frame.groupby(group_columns, dropna=False, sort=False):
        metadata = dict(zip(group_columns, keys))
        for region in definition.labels:
            subset = group[group["spectral_region"] == region]
            if subset.empty:
                continue
            errors = subset["absolute_error"].to_numpy(dtype=float)
            residuals = subset["y_true"].to_numpy(dtype=float) - subset["y_pred"].to_numpy(dtype=float)
            rows.append({
                **metadata,
                "spectral_region": region,
                "n": int(len(subset)),
                "mae": float(np.mean(errors)),
                "median_ae": float(np.median(errors)),
                "std_ae": float(np.std(errors, ddof=1)) if len(errors) > 1 else float("nan"),
                "rmse": float(np.sqrt(np.mean(np.square(residuals)))),
                "q25": float(np.quantile(errors, 0.25)),
                "q75": float(np.quantile(errors, 0.75)),
            })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _metadata_for(path: Path) -> dict[str, Any]:
    """Read experiment metadata adjacent to a prediction file."""
    candidates = [path.parent, path.parent.parent]
    metadata: dict[str, Any] = {}
    for directory in candidates:
        manifest = (
            _read_json(directory / "training_manifest.json")
            or _read_json(directory / "run_manifest.json")
            or _read_json(directory / "manifest.json")
        )
        config = _read_json(directory / "experiment_config.json")
        for source in (manifest, config):
            split_type = source.get("split_type")
            seed = source.get("split_seed", source.get("seed"))
            if split_type is not None and metadata.get("split_type") is None:
                metadata["split_type"] = split_type
            if seed is not None and metadata.get("seed") is None:
                metadata["seed"] = seed
    metadata["experiment"] = path.parent.parent.name if path.parent.name == "predictions" else path.parent.name
    return metadata


def _infer_target(path: Path, frame: pd.DataFrame) -> str | None:
    if "target" in frame and frame["target"].dropna().nunique() == 1:
        value = str(frame["target"].dropna().iloc[0]).lower()
        if value in TARGETS:
            return value
    lower = path.name.lower()
    if "absorption" in lower:
        return "absorption_nm"
    if "emission" in lower:
        return "emission_nm"
    return None


def _parse_conforformer_name(path: Path) -> tuple[str | None, str | None, str | None]:
    parts = path.stem.split("__")
    if len(parts) == 3 and parts[0] in TARGETS:
        return parts[0], parts[1], parts[2]
    return None, None, None


def _selected_estimator(path: Path, target: str, pooling: str | None, feature_set: str | None) -> tuple[str | None, float | None]:
    for directory in (path.parent, path.parent.parent):
        metrics_path = directory / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        mask = metrics.get("target", pd.Series(index=metrics.index, dtype=object)).eq(target)
        if pooling is not None and "pooling_method" in metrics:
            mask &= metrics["pooling_method"].astype(str).eq(pooling)
        if feature_set is not None and "feature_set" in metrics:
            mask &= metrics["feature_set"].astype(str).eq(feature_set)
        matched = metrics[mask]
        if len(matched) == 1:
            return str(matched.iloc[0].get("model", "unknown")), float(matched.iloc[0]["mae"])
    return None, None


def _held_out_rows(frame: pd.DataFrame, path: Path, assume_held_out: bool) -> pd.DataFrame:
    split_column = next((name for name in ("split", "dataset_split", "partition") if name in frame), None)
    if split_column:
        labels = frame[split_column].astype(str).str.lower().str.replace("-", "_", regex=False)
        unsafe = set(labels.unique()).difference(FINAL_SPLIT_LABELS)
        kept = frame[labels.isin(FINAL_SPLIT_LABELS)].copy()
        if kept.empty:
            raise ValueError(f"{path}: no held-out/final-test rows were found")
        if unsafe:
            warnings.warn(f"{path}: excluded non-test split labels: {sorted(unsafe)}", stacklevel=2)
        return kept
    safe_name = path.name.startswith("predictions_") or "final_" in path.name.lower()
    if safe_name or assume_held_out:
        return frame.copy()
    raise ValueError(
        f"{path}: split provenance is absent. Use --assume-held-out only after verifying every row is test data"
    )


def _identity(frame: pd.DataFrame, path: Path) -> tuple[pd.Series, str]:
    for column in ("row_id", "sample_id", "record_id", "id"):
        if column in frame:
            return frame[column].astype(str), column
    stable = [column for column in ("canonical_chromophore_smiles", "canonical_solvent_smiles", "y_true") if column in frame]
    if stable:
        return frame[stable].astype(str).agg("|".join, axis=1), "stable_composite"
    return pd.Series([f"{path.name}:{index}" for index in frame.index], index=frame.index), "implicit_row_index"


def load_prediction_file(
    path: Path,
    definition: RegionDefinition,
    *,
    assume_held_out: bool = False,
    fallback_model: str = "unspecified",
    fallback_representation: str = "unspecified",
    fallback_split_type: str = "unspecified",
    fallback_seed: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Adapt a FluorCast prediction CSV into one normalized row per prediction."""
    frame = pd.read_csv(path)
    frame = _held_out_rows(frame, path, assume_held_out)
    metadata = _metadata_for(path)
    parsed_target, pooling, feature_set = _parse_conforformer_name(path)
    target = parsed_target or _infer_target(path, frame)
    estimator, expected_mae = (None, None)
    if target:
        estimator, expected_mae = _selected_estimator(path, target, pooling, feature_set)
    normalized: list[pd.DataFrame] = []
    notes: list[str] = []

    def append_rows(source: pd.DataFrame, selected_target: str, truth: str, prediction: str, model: str, representation: str) -> None:
        table = source.copy()
        table["y_true"] = table[truth]
        table["y_pred"] = table[prediction]
        table["row_id"], id_source = _identity(table, path)
        table["row_id_source"] = id_source
        table["target"] = selected_target
        table["model"] = model
        table["representation"] = representation
        normalized.append(table)

    if {"y_true", "y_pred"}.issubset(frame.columns):
        if target is None:
            raise ValueError(f"{path}: cannot infer absorption versus emission target")
        representation = REPRESENTATION_LABELS.get(feature_set, feature_set or fallback_representation)
        if pooling and pooling != "not_applicable":
            representation = f"{representation} ({pooling} pooling)"
        append_rows(frame, target, "y_true", "y_pred", estimator or fallback_model, representation)
    else:
        for selected_target in TARGETS:
            prefix = selected_target.removesuffix("_nm")
            truth_candidates = (f"true_{selected_target}", f"known_{selected_target}", f"true_{prefix}_nm", f"known_{prefix}_nm")
            truth = next((column for column in truth_candidates if column in frame), None)
            if truth is None:
                continue
            prediction_columns: list[tuple[str, str]] = []
            if f"hybrid_{selected_target}" in frame:
                prediction_columns.append((f"hybrid_{selected_target}", "hybrid_ensemble"))
            if target == selected_target and "hybrid_prediction" in frame:
                prediction_columns.append(("hybrid_prediction", "hybrid_ensemble"))
            suffix = f"_{selected_target}"
            for column in frame:
                if column.endswith(suffix) and not column.startswith(("true_", "known_", "hybrid_")):
                    prediction_columns.append((column, column.removesuffix(suffix)))
            ordinary = f"predicted_{selected_target}"
            if ordinary in frame:
                prediction_columns.append((ordinary, fallback_model))
            for prediction, model_name in dict(prediction_columns).items():
                append_rows(frame, selected_target, truth, prediction, model_name, fallback_representation)
    if not normalized:
        raise ValueError(f"{path}: no supported prediction-level wavelength schema was found")

    result = calculate_errors(pd.concat(normalized, ignore_index=True), definition)
    result["split_type"] = metadata.get("split_type") or fallback_split_type
    result["seed"] = metadata.get("seed") if metadata.get("seed") is not None else fallback_seed
    result["source_prediction_file"] = str(path.resolve())
    result["experiment"] = metadata["experiment"]
    if expected_mae is not None and len(result):
        actual = float(result["absolute_error"].mean())
        if not np.isclose(actual, expected_mae, rtol=1e-6, atol=1e-6):
            raise ValueError(f"{path}: prediction MAE {actual:.9g} does not match metrics.csv MAE {expected_mae:.9g}")
        notes.append(f"Verified overall MAE against metrics.csv for {path.name}.")
    return result[ERROR_COLUMNS], notes


def discover_prediction_files(inputs: Iterable[Path]) -> list[Path]:
    """Discover only recognized prediction files; never consume aggregate summaries."""
    files: list[Path] = []
    patterns = ("predictions/*.csv", "predictions_*.csv", "final_*predictions.csv", "final_*evaluated_predictions.csv")
    for item in inputs:
        if item.is_file():
            files.append(item)
        elif item.is_dir():
            for pattern in patterns:
                files.extend(item.rglob(pattern))
        else:
            raise FileNotFoundError(f"Input does not exist: {item}")
    unique = sorted({path.resolve() for path in files})
    if not unique:
        raise FileNotFoundError("No prediction-level CSV files were found in the explicit input path(s)")
    return unique


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return cleaned or "unspecified"


def _style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 15,
        "axes.titlesize": 14, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.8,
    })


def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, definition: RegionDefinition, sparse_threshold: int) -> None:
    order = list(definition.labels)
    counts = frame.groupby("spectral_region").size()
    dense_regions = [region for region in order if counts.get(region, 0) >= sparse_threshold]
    dense = frame[frame["spectral_region"].isin(dense_regions)]
    if not dense.empty:
        sns.violinplot(
            data=dense, x="spectral_region", y="absolute_error", hue="spectral_region", order=order,
            palette=REGION_COLORS, inner=None, cut=0, linewidth=0.9, saturation=0.85, ax=ax,
            legend=False,
        )
    sparse = frame[~frame["spectral_region"].isin(dense_regions)]
    if not sparse.empty:
        sns.stripplot(
            data=sparse, x="spectral_region", y="absolute_error", hue="spectral_region", order=order,
            palette=REGION_COLORS, jitter=0.13, size=5, alpha=0.8, ax=ax, legend=False,
        )
    top = max(float(frame["absolute_error"].max()), 1.0)
    for index, region in enumerate(order):
        values = frame.loc[frame["spectral_region"] == region, "absolute_error"]
        if values.empty:
            continue
        median, mean = float(values.median()), float(values.mean())
        ax.hlines(median, index - 0.13, index + 0.13, color="white", linewidth=4.0, zorder=5)
        ax.hlines(median, index - 0.13, index + 0.13, color="black", linewidth=1.6, zorder=6)
        ax.scatter(index, mean, marker="D", s=34, facecolor="black", edgecolor="white", linewidth=0.5, zorder=7)
        ax.text(index, top * 1.035, f"MAE {mean:.1f}\nn = {len(values):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, top * 1.20)
    ax.set_xlabel("Spectral Region")
    ax.set_ylabel("Absolute Prediction Error (nm)")
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    sns.despine(ax=ax)


def _title_for(frame: pd.DataFrame) -> str:
    target = str(frame["target"].iloc[0]).removesuffix("_nm").title()
    representation = str(frame["representation"].iloc[0])
    model = str(frame["model"].iloc[0])
    split = str(frame["split_type"].iloc[0])
    return f"{target} error by spectral region\n{representation}; estimator: {model}; {split} split"


def save_single_plot(frame: pd.DataFrame, definition: RegionDefinition, out_dir: Path, dpi: int, sparse_threshold: int) -> list[Path]:
    _style()
    fig, ax = plt.subplots(figsize=(8.0, 5.7), constrained_layout=True)
    _plot_panel(ax, frame, definition, sparse_threshold)
    ax.set_title(_title_for(frame), pad=12)
    handles = [
        Line2D([0], [0], color="black", linewidth=1.8, label="Median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="black", markersize=6, label="Mean (MAE)"),
    ]
    ax.legend(
        handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.19),
        ncol=2, borderaxespad=0,
    )
    stem = "__".join(_safe_name(frame[column].iloc[0]) for column in ("target", "experiment", "representation", "model", "split_type"))
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def save_panel_plot(
    frames: list[pd.DataFrame], definition: RegionDefinition, out_dir: Path,
    stem: str, heading: str, dpi: int, sparse_threshold: int,
    title_columns: tuple[str, str] = ("representation", "model"),
) -> list[Path]:
    _style()
    fig, axes = plt.subplots(1, len(frames), figsize=(7.0 * len(frames), 5.7), sharey=True, constrained_layout=True, squeeze=False)
    for index, (ax, frame) in enumerate(zip(axes[0], frames)):
        _plot_panel(ax, frame, definition, sparse_threshold)
        ax.set_title(f"{frame[title_columns[0]].iloc[0]}\n{frame[title_columns[1]].iloc[0]}", pad=10)
        if index:
            ax.set_ylabel("")
    fig.suptitle(heading, fontsize=15)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(paths[1], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _matched_model_comparisons(errors: pd.DataFrame) -> list[tuple[str, str, list[pd.DataFrame]]]:
    comparisons: list[tuple[str, str, list[pd.DataFrame]]] = []
    keys = ["target", "experiment", "split_type", "seed"]
    for values, group in errors.groupby(keys, dropna=False, sort=False):
        models = [part for _, part in group.groupby(["representation", "model"], dropna=False, sort=False)]
        if len(models) < 2:
            continue
        if any("implicit_row_index" in set(part["row_id_source"]) for part in models):
            continue
        if any(part["row_id"].duplicated().any() for part in models):
            continue
        id_sets = [set(part["row_id"].astype(str)) for part in models]
        if all(ids == id_sets[0] for ids in id_sets[1:]):
            target, experiment, split_type, seed = values
            stem = f"model_comparison__{_safe_name(target)}__{_safe_name(experiment)}__{_safe_name(split_type)}__seed_{_safe_name(seed)}"
            title = f"{str(target).removesuffix('_nm').title()} model comparison; {split_type} split; seed {seed}"
            comparisons.append((stem, title, models))
    return comparisons


def _split_comparisons(errors: pd.DataFrame) -> list[tuple[str, str, list[pd.DataFrame]]]:
    comparisons: list[tuple[str, str, list[pd.DataFrame]]] = []
    keys = ["target", "representation", "model", "seed"]
    for values, group in errors.groupby(keys, dropna=False, sort=False):
        panels = [part for _, part in group.groupby("split_type", dropna=False, sort=False)]
        if len(panels) < 2:
            continue
        target, representation, model, seed = values
        stem = f"split_comparison__{_safe_name(target)}__{_safe_name(representation)}__{_safe_name(model)}__seed_{_safe_name(seed)}"
        title = f"{str(target).removesuffix('_nm').title()} error across splitting strategies; seed {seed}"
        comparisons.append((stem, title, panels))
    return comparisons


def build_report(errors: pd.DataFrame, summary: pd.DataFrame, notes: list[str], skipped: list[str], generated: list[Path], sparse_threshold: int) -> str:
    lines = ["# Spectral-region prediction-error violin plots", "", "## Scope", ""]
    lines.append(f"Analyzed {len(errors):,} individual held-out predictions from {errors['source_prediction_file'].nunique()} file(s). Regions were assigned from experimental `y_true` values.")
    lines.extend(["", "## Findings", ""])
    for keys, group in summary.groupby(["target", "model", "representation", "split_type", "seed"], dropna=False):
        lowest = group.loc[group["mae"].idxmin()]
        highest = group.loc[group["mae"].idxmax()]
        lines.append(
            f"- `{keys[0]}` / `{keys[2]}` / `{keys[1]}` ({keys[3]}, seed {keys[4]}): "
            f"lowest regional MAE was {lowest['spectral_region']} ({lowest['mae']:.2f} nm; n={int(lowest['n'])}); "
            f"highest was {highest['spectral_region']} ({highest['mae']:.2f} nm; n={int(highest['n'])})."
        )
        sparse = group[group["n"] < sparse_threshold]
        if not sparse.empty:
            labels = ", ".join(f"{row.spectral_region} (n={int(row.n)})" for row in sparse.itertuples())
            lines.append(f"  Sparse regions (shown as observations without a KDE violin): {labels}.")
    lines.extend(["", "## Verification and limitations", ""])
    lines.extend(f"- {note}" for note in notes)
    lines.extend(f"- Skipped: {item}" for item in skipped)
    lines.append("- Model comparisons are generated only when stable row identifiers form identical test cohorts. Split comparisons use separate panels and do not pool distributions.")
    lines.append("- Regional differences are descriptive; no causal explanation is inferred.")
    lines.extend(["", "## Generated files", ""])
    lines.extend(f"- `{path.name}`" for path in generated)
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path, help="Prediction CSV file(s) or explicitly selected experiment directory/directories.")
    parser.add_argument("--target", choices=("absorption", "emission", "both"), default="both")
    parser.add_argument("--models", nargs="*", help="Keep only these estimator/model labels.")
    parser.add_argument("--representations", nargs="*", help="Keep only these exact representation labels.")
    parser.add_argument("--split-types", nargs="*", choices=("random", "molecule", "scaffold"), help="Keep only these splitting strategies.")
    parser.add_argument("--region-scheme", choices=("five", "three"), default="five")
    parser.add_argument("--boundaries", nargs="+", type=float, help="Custom increasing boundaries (four for five regions; two for three).")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/spectral_region_violin_plots"))
    parser.add_argument("--model", default="unspecified", help="Fallback model label when metadata is absent.")
    parser.add_argument("--representation", default="unspecified", help="Fallback representation label when metadata is absent.")
    parser.add_argument("--split-type", choices=("random", "molecule", "scaffold", "unspecified"), default="unspecified", help="Fallback split type when metadata is absent.")
    parser.add_argument("--seed", type=int, help="Fallback seed when metadata is absent.")
    parser.add_argument("--assume-held-out", action="store_true", help="Accept files without split provenance after the user has independently verified they contain only test rows.")
    parser.add_argument("--sparse-threshold", type=int, default=5, help="Minimum n for kernel-density violins; smaller groups are shown as points (default: 5).")
    parser.add_argument("--dpi", type=int, default=400, help="PNG resolution (default: 400).")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> list[Path]:
    definition = region_definition(args.region_scheme, args.boundaries)
    if args.dpi < 300:
        raise ValueError("--dpi must be at least 300 for publication output")
    if args.sparse_threshold < 2:
        raise ValueError("--sparse-threshold must be at least 2")
    paths = discover_prediction_files(args.input)
    tables: list[pd.DataFrame] = []
    notes: list[str] = []
    skipped: list[str] = []
    for path in paths:
        try:
            table, file_notes = load_prediction_file(
                path, definition, assume_held_out=args.assume_held_out,
                fallback_model=args.model, fallback_representation=args.representation,
                fallback_split_type=args.split_type, fallback_seed=args.seed,
            )
            tables.append(table)
            notes.extend(file_notes)
        except ValueError as exc:
            skipped.append(str(exc))
    if not tables:
        detail = "\n".join(f"- {item}" for item in skipped)
        raise ValueError(f"No valid held-out wavelength predictions were loaded.\n{detail}")
    errors = pd.concat(tables, ignore_index=True)
    wanted = TARGETS if args.target == "both" else (f"{args.target}_nm",)
    errors = errors[errors["target"].isin(wanted)]
    if args.models:
        errors = errors[errors["model"].isin(args.models)]
    if args.representations:
        errors = errors[errors["representation"].isin(args.representations)]
    if args.split_types:
        errors = errors[errors["split_type"].isin(args.split_types)]
    if errors.empty:
        raise ValueError("No prediction rows remain after applying the requested filters")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    errors = errors.sort_values(["target", "experiment", "representation", "model", "split_type", "spectral_region", "row_id"])
    summary = summarize_errors(errors, definition)
    error_path = args.out_dir / "spectral_region_absolute_errors.csv"
    summary_path = args.out_dir / "spectral_region_error_summary.csv"
    errors.to_csv(error_path, index=False)
    summary.to_csv(summary_path, index=False)
    generated: list[Path] = [error_path, summary_path]
    group_columns = ["target", "experiment", "representation", "model", "split_type", "seed"]
    for _, frame in errors.groupby(group_columns, dropna=False, sort=False):
        generated.extend(save_single_plot(frame, definition, args.out_dir, args.dpi, args.sparse_threshold))
    model_comparisons = _matched_model_comparisons(errors)
    for stem, title, panels in model_comparisons:
        generated.extend(save_panel_plot(panels, definition, args.out_dir, stem, title, args.dpi, args.sparse_threshold))
    if not model_comparisons:
        skipped.append("Figure C model comparison: no two models had the same target, split, seed, experiment, and identical stable row-ID cohort.")
    split_comparisons = _split_comparisons(errors)
    for stem, title, panels in split_comparisons:
        generated.extend(save_panel_plot(
            panels, definition, args.out_dir, stem, title, args.dpi,
            args.sparse_threshold, title_columns=("split_type", "experiment"),
        ))
    if not split_comparisons:
        skipped.append("Figure D split comparison: no model/representation/target/seed combination had prediction-level results for multiple split types.")
    for target in TARGETS:
        if target not in set(errors["target"]):
            skipped.append(f"{target} violin: no valid held-out prediction-level rows were available after filtering.")
    report_path = args.out_dir / "spectral_region_violin_report.md"
    report_path.write_text(build_report(errors, summary, notes, skipped, generated, args.sparse_threshold), encoding="utf-8")
    generated.append(report_path)
    return generated


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        generated = run(args)
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        try:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            failure_report = args.out_dir / "spectral_region_violin_report.md"
            listed_inputs = "\n".join(f"- `{path}`" for path in args.input)
            failure_report.write_text(
                "# Spectral-region prediction-error violin plots\n\n"
                "## Status\n\nNo figures or numerical summaries were generated because no valid "
                "held-out prediction-level wavelength records could be loaded. Aggregate MAE tables "
                "were not used to reconstruct distributions.\n\n"
                f"Error: `{exc}`\n\n## Explicit inputs searched\n\n{listed_inputs}\n",
                encoding="utf-8",
            )
            print(f"Wrote limitation report: {failure_report}", file=sys.stderr)
        except OSError:
            pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("Generated files:")
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
