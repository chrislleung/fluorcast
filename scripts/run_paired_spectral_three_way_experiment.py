"""Run a shared-split absorption/emission hybrid experiment and calculate Stokes shift."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import build_paired_stokes_dataset as builder  # noqa: E402
import run_hybrid_three_way_experiment as shared  # noqa: E402
import train_combined_predictors as base  # noqa: E402
from chemfluor.hybrid.ensemble import align_features, save_hybrid_ensemble, train_hybrid_ensemble  # noqa: E402
from chemfluor.hybrid.uncertainty import calibration_residuals  # noqa: E402

SPLITS = shared.SPLITS
MODELS = shared.MODELS
TARGETS = ("absorption_nm", "emission_nm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-type", required=True, choices=("random", "molecule", "scaffold"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", required=True, choices=MODELS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-out-dir", required=True, type=Path)
    parser.add_argument("--paired-dataset", type=Path)
    parser.add_argument("--standardized-combined", type=Path)
    parser.add_argument("--solvent-descriptors", type=Path, default=base.DEFAULT_SOLVENT_DESCRIPTORS)
    parser.add_argument("--base-train-fraction", type=float, default=.60)
    parser.add_argument("--meta-train-fraction", type=float, default=.20)
    parser.add_argument("--final-test-fraction", type=float, default=.20)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--invalid-smiles-policy", choices=("drop", "keep-invalid-group"), default="drop")
    return parser.parse_args(argv)


def normalize_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    aliases = {
        "canonical_chromophore_smiles": ("canonical_molecule_smiles", "molecule_smiles", "chromophore_smiles"),
        "canonical_solvent_smiles": ("solvent_smiles",),
        "solvent_original": ("solvent", "solvent_name"),
    }
    for destination, sources in aliases.items():
        if destination not in result:
            source = next((name for name in sources if name in result), None)
            result[destination] = result[source] if source else pd.NA
    if "row_id" not in result:
        result.insert(0, "row_id", np.arange(len(result), dtype=int))
    if result["row_id"].isna().any() or result["row_id"].duplicated().any():
        raise ValueError("row_id must be non-null and unique")
    return result


def load_paired(args: argparse.Namespace) -> tuple[pd.DataFrame, str, int]:
    if args.paired_dataset:
        if not args.paired_dataset.exists():
            raise FileNotFoundError(f"Paired dataset not found: {args.paired_dataset}")
        raw = pd.read_csv(args.paired_dataset, low_memory=False)
        source = str(args.paired_dataset)
    else:
        raw, source = builder.load_source(args.standardized_combined)
    paired, _ = builder.build_paired_dataset(raw, args.max_rows, args.seed)
    return normalize_columns(paired), source, len(paired)


def join_final_predictions(absorption: pd.DataFrame, emission: pd.DataFrame) -> pd.DataFrame:
    """Join paired target predictions by stable row_id, never by row order."""
    absorb = absorption.rename(columns={"hybrid_prediction": "hybrid_absorption_nm"})
    emit = emission.rename(columns={"hybrid_prediction": "hybrid_emission_nm"})
    identity = [column for column in ("row_id", "canonical_chromophore_smiles", "canonical_solvent_smiles", "solvent_original") if column in absorb]
    left = absorb[[*identity, "true_absorption_nm", "hybrid_absorption_nm"]]
    right = emit[["row_id", "true_emission_nm", "hybrid_emission_nm"]]
    joined = left.merge(right, on="row_id", how="inner", validate="one_to_one")
    if len(joined) != len(absorb) or len(joined) != len(emit):
        raise ValueError("Absorption and emission final predictions do not contain identical row_id sets")
    joined["true_stokes_shift_nm"] = joined["true_emission_nm"] - joined["true_absorption_nm"]
    joined["predicted_stokes_shift_nm"] = joined["hybrid_emission_nm"] - joined["hybrid_absorption_nm"]
    joined["true_stokes_shift_cm^-1"] = 1e7 / joined["true_absorption_nm"] - 1e7 / joined["true_emission_nm"]
    joined["predicted_stokes_shift_cm^-1"] = 1e7 / joined["hybrid_absorption_nm"] - 1e7 / joined["hybrid_emission_nm"]
    joined["predicted_physically_valid_stokes"] = joined["hybrid_emission_nm"] > joined["hybrid_absorption_nm"]
    joined["true_physically_valid_stokes"] = joined["true_emission_nm"] > joined["true_absorption_nm"]
    return joined


def stokes_metrics(final: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit in ("nm", "cm^-1"):
        row = shared.metrics_row(
            f"calculated_stokes_shift_{unit}",
            final[f"true_stokes_shift_{unit}"].to_numpy(dtype=float),
            final[f"predicted_stokes_shift_{unit}"].to_numpy(dtype=float),
            "stokes_shift",
        )
        row["unit"] = unit
        row["predicted_negative_or_zero_fraction"] = float((final["predicted_stokes_shift_nm"] <= 0).mean())
        row["true_negative_or_zero_fraction"] = float((final["true_stokes_shift_nm"] <= 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _train_target(target: str, featured: pd.DataFrame, x: np.ndarray, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = featured["split"] == SPLITS[0]
    fitted = {}
    for name in args.models:
        model = base.make_model(name, random_state=args.seed, n_jobs=args.n_jobs)
        model.fit(x[train_mask.to_numpy()], featured.loc[train_mask, target].to_numpy(dtype=float))
        model_dir = args.model_out_dir / "base_models" / target / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        _json(model_dir / "metadata.json", {"model": name, "target": target, "training_split": SPLITS[0], "training_row_ids": featured.loc[train_mask, "row_id"].tolist(), "feature_count": int(x.shape[1]), "n_training_rows": int(train_mask.sum())})
        fitted[name] = model
    prefix = target.removesuffix("_nm")
    tables = {}
    for split, suffix in ((SPLITS[1], "meta_train"), (SPLITS[2], "final_test")):
        mask = featured["split"] == split
        table = shared.prediction_table(featured.loc[mask], {name: model.predict(x[mask.to_numpy()]) for name, model in fitted.items()}, target)
        table.to_csv(args.out_dir / f"{prefix}_base_model_predictions_{suffix}.csv", index=False)
        tables[split] = table
    meta = tables[SPLITS[1]]
    features = shared.meta_features(meta, target, args.models)
    if len(meta) < 5:
        raise ValueError("At least five hybrid_meta_train rows are required")
    train_idx, calibration_idx = train_test_split(np.arange(len(meta)), test_size=max(2, int(np.ceil(.2 * len(meta)))), random_state=args.seed)
    labels = meta[f"true_{target}"].astype(float)
    hybrid = train_hybrid_ensemble(features.iloc[train_idx], labels.iloc[train_idx], target, bright_threshold=.25)
    hybrid_dir = args.model_out_dir / "hybrid_ensemble" / target
    save_hybrid_ensemble(hybrid, list(features.columns), hybrid_dir, {"training_split": SPLITS[1], "training_row_ids": meta.iloc[train_idx]["row_id"].tolist(), "calibration_row_ids": meta.iloc[calibration_idx]["row_id"].tolist(), "n_training_examples": len(train_idx)})
    calibration_residuals(labels.iloc[calibration_idx], hybrid["regressor"].predict(features.iloc[calibration_idx])).to_csv(hybrid_dir / "calibration_residuals.csv", index=False)
    final = tables[SPLITS[2]].copy()
    final_features = align_features(shared.meta_features(final, target, args.models), list(features.columns))
    final["hybrid_prediction"] = hybrid["regressor"].predict(final_features)
    final.to_csv(args.out_dir / f"final_{prefix}_evaluated_predictions.csv", index=False)
    truth = final[f"true_{target}"].to_numpy(dtype=float)
    metrics = pd.DataFrame([
        *[shared.metrics_row(name, truth, final[f"{name}_{target}"].to_numpy(dtype=float), target) for name in args.models],
        shared.metrics_row("prediction_mean", truth, final["prediction_mean"].to_numpy(dtype=float), target),
        shared.metrics_row("hybrid_ensemble", truth, final["hybrid_prediction"].to_numpy(dtype=float), target),
    ])
    metrics.to_csv(args.out_dir / f"{prefix}_metrics_table.csv", index=False)
    return final, metrics


def run(args: argparse.Namespace) -> None:
    fractions = (args.base_train_fraction, args.meta_train_fraction, args.final_test_fraction)
    if any(value <= 0 for value in fractions) or not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    rows, dataset_path, total_paired = load_paired(args)
    rows, invalid_counts = shared.prepare_split_identifiers(rows, args.split_type, args.invalid_smiles_policy, args.out_dir)
    rows["split"] = shared.assign_three_way_splits(rows, args.split_type, fractions, args.seed)
    if rows["split"].isna().any() or any((rows["split"] == name).sum() == 0 for name in SPLITS):
        raise ValueError("Three-way split produced an empty or unassigned split; use more rows/groups")
    leak = shared.leakage_report(rows, args.split_type)
    leak.update({"invalid_row_counts": invalid_counts, "invalid_smiles_policy": args.invalid_smiles_policy})
    _json(args.out_dir / "leakage_check.json", leak)
    if leak["leakage_detected"]:
        raise RuntimeError(f"Leakage detected: {leak['overlap_counts']}")
    rows.to_csv(args.out_dir / "split_assignments.csv", index=False)

    descriptors = base.load_solvent_descriptors(args.solvent_descriptors)
    rows["experiment_row_id"] = rows["row_id"]
    featured, descriptor_columns = base.merge_solvent_descriptors(rows, descriptors)
    featured["row_id"] = featured.pop("experiment_row_id")
    fingerprints = shared.safe_fingerprints(featured)
    train_mask = featured["split"] == SPLITS[0]
    descriptor_values = featured[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    x = base.build_feature_matrix(fingerprints, descriptor_values, descriptor_values.loc[train_mask].median(numeric_only=True))
    absorption, absorption_metrics = _train_target("absorption_nm", featured, x, args)
    emission, emission_metrics = _train_target("emission_nm", featured, x, args)
    final = join_final_predictions(absorption, emission)
    final.to_csv(args.out_dir / "final_paired_spectral_predictions.csv", index=False)
    stokes = stokes_metrics(final)
    stokes.to_csv(args.out_dir / "stokes_metrics_table.csv", index=False)

    counts = {name: int((featured["split"] == name).sum()) for name in SPLITS}
    unique_molecules = {name: int(featured.loc[featured["split"] == name, "canonical_chromophore_smiles"].nunique()) for name in SPLITS}
    best_abs = absorption_metrics[absorption_metrics["model"].isin(args.models)].sort_values("MAE").iloc[0]
    best_em = emission_metrics[emission_metrics["model"].isin(args.models)].sort_values("MAE").iloc[0]
    hybrid_abs = absorption_metrics[absorption_metrics["model"] == "hybrid_ensemble"].iloc[0]
    hybrid_em = emission_metrics[emission_metrics["model"] == "hybrid_ensemble"].iloc[0]
    nm = stokes[stokes["unit"] == "nm"].iloc[0]
    cm = stokes[stokes["unit"] == "cm^-1"].iloc[0]
    scaffold_line = ""
    if args.split_type == "scaffold":
        scaffold_line = f"\n- Unique scaffolds: { {name: int(featured.loc[featured['split'] == name, 'scaffold'].nunique()) for name in SPLITS} }"
    summary = f"""# Paired Spectral Three-way Results

- Split type: {args.split_type}
- Seed: {args.seed}
- Models: {', '.join(args.models)}
- Split row counts: {counts}
- Unique molecules: {unique_molecules}{scaffold_line}
- Best absorption base model by MAE: {best_abs['model']} ({best_abs['MAE']:.6f})
- Absorption hybrid MAE / RMSE / R2: {hybrid_abs['MAE']:.6f} / {hybrid_abs['RMSE']:.6f} / {hybrid_abs['R2']:.6f}
- Absorption hybrid improved over best base: {'yes' if hybrid_abs['MAE'] < best_abs['MAE'] else 'no'}
- Best emission base model by MAE: {best_em['model']} ({best_em['MAE']:.6f})
- Emission hybrid MAE / RMSE / R2: {hybrid_em['MAE']:.6f} / {hybrid_em['RMSE']:.6f} / {hybrid_em['R2']:.6f}
- Emission hybrid improved over best base: {'yes' if hybrid_em['MAE'] < best_em['MAE'] else 'no'}
- Stokes-shift MAE: {nm['MAE']:.6f} nm
- Stokes-shift MAE: {cm['MAE']:.6f} cm^-1
- Fraction of predicted invalid Stokes shifts: {nm['predicted_negative_or_zero_fraction']:.6f}
- Leakage detected: {leak['leakage_detected']}

Stokes shift was calculated from paired absorption and emission predictions; it was not directly modeled. All metrics use only `final_test` rows.
"""
    (args.out_dir / "paired_spectral_metrics_summary.md").write_text(summary, encoding="utf-8")
    _json(args.out_dir / "experiment_config.json", {"split_type": args.split_type, "seed": args.seed, "fractions": dict(zip(SPLITS, fractions)), "models": args.models, "dataset_path": dataset_path, "total_paired_rows": total_paired, "rows_retained_after_invalid_handling": len(featured), "split_row_counts": counts, "invalid_row_counts": invalid_counts, "invalid_smiles_policy": args.invalid_smiles_policy, "timestamp": datetime.now(timezone.utc).isoformat(), "feature_columns": [*[f"fingerprint_{i}" for i in range(2048)], *descriptor_columns], "package_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__}})


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
