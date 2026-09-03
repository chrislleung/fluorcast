from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]

for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_combined_predictors as base

from chemfluor.conforformer.downstream import (
    _feature_matrix,
    _feature_set_mask,
    _target_rows,
    build_feature_bundle,
)
from chemfluor.hybrid.ensemble import (
    _regression_pipeline,
    train_hybrid_ensemble,
)


MODELS = ("rf", "extratrees", "histgb", "gbdt", "mlp")
TARGETS = (
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "stokes_shift_nm",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--dataset-csv", type=Path)
    p.add_argument("--embedding-run-root", type=Path)
    p.add_argument("--solvent-descriptors", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def prediction_table(rows, predictions, target):
    keep = [
        "row_id",
        "canonical_chromophore_smiles",
        "canonical_solvent_smiles",
        "split",
    ]
    out = rows[keep].reset_index(drop=True).copy()
    out[f"true_{target}"] = rows[target].to_numpy(dtype=float)

    pred_cols = []

    for name, pred in predictions.items():
        col = f"{name}_{target}"
        out[col] = pred
        pred_cols.append(col)

    vals = out[pred_cols]

    out["prediction_mean"] = vals.mean(axis=1)
    out["prediction_std"] = vals.std(axis=1, ddof=0)
    out["prediction_min"] = vals.min(axis=1)
    out["prediction_max"] = vals.max(axis=1)
    out["prediction_range"] = out["prediction_max"] - out["prediction_min"]
    out["prediction_count"] = vals.notna().sum(axis=1)

    return out


def meta_features(table, target):
    cols = [
        *(f"{name}_{target}" for name in MODELS),
        "prediction_mean",
        "prediction_std",
        "prediction_min",
        "prediction_max",
        "prediction_range",
        "prediction_count",
    ]
    return table[cols].apply(pd.to_numeric, errors="coerce")


def metric_row(split_name, target, model, truth, prediction):
    return {
        "split": split_name,
        "target": target,
        "model": model,
        "count": int(len(truth)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
    }


def require_same_reference_rows(run_dir, target, final_rows):
    ref_path = (
        run_dir
        / "predictions"
        / f"{target}__not_applicable__morgan_solvent.csv"
    )

    if not ref_path.exists():
        raise FileNotFoundError(ref_path)

    ref = pd.read_csv(ref_path)

    a = final_rows[["row_id", target]].copy()
    b = ref[["row_id", "y_true"]].copy()

    a = a.sort_values("row_id").reset_index(drop=True)
    b = b.sort_values("row_id").reset_index(drop=True)

    if not a["row_id"].equals(b["row_id"]):
        raise RuntimeError(
            f"{target}: new Morgan ensemble final rows do not match "
            "existing CF benchmark rows"
        )

    if not np.allclose(
        a[target].to_numpy(dtype=float),
        b["y_true"].to_numpy(dtype=float),
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{target}: y_true differs from existing CF benchmark"
        )

    return len(ref)


def main():
    args = parse_args()

    run_dir = args.run_dir.resolve()

    split_type = run_dir.name.split("_seed", 1)[0]

    if split_type not in {"random", "molecule", "scaffold"}:
        raise ValueError(
            f"Could not infer split type from {run_dir.name}"
        )

    dataset_csv = (
        args.dataset_csv
        if args.dataset_csv
        else ROOT
        / "data"
        / "processed"
        / "fluodb_lite"
        / "combined_deduplicated.csv"
    )

    embedding_run_root = (
        args.embedding_run_root
        if args.embedding_run_root
        else ROOT
        / "outputs"
        / "conforformer"
        / "full_dataset"
        / "full_20260806T004836Z"
    )

    solvent_descriptors = (
        args.solvent_descriptors
        if args.solvent_descriptors
        else base.DEFAULT_SOLVENT_DESCRIPTORS
    )

    print("Dataset:", dataset_csv)
    print("Embedding root:", embedding_run_root)
    print("Solvent descriptors:", solvent_descriptors)
    print("Existing benchmark:", run_dir)
    print("Split type:", split_type)

    bundle, excluded, morgan_excluded = build_feature_bundle(
        dataset_csv=dataset_csv,
        embedding_run_root=embedding_run_root,
        solvent_descriptors=solvent_descriptors,
        n_bits=2048,
        radius=2,
    )

    saved = pd.read_csv(run_dir / "split_assignments.csv")

    if saved["row_id"].duplicated().any():
        raise RuntimeError("Duplicate row_id in saved split assignments")

    original_order = bundle.rows["row_id"].tolist()

    rows = bundle.rows.merge(
        saved[["row_id", "split"]],
        how="left",
        on="row_id",
        validate="one_to_one",
        sort=False,
    )

    if rows["row_id"].tolist() != original_order:
        raise RuntimeError("Row ordering changed while attaching splits")

    if rows["split"].isna().any():
        missing = int(rows["split"].isna().sum())
        raise RuntimeError(
            f"{missing} bundle rows have no saved split assignment"
        )

    bundle = replace(bundle, rows=rows)

    feature_mask = _feature_set_mask(bundle, "morgan_solvent")
    cohort_rows = bundle.rows.loc[feature_mask].copy()

    full_x = _feature_matrix(
        bundle,
        pooling="not_applicable",
        feature_set="morgan_solvent",
    )

    out_dir = run_dir / "morgan_ensemble_same_split"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []

    for target in TARGETS:
        print("\n" + "=" * 100)
        print(split_type.upper(), target)
        print("=" * 100)

        target_rows, _ = _target_rows(cohort_rows, target)

        idx = target_rows.index.to_numpy()
        y = pd.to_numeric(
            target_rows[target], errors="coerce"
        ).to_numpy(dtype=float)

        splits = target_rows["split"].to_numpy()

        base_mask = splits == "base_train"
        meta_mask = splits == "model_selection"
        final_mask = splits == "final_test"

        if not base_mask.any():
            raise RuntimeError(f"{target}: no base_train rows")
        if not meta_mask.any():
            raise RuntimeError(f"{target}: no model_selection rows")
        if not final_mask.any():
            raise RuntimeError(f"{target}: no final_test rows")

        final_rows = target_rows.loc[final_mask].copy()

        expected_n = require_same_reference_rows(
            run_dir,
            target,
            final_rows,
        )

        print(
            f"Rows: base={base_mask.sum()} "
            f"meta={meta_mask.sum()} "
            f"final={final_mask.sum()}"
        )
        print(
            f"Existing CF final-test reference: n={expected_n} — PASS"
        )

        imputer = SimpleImputer(strategy="median")

        x_base = imputer.fit_transform(
            full_x[idx[base_mask]]
        )
        x_meta = imputer.transform(
            full_x[idx[meta_mask]]
        )
        x_final = imputer.transform(
            full_x[idx[final_mask]]
        )

        y_base = y[base_mask]

        fitted = {}

        for name in MODELS:
            print("Training base model:", name)

            model = base.make_model(
                name,
                random_state=args.seed,
                n_jobs=args.n_jobs,
            )

            model.fit(x_base, y_base)
            fitted[name] = model

        meta_predictions = {
            name: model.predict(x_meta)
            for name, model in fitted.items()
        }

        final_predictions = {
            name: model.predict(x_final)
            for name, model in fitted.items()
        }

        meta_table = prediction_table(
            target_rows.loc[meta_mask],
            meta_predictions,
            target,
        )

        final_table = prediction_table(
            final_rows,
            final_predictions,
            target,
        )

        meta_x = meta_features(meta_table, target)
        meta_y = meta_table[f"true_{target}"].astype(float)

        train_idx, calibration_idx = train_test_split(
            np.arange(len(meta_table)),
            test_size=max(
                2,
                int(np.ceil(0.2 * len(meta_table))),
            ),
            random_state=args.seed,
        )

        if target == "stokes_shift_nm":
            # Same regression architecture used by the FluorCast hybrid
            # ensemble. Stokes simply does not need the QY classifier.
            hybrid_regressor = _regression_pipeline().fit(
                meta_x.iloc[train_idx],
                meta_y.iloc[train_idx],
            )
            hybrid_object = {
                "target_name": target,
                "regressor": hybrid_regressor,
                "classifier": None,
            }
        else:
            hybrid_object = train_hybrid_ensemble(
                meta_x.iloc[train_idx],
                meta_y.iloc[train_idx],
                target,
                bright_threshold=0.25,
            )
            hybrid_regressor = hybrid_object["regressor"]

        final_x = meta_features(final_table, target)

        hybrid_prediction = hybrid_regressor.predict(final_x)

        final_table["hybrid_prediction"] = hybrid_prediction

        target_dir = out_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        meta_table.to_csv(
            target_dir / "base_model_predictions_meta_train.csv",
            index=False,
        )

        final_table.to_csv(
            target_dir / "final_evaluated_predictions.csv",
            index=False,
        )

        joblib.dump(
            {
                "imputer": imputer,
                "base_models": fitted,
                "hybrid": hybrid_object,
                "meta_training_indices": train_idx,
                "calibration_indices": calibration_idx,
            },
            target_dir / "ensemble.joblib",
        )

        truth = final_table[f"true_{target}"].to_numpy(dtype=float)

        rows_for_target = []

        for name in MODELS:
            row = metric_row(
                split_type,
                target,
                name,
                truth,
                final_table[f"{name}_{target}"].to_numpy(dtype=float),
            )
            rows_for_target.append(row)

        rows_for_target.append(
            metric_row(
                split_type,
                target,
                "prediction_mean",
                truth,
                final_table["prediction_mean"].to_numpy(dtype=float),
            )
        )

        rows_for_target.append(
            metric_row(
                split_type,
                target,
                "hybrid_ensemble",
                truth,
                hybrid_prediction,
            )
        )

        metrics = pd.DataFrame(rows_for_target)

        metrics.to_csv(
            target_dir / "metrics_table.csv",
            index=False,
        )

        print(metrics.to_string(index=False))

        best_solo = (
            metrics[
                metrics["model"].isin(MODELS)
            ]
            .sort_values("mae")
            .iloc[0]
        )

        hybrid_row = metrics[
            metrics["model"] == "hybrid_ensemble"
        ].iloc[0]

        print()
        print(
            f"BEST SOLO: {best_solo['model']} "
            f"MAE={best_solo['mae']:.6f}"
        )
        print(
            f"HYBRID: MAE={hybrid_row['mae']:.6f} "
            f"RMSE={hybrid_row['rmse']:.6f} "
            f"R2={hybrid_row['r2']:.6f}"
        )

        if int(hybrid_row["count"]) != expected_n:
            raise RuntimeError(
                f"{target}: hybrid N={hybrid_row['count']} "
                f"but reference N={expected_n}"
            )

        all_metrics.extend(rows_for_target)

    combined = pd.DataFrame(all_metrics)

    combined.to_csv(
        out_dir / "metrics_all_targets.csv",
        index=False,
    )

    summary_rows = []

    for target in TARGETS:
        part = combined[combined["target"] == target]

        solo = (
            part[part["model"].isin(MODELS)]
            .sort_values("mae")
            .iloc[0]
        )

        hybrid = part[
            part["model"] == "hybrid_ensemble"
        ].iloc[0]

        summary_rows.append(
            {
                "split": split_type,
                "target": target,
                "best_morgan_solo_model": solo["model"],
                "best_morgan_solo_n": int(solo["count"]),
                "best_morgan_solo_mae": solo["mae"],
                "best_morgan_solo_rmse": solo["rmse"],
                "best_morgan_solo_r2": solo["r2"],
                "morgan_hybrid_n": int(hybrid["count"]),
                "morgan_hybrid_mae": hybrid["mae"],
                "morgan_hybrid_rmse": hybrid["rmse"],
                "morgan_hybrid_r2": hybrid["r2"],
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print(summary.to_string(index=False))

    print("\nALL SAME-ROW VALIDATION CHECKS PASSED")


if __name__ == "__main__":
    main()
