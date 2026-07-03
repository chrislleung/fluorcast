from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


MODEL_TOKENS = [
    "graph_mpnn",
    "graph_gin",
    "graph_gcn",
    "extratrees",
    "extra_trees",
    "lightgbm",
    "xgboost",
    "histgb",
    "gbdt",
    "mlp",
    "rf",
    "random_forest",
]

OPTIONAL_DIAGNOSTIC_COLUMNS = [
    "nearest_training_similarity",
    "molecule_seen_score",
    "solvent_seen_score",
    "pair_seen_score",
    "label_consistency_score",
    "model_agreement_score",
    "overall_confidence_score",
    "outside_applicability_domain",
]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _find_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_norm(c): c for c in columns}

    for candidate in candidates:
        key = _norm(candidate)
        if key in normalized:
            return normalized[key]

    return None


def _find_prediction_column(columns: list[str], target_name: str) -> str | None:
    target = _norm(target_name)

    direct_candidates = [
        f"predicted_{target}",
        f"{target}_predicted",
        f"pred_{target}",
        f"{target}_pred",
        "y_pred",
        "prediction",
        "predicted",
        "pred",
        "prediction_value",
        "predicted_value",
    ]

    found = _find_column(columns, direct_candidates)
    if found:
        return found

    for col in columns:
        n = _norm(col)
        if "pred" in n and target in n:
            return col

    return None


def _find_true_column(columns: list[str], target_name: str) -> str | None:
    target = _norm(target_name)

    direct_candidates = [
        f"true_{target}",
        f"{target}_true",
        f"actual_{target}",
        f"{target}_actual",
        "y_true",
        "actual",
        "actual_value",
        "target",
        "label",
        target,
    ]

    found = _find_column(columns, direct_candidates)
    if found:
        return found

    for col in columns:
        n = _norm(col)
        if ("true" in n or "actual" in n) and target in n:
            return col

    return None


def infer_model_name(path: Path) -> str:
    stem = path.stem.lower()

    if "__" in stem:
        parts = stem.split("__")
        if len(parts) >= 2:
            return _norm(parts[1]).replace("extra_trees", "extratrees").replace("random_forest", "rf")

    for token in MODEL_TOKENS:
        if re.search(rf"(^|[_\-]){re.escape(token)}($|[_\-])", stem):
            return _norm(token).replace("extra_trees", "extratrees").replace("random_forest", "rf")

    return _norm(stem)


def find_prediction_files(
    prediction_dir: Path,
    target_name: str,
    split: str | None,
    seed: str | None,
) -> list[Path]:
    if not prediction_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {prediction_dir}")

    files = sorted(prediction_dir.rglob("*.csv"))
    target_key = _norm(target_name)

    matched: list[Path] = []
    for path in files:
        stem = _norm(path.stem)

        if target_key not in stem:
            continue

        if split and _norm(split) not in stem:
            continue

        if seed is not None:
            seed_patterns = [
                f"seed{seed}",
                f"seed_{seed}",
                f"seed-{seed}",
            ]
            if not any(pattern in stem for pattern in seed_patterns):
                continue

        matched.append(path)

    if not matched:
        raise FileNotFoundError(
            f"No prediction CSVs found in {prediction_dir} for target={target_name}, "
            f"split={split}, seed={seed}."
        )

    return matched


def load_prediction_file(path: Path, target_name: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Prediction file is empty: {path}")

    columns = list(raw.columns)

    true_col = _find_true_column(columns, target_name)
    pred_col = _find_prediction_column(columns, target_name)

    if true_col is None:
        raise ValueError(
            f"Could not detect true target column in {path}. "
            f"Available columns: {columns}"
        )

    if pred_col is None:
        raise ValueError(
            f"Could not detect prediction column in {path}. "
            f"Available columns: {columns}"
        )

    row_id_col = _find_column(
        columns,
        ["row_id", "test_index", "original_index", "index", "idx"],
    )

    molecule_col = _find_column(
        columns,
        [
            "canonical_smiles",
            "canonical_molecule_smiles",
            "smiles",
            "molecule_smiles",
            "chromophore_smiles",
            "chromophore",
        ],
    )

    solvent_col = _find_column(
        columns,
        [
            "solvent",
            "solvent_smiles",
            "canonical_solvent_smiles",
            "solvent_name",
        ],
    )

    model_name = infer_model_name(path)
    target_key = _norm(target_name)
    pred_out_col = f"{model_name}_{target_key}"

    out = pd.DataFrame()
    out["__row_order"] = np.arange(len(raw))

    if row_id_col:
        out["row_id"] = raw[row_id_col]

    if molecule_col:
        out["canonical_smiles"] = raw[molecule_col]

    if solvent_col:
        out["solvent"] = raw[solvent_col]

    out[f"true_{target_key}"] = pd.to_numeric(raw[true_col], errors="coerce")
    out[pred_out_col] = pd.to_numeric(raw[pred_col], errors="coerce")

    for optional_col in OPTIONAL_DIAGNOSTIC_COLUMNS:
        existing = _find_column(columns, [optional_col])
        if existing:
            out[optional_col] = raw[existing]

    out = out.dropna(subset=[f"true_{target_key}", pred_out_col]).reset_index(drop=True)
    out["__source_file"] = str(path)

    return out


def choose_merge_keys(tables: list[pd.DataFrame]) -> list[str]:
    if all("row_id" in table.columns for table in tables):
        return ["row_id"]

    if all({"canonical_smiles", "solvent"}.issubset(table.columns) for table in tables):
        return ["canonical_smiles", "solvent"]

    row_counts = {len(table) for table in tables}
    if len(row_counts) == 1:
        return ["__row_order"]

    raise ValueError(
        "Could not determine safe merge keys. Need row_id/test_index, "
        "canonical_smiles + solvent, or identical row counts for row-order merge."
    )


def merge_prediction_tables(tables: list[pd.DataFrame], target_name: str) -> pd.DataFrame:
    target_key = _norm(target_name)
    true_col = f"true_{target_key}"
    merge_keys = choose_merge_keys(tables)

    base_columns = list(dict.fromkeys(merge_keys + [true_col]))

    for col in ["canonical_smiles", "solvent"]:
        if col not in base_columns and col in tables[0].columns:
            base_columns.append(col)

    for optional_col in OPTIONAL_DIAGNOSTIC_COLUMNS:
        if optional_col in tables[0].columns:
            base_columns.append(optional_col)

    merged = tables[0][base_columns].copy()

    prediction_columns: list[str] = []

    for table in tables:
        model_pred_cols = [
            col
            for col in table.columns
            if col.endswith(f"_{target_key}")
            and col != true_col
        ]

        if not model_pred_cols:
            continue

        pred_col = model_pred_cols[0]
        prediction_columns.append(pred_col)

        add_cols = merge_keys + [pred_col]
        merged = merged.merge(
            table[add_cols],
            on=merge_keys,
            how="inner",
            validate="one_to_one",
        )

    prediction_columns = [col for col in prediction_columns if col in merged.columns]

    if not prediction_columns:
        raise ValueError("No model prediction columns were produced after merging.")

    preds = merged[prediction_columns]

    merged["prediction_mean"] = preds.mean(axis=1)
    merged["prediction_std"] = preds.std(axis=1).fillna(0.0)
    merged["prediction_min"] = preds.min(axis=1)
    merged["prediction_max"] = preds.max(axis=1)
    merged["prediction_range"] = merged["prediction_max"] - merged["prediction_min"]
    merged["prediction_count"] = preds.notna().sum(axis=1)

    if "__row_order" in merged.columns:
        merged = merged.drop(columns=["__row_order"])

    return merged.reset_index(drop=True)


def split_train_validation(
    table: pd.DataFrame,
    validation_fraction: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1.")

    if "canonical_smiles" in table.columns and table["canonical_smiles"].nunique() > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction,
            random_state=random_state,
        )
        train_idx, val_idx = next(
            splitter.split(table, groups=table["canonical_smiles"])
        )
        train = table.iloc[train_idx].copy()
        val = table.iloc[val_idx].copy()
    else:
        train, val = train_test_split(
            table,
            test_size=validation_fraction,
            random_state=random_state,
            shuffle=True,
        )

    return train.reset_index(drop=True), val.reset_index(drop=True)


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a hybrid ensemble meta-training table from per-model prediction CSVs."
    )
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--target-name", required=True, choices=["emission_nm", "quantum_yield", "absorption_nm"])
    parser.add_argument("--split", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--train-out", default=None)
    parser.add_argument("--validation-out", default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=0)

    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir)
    files = find_prediction_files(
        prediction_dir=prediction_dir,
        target_name=args.target_name,
        split=args.split,
        seed=args.seed,
    )

    print("Found prediction files:")
    for path in files:
        print(f"  - {path}")

    tables = [load_prediction_file(path, args.target_name) for path in files]
    merged = merge_prediction_tables(tables, args.target_name)

    if args.out:
        write_csv(merged, Path(args.out))
        print(f"Wrote merged table: {args.out}")

    if args.train_out and args.validation_out:
        train, val = split_train_validation(
            merged,
            validation_fraction=args.validation_fraction,
            random_state=args.random_state,
        )
        write_csv(train, Path(args.train_out))
        write_csv(val, Path(args.validation_out))
        print(f"Wrote training table: {args.train_out} rows={len(train)}")
        print(f"Wrote validation table: {args.validation_out} rows={len(val)}")

    if not args.out and not (args.train_out and args.validation_out):
        raise SystemExit("Provide --out or both --train-out and --validation-out.")

    print(f"Merged shape: {merged.shape}")
    print("Columns:")
    for col in merged.columns:
        print(f"  - {col}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())