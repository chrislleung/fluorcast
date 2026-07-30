"""Downstream training on finalized ConforFormer embeddings."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .embedding_store import EXPECTED_EMBEDDING_DIM
from .inventory import atomic_write_text, fluorcast_git_commit, sha256_file

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover
    Chem = None
    DataStructs = None
    AllChem = None
    MurckoScaffold = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")


TARGETS = ["absorption_nm", "emission_nm", "quantum_yield", "stokes_shift_nm"]
POOLING_METHODS = ["mean", "lowest_energy", "boltzmann_298k"]
FEATURE_SETS = ["conforformer_solvent", "morgan_solvent", "conforformer_morgan_solvent"]
TARGET_COLUMNS = set(TARGETS) | {"stokes_shift_cm^-1"}
IDENTITY_DESCRIPTOR_COLUMNS = {
    "solvent",
    "solvent_original",
    "canonical_smiles",
    "canonical_solvent_smiles",
    "is_valid_rdkit",
    "is_environment_label",
    "deep4chem_row_count",
    "existing_solvent_match",
    "existing_canonical_solvent_smiles",
    "descriptor_canonical_key",
    "descriptor_solvent_key",
}


@dataclass(frozen=True)
class FeatureBundle:
    rows: pd.DataFrame
    embeddings_by_pooling: dict[str, np.ndarray]
    morgan: np.ndarray
    morgan_valid: np.ndarray
    solvent_values: pd.DataFrame
    solvent_columns: list[str]


def require_rdkit() -> None:
    if Chem is None or DataStructs is None or AllChem is None:
        raise ImportError("RDKit is required for Morgan fingerprints and scaffold splits") from _RDKIT_IMPORT_ERROR


def load_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_solvent_descriptors(path: Path | str) -> pd.DataFrame:
    descriptors = pd.read_csv(path, low_memory=False)
    if "canonical_solvent_smiles" not in descriptors.columns:
        descriptors["canonical_solvent_smiles"] = descriptors.get("canonical_smiles", pd.NA)
    if "solvent_original" not in descriptors.columns:
        descriptors["solvent_original"] = descriptors.get("solvent", pd.NA)
    descriptors["descriptor_canonical_key"] = descriptors["canonical_solvent_smiles"].astype("string")
    descriptors["descriptor_solvent_key"] = descriptors["solvent_original"].astype("string").str.lower()
    return descriptors


def choose_solvent_descriptor_columns(descriptors: pd.DataFrame) -> list[str]:
    excluded = IDENTITY_DESCRIPTOR_COLUMNS | TARGET_COLUMNS
    columns: list[str] = []
    for column in descriptors.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(descriptors[column], errors="coerce")
        if numeric.notna().any():
            descriptors[column] = numeric
            columns.append(column)
    return columns


def merge_solvent_descriptors(rows: pd.DataFrame, descriptors: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    descriptor_columns = choose_solvent_descriptor_columns(descriptors)
    canonical = descriptors.dropna(subset=["descriptor_canonical_key"]).drop_duplicates(
        "descriptor_canonical_key", keep="first"
    )[["descriptor_canonical_key", *descriptor_columns]]
    labels = descriptors.dropna(subset=["descriptor_solvent_key"]).drop_duplicates(
        "descriptor_solvent_key", keep="first"
    )[["descriptor_solvent_key", *descriptor_columns]]
    merged = rows.copy()
    merged["_descriptor_merge_row"] = np.arange(len(merged))
    merged["descriptor_canonical_key"] = merged["canonical_solvent_smiles"].astype("string")
    merged["descriptor_solvent_key"] = merged.get("solvent_original", pd.Series(pd.NA, index=merged.index)).astype("string").str.lower()
    merged = merged.merge(canonical, how="left", on="descriptor_canonical_key")
    unmatched = merged[descriptor_columns].isna().all(axis=1) if descriptor_columns else pd.Series(False, index=merged.index)
    if unmatched.any():
        fallback = merged.loc[unmatched, ["_descriptor_merge_row", "descriptor_solvent_key"]].merge(labels, how="left", on="descriptor_solvent_key").set_index("_descriptor_merge_row")
        for column in descriptor_columns:
            merged.loc[unmatched, column] = merged.loc[unmatched, "_descriptor_merge_row"].map(fallback[column])
    return merged.drop(columns=["_descriptor_merge_row", "descriptor_canonical_key", "descriptor_solvent_key"]), descriptor_columns


def morgan_fingerprint(smiles: str, *, radius: int = 2, n_bits: int = 2048) -> np.ndarray | None:
    require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    bit_vector = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    array = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bit_vector, array)
    return array


def scaffold_for_smiles(smiles: str) -> str:
    require_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _assign_groups(groups: pd.Series, *, seed: int) -> dict[str, str]:
    counts = groups.value_counts().to_dict()
    ordered = sorted(counts)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: counts[group], reverse=True)
    total = int(sum(counts.values()))
    targets = {"base_train": 0.6 * total, "model_selection": 0.2 * total, "final_test": 0.2 * total}
    split_counts = {"base_train": 0, "model_selection": 0, "final_test": 0}
    assignment: dict[str, str] = {}
    for group in ordered:
        split = min(split_counts, key=lambda name: (split_counts[name] / targets[name], split_counts[name]))
        assignment[group] = split
        split_counts[split] += int(counts[group])
    return assignment


def make_split_assignments(rows: pd.DataFrame, *, split_type: str = "molecule", seed: int = 0) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split_type not in {"molecule", "scaffold"}:
        raise ValueError("split_type must be molecule or scaffold")
    frame = rows.copy()
    if split_type == "molecule":
        frame["split_group"] = frame["canonical_chromophore_smiles"].astype(str)
    else:
        frame["split_group"] = frame["canonical_chromophore_smiles"].map(scaffold_for_smiles)
    mapping = _assign_groups(frame["split_group"], seed=seed)
    out = pd.DataFrame(
        {
            "row_id": frame["row_id"].to_numpy(),
            "canonical_chromophore_smiles": frame["canonical_chromophore_smiles"].astype(str).to_numpy(),
            "split_group": frame["split_group"].astype(str).to_numpy(),
            "split": frame["split_group"].map(mapping).to_numpy(),
        }
    )
    leakage = leakage_check(out, split_type=split_type)
    if leakage["leakage_group_count"] != 0:
        raise RuntimeError(f"{split_type} leakage detected")
    return out, leakage


def leakage_check(assignments: pd.DataFrame, *, split_type: str) -> dict[str, Any]:
    leaked = []
    for group, part in assignments.groupby("split_group"):
        splits = sorted(part["split"].unique())
        if len(splits) > 1:
            leaked.append({"group": group, "splits": splits})
    return {
        "split_type": split_type,
        "leakage_group_count": len(leaked),
        "leaked_groups": leaked[:20],
        "split_counts": assignments["split"].value_counts().to_dict(),
    }


def load_finalized_embeddings(embedding_run_root: Path | str, pooling_methods: list[str] | None = None) -> pd.DataFrame:
    root = Path(embedding_run_root)
    index = pd.read_csv(root / "embedding_index.csv")
    pooling_methods = pooling_methods or POOLING_METHODS
    rows: list[dict[str, Any]] = []
    for shard_index, shard_rows in index.groupby("shard_index", sort=True):
        data = np.load(root / "embeddings" / f"shard_{int(shard_index):05d}.npz", allow_pickle=False)
        status = data["statuses"].astype(str)
        for _, row in shard_rows.iterrows():
            local = int(row["molecule_row"])
            record: dict[str, Any] = {
                "molecule_id": row["molecule_id"],
                "canonical_chromophore_smiles": row["canonical_chromophore_smiles"],
                "embedding_status": status[local],
            }
            for method in pooling_methods:
                key = f"{method}_embeddings"
                record[method] = data[key][local].astype(np.float32) if key in data.files else None
            rows.append(record)
        data.close()
    return pd.DataFrame(rows)


def join_embeddings(dataset: pd.DataFrame, embeddings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if embeddings["canonical_chromophore_smiles"].duplicated().any():
        raise ValueError("embedding table must contain one row per chromophore")
    rows = dataset.copy()
    rows["row_id"] = np.arange(len(rows))
    joined = rows.merge(embeddings, how="left", on="canonical_chromophore_smiles", validate="many_to_one")
    missing = joined["embedding_status"].isna()
    failed = joined["embedding_status"].eq("terminal_failure")
    excluded = joined[missing | failed].copy()
    included = joined[~missing & ~failed].reset_index(drop=True)
    return included, excluded


def build_feature_bundle(
    *,
    dataset_csv: Path | str,
    embedding_run_root: Path | str,
    solvent_descriptors: Path | str,
    n_bits: int = 2048,
    radius: int = 2,
    include_missing_indicators: bool = True,
) -> tuple[FeatureBundle, pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv(dataset_csv, low_memory=False)
    dataset = dataset.dropna(subset=["canonical_chromophore_smiles"]).reset_index(drop=True)
    embeddings = load_finalized_embeddings(embedding_run_root)
    joined, excluded = join_embeddings(dataset, embeddings)
    descriptors = load_solvent_descriptors(solvent_descriptors)
    rows, solvent_columns = merge_solvent_descriptors(joined, descriptors)
    solvent_values = rows[solvent_columns].apply(pd.to_numeric, errors="coerce") if solvent_columns else pd.DataFrame(index=rows.index)
    if include_missing_indicators and solvent_columns:
        indicators = solvent_values.isna().astype(np.float32)
        indicators.columns = [f"{column}__missing" for column in solvent_columns]
        solvent_values = pd.concat([solvent_values, indicators], axis=1)
        solvent_columns = list(solvent_values.columns)
    fingerprints: list[np.ndarray] = []
    morgan_valid: list[bool] = []
    morgan_exclusions: list[dict[str, Any]] = []
    for idx, smiles in rows["canonical_chromophore_smiles"].items():
        fp = morgan_fingerprint(str(smiles), radius=radius, n_bits=n_bits)
        if fp is not None:
            fingerprints.append(fp)
            morgan_valid.append(True)
        else:
            fingerprints.append(np.zeros((n_bits,), dtype=np.float32))
            morgan_valid.append(False)
            morgan_exclusions.append(
                {
                    "row_id": rows.loc[idx, "row_id"],
                    "canonical_chromophore_smiles": rows.loc[idx, "canonical_chromophore_smiles"],
                    "exclusion_reason": "morgan_fingerprint_failed",
                }
            )
    embeddings_by_pooling = {
        method: np.vstack(rows[method].to_list()).astype(np.float32)
        for method in POOLING_METHODS
    }
    return (
        FeatureBundle(
            rows=rows,
            embeddings_by_pooling=embeddings_by_pooling,
            morgan=np.vstack(fingerprints).astype(np.float32),
            morgan_valid=np.asarray(morgan_valid, dtype=bool),
            solvent_values=solvent_values,
            solvent_columns=solvent_columns,
        ),
        excluded,
        pd.DataFrame(morgan_exclusions),
    )


def make_candidates(seed: int, n_jobs: int) -> dict[str, Any]:
    return {
        "ridge_alpha_0.1": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=0.1))]),
        "ridge_alpha_1": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "ridge_alpha_10": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "extratrees": ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=n_jobs),
        "histgb": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, random_state=seed),
        "mlp": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        alpha=1e-4,
                        early_stopping=True,
                        max_iter=200,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "count": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "median_absolute_error": float(median_absolute_error(y_true, y_pred)),
        "mean_signed_error_bias": float(np.mean(y_pred - y_true)),
    }


def _feature_matrix(bundle: FeatureBundle, *, pooling: str, feature_set: str) -> np.ndarray:
    solvent = bundle.solvent_values.to_numpy(dtype=np.float32, copy=True)
    parts: list[np.ndarray] = []
    if feature_set in {"conforformer_solvent", "conforformer_morgan_solvent"}:
        parts.append(bundle.embeddings_by_pooling[pooling])
    if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"}:
        parts.append(bundle.morgan)
    parts.append(solvent)
    return np.hstack(parts).astype(np.float32)


def _feature_set_mask(bundle: FeatureBundle, feature_set: str) -> np.ndarray:
    if feature_set == "conforformer_solvent":
        return np.ones(len(bundle.rows), dtype=bool)
    if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"}:
        return bundle.morgan_valid.copy()
    raise ValueError(f"unknown feature set: {feature_set}")


def _poolings_for_feature_set(pooling_methods: list[str], feature_set: str) -> list[str]:
    if feature_set == "morgan_solvent":
        return ["not_applicable"]
    return pooling_methods


def feature_names(*, pooling: str, feature_set: str, solvent_columns: list[str], n_bits: int) -> list[str]:
    names: list[str] = []
    if feature_set in {"conforformer_solvent", "conforformer_morgan_solvent"}:
        names.extend([f"conforformer_{pooling}_{idx:03d}" for idx in range(EXPECTED_EMBEDDING_DIM)])
    if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"}:
        names.extend([f"morgan_{idx:04d}" for idx in range(n_bits)])
    names.extend(solvent_columns)
    return names


def _target_rows(rows: pd.DataFrame, target: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    info: dict[str, Any] = {}
    if target == "stokes_shift_nm":
        paired = rows[pd.to_numeric(rows["absorption_nm"], errors="coerce").notna() & pd.to_numeric(rows["emission_nm"], errors="coerce").notna()].copy()
        paired["stokes_shift_nm"] = pd.to_numeric(paired["emission_nm"], errors="coerce") - pd.to_numeric(paired["absorption_nm"], errors="coerce")
        info["paired_row_count"] = int(len(paired))
        info["nonpositive_stokes_excluded"] = int((paired["stokes_shift_nm"] <= 0).sum())
        return paired[paired["stokes_shift_nm"] > 0].copy(), info
    filtered = rows[pd.to_numeric(rows[target], errors="coerce").notna()].copy()
    return filtered, info


def train_downstream(
    *,
    dataset_csv: Path | str,
    embedding_run_root: Path | str,
    solvent_descriptors: Path | str,
    out_dir: Path | str,
    model_out_dir: Path | str,
    split_type: str = "molecule",
    seed: int = 0,
    n_jobs: int = -1,
    n_bits: int = 2048,
    radius: int = 2,
    feature_sets: list[str] | None = None,
    pooling_methods: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    model_out_dir = Path(model_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    feature_sets = feature_sets or ["conforformer_solvent", "morgan_solvent"]
    pooling_methods = pooling_methods or POOLING_METHODS
    bundle, excluded, morgan_excluded = build_feature_bundle(
        dataset_csv=dataset_csv,
        embedding_run_root=embedding_run_root,
        solvent_descriptors=solvent_descriptors,
        n_bits=n_bits,
        radius=radius,
    )
    split_assignments, leakage = make_split_assignments(bundle.rows, split_type=split_type, seed=seed)
    bundle = replace(
        bundle,
        rows=bundle.rows.merge(split_assignments[["row_id", "split"]], how="left", on="row_id", validate="one_to_one"),
    )
    split_assignments.to_csv(out_dir / "split_assignments.csv", index=False)
    excluded.to_csv(out_dir / "excluded_rows.csv", index=False)
    morgan_excluded.to_csv(out_dir / "morgan_excluded_rows.csv", index=False)
    atomic_write_text(out_dir / "leakage_check.json", json.dumps(leakage, indent=2, sort_keys=True) + "\n")

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selected: dict[str, Any] = {}
    predictions_by_key: dict[tuple[str, str, str], pd.DataFrame] = {}

    cohort_counts: dict[str, Any] = {}
    target_counts: dict[str, Any] = {}

    for feature_set in feature_sets:
        feature_mask = _feature_set_mask(bundle, feature_set)
        cohort_rows = bundle.rows.loc[feature_mask]
        cohort_counts[feature_set] = {
            "primary_conforformer_row_count": int(len(bundle.rows)),
            "feature_set_row_count": int(feature_mask.sum()),
            "morgan_excluded_row_count": int((~bundle.morgan_valid).sum()) if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"} else 0,
            "comparison_cohort": "matched_intersection" if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"} else "full_primary",
        }
        for pooling in _poolings_for_feature_set(pooling_methods, feature_set):
            full_x = _feature_matrix(bundle, pooling=pooling, feature_set=feature_set)
            names = feature_names(pooling=pooling, feature_set=feature_set, solvent_columns=bundle.solvent_columns, n_bits=n_bits)
            for target in TARGETS:
                target_rows, target_info = _target_rows(cohort_rows, target)
                target_counts[f"{target}/{pooling}/{feature_set}"] = {
                    "before_target_missingness": int(len(cohort_rows)),
                    "after_target_missingness": int(len(target_rows)),
                }
                if len(target_rows) < 6:
                    continue
                idx = target_rows.index.to_numpy()
                y = pd.to_numeric(target_rows[target], errors="coerce").to_numpy(dtype=float)
                split = target_rows["split"].to_numpy()
                masks = {name: split == name for name in ["base_train", "model_selection", "final_test"]}
                if any(masks[name].sum() == 0 for name in masks):
                    continue
                train_sel = masks["base_train"] | masks["model_selection"]
                imputer = SimpleImputer(strategy="median")
                x_base = imputer.fit_transform(full_x[idx[masks["base_train"]]])
                x_val = imputer.transform(full_x[idx[masks["model_selection"]]])
                candidates = make_candidates(seed=seed, n_jobs=n_jobs)
                best_name = ""
                best_mae = float("inf")
                for name, candidate in candidates.items():
                    candidate.fit(x_base, y[masks["base_train"]])
                    pred = candidate.predict(x_val)
                    mae = float(mean_absolute_error(y[masks["model_selection"]], pred))
                    selection_rows.append(
                        {
                            "target": target,
                            "pooling_method": pooling,
                            "feature_set": feature_set,
                            "comparison_cohort": cohort_counts[feature_set]["comparison_cohort"],
                            "candidate": name,
                            "validation_mae": mae,
                            **target_info,
                        }
                    )
                    if mae < best_mae:
                        best_mae = mae
                        best_name = name
                final_imputer = SimpleImputer(strategy="median")
                x_train_sel = final_imputer.fit_transform(full_x[idx[train_sel]])
                x_test = final_imputer.transform(full_x[idx[masks["final_test"]]])
                model = make_candidates(seed=seed, n_jobs=n_jobs)[best_name]
                model.fit(x_train_sel, y[train_sel])
                pred = model.predict(x_test)
                test_rows = target_rows.loc[masks["final_test"]].copy()
                pred_df = test_rows[["row_id", "canonical_chromophore_smiles", "canonical_solvent_smiles", "split"]].copy()
                pred_df["target"] = target
                pred_df["y_true"] = y[masks["final_test"]]
                pred_df["y_pred"] = pred
                pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
                pred_path = out_dir / "predictions" / f"{target}__{pooling}__{feature_set}.csv"
                pred_df.to_csv(pred_path, index=False)
                predictions_by_key[(target, pooling, feature_set)] = pred_df
                row_metrics = {
                    "target": target,
                    "pooling_method": pooling,
                    "feature_set": feature_set,
                    "comparison_cohort": cohort_counts[feature_set]["comparison_cohort"],
                    "model": best_name,
                    "split": "final_test",
                    **metrics(pred_df["y_true"].to_numpy(dtype=float), pred_df["y_pred"].to_numpy(dtype=float)),
                    **target_info,
                    "prediction_path": str(pred_path),
                }
                if target == "quantum_yield":
                    clipped = np.clip(pred, 0.0, 1.0)
                    row_metrics.update(
                        {
                            "raw_mae": row_metrics["mae"],
                            "raw_rmse": row_metrics["rmse"],
                            "clipped_mae": float(mean_absolute_error(pred_df["y_true"], clipped)),
                            "clipped_rmse": float(np.sqrt(mean_squared_error(pred_df["y_true"], clipped))),
                            "raw_fraction_below_0": float(np.mean(pred < 0)),
                            "raw_fraction_above_1": float(np.mean(pred > 1)),
                        }
                    )
                    pred_df["y_pred_clipped"] = clipped
                    pred_df.to_csv(pred_path, index=False)
                metric_rows.append(row_metrics)
                model_dir = model_out_dir / split_type / pooling / feature_set / target
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(Pipeline([("imputer", final_imputer), ("model", model)]), model_dir / "model.joblib")
                metadata = {
                    "feature_order": names,
                    "selected_candidate": best_name,
                    "imputer": "train_only_median_refit_on_base_train_plus_model_selection",
                    "scaling": "inside Ridge/MLP candidate pipelines only",
                    "target": target,
                    "pooling_method": pooling,
                    "feature_set": feature_set,
                    "comparison_cohort": cohort_counts[feature_set]["comparison_cohort"],
                }
                atomic_write_text(model_dir / "feature_metadata.json", json.dumps(metadata, indent=2, sort_keys=True) + "\n")
                atomic_write_text(model_dir / "metrics.json", json.dumps(row_metrics, indent=2, sort_keys=True) + "\n")
                selected[f"{target}/{pooling}/{feature_set}"] = best_name

    stokes_rows = _derived_stokes_metrics(predictions_by_key)
    metric_rows.extend(stokes_rows)
    pd.DataFrame(selection_rows).to_csv(out_dir / "selection_results.csv", index=False)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    atomic_write_text(out_dir / "metrics.json", metrics_df.to_json(orient="records", indent=2) + "\n")
    manifest = {
        "dataset_csv": str(dataset_csv),
        "dataset_sha256": sha256_file(dataset_csv),
        "embedding_manifest_sha256": sha256_file(Path(embedding_run_root) / "embedding_manifest.json"),
        "fluorcast_git_commit": fluorcast_git_commit(),
        "split_seed": seed,
        "split_type": split_type,
        "split_assignments_sha256": sha256_file(out_dir / "split_assignments.csv"),
        "target_definitions": {
            "stokes_shift_nm": "emission_nm - absorption_nm; positive finite primary rows only",
            "quantum_yield": "raw numeric target; predictions also clipped to [0, 1] for reporting",
        },
        "feature_sets": feature_sets,
        "pooling_methods": pooling_methods,
        "effective_pooling_methods_by_feature_set": {
            feature_set: _poolings_for_feature_set(pooling_methods, feature_set)
            for feature_set in feature_sets
        },
        "feature_set_row_counts": cohort_counts,
        "target_row_counts": target_counts,
        "morgan_exclusion_report": str(out_dir / "morgan_excluded_rows.csv"),
        "solvent_feature_names": bundle.solvent_columns,
        "imputation": "SimpleImputer(strategy='median') fit only on the training rows used for the fit",
        "candidate_definitions": list(make_candidates(seed=seed, n_jobs=n_jobs).keys()),
        "selected_candidates": selected,
        "package_versions": _package_versions(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(out_dir / "training_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _derived_stokes_metrics(predictions: dict[tuple[str, str, str], pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = {(pooling, feature_set) for target, pooling, feature_set in predictions if target in {"absorption_nm", "emission_nm", "stokes_shift_nm"}}
    for pooling, feature_set in sorted(keys):
        a = predictions.get(("absorption_nm", pooling, feature_set))
        e = predictions.get(("emission_nm", pooling, feature_set))
        d = predictions.get(("stokes_shift_nm", pooling, feature_set))
        if a is None or e is None or d is None:
            continue
        paired = d[["row_id", "y_true"]].merge(
            a[["row_id", "y_pred"]].rename(columns={"y_pred": "absorption_pred"}),
            on="row_id",
            how="inner",
        ).merge(
            e[["row_id", "y_pred"]].rename(columns={"y_pred": "emission_pred"}),
            on="row_id",
            how="inner",
        )
        if paired.empty:
            continue
        derived = paired["emission_pred"].to_numpy(dtype=float) - paired["absorption_pred"].to_numpy(dtype=float)
        row = {
            "target": "stokes_shift_nm",
            "pooling_method": pooling,
            "feature_set": feature_set,
            "model": "absorption_emission_derived",
            "split": "final_test_identical_rows",
            **metrics(paired["y_true"].to_numpy(dtype=float), derived),
            "physically_invalid_derived_prediction_fraction": float(np.mean(derived <= 0)),
            "identical_row_count": int(len(paired)),
            "comparison_cohort": "matched_intersection" if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"} else "full_primary",
        }
        rows.append(row)
    return rows


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ["numpy", "pandas", "sklearn", "joblib"]:
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions
