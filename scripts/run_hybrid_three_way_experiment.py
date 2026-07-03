"""Run a leakage-safe three-way base-model/hybrid ensemble experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_combined_predictors as base  # noqa: E402
from chemfluor.hybrid.ensemble import (  # noqa: E402
    align_features, predict_hybrid_ensemble, save_hybrid_ensemble,
    train_hybrid_ensemble,
)
from chemfluor.hybrid.uncertainty import calibration_residuals  # noqa: E402

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:  # pragma: no cover
    Chem = MurckoScaffold = rdBase = None

SPLITS = ("base_model_train", "hybrid_meta_train", "final_test")
MODELS = ("rf", "extratrees", "histgb", "gbdt", "mlp")
TARGETS = ("emission_nm", "quantum_yield", "absorption_nm")
DEFAULT_STANDARDIZED = ROOT / "data" / "processed" / "fluodb_lite" / "combined_deduplicated.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-name", required=True, choices=TARGETS)
    parser.add_argument("--split-type", required=True, choices=("random", "molecule", "scaffold"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", required=True, choices=MODELS)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-out-dir", required=True, type=Path)
    parser.add_argument("--base-train-fraction", type=float, default=.60)
    parser.add_argument("--meta-train-fraction", type=float, default=.20)
    parser.add_argument("--final-test-fraction", type=float, default=.20)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--standardized-combined", type=Path)
    parser.add_argument("--solvent-descriptors", type=Path, default=base.DEFAULT_SOLVENT_DESCRIPTORS)
    parser.add_argument(
        "--invalid-smiles-policy",
        choices=("drop", "keep-invalid-group"),
        default="drop",
        help="Drop invalid molecule/scaffold rows (default), or keep them in one non-leaking group.",
    )
    return parser.parse_args(argv)


def _fractions(args: argparse.Namespace) -> tuple[float, float, float]:
    values = (args.base_train_fraction, args.meta_train_fraction, args.final_test_fraction)
    if any(value <= 0 for value in values) or not np.isclose(sum(values), 1.0):
        raise ValueError("Split fractions must be positive and sum to 1.0")
    return values


def safe_mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Parse and sanitize SMILES without allowing RDKit exceptions to escape."""
    if Chem is None:
        return None
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    text = str(smiles).strip()
    try:
        mol = Chem.MolFromSmiles(text)
        if mol is not None:
            return mol
    except Exception:
        pass
    try:
        mol = Chem.MolFromSmiles(text, sanitize=False)
        if mol is None:
            return None
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.DOUBLE:
                bond.SetStereo(Chem.BondStereo.STEREONONE)
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def safe_canonical_smiles(smiles: str) -> str | None:
    """Return canonical SMILES, or None for any parse/canonicalization failure."""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        canonical = Chem.MolToSmiles(mol, canonical=True)
        # Exercise CanonSmiles too: some stereo failures occur only in canonicalization.
        canonical = Chem.CanonSmiles(canonical)
        return canonical or None
    except Exception:
        return None


def safe_murcko_scaffold(smiles: str) -> str | None:
    """Return a Murcko scaffold without propagating RDKit precondition errors."""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None
    try:
        scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold = Chem.MolToSmiles(scaffold_mol, canonical=True)
        if scaffold:
            return Chem.CanonSmiles(scaffold)
        return "ACYCLIC"
    except Exception:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=safe_canonical_smiles(smiles), includeChirality=False
            )
            return scaffold or "ACYCLIC"
        except Exception:
            return None


def prepare_split_identifiers(
    rows: pd.DataFrame, split_type: str, policy: str, out_dir: Path
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Safely canonicalize split identifiers and apply the invalid-row policy."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prepared = rows.copy()
    counts = {"invalid_molecule_rows": 0, "invalid_scaffold_rows": 0}
    if split_type not in {"molecule", "scaffold"}:
        return prepared, counts
    canonical = prepared["canonical_chromophore_smiles"].map(safe_canonical_smiles)
    invalid_molecule = canonical.isna()
    counts["invalid_molecule_rows"] = int(invalid_molecule.sum())
    if invalid_molecule.any():
        prepared.loc[invalid_molecule].to_csv(out_dir / "invalid_molecule_rows.csv", index=False)
    prepared["canonical_chromophore_smiles"] = canonical

    invalid = invalid_molecule.copy()
    invalid_path = out_dir / "invalid_molecule_rows.csv"
    identifier = "canonical molecule SMILES"
    if split_type == "scaffold":
        prepared["scaffold"] = prepared["canonical_chromophore_smiles"].map(safe_murcko_scaffold)
        invalid_scaffold = prepared["scaffold"].isna()
        counts["invalid_scaffold_rows"] = int(invalid_scaffold.sum())
        invalid = invalid_scaffold
        invalid_path = out_dir / "invalid_scaffold_rows.csv"
        identifier = "Murcko scaffold"
        if invalid.any():
            prepared.loc[invalid].to_csv(invalid_path, index=False)
    if invalid.all():
        raise ValueError(f"All rows have invalid {identifier}; no rows remain for splitting")
    if invalid.any() and policy == "drop":
        warnings.warn(f"Dropped {int(invalid.sum())} row(s) with invalid {identifier}; saved to {invalid_path}")
        prepared = prepared.loc[~invalid].copy()
    elif invalid.any():
        group_column = "scaffold" if split_type == "scaffold" else "canonical_chromophore_smiles"
        sentinel = "INVALID_SCAFFOLD" if split_type == "scaffold" else "INVALID_MOLECULE"
        prepared.loc[invalid, group_column] = sentinel
        warnings.warn(f"Kept {int(invalid.sum())} invalid row(s) together as {sentinel}")
    return prepared.reset_index(drop=True), counts


def safe_fingerprints(rows: pd.DataFrame, n_bits: int = 2048) -> np.ndarray:
    """Build fingerprints safely; kept invalid-group rows receive an all-zero vector."""
    vectors: list[np.ndarray] = []
    zero_count = 0
    for smiles in rows["canonical_chromophore_smiles"]:
        try:
            vector = None if str(smiles).startswith("INVALID_") else base.morgan_fingerprint(str(smiles), 2, n_bits)
        except Exception:
            vector = None
        if vector is None:
            vector = np.zeros(n_bits, dtype=np.float32)
            zero_count += 1
        vectors.append(vector)
    if zero_count:
        warnings.warn(f"Used zero molecule fingerprints for {zero_count} kept invalid row(s)")
    return np.vstack(vectors)


def assign_three_way_splits(rows: pd.DataFrame, split_type: str, fractions: tuple[float, float, float], seed: int) -> pd.Series:
    """Assign every row once, keeping requested groups intact."""
    rng = np.random.RandomState(seed)
    labels = pd.Series(index=rows.index, dtype="object")
    if split_type == "random":
        order = rng.permutation(rows.index.to_numpy())
        cut1 = int(round(fractions[0] * len(order)))
        cut2 = cut1 + int(round(fractions[1] * len(order)))
        for name, indices in zip(SPLITS, (order[:cut1], order[cut1:cut2], order[cut2:])):
            labels.loc[indices] = name
        return labels
    group_column = "canonical_chromophore_smiles" if split_type == "molecule" else "scaffold"
    if rows[group_column].isna().any():
        raise ValueError(f"Missing values in required grouping column {group_column}")
    sizes = rows.groupby(group_column, sort=False).size()
    groups = sizes.index.to_numpy(dtype=object)
    rng.shuffle(groups)
    targets = np.asarray(fractions) * len(rows)
    totals = np.zeros(3, dtype=int)
    for group in groups:
        # Put the group in the split furthest below its requested row target.
        destination = int(np.argmax((targets - totals) / np.maximum(targets, 1)))
        labels.loc[rows[group_column] == group] = SPLITS[destination]
        totals[destination] += int(sizes.loc[group])
    return labels


def leakage_report(rows: pd.DataFrame, split_type: str) -> dict[str, Any]:
    column = "canonical_chromophore_smiles" if split_type == "molecule" else "scaffold" if split_type == "scaffold" else "row_id"
    sets = {name: set(rows.loc[rows["split"] == name, column].dropna()) for name in SPLITS}
    overlaps = {f"{a}__{b}": sorted(map(str, sets[a] & sets[b])) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]}
    detected = any(overlaps.values())
    return {"split_type": split_type, "checked_identifier": column, "leakage_detected": detected,
            "overlap_counts": {key: len(value) for key, value in overlaps.items()}, "overlaps": overlaps}


def prediction_table(rows: pd.DataFrame, predictions: dict[str, np.ndarray], target: str) -> pd.DataFrame:
    columns = ["row_id", "canonical_chromophore_smiles", "canonical_solvent_smiles", "solvent_original"]
    result = rows[[column for column in columns if column in rows]].reset_index(drop=True).copy()
    result[f"true_{target}"] = rows[target].to_numpy(dtype=float)
    prediction_columns = []
    for name, values in predictions.items():
        column = f"{name}_{target}"
        result[column] = values
        prediction_columns.append(column)
    values = result[prediction_columns]
    result["prediction_mean"] = values.mean(axis=1)
    result["prediction_std"] = values.std(axis=1, ddof=0)
    result["prediction_min"] = values.min(axis=1)
    result["prediction_max"] = values.max(axis=1)
    result["prediction_range"] = result["prediction_max"] - result["prediction_min"]
    result["prediction_count"] = values.notna().sum(axis=1)
    return result


def meta_features(table: pd.DataFrame, target: str, models: list[str]) -> pd.DataFrame:
    return table[[*(f"{name}_{target}" for name in models), "prediction_mean", "prediction_std",
                  "prediction_min", "prediction_max", "prediction_range", "prediction_count"]].apply(pd.to_numeric, errors="coerce")


def metrics_row(name: str, truth: np.ndarray, prediction: np.ndarray, target: str) -> dict[str, Any]:
    row = {"model": name, "MAE": float(mean_absolute_error(truth, prediction)),
           "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
           "R2": float(r2_score(truth, prediction)) if len(truth) > 1 else np.nan, "N": int(len(truth))}
    if target == "quantum_yield":
        actual, predicted = truth > .25, prediction > .25
        row.update({"accuracy": float(accuracy_score(actual, predicted)),
                    "macro_F1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
                    "weighted_F1": float(f1_score(actual, predicted, average="weighted", zero_division=0))})
    return row


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    fractions = _fractions(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    standardized_path = args.standardized_combined or (DEFAULT_STANDARDIZED if DEFAULT_STANDARDIZED.exists() else None)
    if standardized_path:
        rows = base.load_standardized_combined(standardized_path)
        dataset_path = standardized_path
    else:
        rows = base.load_combined_rows(base.DEFAULT_DEEP4CHEM, base.DEFAULT_CHEMFLUOR)
        dataset_path = "combined default ChemFluor + Deep4Chem"
    rows[args.target_name] = pd.to_numeric(rows[args.target_name], errors="coerce")
    rows = rows[np.isfinite(rows[args.target_name])].copy().reset_index(drop=True)
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    rows.insert(0, "row_id", np.arange(len(rows), dtype=int))
    rows, invalid_counts = prepare_split_identifiers(
        rows, args.split_type, args.invalid_smiles_policy, args.out_dir
    )
    rows["split"] = assign_three_way_splits(rows, args.split_type, fractions, args.seed)
    if rows["split"].isna().any() or any((rows["split"] == name).sum() == 0 for name in SPLITS):
        raise ValueError("Three-way split produced an empty or unassigned split; use more rows/groups")
    leak = leakage_report(rows, args.split_type)
    leak["invalid_row_counts"] = invalid_counts
    leak["invalid_smiles_policy"] = args.invalid_smiles_policy
    _json(args.out_dir / "leakage_check.json", leak)
    if leak["leakage_detected"]:
        raise RuntimeError(f"Leakage detected: {leak['overlap_counts']}")
    rows.to_csv(args.out_dir / "split_assignments.csv", index=False)

    descriptors = base.load_solvent_descriptors(args.solvent_descriptors)
    rows["experiment_row_id"] = rows["row_id"]
    featured, descriptor_columns = base.merge_solvent_descriptors(rows, descriptors)
    featured["row_id"] = featured.pop("experiment_row_id").astype(int)
    fingerprints = safe_fingerprints(featured)
    train_mask = featured["split"] == SPLITS[0]
    descriptor_values = featured[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    medians = descriptor_values.loc[train_mask].median(numeric_only=True)
    x = base.build_feature_matrix(fingerprints, descriptor_values, medians)
    y = featured[args.target_name].to_numpy(dtype=float)
    fitted: dict[str, Any] = {}
    for name in args.models:
        model = base.make_model(name, random_state=args.seed, n_jobs=args.n_jobs)
        model.fit(x[train_mask.to_numpy()], y[train_mask.to_numpy()])
        model_dir = args.model_out_dir / "base_models" / name
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_dir / "model.joblib")
        _json(model_dir / "metadata.json", {"model": name, "target": args.target_name,
              "training_split": SPLITS[0], "training_row_ids": featured.loc[train_mask, "row_id"].tolist(),
              "feature_count": int(x.shape[1]), "n_training_rows": int(train_mask.sum())})
        fitted[name] = model
    tables = {}
    for split, filename in ((SPLITS[1], "base_model_predictions_meta_train.csv"), (SPLITS[2], "base_model_predictions_final_test.csv")):
        mask = featured["split"] == split
        tables[split] = prediction_table(featured.loc[mask], {name: model.predict(x[mask.to_numpy()]) for name, model in fitted.items()}, args.target_name)
        tables[split].to_csv(args.out_dir / filename, index=False)

    meta = tables[SPLITS[1]]
    features = meta_features(meta, args.target_name, args.models)
    labels = meta[f"true_{args.target_name}"].astype(float)
    if len(meta) < 5:
        raise ValueError("At least five hybrid_meta_train rows are required")
    train_idx, calibration_idx = train_test_split(np.arange(len(meta)), test_size=max(2, int(np.ceil(.2 * len(meta)))), random_state=args.seed)
    hybrid = train_hybrid_ensemble(features.iloc[train_idx], labels.iloc[train_idx], args.target_name, bright_threshold=.25)
    hybrid_dir = args.model_out_dir / "hybrid_ensemble"
    save_hybrid_ensemble(hybrid, list(features.columns), hybrid_dir,
                         {"training_split": SPLITS[1], "training_row_ids": meta.iloc[train_idx]["row_id"].tolist(),
                          "calibration_row_ids": meta.iloc[calibration_idx]["row_id"].tolist(), "n_training_examples": len(train_idx)})
    calibration_prediction = hybrid["regressor"].predict(features.iloc[calibration_idx])
    calibration_residuals(labels.iloc[calibration_idx], calibration_prediction).to_csv(hybrid_dir / "calibration_residuals.csv", index=False)

    final = tables[SPLITS[2]].copy()
    final_features = align_features(meta_features(final, args.target_name, args.models), list(features.columns))
    final["hybrid_prediction"] = hybrid["regressor"].predict(final_features)
    final.to_csv(args.out_dir / "final_evaluated_predictions.csv", index=False)
    truth = final[f"true_{args.target_name}"].to_numpy(dtype=float)
    metric_rows = [metrics_row(name, truth, final[f"{name}_{args.target_name}"].to_numpy(), args.target_name) for name in args.models]
    metric_rows += [metrics_row("prediction_mean", truth, final["prediction_mean"].to_numpy(), args.target_name),
                    metrics_row("hybrid_ensemble", truth, final["hybrid_prediction"].to_numpy(), args.target_name)]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "metrics_table.csv", index=False)
    best = metrics[metrics["model"].isin(args.models)].sort_values("MAE").iloc[0]
    hybrid_metrics = metrics[metrics["model"] == "hybrid_ensemble"].iloc[0]
    difference = float(best["MAE"] - hybrid_metrics["MAE"])
    counts = {name: int((featured["split"] == name).sum()) for name in SPLITS}
    unique_molecules = {name: int(featured.loc[featured["split"] == name, "canonical_chromophore_smiles"].nunique()) for name in SPLITS}
    scaffold_text = ""
    if "scaffold" in featured:
        scaffold_text = "\n" + "\n".join(f"- Unique scaffolds in {name}: {featured.loc[featured['split'] == name, 'scaffold'].nunique()}" for name in SPLITS)
    summary = f"""# Three-way Hybrid Ensemble Results

- Target: {args.target_name}
- Split type: {args.split_type}
- Seed: {args.seed}
- Rows: {counts}
- Unique molecules: {unique_molecules}{scaffold_text}
- Best base model by MAE: {best['model']} ({best['MAE']:.6f})
- Hybrid MAE / RMSE / R2: {hybrid_metrics['MAE']:.6f} / {hybrid_metrics['RMSE']:.6f} / {hybrid_metrics['R2']:.6f}
- Hybrid improved over best base model: {'yes' if difference > 0 else 'no'}
- Exact MAE improvement (best base minus hybrid): {difference:.6f}
- Leakage detected: {leak['leakage_detected']}

Final metrics are calculated only on the untouched `final_test` split.
"""
    (args.out_dir / "metrics_summary.md").write_text(summary, encoding="utf-8")
    config = {"target_name": args.target_name, "split_type": args.split_type, "seed": args.seed,
              "fractions": dict(zip(SPLITS, fractions)), "models": args.models,
              "input_dataset": str(dataset_path), "solvent_descriptors": str(args.solvent_descriptors),
              "invalid_smiles_policy": args.invalid_smiles_policy,
              "invalid_row_counts": invalid_counts,
              "row_counts": counts, "feature_columns": [*[f"fingerprint_{i}" for i in range(2048)], *descriptor_columns],
              "timestamp": datetime.now(timezone.utc).isoformat(),
              "package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                                   "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                                   "rdkit": getattr(rdBase, "rdkitVersion", None)}}
    _json(args.out_dir / "experiment_config.json", config)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
