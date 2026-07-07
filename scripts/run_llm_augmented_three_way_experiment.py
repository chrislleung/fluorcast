"""Compare leakage-safe hybrid ensembles with LLM-augmented variants."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_hybrid_three_way_experiment as three_way  # noqa: E402
import train_combined_predictors as base  # noqa: E402
from chemfluor.hybrid.ensemble import train_hybrid_ensemble  # noqa: E402
from chemfluor.llm_prediction.feature_encoding import encode_llm_features  # noqa: E402
from chemfluor.llm_prediction.schema import LLMOutput  # noqa: E402

VARIANTS = {
    "non_llm_hybrid_baseline": (),
    "llm_numeric_only_hybrid": ("numeric",),
    "llm_descriptor_only_hybrid": ("descriptors",),
    "llm_numeric_plus_descriptor_hybrid": ("numeric", "descriptors"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-name", required=True, choices=three_way.TARGETS)
    parser.add_argument("--split-type", required=True, choices=("random", "molecule", "scaffold"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="+", required=True, choices=three_way.MODELS)
    parser.add_argument("--llm-mode", choices=("template", "ollama", "openai"), default="template")
    parser.add_argument("--llm-feature-mode", nargs="+", choices=("descriptors", "numeric", "both"),
                        default=["descriptors", "numeric", "both"])
    parser.add_argument("--llm-model", help="Provider model name (uses provider default if omitted)")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--invalid-smiles-policy", choices=("drop", "keep-invalid-group"), default="drop")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-out-dir", required=True, type=Path)
    parser.add_argument("--base-train-fraction", type=float, default=.60)
    parser.add_argument("--meta-train-fraction", type=float, default=.20)
    parser.add_argument("--final-test-fraction", type=float, default=.20)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--standardized-combined", type=Path)
    parser.add_argument("--solvent-descriptors", type=Path, default=base.DEFAULT_SOLVENT_DESCRIPTORS)
    return parser.parse_args(argv)


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")


def _provider(args: argparse.Namespace) -> Callable[[str, str, str | None], LLMOutput]:
    if args.llm_mode == "template":
        from chemfluor.llm_prediction.template_stub import predict
        return predict
    if args.llm_mode == "ollama":
        from chemfluor.llm_prediction.ollama_client import predict
        return lambda target, smiles, solvent: predict(target, smiles, solvent,
            **({"model": args.llm_model} if args.llm_model else {}), base_url=args.ollama_base_url)
    from chemfluor.llm_prediction.openai_client import predict
    return lambda target, smiles, solvent: predict(target, smiles, solvent,
        **({"model": args.llm_model} if args.llm_model else {}))


def cached_llm_outputs(rows: pd.DataFrame, target: str, cache_path: Path,
                       provider: Callable[[str, str, str | None], LLMOutput]) -> list[dict[str, Any]]:
    """Return one safe response per row, appending only cache misses to JSONL."""
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                cached[str(item["row_id"])] = item
            except (ValueError, KeyError, TypeError):
                continue
    result = []
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as stream:
        for _, row in rows.reset_index(drop=True).iterrows():
            key = str(row["row_id"])
            item = cached.get(key)
            if item is None:
                try:
                    output = provider(target, str(row.get("canonical_chromophore_smiles", "")),
                                      row.get("canonical_solvent_smiles"))
                    output = output if isinstance(output, LLMOutput) else LLMOutput.from_dict(output, target)
                except Exception as exc:  # a provider failure is represented as missing features
                    output = LLMOutput.empty(target, f"Provider error: {type(exc).__name__}: {exc}")
                item = {"row_id": row["row_id"], **output.to_dict()}
                stream.write(json.dumps(item, allow_nan=False, default=str) + "\n")
                stream.flush()
            result.append(LLMOutput.from_dict(item, target).to_dict())
    return result


def augmented_features(base_features: pd.DataFrame, outputs: list[dict[str, Any]], mode: str) -> pd.DataFrame:
    """Add requested LLM columns; malformed/missing values remain safely imputable NaNs."""
    llm = encode_llm_features(outputs).reset_index(drop=True)
    result = base_features.reset_index(drop=True).copy()
    if mode in {"numeric", "both"}:
        result[["llm_numeric_prediction", "llm_confidence"]] = llm[["llm_numeric_prediction", "llm_confidence"]]
    if mode in {"descriptors", "both"}:
        columns = [column for column in llm if column.startswith("llm_descriptor_")]
        result[columns] = llm[columns]
    return result


def _enabled_variants(modes: list[str]) -> list[str]:
    enabled = ["non_llm_hybrid_baseline"]
    if "numeric" in modes or "both" in modes:
        enabled.append("llm_numeric_only_hybrid")
    if "descriptors" in modes or "both" in modes:
        enabled.append("llm_descriptor_only_hybrid")
    if "both" in modes or {"numeric", "descriptors"} <= set(modes):
        enabled.append("llm_numeric_plus_descriptor_hybrid")
    return enabled


def run(args: argparse.Namespace) -> None:
    fractions = three_way._fractions(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.model_out_dir.mkdir(parents=True, exist_ok=True)
    standardized = args.standardized_combined or (three_way.DEFAULT_STANDARDIZED if three_way.DEFAULT_STANDARDIZED.exists() else None)
    if standardized:
        rows, dataset = base.load_standardized_combined(standardized), standardized
    else:
        rows, dataset = base.load_combined_rows(base.DEFAULT_DEEP4CHEM, base.DEFAULT_CHEMFLUOR), "combined defaults"
    rows[args.target_name] = pd.to_numeric(rows[args.target_name], errors="coerce")
    rows = rows[np.isfinite(rows[args.target_name])].reset_index(drop=True)
    if args.max_rows and len(rows) > args.max_rows:
        rows = rows.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    rows.insert(0, "row_id", np.arange(len(rows)))
    rows, invalid_counts = three_way.prepare_split_identifiers(rows, args.split_type,
        args.invalid_smiles_policy, args.out_dir)
    rows["split"] = three_way.assign_three_way_splits(rows, args.split_type, fractions, args.seed)
    if rows["split"].isna().any() or any(not (rows["split"] == split).any() for split in three_way.SPLITS):
        raise ValueError("Three-way split produced an empty or unassigned split; use more rows/groups")
    leak = three_way.leakage_report(rows, args.split_type)
    leak.update({"invalid_row_counts": invalid_counts, "invalid_smiles_policy": args.invalid_smiles_policy,
                 "llm_splits": ["hybrid_meta_train", "final_test"],
                 "final_test_labels_supplied_to_llm": False})
    _json(args.out_dir / "leakage_check.json", leak)
    if leak["leakage_detected"]:
        raise RuntimeError(f"Leakage detected: {leak['overlap_counts']}")
    rows.to_csv(args.out_dir / "split_assignments.csv", index=False)

    solvent = base.load_solvent_descriptors(args.solvent_descriptors)
    rows["experiment_row_id"] = rows["row_id"]
    featured, descriptor_columns = base.merge_solvent_descriptors(rows, solvent)
    featured["row_id"] = featured.pop("experiment_row_id").astype(int)
    fingerprints = three_way.safe_fingerprints(featured)
    train_mask = featured["split"] == "base_model_train"
    values = featured[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    x = base.build_feature_matrix(fingerprints, values, values.loc[train_mask].median(numeric_only=True))
    y = featured[args.target_name].to_numpy(float)
    fitted = {}
    for name in args.models:
        model = base.make_model(name, random_state=args.seed, n_jobs=args.n_jobs)
        model.fit(x[train_mask.to_numpy()], y[train_mask.to_numpy()])
        directory = args.model_out_dir / "base_models" / name
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / "model.joblib")
        fitted[name] = model
    tables = {}
    for split in three_way.SPLITS:  # includes base_model_train as an explicit generated prediction set
        mask = featured["split"] == split
        tables[split] = three_way.prediction_table(featured.loc[mask],
            {name: model.predict(x[mask.to_numpy()]) for name, model in fitted.items()}, args.target_name)
    meta, final = tables["hybrid_meta_train"], tables["final_test"]
    meta.to_csv(args.out_dir / "base_model_predictions_meta_train.csv", index=False)
    final.to_csv(args.out_dir / "base_model_predictions_final_test.csv", index=False)

    provider = _provider(args)
    meta_outputs = cached_llm_outputs(meta, args.target_name, args.out_dir / "llm_outputs_meta_train.jsonl", provider)
    final_outputs = cached_llm_outputs(final, args.target_name, args.out_dir / "llm_outputs_final_test.jsonl", provider)
    base_meta = three_way.meta_features(meta, args.target_name, args.models)
    base_final = three_way.meta_features(final, args.target_name, args.models)
    labels = meta[f"true_{args.target_name}"].astype(float)
    if len(meta) < 2:
        raise ValueError("At least two hybrid_meta_train rows are required")
    base_meta.assign(**{f"true_{args.target_name}": labels}).to_csv(
        args.out_dir / "hybrid_meta_training_table_non_llm.csv", index=False)
    augmented_features(base_meta, meta_outputs, "both").assign(**{f"true_{args.target_name}": labels}).to_csv(
        args.out_dir / "hybrid_meta_training_table_llm.csv", index=False)

    variant_predictions = {}
    for variant in _enabled_variants(args.llm_feature_mode):
        kinds = VARIANTS[variant]
        mode = "both" if len(kinds) == 2 else kinds[0] if kinds else "none"
        train_features = base_meta if mode == "none" else augmented_features(base_meta, meta_outputs, mode)
        test_features = base_final if mode == "none" else augmented_features(base_final, final_outputs, mode)
        model = train_hybrid_ensemble(train_features, labels, args.target_name)
        directory = args.model_out_dir / variant
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, directory / "model.joblib")
        _json(directory / "feature_columns.json", list(train_features.columns))
        variant_predictions[variant] = model["regressor"].predict(test_features)

    evaluated = final.copy()
    for name, prediction in variant_predictions.items():
        evaluated[name] = prediction
    evaluated.to_csv(args.out_dir / "final_evaluated_predictions.csv", index=False)
    truth = evaluated[f"true_{args.target_name}"].to_numpy(float)
    metric_rows = [three_way.metrics_row(name, truth, evaluated[f"{name}_{args.target_name}"].to_numpy(float), args.target_name)
                   for name in args.models]
    metric_rows += [three_way.metrics_row(name, truth, prediction, args.target_name)
                    for name, prediction in variant_predictions.items()]
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.out_dir / "metrics_table.csv", index=False)
    best = metrics[metrics.model.isin(args.models)].sort_values("MAE").iloc[0]
    baseline = metrics[metrics.model == "non_llm_hybrid_baseline"].iloc[0]
    lines = ["# LLM-Augmented Three-Way Experiment", "", "All metrics below use only the untouched `final_test` split.", "",
             f"- Best base model MAE: {best['model']} ({best['MAE']:.6f})",
             f"- Non-LLM hybrid MAE / RMSE / R2: {baseline['MAE']:.6f} / {baseline['RMSE']:.6f} / {baseline['R2']:.6f}"]
    labels_by_variant = {"llm_numeric_only_hybrid": "LLM numeric hybrid",
        "llm_descriptor_only_hybrid": "LLM descriptor hybrid",
        "llm_numeric_plus_descriptor_hybrid": "LLM numeric + descriptor hybrid"}
    improvements = []
    for name, label in labels_by_variant.items():
        selected = metrics[metrics.model == name]
        if selected.empty:
            lines.append(f"- {label}: not requested")
            continue
        row = selected.iloc[0]
        delta = float(baseline.MAE - row.MAE)
        improvements.append(delta)
        lines.append(f"- {label} MAE / RMSE / R2: {row.MAE:.6f} / {row.RMSE:.6f} / {row.R2:.6f}")
        lines.append(f"  - Exact MAE improvement vs non-LLM hybrid: {delta:+.6f} "
                     f"({'improvement' if delta > 0 else 'degradation' if delta < 0 else 'no change'})")
    lines += [f"- Any LLM-augmented model improved: {'yes' if any(x > 0 for x in improvements) else 'no'}",
              f"- Leakage detected: {leak['leakage_detected']}", ""]
    (args.out_dir / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")
    counts = {split: int((featured.split == split).sum()) for split in three_way.SPLITS}
    _json(args.out_dir / "experiment_config.json", {"target_name": args.target_name,
        "split_type": args.split_type, "seed": args.seed, "fractions": dict(zip(three_way.SPLITS, fractions)),
        "models": args.models, "llm_mode": args.llm_mode, "llm_feature_mode": args.llm_feature_mode,
        "llm_model": args.llm_model, "input_dataset": str(dataset), "row_counts": counts,
        "invalid_smiles_policy": args.invalid_smiles_policy, "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "pandas": pd.__version__, "scikit_learn": sklearn.__version__}})


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
