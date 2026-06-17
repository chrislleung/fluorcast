"""Predict one molecule with every available ChemFluor model family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from chemfluor.combined_prediction import (  # noqa: E402
    applicability_domain_payload,
    build_single_feature_matrix,
    canonicalize_required,
    compute_nearest_training_similarity,
    get_solvent_descriptor_row,
    load_or_infer_feature_metadata,
    load_reference_fingerprints,
    load_solvent_descriptors,
    require_rdkit,
    similarity_confidence_label,
)
from chemfluor.data_standardization import TARGET_COLUMNS  # noqa: E402
from chemfluor.graph_features import mol_to_graph  # noqa: E402


DEFAULT_SOLVENT_DESCRIPTORS = Path("data/solvent_descriptors_expanded_deep4chem.csv")
DEFAULT_STANDARDIZED_COMBINED = Path("data/processed/fluodb_lite/combined_deduplicated.csv")
DEFAULT_TREE_MODEL_DIR = Path("models/experiments_fluodb")
DEFAULT_NEURAL_MODEL_DIR = Path("models/neural_experiments_fluodb")
DEFAULT_GRAPH_MODEL_DIR = Path("models/graph_experiments_fluodb")
DEFAULT_APPLICABILITY_THRESHOLD = 0.50
PREDICTION_TARGETS = ["emission_nm", "quantum_yield"]
OUTPUT_COLUMNS = [
    "model",
    "model_family",
    "target",
    "seed",
    "predicted_value",
    "predicted_emission_nm",
    "predicted_quantum_yield",
    "nearest_training_similarity",
    "nearest_training_smiles",
    "confidence_label",
    "outside_applicability_domain",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(
        description=(
            "Predict one chromophore/solvent pair with every available trained "
            "ChemFluor tree, neural, and graph model."
        )
    )
    parser.add_argument("--smiles", required=True, help="Chromophore SMILES.")
    parser.add_argument("--solvent", default=None, help="Solvent name, for example ethanol.")
    parser.add_argument("--solvent-smiles", default=None, help="Solvent SMILES, for example CCO.")
    parser.add_argument(
        "--solvent-descriptors",
        default=DEFAULT_SOLVENT_DESCRIPTORS,
        type=Path,
        help="Solvent descriptor CSV.",
    )
    parser.add_argument(
        "--standardized-combined",
        default=DEFAULT_STANDARDIZED_COMBINED,
        type=Path,
        help="Fallback standardized combined CSV for applicability-domain scoring.",
    )
    parser.add_argument("--tree-model-dir", default=DEFAULT_TREE_MODEL_DIR, type=Path)
    parser.add_argument("--neural-model-dir", default=DEFAULT_NEURAL_MODEL_DIR, type=Path)
    parser.add_argument(
        "--graph-model-dirs",
        nargs="*",
        default=None,
        type=Path,
        help="Optional graph model directories. Defaults to subdirectories of models/graph_experiments_fluodb.",
    )
    parser.add_argument("--out", default=None, type=Path, help="Optional CSV output path.")
    parser.add_argument(
        "--applicability-threshold",
        default=DEFAULT_APPLICABILITY_THRESHOLD,
        type=float,
        help="Nearest-training similarity threshold for outside-domain flagging.",
    )
    return parser.parse_args()


def discover_model_dirs(root: Path) -> list[Path]:
    """Return model directories under root, accepting root itself as a model dir."""
    if not root.exists():
        return []
    if (root / "feature_metadata.json").exists() or any(root.glob("*.joblib")):
        return [root]
    return sorted(
        child
        for child in root.iterdir()
        if child.is_dir()
        and ((child / "feature_metadata.json").exists() or any(child.glob("*.joblib")))
    )


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON if present."""
    return json.loads(path.read_text(encoding="utf-8"))


def model_name_from_dir(model_dir: Path) -> str:
    """Infer a stable model name from metadata or directory name."""
    metadata_path = model_dir / "feature_metadata.json"
    if metadata_path.exists():
        metadata = load_json(metadata_path)
        model_type = metadata.get("model_type")
        if model_type:
            return str(model_type)
    return model_dir.name


def model_seed(model_dir: Path) -> int | None:
    """Infer model seed from metadata or metrics."""
    for filename in ["feature_metadata.json", "metrics.json"]:
        path = model_dir / filename
        if not path.exists():
            continue
        payload = load_json(path)
        if "seed" in payload and payload["seed"] is not None:
            return int(payload["seed"])
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, dict) and value.get("seed") is not None:
                    return int(value["seed"])
    return None


def resolve_solvent(
    solvent: str | None,
    solvent_smiles: str | None,
    descriptors: pd.DataFrame,
) -> tuple[str, str | None, pd.Series | None]:
    """Resolve solvent name or SMILES to a descriptor row and canonical solvent SMILES."""
    if not solvent and not solvent_smiles:
        raise ValueError("Provide either --solvent or --solvent-smiles.")

    if solvent_smiles:
        canonical_solvent = canonicalize_required(solvent_smiles, "solvent")
        row = get_solvent_descriptor_row(
            descriptors=descriptors,
            solvent_smiles=solvent_smiles,
            canonical_solvent_smiles=canonical_solvent,
        )
        return solvent_smiles, canonical_solvent, row

    assert solvent is not None
    label = solvent.strip()
    for column in ["solvent_original", "solvent"]:
        if column not in descriptors.columns:
            continue
        matches = descriptors[
            descriptors[column].astype("string").str.strip().str.lower()
            == label.lower()
        ]
        if not matches.empty:
            row = matches.iloc[0]
            canonical = row.get("canonical_solvent_smiles")
            canonical_text = None if pd.isna(canonical) else str(canonical).strip()
            return label, canonical_text or None, row
    return label, None, None


def first_model_feature_count(model_dir: Path, model_name: str) -> int | None:
    """Return n_features_in_ from the first matching joblib model."""
    for path in sorted(model_dir.glob(f"*_{model_name}.joblib")):
        try:
            return int(getattr(joblib.load(path), "n_features_in_", None))
        except Exception:
            continue
    return None


def build_features_for_target(
    canonical_smiles: str,
    solvent_row: pd.Series | None,
    metadata: dict[str, Any],
    target: str,
) -> np.ndarray:
    """Build the combined fingerprint plus solvent-descriptor feature matrix."""
    return build_single_feature_matrix(
        canonical_smiles=canonical_smiles,
        solvent_descriptor_row=solvent_row,
        descriptor_columns=list(metadata.get("solvent_descriptor_columns_used", [])),
        medians=metadata.get("median_values_used_for_imputation", {}).get(target, {}),
        radius=int(metadata.get("fingerprint_radius", 2)),
        n_bits=int(metadata.get("fingerprint_n_bits", 2048)),
    )


def predict_joblib_targets(
    model_dir: Path,
    model_name: str,
    canonical_smiles: str,
    solvent_row: pd.Series | None,
    solvent_descriptors: pd.DataFrame,
) -> tuple[dict[str, float], list[str]]:
    """Predict available targets for sklearn/joblib models."""
    warnings: list[str] = []
    metadata, metadata_warnings = load_or_infer_feature_metadata(
        model_dir=model_dir,
        solvent_descriptors=solvent_descriptors,
        model_feature_count=first_model_feature_count(model_dir, model_name),
    )
    warnings.extend(metadata_warnings)
    predictions: dict[str, float] = {}
    for target in PREDICTION_TARGETS:
        path = model_dir / f"{target}_{model_name}.joblib"
        if not path.exists():
            warnings.append(f"Model file not found; skipping: {path}")
            continue
        try:
            model = joblib.load(path)
            features = build_features_for_target(canonical_smiles, solvent_row, metadata, target)
            predictions[target] = float(model.predict(features)[0])
        except Exception as exc:
            warnings.append(f"Failed to predict {target} with {model_name}; skipping: {exc}")
    return predictions, warnings


def import_neural_helpers() -> Any | None:
    """Import neural experiment helpers lazily."""
    try:
        import run_neural_model_experiments as neural_helpers
    except ImportError:
        return None
    return neural_helpers


def import_graph_helpers() -> Any | None:
    """Import graph experiment helpers lazily."""
    try:
        import run_graph_model_experiments as graph_helpers
    except ImportError:
        return None
    return graph_helpers


def predict_neural_targets(
    model_dir: Path,
    model_name: str,
    canonical_smiles: str,
    solvent_row: pd.Series | None,
) -> tuple[dict[str, float], list[str]]:
    """Predict available targets for neural model directories."""
    warnings: list[str] = []
    helpers = import_neural_helpers()
    if helpers is None:
        return {}, [f"Neural helpers unavailable; skipping: {model_dir}"]
    metadata_path = model_dir / "feature_metadata.json"
    if not metadata_path.exists():
        return {}, [f"feature_metadata.json not found; skipping neural model: {model_dir}"]
    metadata = load_json(metadata_path)
    predictions: dict[str, float] = {}
    for target in PREDICTION_TARGETS:
        features = build_features_for_target(canonical_smiles, solvent_row, metadata, target)
        try:
            if model_name == helpers.TORCH_MODEL_NAME:
                predicted = helpers.predict_pytorch_benchmark(model_dir, target, features)
            else:
                predicted = helpers.predict_sklearn_benchmark(
                    model_dir, model_name, target, features
                )
        except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
            warnings.append(f"Failed to predict {target} with {model_name}; skipping: {exc}")
            continue
        if predicted is None:
            warnings.append(f"Model file not found; skipping {model_name} {target}.")
            continue
        predictions[target] = float(predicted)
    return predictions, warnings


def solvent_vector_for_graph(
    metadata: dict[str, Any],
    target: str,
    solvent_row: pd.Series | None,
) -> np.ndarray:
    """Build the raw solvent descriptor vector expected by graph helpers."""
    values: list[float] = []
    medians = metadata.get("median_values_used_for_imputation", {}).get(target, {})
    for column in metadata.get("solvent_descriptor_columns_used", []):
        value = pd.NA if solvent_row is None or column not in solvent_row.index else solvent_row[column]
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            numeric = medians.get(column, 0.0)
        values.append(float(0.0 if pd.isna(numeric) else numeric))
    return np.asarray(values, dtype=np.float32)


def predict_graph_targets(
    model_dir: Path,
    model_name: str,
    canonical_smiles: str,
    solvent_row: pd.Series | None,
) -> tuple[dict[str, float], list[str]]:
    """Predict available targets for graph neural model directories."""
    warnings: list[str] = []
    helpers = import_graph_helpers()
    if helpers is None:
        return {}, [f"Graph helpers unavailable; skipping: {model_dir}"]
    torch = helpers.import_torch()
    if torch is None:
        return {}, [f"PyTorch unavailable; skipping graph model: {model_dir}"]
    metadata_path = model_dir / "feature_metadata.json"
    if not metadata_path.exists():
        return {}, [f"feature_metadata.json not found; skipping graph model: {model_dir}"]
    graph = mol_to_graph(canonical_smiles)
    if graph is None:
        raise ValueError(f"Could not convert molecule to graph: {canonical_smiles}")
    metadata = load_json(metadata_path)
    predictions: dict[str, float] = {}
    for target in PREDICTION_TARGETS:
        vector = solvent_vector_for_graph(metadata, target, solvent_row)
        try:
            predicted = helpers.predict_graph_target(
                model_dir, model_name, target, graph, vector, torch
            )
        except (FileNotFoundError, ValueError, KeyError, ImportError) as exc:
            warnings.append(f"Failed to predict {target} with {model_name}; skipping: {exc}")
            continue
        if predicted is None:
            warnings.append(f"Model file not found; skipping {model_name} {target}.")
            continue
        predictions[target] = float(predicted)
    return predictions, warnings


def fallback_applicability_domain(
    canonical_smiles: str,
    standardized_combined: Path,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    """Score applicability against the standardized combined CSV when model refs are missing."""
    if not standardized_combined.exists():
        return (
            {"confidence_label": "unknown"},
            ["No applicability reference CSV found; continuing without scoring."],
        )
    fps, smiles = load_reference_fingerprints(standardized_combined, radius=2, n_bits=2048)
    similarity, nearest = compute_nearest_training_similarity(
        canonical_smiles, fps, smiles, radius=2, n_bits=2048
    )
    label = similarity_confidence_label(similarity)
    return (
        {
            "nearest_training_similarity": similarity,
            "nearest_training_smiles": nearest,
            "confidence_label": label,
            "outside_applicability_domain": bool(pd.isna(similarity) or similarity < threshold),
        },
        [],
    )


def compute_domain(
    canonical_smiles: str,
    model_dir: Path,
    standardized_combined: Path,
    threshold: float,
) -> tuple[dict[str, Any], list[str]]:
    """Compute applicability-domain payload, with standardized-combined fallback."""
    domain, warnings = applicability_domain_payload(
        canonical_smiles=canonical_smiles,
        model_dir=model_dir,
        threshold=threshold,
        radius=2,
        n_bits=2048,
        disabled=False,
    )
    if domain.get("nearest_training_similarity") is not None:
        return domain, warnings
    fallback_domain, fallback_warnings = fallback_applicability_domain(
        canonical_smiles, standardized_combined, threshold
    )
    return fallback_domain, [*warnings, *fallback_warnings]


def rows_for_model(
    model: str,
    model_family: str,
    seed: int | None,
    predictions: dict[str, float],
    domain: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create one output row per predicted target."""
    rows = []
    for target, predicted in predictions.items():
        rows.append(
            {
                "model": model,
                "model_family": model_family,
                "target": target,
                "seed": seed,
                "predicted_value": predicted,
                "predicted_emission_nm": predictions.get("emission_nm"),
                "predicted_quantum_yield": predictions.get("quantum_yield"),
                "nearest_training_similarity": domain.get("nearest_training_similarity"),
                "nearest_training_smiles": domain.get("nearest_training_smiles"),
                "confidence_label": domain.get("confidence_label"),
                "outside_applicability_domain": domain.get("outside_applicability_domain"),
            }
        )
    return rows


def disagreement_summary(values: pd.Series) -> dict[str, float | None]:
    """Compute model-disagreement summary statistics."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {key: None for key in ["mean", "median", "std", "min", "max", "range"]}
    return {
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "range": float(numeric.max() - numeric.min()),
    }


def compute_disagreement_summaries(table: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    """Compute emission and quantum-yield disagreement across unique models."""
    if table.empty:
        return {
            "emission_nm": disagreement_summary(pd.Series(dtype=float)),
            "quantum_yield": disagreement_summary(pd.Series(dtype=float)),
        }
    model_level = table.drop_duplicates(["model", "model_family", "seed"])
    return {
        "emission_nm": disagreement_summary(model_level["predicted_emission_nm"]),
        "quantum_yield": disagreement_summary(model_level["predicted_quantum_yield"]),
    }


def print_report(
    canonical_smiles: str,
    solvent_label: str,
    canonical_solvent: str | None,
    table: pd.DataFrame,
    summaries: dict[str, dict[str, float | None]],
    warnings: list[str],
) -> None:
    """Print a readable terminal report."""
    print("\nAll-model ChemFluor prediction")
    print(f"Canonical SMILES: {canonical_smiles}")
    print(f"Solvent input: {solvent_label}")
    if canonical_solvent:
        print(f"Canonical solvent SMILES: {canonical_solvent}")
    print("\nPredictions:")
    if table.empty:
        print("  (no available model predictions)")
    else:
        display = table[OUTPUT_COLUMNS].copy()
        print(display.to_string(index=False))
    for label, key in [
        ("Emission disagreement (nm)", "emission_nm"),
        ("Quantum-yield disagreement", "quantum_yield"),
    ]:
        stats = summaries[key]
        print(f"\n{label}:")
        print(
            "  "
            + ", ".join(
                f"{name}={'NA' if value is None else f'{value:.6g}'}"
                for name, value in stats.items()
            )
        )
    for warning in warnings:
        print(f"WARNING: {warning}")


def collect_predictions(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], str, str | None, str]:
    """Collect all available model predictions."""
    require_rdkit()
    canonical_smiles = canonicalize_required(args.smiles, "molecule")
    solvent_descriptors = load_solvent_descriptors(args.solvent_descriptors)
    solvent_label, canonical_solvent, solvent_row = resolve_solvent(
        args.solvent, args.solvent_smiles, solvent_descriptors
    )
    warnings: list[str] = []
    if solvent_row is None:
        warnings.append(
            "Solvent descriptors not found; available models will use training medians."
        )

    rows: list[dict[str, Any]] = []
    for model_dir in discover_model_dirs(args.tree_model_dir):
        model_name = model_name_from_dir(model_dir)
        predictions, model_warnings = predict_joblib_targets(
            model_dir, model_name, canonical_smiles, solvent_row, solvent_descriptors
        )
        warnings.extend(model_warnings)
        if not predictions:
            continue
        domain, domain_warnings = compute_domain(
            canonical_smiles, model_dir, args.standardized_combined, args.applicability_threshold
        )
        warnings.extend(domain_warnings)
        rows.extend(rows_for_model(model_name, "tree", model_seed(model_dir), predictions, domain))

    for model_dir in discover_model_dirs(args.neural_model_dir):
        model_name = model_name_from_dir(model_dir)
        predictions, model_warnings = predict_neural_targets(
            model_dir, model_name, canonical_smiles, solvent_row
        )
        warnings.extend(model_warnings)
        if not predictions:
            continue
        domain, domain_warnings = compute_domain(
            canonical_smiles, model_dir, args.standardized_combined, args.applicability_threshold
        )
        warnings.extend(domain_warnings)
        rows.extend(rows_for_model(model_name, "neural", model_seed(model_dir), predictions, domain))

    graph_dirs = args.graph_model_dirs
    if graph_dirs is None:
        graph_dirs = discover_model_dirs(DEFAULT_GRAPH_MODEL_DIR)
    for model_dir in graph_dirs:
        model_name = model_name_from_dir(model_dir)
        predictions, model_warnings = predict_graph_targets(
            model_dir, model_name, canonical_smiles, solvent_row
        )
        warnings.extend(model_warnings)
        if not predictions:
            continue
        domain, domain_warnings = compute_domain(
            canonical_smiles, model_dir, args.standardized_combined, args.applicability_threshold
        )
        warnings.extend(domain_warnings)
        rows.extend(rows_for_model(model_name, "graph_neural", model_seed(model_dir), predictions, domain))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS), warnings, canonical_smiles, canonical_solvent, solvent_label


def main() -> int:
    """Run all-model prediction."""
    args = parse_args()
    try:
        table, warnings, canonical_smiles, canonical_solvent, solvent_label = collect_predictions(args)
        summaries = compute_disagreement_summaries(table)
        print_report(
            canonical_smiles,
            solvent_label,
            canonical_solvent,
            table,
            summaries,
            warnings,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(args.out, index=False)
            print(f"\nSaved prediction table to: {args.out}")
        return 0
    except (FileNotFoundError, ImportError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
