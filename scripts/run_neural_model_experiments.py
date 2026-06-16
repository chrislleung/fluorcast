"""Run neural-network experiment comparisons for combined ChemFluor predictors."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    get_solvent_descriptor_row,
    load_solvent_descriptors as load_prediction_solvent_descriptors,
    require_rdkit,
)
from chemfluor.data_standardization import TARGET_COLUMNS  # noqa: E402
import run_combined_model_experiments as combined_experiments  # noqa: E402
import train_combined_predictors as trainer  # noqa: E402


DEFAULT_OUT_ROOT = Path("models/neural_experiments_fluodb")
DEFAULT_COMPARE_OUT = Path("outputs/neural_model_experiments_fluodb")
DEFAULT_MODELS = "mlp_small,mlp_medium,mlp_large,pytorch_mlp"
DEFAULT_TARGETS = "emission_nm,quantum_yield"
MLP_ARCHITECTURES = {
    "mlp_small": (256, 128),
    "mlp_medium": (512, 256),
    "mlp_large": (1024, 512, 256),
}
MLP_ALPHAS = (1e-3, 1e-4, 1e-5)
TORCH_MODEL_NAME = "pytorch_mlp"
DEFAULT_APPLICABILITY_THRESHOLD = 0.50


@dataclass(frozen=True)
class TargetData:
    """Train/test matrices and metadata for one target."""

    target_rows: pd.DataFrame
    descriptor_values: pd.DataFrame
    train_index: np.ndarray
    test_index: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    medians: pd.Series


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and compare neural ChemFluor model families."
    )
    parser.add_argument("--standardized-combined", required=True, type=Path)
    parser.add_argument(
        "--solvent-descriptors",
        default=trainer.DEFAULT_SOLVENT_DESCRIPTORS,
        type=Path,
    )
    parser.add_argument(
        "--tree-compare-dir",
        default=combined_experiments.DEFAULT_COMPARE_OUT,
        type=Path,
    )
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, type=Path)
    parser.add_argument("--compare-out", default=DEFAULT_COMPARE_OUT, type=Path)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--max-train-rows", default=None, type=int)
    parser.add_argument("--n-bits", default=2048, type=int)
    parser.add_argument("--radius", default=2, type=int)
    parser.add_argument("--max-iter", default=500, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--benchmark-smiles", default=None)
    parser.add_argument("--benchmark-solvent-smiles", default=None)
    parser.add_argument("--known-emission-nm", default=None, type=float)
    parser.add_argument("--known-quantum-yield", default=None, type=float)
    return parser.parse_args()


def parse_csv_list(text: str) -> list[str]:
    """Parse a comma-separated CLI list."""
    return [item.strip() for item in text.split(",") if item.strip()]


def alpha_label(alpha: float) -> str:
    """Format alpha for stable path and model names."""
    return f"{alpha:.0e}"


def requested_model_names(models: list[str], skip_pytorch: bool) -> list[str]:
    """Expand architecture names into concrete trainable model names."""
    expanded: list[str] = []
    for model in models:
        if model in MLP_ARCHITECTURES:
            expanded.extend(f"{model}_alpha_{alpha_label(alpha)}" for alpha in MLP_ALPHAS)
        elif model == TORCH_MODEL_NAME:
            if not skip_pytorch:
                expanded.append(model)
        else:
            raise ValueError(
                f"Unknown neural model '{model}'. Valid models: "
                + ", ".join([*MLP_ARCHITECTURES, TORCH_MODEL_NAME])
            )
    return expanded


def parse_neural_model_name(model_name: str) -> tuple[str, tuple[int, ...], float]:
    """Return architecture key, hidden layers, and alpha for a sklearn MLP variant."""
    for architecture, hidden_layers in MLP_ARCHITECTURES.items():
        prefix = f"{architecture}_alpha_"
        if model_name.startswith(prefix):
            return architecture, hidden_layers, float(model_name.removeprefix(prefix))
    raise ValueError(f"Not a sklearn MLP variant: {model_name}")


def load_training_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """Load combined rows, solvent descriptors, modeling rows, and fingerprints."""
    require_rdkit()
    combined_rows = trainer.load_standardized_combined(args.standardized_combined)
    if args.max_train_rows is not None and len(combined_rows) > args.max_train_rows:
        combined_rows = combined_rows.sample(
            n=args.max_train_rows, random_state=args.random_state
        ).reset_index(drop=True)
        print(f"Using {len(combined_rows)} sampled row(s) due to --max-train-rows.")

    solvent_descriptors = trainer.load_solvent_descriptors(args.solvent_descriptors)
    modeling_rows, descriptor_columns = trainer.merge_solvent_descriptors(
        combined_rows, solvent_descriptors
    )
    modeling_rows, fingerprints = trainer.add_fingerprints(
        modeling_rows, radius=args.radius, n_bits=args.n_bits
    )
    return modeling_rows, descriptor_columns, fingerprints


def prepare_target_data(
    target: str,
    rows: pd.DataFrame,
    fingerprints: np.ndarray,
    descriptor_columns: list[str],
    test_size: float,
    random_state: int,
) -> TargetData | None:
    """Build target-specific grouped train/test data."""
    target_rows = rows[rows[target].notna()].copy()
    if len(target_rows) < 4:
        print(f"WARNING: skipping {target}; only {len(target_rows)} usable rows.")
        return None

    groups = target_rows["canonical_chromophore_smiles"].to_numpy()
    if pd.Series(groups).nunique() < 2:
        print(f"WARNING: skipping {target}; fewer than 2 chromophore groups.")
        return None

    splitter = GroupShuffleSplit(
        test_size=test_size, random_state=random_state, n_splits=1
    )
    train_index, test_index = next(splitter.split(target_rows, groups=groups))
    descriptor_values = target_rows[descriptor_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    medians = descriptor_values.iloc[train_index].median(numeric_only=True)
    target_fingerprints = fingerprints[target_rows.index.to_numpy()]
    x_train = trainer.build_feature_matrix(
        target_fingerprints[train_index],
        descriptor_values.iloc[train_index],
        medians,
    )
    x_test = trainer.build_feature_matrix(
        target_fingerprints[test_index],
        descriptor_values.iloc[test_index],
        medians,
    )
    return TargetData(
        target_rows=target_rows,
        descriptor_values=descriptor_values,
        train_index=train_index,
        test_index=test_index,
        x_train=x_train,
        x_test=x_test,
        y_train=target_rows[target].iloc[train_index].to_numpy(dtype=float),
        y_test=target_rows[target].iloc[test_index].to_numpy(dtype=float),
        medians=medians,
    )


def metric_payload(
    target: str,
    model_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    train_rows: int,
    test_rows: int,
    model_path: Path,
    prediction_path: Path,
) -> dict[str, Any]:
    """Build a standard metrics payload."""
    return {
        "target": target,
        "model_type": model_name,
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    }


def save_predictions(
    target_data: TargetData,
    y_pred: np.ndarray,
    target: str,
    out_dir: Path,
) -> Path:
    """Save target test predictions in the combined workflow schema."""
    predictions = target_data.target_rows.iloc[target_data.test_index][
        [
            "canonical_chromophore_smiles",
            "solvent_original",
            "canonical_solvent_smiles",
            "source_dataset",
        ]
    ].copy()
    predictions["y_true"] = target_data.y_test
    predictions["y_pred"] = y_pred
    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    prediction_path = out_dir / f"predictions_{target}.csv"
    predictions.to_csv(prediction_path, index=False)
    predictions.to_csv(out_dir / f"predictions_test_{target}.csv", index=False)
    return prediction_path


def make_sklearn_mlp(
    hidden_layers: tuple[int, ...],
    alpha: float,
    max_iter: int,
    random_state: int,
) -> Pipeline:
    """Create a scaled sklearn MLP regressor."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    alpha=alpha,
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    validation_fraction=0.1,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_sklearn_model(
    model_name: str,
    target_data_by_target: dict[str, TargetData],
    args: argparse.Namespace,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, dict[str, float | None]]]:
    """Train one sklearn MLP variant for all requested targets."""
    _, hidden_layers, alpha = parse_neural_model_name(model_name)
    out_dir = args.out_root / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_target: dict[str, dict[str, Any]] = {}
    medians_by_target: dict[str, dict[str, float | None]] = {}

    for target, target_data in target_data_by_target.items():
        print(f"Training {model_name} for {target}")
        model = make_sklearn_mlp(
            hidden_layers=hidden_layers,
            alpha=alpha,
            max_iter=args.max_iter,
            random_state=args.random_state,
        )
        model.fit(target_data.x_train, target_data.y_train)
        y_pred = model.predict(target_data.x_test)
        model_path = out_dir / f"{target}_{model_name}.joblib"
        joblib.dump(model, model_path)
        prediction_path = save_predictions(target_data, y_pred, target, out_dir)
        metrics_by_target[target] = metric_payload(
            target=target,
            model_name=model_name,
            y_test=target_data.y_test,
            y_pred=y_pred,
            train_rows=len(target_data.train_index),
            test_rows=len(target_data.test_index),
            model_path=model_path,
            prediction_path=prediction_path,
        )
        medians_by_target[target] = {
            key: (None if pd.isna(value) else float(value))
            for key, value in target_data.medians.items()
        }
    return out_dir, metrics_by_target, medians_by_target


def import_torch() -> Any | None:
    """Import torch lazily and return None when unavailable."""
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def train_pytorch_single_target(
    target_data: TargetData,
    target: str,
    model_name: str,
    out_dir: Path,
    args: argparse.Namespace,
    torch: Any,
) -> dict[str, Any]:
    """Train and save one PyTorch MLP target model."""
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(target_data.x_train).astype(np.float32)
    x_test_scaled = x_scaler.transform(target_data.x_test).astype(np.float32)
    y_train_scaled = y_scaler.fit_transform(
        target_data.y_train.reshape(-1, 1)
    ).astype(np.float32)

    train_idx, val_idx = train_test_split(
        np.arange(len(x_train_scaled)),
        test_size=0.1,
        random_state=args.random_state,
    )

    nn = torch.nn
    model = nn.Sequential(
        nn.Linear(x_train_scaled.shape[1], 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    loss_fn = nn.SmoothL1Loss()

    x_train_tensor = torch.tensor(x_train_scaled[train_idx], device=device)
    y_train_tensor = torch.tensor(y_train_scaled[train_idx], device=device)
    x_val_tensor = torch.tensor(x_train_scaled[val_idx], device=device)
    x_test_tensor = torch.tensor(x_test_scaled, device=device)
    train_count = len(train_idx)
    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    stale_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(train_count, device=device)
        for start in range(0, train_count, args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(x_train_tensor[batch_idx]), y_train_tensor[batch_idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scaled = model(x_val_tensor).detach().cpu().numpy()
        val_pred = y_scaler.inverse_transform(val_scaled).ravel()
        val_mae = mean_absolute_error(target_data.y_train[val_idx], val_pred)
        if val_mae < best_val_mae:
            best_val_mae = float(val_mae)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"Early stopping {model_name} {target} at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_scaled = model(x_test_tensor).detach().cpu().numpy()
    y_pred = y_scaler.inverse_transform(test_scaled).ravel()

    model_path = out_dir / f"{target}_{model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(x_train_scaled.shape[1]),
            "target": target,
            "model_name": model_name,
        },
        model_path,
    )
    joblib.dump(x_scaler, out_dir / f"{target}_{model_name}_feature_scaler.joblib")
    joblib.dump(y_scaler, out_dir / f"{target}_{model_name}_target_scaler.joblib")
    prediction_path = save_predictions(target_data, y_pred, target, out_dir)
    return metric_payload(
        target=target,
        model_name=model_name,
        y_test=target_data.y_test,
        y_pred=y_pred,
        train_rows=len(target_data.train_index),
        test_rows=len(target_data.test_index),
        model_path=model_path,
        prediction_path=prediction_path,
    )


def train_pytorch_model(
    target_data_by_target: dict[str, TargetData],
    args: argparse.Namespace,
) -> tuple[Path | None, dict[str, dict[str, Any]], dict[str, dict[str, float | None]]]:
    """Train PyTorch MLP models when torch is installed."""
    torch = import_torch()
    if torch is None:
        print("WARNING: torch is not installed; skipping pytorch_mlp.")
        return None, {}, {}
    out_dir = args.out_root / TORCH_MODEL_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_by_target: dict[str, dict[str, Any]] = {}
    medians_by_target: dict[str, dict[str, float | None]] = {}
    for target, target_data in target_data_by_target.items():
        print(f"Training {TORCH_MODEL_NAME} for {target}")
        metrics_by_target[target] = train_pytorch_single_target(
            target_data=target_data,
            target=target,
            model_name=TORCH_MODEL_NAME,
            out_dir=out_dir,
            args=args,
            torch=torch,
        )
        medians_by_target[target] = {
            key: (None if pd.isna(value) else float(value))
            for key, value in target_data.medians.items()
        }
    return out_dir, metrics_by_target, medians_by_target


def save_model_metadata(
    out_dir: Path,
    model_name: str,
    metrics_by_target: dict[str, dict[str, Any]],
    medians_by_target: dict[str, dict[str, float | None]],
    selected_targets: list[str],
    descriptor_columns: list[str],
    args: argparse.Namespace,
    modeling_rows_source: Path,
) -> None:
    """Save metrics and feature metadata in the combined-model layout."""
    metadata = {
        "fingerprint_radius": args.radius,
        "fingerprint_n_bits": args.n_bits,
        "solvent_descriptor_columns_used": descriptor_columns,
        "target_columns": selected_targets,
        "model_type": model_name,
        "median_values_used_for_imputation": medians_by_target,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_by_target, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    shutil.copyfile(
        modeling_rows_source,
        out_dir / "combined_modeling_rows_after_feature_merge.csv",
    )


def collect_model_metrics(model_dirs: dict[str, Path]) -> pd.DataFrame:
    """Collect neural model metrics into one comparison table."""
    rows: list[dict[str, Any]] = []
    for model_name, model_dir in model_dirs.items():
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for metric in metrics.values():
            rows.append(
                {
                    "model": model_name,
                    "model_family": "neural",
                    "target": metric.get("target"),
                    "mae": metric.get("mae"),
                    "rmse": metric.get("rmse"),
                    "r2": metric.get("r2"),
                    "train_rows": metric.get("train_rows"),
                    "test_rows": metric.get("test_rows"),
                    "best_variant_for_architecture": False,
                }
            )
    comparison = pd.DataFrame(rows)
    if comparison.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "model_family",
                "target",
                "mae",
                "rmse",
                "r2",
                "train_rows",
                "test_rows",
                "best_variant_for_architecture",
            ]
        )
    comparison = mark_best_variants(comparison)
    return sort_comparison(comparison)


def architecture_from_model_name(model_name: str) -> str:
    """Return the architecture key for best-variant marking."""
    for architecture in MLP_ARCHITECTURES:
        if model_name.startswith(f"{architecture}_alpha_"):
            return architecture
    return model_name


def mark_best_variants(comparison: pd.DataFrame) -> pd.DataFrame:
    """Mark the best alpha variant for each sklearn architecture and target."""
    marked = comparison.copy()
    marked["architecture"] = marked["model"].map(architecture_from_model_name)
    marked["best_variant_for_architecture"] = False
    for (_, target, architecture), group in marked.groupby(
        ["model_family", "target", "architecture"], dropna=False
    ):
        if len(group) <= 1:
            continue
        best_index = group["mae"].astype(float).idxmin()
        marked.loc[best_index, "best_variant_for_architecture"] = True
    return marked.drop(columns=["architecture"])


def sort_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Sort comparison tables with emission first and MAE ascending within target."""
    if comparison.empty:
        return comparison
    working = comparison.copy()
    working["_target_rank"] = working["target"].map(
        {"emission_nm": 0, "quantum_yield": 1}
    ).fillna(2)
    working["mae"] = pd.to_numeric(working["mae"], errors="coerce")
    return (
        working.sort_values(["_target_rank", "target", "mae"], kind="mergesort")
        .drop(columns=["_target_rank"])
        .reset_index(drop=True)
    )


def write_markdown_comparison(
    comparison: pd.DataFrame, path: Path, title: str = "Neural Model Comparison"
) -> None:
    """Write a compact markdown table."""
    if comparison.empty:
        path.write_text(f"# {title}\n\nNo metrics were collected.\n", encoding="utf-8")
        return
    rounded = comparison.copy()
    for column in ["mae", "rmse", "r2"]:
        if column in rounded.columns:
            rounded[column] = rounded[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.4f}"
            )
    headers = list(rounded.columns)
    rows = [[str(value) for value in row] for row in rounded.to_numpy()]
    table = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )
    path.write_text(
        f"# {title}\n\nModels are ranked by target and MAE.\n\n{table}\n",
        encoding="utf-8",
    )


def predict_sklearn_benchmark(
    model_dir: Path,
    model_name: str,
    target: str,
    features: np.ndarray,
) -> float | None:
    """Predict benchmark value with a saved sklearn model."""
    model_path = model_dir / f"{target}_{model_name}.joblib"
    if not model_path.exists():
        return None
    model = joblib.load(model_path)
    return float(model.predict(features)[0])


def predict_pytorch_benchmark(
    model_dir: Path,
    target: str,
    features: np.ndarray,
) -> float | None:
    """Predict benchmark value with a saved PyTorch MLP."""
    torch = import_torch()
    model_path = model_dir / f"{target}_{TORCH_MODEL_NAME}.pt"
    if torch is None or not model_path.exists():
        return None
    payload = torch.load(model_path, map_location="cpu")
    nn = torch.nn
    model = nn.Sequential(
        nn.Linear(int(payload["input_dim"]), 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_scaler = joblib.load(model_dir / f"{target}_{TORCH_MODEL_NAME}_feature_scaler.joblib")
    y_scaler = joblib.load(model_dir / f"{target}_{TORCH_MODEL_NAME}_target_scaler.joblib")
    scaled = x_scaler.transform(features).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.tensor(scaled)).detach().numpy()
    return float(y_scaler.inverse_transform(pred_scaled).ravel()[0])


def benchmark_prediction_for_model(
    model_name: str,
    model_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Predict the optional benchmark molecule with one neural model directory."""
    require_rdkit()
    canonical_smiles = canonicalize_required(args.benchmark_smiles, "molecule")
    canonical_solvent = canonicalize_required(args.benchmark_solvent_smiles, "solvent")
    metadata = json.loads((model_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    solvent_descriptors = load_prediction_solvent_descriptors(args.solvent_descriptors)
    solvent_row = get_solvent_descriptor_row(
        solvent_descriptors,
        solvent_smiles=args.benchmark_solvent_smiles,
        canonical_solvent_smiles=canonical_solvent,
    )

    predictions: dict[str, float | None] = {}
    for target in ["emission_nm", "quantum_yield"]:
        medians = metadata.get("median_values_used_for_imputation", {}).get(target, {})
        features = build_single_feature_matrix(
            canonical_smiles=canonical_smiles,
            solvent_descriptor_row=solvent_row,
            descriptor_columns=list(metadata.get("solvent_descriptor_columns_used", [])),
            medians=medians,
            radius=int(metadata.get("fingerprint_radius", 2)),
            n_bits=int(metadata.get("fingerprint_n_bits", 2048)),
        )
        if model_name == TORCH_MODEL_NAME:
            predictions[target] = predict_pytorch_benchmark(model_dir, target, features)
        else:
            predictions[target] = predict_sklearn_benchmark(
                model_dir, model_name, target, features
            )

    domain, domain_warnings = applicability_domain_payload(
        canonical_smiles=canonical_smiles,
        model_dir=model_dir,
        threshold=DEFAULT_APPLICABILITY_THRESHOLD,
        radius=int(metadata.get("fingerprint_radius", 2)),
        n_bits=int(metadata.get("fingerprint_n_bits", 2048)),
        disabled=False,
    )
    for warning in domain_warnings:
        print(f"WARNING: {warning}")

    predicted_emission = predictions.get("emission_nm")
    predicted_qy = predictions.get("quantum_yield")
    return {
        "model": model_name,
        "model_family": "neural",
        "predicted_emission_nm": predicted_emission,
        "known_emission_nm": args.known_emission_nm,
        "emission_absolute_error": (
            None
            if predicted_emission is None or args.known_emission_nm is None
            else abs(predicted_emission - args.known_emission_nm)
        ),
        "predicted_quantum_yield": predicted_qy,
        "known_quantum_yield": args.known_quantum_yield,
        "quantum_yield_absolute_error": (
            None
            if predicted_qy is None or args.known_quantum_yield is None
            else abs(predicted_qy - args.known_quantum_yield)
        ),
        "nearest_training_similarity": domain.get("nearest_training_similarity"),
        "confidence_label": domain.get("confidence_label"),
        "outside_applicability_domain": domain.get("outside_applicability_domain"),
    }


def collect_benchmark_predictions(
    model_dirs: dict[str, Path], args: argparse.Namespace
) -> pd.DataFrame:
    """Run benchmark predictions when benchmark inputs are provided."""
    if not args.benchmark_smiles or not args.benchmark_solvent_smiles:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for model_name, model_dir in model_dirs.items():
        try:
            rows.append(benchmark_prediction_for_model(model_name, model_dir, args))
        except (FileNotFoundError, ValueError, ImportError, KeyError) as exc:
            print(f"WARNING: benchmark failed for {model_name}: {exc}")
    return pd.DataFrame(rows)


def read_tree_output(path: Path) -> pd.DataFrame:
    """Read an existing tree comparison CSV when present."""
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "model_family" not in table.columns and "model" in table.columns:
        table.insert(1, "model_family", "tree")
    return table


def combine_tables(neural: pd.DataFrame, tree: pd.DataFrame) -> pd.DataFrame:
    """Combine neural and tree comparison tables with compatible columns."""
    if tree.empty:
        return neural.copy()
    if neural.empty:
        return tree.copy()
    columns = list(dict.fromkeys([*tree.columns, *neural.columns]))
    return pd.concat(
        [tree.reindex(columns=columns), neural.reindex(columns=columns)],
        ignore_index=True,
    )


def write_outputs(
    compare_out: Path,
    tree_compare_dir: Path,
    comparison: pd.DataFrame,
    region_comparison: pd.DataFrame,
    benchmark_comparison: pd.DataFrame,
) -> None:
    """Write neural-only and combined tree-plus-neural comparison outputs."""
    compare_out.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(compare_out / "neural_model_comparison.csv", index=False)
    write_markdown_comparison(
        comparison,
        compare_out / "neural_model_comparison.md",
        title="Neural Model Comparison",
    )
    region_comparison.to_csv(
        compare_out / "neural_error_by_region_comparison.csv", index=False
    )
    benchmark_comparison.to_csv(
        compare_out / "neural_benchmark_prediction_comparison.csv", index=False
    )

    tree_comparison = read_tree_output(tree_compare_dir / "model_comparison.csv")
    all_comparison = combine_tables(comparison, tree_comparison)
    if {"target", "mae"}.issubset(all_comparison.columns):
        all_comparison = sort_comparison(all_comparison)
    all_comparison.to_csv(compare_out / "all_model_comparison.csv", index=False)
    write_markdown_comparison(
        all_comparison,
        compare_out / "all_model_comparison.md",
        title="All Model Comparison",
    )

    tree_region = read_tree_output(tree_compare_dir / "error_by_region_comparison.csv")
    all_region = combine_tables(region_comparison, tree_region)
    all_region.to_csv(compare_out / "all_error_by_region_comparison.csv", index=False)

    tree_benchmark = read_tree_output(
        tree_compare_dir / "benchmark_prediction_comparison.csv"
    )
    all_benchmark = combine_tables(benchmark_comparison, tree_benchmark)
    all_benchmark.to_csv(
        compare_out / "all_benchmark_prediction_comparison.csv", index=False
    )


def main() -> int:
    """Run all requested neural model experiments."""
    args = parse_args()
    try:
        selected_models = requested_model_names(
            parse_csv_list(args.models), skip_pytorch=args.skip_pytorch
        )
        selected_targets = trainer.parse_targets(args.targets)
        args.out_root.mkdir(parents=True, exist_ok=True)

        modeling_rows, descriptor_columns, fingerprints = load_training_inputs(args)
        modeling_rows_source = args.out_root / "combined_modeling_rows_after_feature_merge.csv"
        modeling_rows.to_csv(modeling_rows_source, index=False)
        target_data_by_target = {
            target: target_data
            for target in selected_targets
            if (
                target_data := prepare_target_data(
                    target=target,
                    rows=modeling_rows,
                    fingerprints=fingerprints,
                    descriptor_columns=descriptor_columns,
                    test_size=args.test_size,
                    random_state=args.random_state,
                )
            )
            is not None
        }
        if not target_data_by_target:
            raise ValueError("No requested targets had enough usable rows to train.")

        model_dirs: dict[str, Path] = {}
        for model_name in selected_models:
            if model_name == TORCH_MODEL_NAME:
                model_dir, metrics_by_target, medians_by_target = train_pytorch_model(
                    target_data_by_target, args
                )
                if model_dir is None:
                    continue
            else:
                model_dir, metrics_by_target, medians_by_target = train_sklearn_model(
                    model_name, target_data_by_target, args
                )
            if metrics_by_target:
                save_model_metadata(
                    out_dir=model_dir,
                    model_name=model_name,
                    metrics_by_target=metrics_by_target,
                    medians_by_target=medians_by_target,
                    selected_targets=selected_targets,
                    descriptor_columns=descriptor_columns,
                    args=args,
                    modeling_rows_source=modeling_rows_source,
                )
                model_dirs[model_name] = model_dir

        comparison = collect_model_metrics(model_dirs)
        region_comparison = combined_experiments.collect_error_by_region(model_dirs)
        if not region_comparison.empty:
            region_comparison.insert(1, "model_family", "neural")
        benchmark_comparison = collect_benchmark_predictions(model_dirs, args)
        write_outputs(
            compare_out=args.compare_out,
            tree_compare_dir=args.tree_compare_dir,
            comparison=comparison,
            region_comparison=region_comparison,
            benchmark_comparison=benchmark_comparison,
        )
        print(f"Saved neural comparison to: {args.compare_out / 'neural_model_comparison.csv'}")
        print(f"Saved all-model comparison to: {args.compare_out / 'all_model_comparison.csv'}")
        return 0
    except (FileNotFoundError, ImportError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
