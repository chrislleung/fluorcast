"""Run graph neural-network experiments for combined ChemFluor predictors."""

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
from sklearn.model_selection import GroupShuffleSplit
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
    canonicalize_required,
    compute_nearest_training_similarity,
    get_solvent_descriptor_row,
    load_solvent_descriptors as load_prediction_solvent_descriptors,
    morgan_bitvect,
    require_rdkit,
)
from chemfluor.graph_features import (  # noqa: E402
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    GraphData,
    mol_to_graph,
)
import run_combined_model_experiments as combined_experiments  # noqa: E402
import train_combined_predictors as trainer  # noqa: E402


DEFAULT_OUT_ROOT = Path("models/graph_experiments_fluodb")
DEFAULT_COMPARE_OUT = Path("outputs/graph_model_experiments_fluodb")
DEFAULT_MODELS = "graph_gcn,graph_mpnn,graph_gin"
DEFAULT_TARGETS = "emission_nm,quantum_yield"
GRAPH_MODEL_NAMES = {"graph_gcn", "graph_mpnn", "graph_gin"}
SIMILARITY_BINS = [0.0, 0.30, 0.50, 0.70, 0.85, 1.000001]
SIMILARITY_LABELS = ["0.00-0.30", "0.30-0.50", "0.50-0.70", "0.70-0.85", "0.85-1.00"]


@dataclass(frozen=True)
class GraphSample:
    """One graph-modeling row."""

    graph: GraphData
    solvent: np.ndarray
    y: float
    row: dict[str, Any]


@dataclass(frozen=True)
class GraphTargetData:
    """Target-specific graph samples and split indices."""

    target_rows: pd.DataFrame
    samples: list[GraphSample]
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    solvent_scaler: StandardScaler
    target_scaler: StandardScaler
    descriptor_medians: pd.Series
    train_reference_fps: list[Any]
    train_reference_smiles: list[str]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and compare graph neural ChemFluor model families."
    )
    parser.add_argument("--standardized-combined", required=True, type=Path)
    parser.add_argument("--solvent-descriptors", default=trainer.DEFAULT_SOLVENT_DESCRIPTORS, type=Path)
    parser.add_argument("--tree-compare-dir", default=combined_experiments.DEFAULT_COMPARE_OUT, type=Path)
    parser.add_argument("--neural-compare-dir", default=Path("outputs/neural_model_experiments_fluodb"), type=Path)
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, type=Path)
    parser.add_argument("--compare-out", default=DEFAULT_COMPARE_OUT, type=Path)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--val-size", default=0.1, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--hidden-dim", default=256, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=25, type=int)
    parser.add_argument("--max-train-rows", default=None, type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--benchmark-smiles", default=None)
    parser.add_argument("--benchmark-solvent-smiles", default=None)
    parser.add_argument("--known-emission-nm", default=None, type=float)
    parser.add_argument("--known-quantum-yield", default=None, type=float)
    parser.add_argument("--applicability-threshold", default=0.50, type=float)
    return parser.parse_args()


def import_torch() -> Any | None:
    """Import PyTorch lazily."""
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def parse_csv_list(text: str) -> list[str]:
    """Parse a comma-separated CLI list."""
    return [item.strip() for item in text.split(",") if item.strip()]


def validate_graph_models(models: list[str]) -> list[str]:
    """Validate requested graph model names."""
    invalid = [model for model in models if model not in GRAPH_MODEL_NAMES]
    if invalid:
        raise ValueError(
            "Unknown graph model(s): "
            + ", ".join(invalid)
            + ". Valid models: "
            + ", ".join(sorted(GRAPH_MODEL_NAMES))
        )
    return models


def load_graph_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Load rows and merge solvent descriptors using the combined workflow."""
    require_rdkit()
    combined_rows = trainer.load_standardized_combined(args.standardized_combined)
    if args.max_train_rows is not None and len(combined_rows) > args.max_train_rows:
        combined_rows = combined_rows.sample(
            n=args.max_train_rows, random_state=args.random_state
        ).reset_index(drop=True)
        print(f"Using {len(combined_rows)} sampled row(s) due to --max-train-rows.")
    solvent_descriptors = trainer.load_solvent_descriptors(args.solvent_descriptors)
    return trainer.merge_solvent_descriptors(combined_rows, solvent_descriptors)


def split_train_val_test(
    target_rows: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grouped chromophore train/validation/test indices."""
    groups = target_rows["canonical_chromophore_smiles"].to_numpy()
    splitter = GroupShuffleSplit(test_size=test_size, random_state=random_state, n_splits=1)
    train_val_index, test_index = next(splitter.split(target_rows, groups=groups))
    train_val_rows = target_rows.iloc[train_val_index]
    train_val_groups = train_val_rows["canonical_chromophore_smiles"].to_numpy()
    if pd.Series(train_val_groups).nunique() < 2 or len(train_val_index) < 4:
        return train_val_index, train_val_index[:0], test_index
    val_fraction = min(0.5, max(0.05, val_size / max(1e-9, 1.0 - test_size)))
    val_splitter = GroupShuffleSplit(
        test_size=val_fraction, random_state=random_state + 1, n_splits=1
    )
    train_local, val_local = next(
        val_splitter.split(train_val_rows, groups=train_val_groups)
    )
    return train_val_index[train_local], train_val_index[val_local], test_index


def prepare_target_data(
    target: str,
    rows: pd.DataFrame,
    descriptor_columns: list[str],
    args: argparse.Namespace,
) -> GraphTargetData | None:
    """Build graph samples, solvent features, and split/scaling artifacts."""
    target_rows = rows[rows[target].notna()].copy().reset_index(drop=True)
    if len(target_rows) < 4:
        print(f"WARNING: skipping {target}; only {len(target_rows)} usable rows.")
        return None
    if target_rows["canonical_chromophore_smiles"].nunique() < 2:
        print(f"WARNING: skipping {target}; fewer than 2 chromophore groups.")
        return None

    train_index, val_index, test_index = split_train_val_test(
        target_rows, args.test_size, args.val_size, args.random_state
    )
    descriptor_values = target_rows[descriptor_columns].apply(pd.to_numeric, errors="coerce")
    medians = descriptor_values.iloc[train_index].median(numeric_only=True)
    solvent_matrix = descriptor_values.fillna(medians).fillna(0.0).to_numpy(dtype=np.float32)
    solvent_scaler = StandardScaler()
    solvent_matrix[train_index] = solvent_scaler.fit_transform(solvent_matrix[train_index])
    if len(val_index):
        solvent_matrix[val_index] = solvent_scaler.transform(solvent_matrix[val_index])
    solvent_matrix[test_index] = solvent_scaler.transform(solvent_matrix[test_index])
    target_scaler = StandardScaler()
    target_scaler.fit(target_rows[target].iloc[train_index].to_numpy(dtype=float).reshape(-1, 1))

    samples: list[GraphSample] = []
    kept_indices: list[int] = []
    for index, row in target_rows.iterrows():
        graph = mol_to_graph(str(row["canonical_chromophore_smiles"]))
        if graph is None:
            continue
        samples.append(
            GraphSample(
                graph=graph,
                solvent=solvent_matrix[index],
                y=float(row[target]),
                row=row.to_dict(),
            )
        )
        kept_indices.append(index)
    if len(kept_indices) != len(target_rows):
        index_map = {old: new for new, old in enumerate(kept_indices)}
        train_index = np.asarray([index_map[i] for i in train_index if i in index_map])
        val_index = np.asarray([index_map[i] for i in val_index if i in index_map])
        test_index = np.asarray([index_map[i] for i in test_index if i in index_map])
        target_rows = target_rows.iloc[kept_indices].reset_index(drop=True)
    if len(train_index) == 0 or len(test_index) == 0:
        print(f"WARNING: skipping {target}; split produced empty train or test rows.")
        return None

    train_smiles = sorted(
        set(target_rows.iloc[train_index]["canonical_chromophore_smiles"].astype(str))
    )
    train_fps: list[Any] = []
    train_fp_smiles: list[str] = []
    for smiles in train_smiles:
        fp = morgan_bitvect(smiles, radius=2, n_bits=2048)
        if fp is not None:
            train_fps.append(fp)
            train_fp_smiles.append(smiles)
    return GraphTargetData(
        target_rows=target_rows,
        samples=samples,
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        solvent_scaler=solvent_scaler,
        target_scaler=target_scaler,
        descriptor_medians=medians,
        train_reference_fps=train_fps,
        train_reference_smiles=train_fp_smiles,
    )


def subset_samples(data: GraphTargetData, indices: np.ndarray) -> list[GraphSample]:
    """Return split samples by index."""
    return [data.samples[int(index)] for index in indices]


def collate_graph_batch(samples: list[GraphSample], torch: Any | None = None) -> dict[str, Any]:
    """Collate variable-size molecular graphs into one concatenated batch."""
    torch = torch or import_torch()
    if torch is None:
        raise ImportError("PyTorch is required for graph batching.")
    node_tensors = []
    edge_indices = []
    edge_attrs = []
    batch_parts = []
    solvents = []
    targets = []
    offset = 0
    for graph_index, sample in enumerate(samples):
        node_count = sample.graph.num_nodes
        node_tensors.append(torch.tensor(sample.graph.x, dtype=torch.float32))
        edge_indices.append(torch.tensor(sample.graph.edge_index + offset, dtype=torch.long))
        edge_attrs.append(torch.tensor(sample.graph.edge_attr, dtype=torch.float32))
        batch_parts.append(torch.full((node_count,), graph_index, dtype=torch.long))
        solvents.append(torch.tensor(sample.solvent, dtype=torch.float32))
        targets.append(float(sample.y))
        offset += node_count
    return {
        "x": torch.cat(node_tensors, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1) if edge_indices else torch.zeros((2, 0), dtype=torch.long),
        "edge_attr": torch.cat(edge_attrs, dim=0) if edge_attrs else torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32),
        "batch": torch.cat(batch_parts, dim=0),
        "solvent": torch.stack(solvents, dim=0),
        "y": torch.tensor(targets, dtype=torch.float32).view(-1, 1),
    }


def make_loader(samples: list[GraphSample], batch_size: int, shuffle: bool, torch: Any) -> Any:
    """Create a PyTorch DataLoader for graph samples."""
    data_mod = importlib.import_module("torch.utils.data")
    return data_mod.DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_graph_batch(batch, torch=torch),
    )


def build_model_class(torch: Any) -> Any:
    """Create the pure-PyTorch graph regressor class."""
    nn = torch.nn

    class GraphRegressor(nn.Module):
        def __init__(
            self,
            model_type: str,
            node_dim: int,
            edge_dim: int,
            solvent_dim: int,
            hidden_dim: int,
            num_layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.model_type = model_type
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
            )
            self.message_layers = nn.ModuleList(
                [nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(num_layers)]
            )
            self.gin_mlps = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                    for _ in range(num_layers)
                ]
            )
            self.gru = nn.GRUCell(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2 + solvent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def aggregate(self, messages: Any, dst: Any, node_count: int) -> Any:
            out = torch.zeros((node_count, messages.shape[1]), device=messages.device)
            out.index_add_(0, dst, messages)
            counts = torch.zeros((node_count, 1), device=messages.device)
            counts.index_add_(0, dst, torch.ones((len(dst), 1), device=messages.device))
            if self.model_type == "graph_gcn":
                out = out / counts.clamp_min(1.0)
            return out

        def pool(self, h: Any, batch: Any, graph_count: int) -> Any:
            mean = torch.zeros((graph_count, h.shape[1]), device=h.device)
            mean.index_add_(0, batch, h)
            counts = torch.zeros((graph_count, 1), device=h.device)
            counts.index_add_(0, batch, torch.ones((len(batch), 1), device=h.device))
            mean = mean / counts.clamp_min(1.0)
            max_rows = []
            for graph_index in range(graph_count):
                graph_h = h[batch == graph_index]
                max_rows.append(graph_h.max(dim=0).values)
            return torch.cat([mean, torch.stack(max_rows, dim=0)], dim=1)

        def forward(self, batch: dict[str, Any]) -> Any:
            h = torch.relu(self.node_proj(batch["x"]))
            src, dst = batch["edge_index"]
            edge_hidden = torch.relu(self.edge_proj(batch["edge_attr"]))
            for index, layer in enumerate(self.layers):
                if src.numel() == 0:
                    aggregated = torch.zeros_like(h)
                elif self.model_type == "graph_mpnn":
                    messages = torch.relu(self.message_layers[index](torch.cat([h[src], edge_hidden], dim=1)))
                    aggregated = self.aggregate(messages, dst, h.shape[0])
                    h = self.gru(aggregated, h)
                    h = self.dropout(h)
                    continue
                elif self.model_type == "graph_gin":
                    aggregated = self.aggregate(h[src], dst, h.shape[0])
                    h = self.gin_mlps[index](h + aggregated)
                    h = self.dropout(torch.relu(h))
                    continue
                else:
                    aggregated = self.aggregate(h[src], dst, h.shape[0])
                h = self.dropout(torch.relu(layer(h + aggregated)))
            pooled = self.pool(h, batch["batch"], batch["solvent"].shape[0])
            return self.head(torch.cat([pooled, batch["solvent"]], dim=1))

    return GraphRegressor


def move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    """Move all tensors in a graph batch to a device."""
    return {key: value.to(device) for key, value in batch.items()}


def evaluate_model(model: Any, loader: Any, data: GraphTargetData, torch: Any, device: Any) -> tuple[np.ndarray, np.ndarray, float]:
    """Evaluate a graph model and return original-unit predictions and MAE."""
    model.eval()
    predictions: list[float] = []
    y_true: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            pred_scaled = model(batch).detach().cpu().numpy()
            pred = data.target_scaler.inverse_transform(pred_scaled).ravel()
            predictions.extend(pred.tolist())
            y_true.extend(batch["y"].detach().cpu().numpy().ravel().tolist())
    pred_array = np.asarray(predictions, dtype=float)
    true_array = np.asarray(y_true, dtype=float)
    return true_array, pred_array, float(mean_absolute_error(true_array, pred_array))


def train_one_graph_target(
    model_name: str,
    target: str,
    data: GraphTargetData,
    out_dir: Path,
    args: argparse.Namespace,
    torch: Any,
) -> dict[str, Any]:
    """Train one graph model for one target."""
    torch.manual_seed(args.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GraphRegressor = build_model_class(torch)
    solvent_dim = int(data.samples[0].solvent.shape[0])
    model = GraphRegressor(
        model_type=model_name,
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        solvent_dim=solvent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    loss_fn = torch.nn.SmoothL1Loss()
    train_loader = make_loader(subset_samples(data, data.train_index), args.batch_size, True, torch)
    val_samples = subset_samples(data, data.val_index if len(data.val_index) else data.test_index)
    val_loader = make_loader(val_samples, args.batch_size, False, torch)
    test_loader = make_loader(subset_samples(data, data.test_index), args.batch_size, False, torch)

    best_state: dict[str, Any] | None = None
    best_val_mae = float("inf")
    stale_epochs = 0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            y_scaled = torch.tensor(
                data.target_scaler.transform(batch["y"].detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )
            optimizer.zero_grad()
            loss = loss_fn(model(batch), y_scaled)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        _, _, val_mae = evaluate_model(model, val_loader, data, torch, device)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        print(
            f"{model_name} {target} epoch {epoch + 1}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_mae={val_mae:.4f} best={best_val_mae:.4f}"
        )
        if val_mae < best_val_mae:
            best_val_mae = val_mae
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
    y_test, y_pred, _ = evaluate_model(model, test_loader, data, torch, device)
    model_path = out_dir / f"{target}_{model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": model_name,
            "node_dim": ATOM_FEATURE_DIM,
            "edge_dim": BOND_FEATURE_DIM,
            "solvent_dim": solvent_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        model_path,
    )
    joblib.dump(data.solvent_scaler, out_dir / f"{target}_{model_name}_solvent_scaler.joblib")
    joblib.dump(data.target_scaler, out_dir / f"{target}_{model_name}_target_scaler.joblib")
    prediction_path = save_predictions(data, y_test, y_pred, target, model_name, out_dir)
    return {
        "target": target,
        "model_type": model_name,
        "model_family": "graph_neural",
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
        "r2": float(r2_score(y_test, y_pred)) if len(y_test) > 1 else float("nan"),
        "train_rows": int(len(data.train_index)),
        "test_rows": int(len(data.test_index)),
        "model_path": str(model_path),
        "prediction_path": str(prediction_path),
    }


def save_predictions(
    data: GraphTargetData,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    model_name: str,
    out_dir: Path,
) -> Path:
    """Save graph test predictions with similarity values."""
    rows = data.target_rows.iloc[data.test_index][
        ["canonical_chromophore_smiles", "solvent_original", "canonical_solvent_smiles", "source_dataset"]
    ].copy()
    rows["y_true"] = y_test
    rows["y_pred"] = y_pred
    rows["residual"] = rows["y_true"] - rows["y_pred"]
    similarities = []
    nearest = []
    for smiles in rows["canonical_chromophore_smiles"].astype(str):
        if data.train_reference_fps:
            sim, near = compute_nearest_training_similarity(
                smiles, data.train_reference_fps, data.train_reference_smiles, radius=2, n_bits=2048
            )
        else:
            sim, near = float("nan"), ""
        similarities.append(sim)
        nearest.append(near)
    rows["nearest_training_similarity"] = similarities
    rows["nearest_training_smiles"] = nearest
    path = out_dir / f"predictions_{target}.csv"
    rows.to_csv(path, index=False)
    rows.to_csv(out_dir / f"predictions_test_{target}.csv", index=False)
    return path


def train_graph_model(
    model_name: str,
    target_data_by_target: dict[str, GraphTargetData],
    args: argparse.Namespace,
    torch: Any,
) -> Path:
    """Train one graph model family across targets."""
    out_dir = args.out_root / model_name
    if args.skip_existing and (out_dir / "metrics.json").exists():
        print(f"Skipping existing graph model: {model_name}")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, Any]] = {}
    medians: dict[str, dict[str, float | None]] = {}
    for target, data in target_data_by_target.items():
        metrics[target] = train_one_graph_target(model_name, target, data, out_dir, args, torch)
        medians[target] = {
            key: (None if pd.isna(value) else float(value))
            for key, value in data.descriptor_medians.items()
        }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    metadata = {
        "model_type": model_name,
        "model_family": "graph_neural",
        "target_columns": list(target_data_by_target),
        "solvent_descriptor_columns_used": getattr(args, "_descriptor_columns", []),
        "median_values_used_for_imputation": medians,
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "bond_feature_dim": BOND_FEATURE_DIM,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
    }
    (out_dir / "feature_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    shutil.copyfile(args._modeling_rows_source, out_dir / "combined_modeling_rows_after_feature_merge.csv")
    return out_dir


def collect_model_metrics(model_dirs: dict[str, Path]) -> pd.DataFrame:
    """Collect graph model metrics."""
    rows = []
    for model_name, model_dir in model_dirs.items():
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for metric in metrics.values():
            rows.append(
                {
                    "model": model_name,
                    "model_family": "graph_neural",
                    "target": metric.get("target"),
                    "mae": metric.get("mae"),
                    "rmse": metric.get("rmse"),
                    "r2": metric.get("r2"),
                    "train_rows": metric.get("train_rows"),
                    "test_rows": metric.get("test_rows"),
                }
            )
    return sort_comparison(pd.DataFrame(rows))


def sort_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Sort comparison by target and MAE."""
    if comparison.empty:
        return pd.DataFrame(columns=["model", "model_family", "target", "mae", "rmse", "r2", "train_rows", "test_rows"])
    working = comparison.copy()
    working["_target_rank"] = working["target"].map({"emission_nm": 0, "quantum_yield": 1}).fillna(2)
    working["mae"] = pd.to_numeric(working["mae"], errors="coerce")
    return working.sort_values(["_target_rank", "target", "mae"], kind="mergesort").drop(columns=["_target_rank"]).reset_index(drop=True)


def write_markdown_table(table: pd.DataFrame, path: Path, title: str, note: str = "") -> None:
    """Write a simple markdown table."""
    if table.empty:
        path.write_text(f"# {title}\n\nNo rows were collected.\n", encoding="utf-8")
        return
    rounded = table.copy()
    for column in ["mae", "rmse", "r2", "mean_absolute_error"]:
        if column in rounded.columns:
            rounded[column] = rounded[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    headers = list(rounded.columns)
    rows = [[str(value) for value in row] for row in rounded.to_numpy()]
    markdown = "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(row) + " |" for row in rows],
        ]
    )
    path.write_text(f"# {title}\n\n{note}\n\n{markdown}\n", encoding="utf-8")


def load_existing_table(path: Path, family: str) -> pd.DataFrame:
    """Load an existing comparison file and add model family when needed."""
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if "model_family" not in table.columns and "model" in table.columns:
        table.insert(1, "model_family", family)
    return table


def combine_tables(*tables: pd.DataFrame) -> pd.DataFrame:
    """Combine possibly-empty tables with compatible columns."""
    nonempty = [table for table in tables if not table.empty]
    if not nonempty:
        return pd.DataFrame()
    columns = list(dict.fromkeys(column for table in nonempty for column in table.columns))
    return pd.concat([table.reindex(columns=columns) for table in nonempty], ignore_index=True)


def compute_similarity_bin_outputs(model_dirs: dict[str, Path]) -> pd.DataFrame:
    """Compute MAE by nearest-training-similarity bin."""
    rows = []
    for model_name, model_dir in model_dirs.items():
        for prediction_path in model_dir.glob("predictions_*.csv"):
            target = prediction_path.stem.removeprefix("predictions_")
            predictions = pd.read_csv(prediction_path)
            if "nearest_training_similarity" not in predictions.columns:
                continue
            working = predictions.copy()
            working["absolute_error"] = (working["y_true"] - working["y_pred"]).abs()
            working["similarity_bin"] = pd.cut(
                working["nearest_training_similarity"],
                bins=SIMILARITY_BINS,
                labels=SIMILARITY_LABELS,
                include_lowest=True,
                right=False,
            )
            for label in SIMILARITY_LABELS:
                subset = working[working["similarity_bin"] == label]
                if subset.empty:
                    continue
                rows.append(
                    {
                        "model": model_name,
                        "model_family": "graph_neural",
                        "target": target,
                        "similarity_bin": label,
                        "rows": int(len(subset)),
                        "mean_absolute_error": float(subset["absolute_error"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def write_similarity_markdown(table: pd.DataFrame, path: Path) -> None:
    """Write similarity-bin interpretation markdown."""
    if table.empty:
        path.write_text(
            "# Performance By Similarity Bin\n\nNo similarity-bin rows were available.\n",
            encoding="utf-8",
        )
        return
    low = table[table["similarity_bin"].isin(["0.00-0.30", "0.30-0.50"])]
    high = table[table["similarity_bin"].isin(["0.70-0.85", "0.85-1.00"])]
    low_best = low.sort_values("mean_absolute_error").head(1)
    high_best = high.sort_values("mean_absolute_error").head(1)
    note_lines = [
        "This table asks whether graph models help unfamiliar molecules rather than only improving global MAE.",
        "Low-similarity performance is the main extrapolation signal; high-similarity performance is closer to interpolation.",
    ]
    if not low_best.empty:
        row = low_best.iloc[0]
        note_lines.append(
            f"Best low-similarity graph result: {row['model']} on {row['target']} "
            f"({row['similarity_bin']} MAE {row['mean_absolute_error']:.4f})."
        )
    if not high_best.empty:
        row = high_best.iloc[0]
        note_lines.append(
            f"Best high-similarity graph result: {row['model']} on {row['target']} "
            f"({row['similarity_bin']} MAE {row['mean_absolute_error']:.4f})."
        )
    write_markdown_table(
        table,
        path,
        "Performance By Similarity Bin",
        note="\n\n".join(note_lines),
    )


def predict_graph_target(
    model_dir: Path,
    model_name: str,
    target: str,
    graph: GraphData,
    solvent_vector: np.ndarray,
    torch: Any,
) -> float | None:
    """Predict one target from a saved graph model."""
    model_path = model_dir / f"{target}_{model_name}.pt"
    if not model_path.exists():
        return None
    payload = torch.load(model_path, map_location="cpu")
    GraphRegressor = build_model_class(torch)
    model = GraphRegressor(
        model_type=model_name,
        node_dim=int(payload["node_dim"]),
        edge_dim=int(payload["edge_dim"]),
        solvent_dim=int(payload["solvent_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["num_layers"]),
        dropout=float(payload["dropout"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    solvent_scaler = joblib.load(model_dir / f"{target}_{model_name}_solvent_scaler.joblib")
    target_scaler = joblib.load(model_dir / f"{target}_{model_name}_target_scaler.joblib")
    sample = GraphSample(
        graph=graph,
        solvent=solvent_scaler.transform(solvent_vector.reshape(1, -1)).astype(np.float32)[0],
        y=0.0,
        row={},
    )
    batch = collate_graph_batch([sample], torch=torch)
    with torch.no_grad():
        pred_scaled = model(batch).detach().numpy()
    return float(target_scaler.inverse_transform(pred_scaled).ravel()[0])


def benchmark_prediction_for_model(model_name: str, model_dir: Path, args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Predict the benchmark molecule for one graph model."""
    canonical_smiles = canonicalize_required(args.benchmark_smiles, "molecule")
    canonical_solvent = canonicalize_required(args.benchmark_solvent_smiles, "solvent")
    graph = mol_to_graph(canonical_smiles)
    if graph is None:
        raise ValueError(f"Could not graph benchmark SMILES: {args.benchmark_smiles}")
    metadata = json.loads((model_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    descriptor_columns = list(metadata.get("solvent_descriptor_columns_used", []))
    solvent_descriptors = load_prediction_solvent_descriptors(args.solvent_descriptors)
    solvent_row = get_solvent_descriptor_row(solvent_descriptors, args.benchmark_solvent_smiles, canonical_solvent)

    predictions: dict[str, float | None] = {}
    for target in ["emission_nm", "quantum_yield"]:
        medians = metadata.get("median_values_used_for_imputation", {}).get(target, {})
        values = []
        for column in descriptor_columns:
            value = pd.NA if solvent_row is None or column not in solvent_row.index else solvent_row[column]
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.isna(numeric):
                numeric = medians.get(column, 0.0)
            values.append(float(0.0 if pd.isna(numeric) else numeric))
        predictions[target] = predict_graph_target(
            model_dir, model_name, target, graph, np.asarray(values, dtype=np.float32), torch
        )
    domain, warnings = applicability_domain_payload(
        canonical_smiles=canonical_smiles,
        model_dir=model_dir,
        threshold=args.applicability_threshold,
        radius=2,
        n_bits=2048,
        disabled=False,
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    emission = predictions.get("emission_nm")
    qy = predictions.get("quantum_yield")
    return {
        "model": model_name,
        "model_family": "graph_neural",
        "predicted_emission_nm": emission,
        "known_emission_nm": args.known_emission_nm,
        "emission_absolute_error": None if emission is None or args.known_emission_nm is None else abs(emission - args.known_emission_nm),
        "predicted_quantum_yield": qy,
        "known_quantum_yield": args.known_quantum_yield,
        "quantum_yield_absolute_error": None if qy is None or args.known_quantum_yield is None else abs(qy - args.known_quantum_yield),
        "nearest_training_similarity": domain.get("nearest_training_similarity"),
        "confidence_label": domain.get("confidence_label"),
        "outside_applicability_domain": domain.get("outside_applicability_domain"),
    }


def collect_benchmark_predictions(model_dirs: dict[str, Path], args: argparse.Namespace, torch: Any) -> pd.DataFrame:
    """Collect benchmark predictions for graph models."""
    if not args.benchmark_smiles or not args.benchmark_solvent_smiles:
        return pd.DataFrame()
    rows = []
    for model_name, model_dir in model_dirs.items():
        try:
            rows.append(benchmark_prediction_for_model(model_name, model_dir, args, torch))
        except (FileNotFoundError, ImportError, ValueError, KeyError) as exc:
            print(f"WARNING: benchmark failed for {model_name}: {exc}")
    table = pd.DataFrame(rows)
    if "emission_absolute_error" in table.columns:
        table = table.sort_values("emission_absolute_error", na_position="last").reset_index(drop=True)
    return table


def write_disagreement_summary(compare_out: Path, benchmark: pd.DataFrame, all_benchmark: pd.DataFrame) -> None:
    """Write benchmark-level model-family disagreement summary."""
    source = all_benchmark if not all_benchmark.empty else benchmark
    if source.empty:
        pd.DataFrame().to_csv(compare_out / "model_disagreement_summary.csv", index=False)
        return
    emission_spread = pd.to_numeric(source.get("predicted_emission_nm"), errors="coerce").max() - pd.to_numeric(source.get("predicted_emission_nm"), errors="coerce").min()
    qy_spread = pd.to_numeric(source.get("predicted_quantum_yield"), errors="coerce").max() - pd.to_numeric(source.get("predicted_quantum_yield"), errors="coerce").min()
    summary = source[[
        column for column in [
            "model",
            "model_family",
            "predicted_emission_nm",
            "predicted_quantum_yield",
            "nearest_training_similarity",
            "outside_applicability_domain",
        ]
        if column in source.columns
    ]].copy()
    summary["emission_spread_across_models"] = emission_spread
    summary["quantum_yield_spread_across_models"] = qy_spread
    summary.to_csv(compare_out / "model_disagreement_summary.csv", index=False)


def write_outputs(
    compare_out: Path,
    tree_compare_dir: Path,
    neural_compare_dir: Path,
    graph_comparison: pd.DataFrame,
    graph_region: pd.DataFrame,
    graph_benchmark: pd.DataFrame,
    similarity_table: pd.DataFrame,
) -> None:
    """Write graph-only and merged comparison outputs."""
    compare_out.mkdir(parents=True, exist_ok=True)
    graph_comparison.to_csv(compare_out / "graph_model_comparison.csv", index=False)
    write_markdown_table(graph_comparison, compare_out / "graph_model_comparison.md", "Graph Model Comparison")
    graph_region.to_csv(compare_out / "graph_error_by_region_comparison.csv", index=False)
    graph_benchmark.to_csv(compare_out / "graph_benchmark_prediction_comparison.csv", index=False)
    similarity_table.to_csv(compare_out / "performance_by_similarity_bin.csv", index=False)
    write_similarity_markdown(similarity_table, compare_out / "performance_by_similarity_bin.md")

    tree_models = load_existing_table(tree_compare_dir / "model_comparison.csv", "tree")
    neural_models = load_existing_table(neural_compare_dir / "neural_model_comparison.csv", "neural")
    all_models = combine_tables(tree_models, neural_models, graph_comparison)
    if {"target", "mae"}.issubset(all_models.columns):
        all_models = sort_comparison(all_models)
    all_models.to_csv(compare_out / "all_model_comparison.csv", index=False)
    write_markdown_table(all_models, compare_out / "all_model_comparison.md", "All Model Comparison")

    all_region = combine_tables(
        load_existing_table(tree_compare_dir / "error_by_region_comparison.csv", "tree"),
        load_existing_table(neural_compare_dir / "neural_error_by_region_comparison.csv", "neural"),
        graph_region,
    )
    all_region.to_csv(compare_out / "all_error_by_region_comparison.csv", index=False)

    all_benchmark = combine_tables(
        load_existing_table(tree_compare_dir / "benchmark_prediction_comparison.csv", "tree"),
        load_existing_table(neural_compare_dir / "neural_benchmark_prediction_comparison.csv", "neural"),
        graph_benchmark,
    )
    if "emission_absolute_error" in all_benchmark.columns:
        all_benchmark["emission_absolute_error"] = pd.to_numeric(all_benchmark["emission_absolute_error"], errors="coerce")
        all_benchmark = all_benchmark.sort_values("emission_absolute_error", na_position="last").reset_index(drop=True)
    all_benchmark.to_csv(compare_out / "all_benchmark_prediction_comparison.csv", index=False)
    write_disagreement_summary(compare_out, graph_benchmark, all_benchmark)


def main() -> int:
    """Run graph model experiments."""
    args = parse_args()
    try:
        selected_models = validate_graph_models(parse_csv_list(args.models))
        selected_targets = trainer.parse_targets(args.targets)
        torch = import_torch()
        if torch is None:
            print(
                "ERROR: graph neural experiments require PyTorch. Install torch in this environment, "
                "or run this script on Nibi with a PyTorch-enabled environment.",
                file=sys.stderr,
            )
            return 1
        args.out_root.mkdir(parents=True, exist_ok=True)
        modeling_rows, descriptor_columns = load_graph_inputs(args)
        modeling_rows_source = args.out_root / "combined_modeling_rows_after_feature_merge.csv"
        modeling_rows.to_csv(modeling_rows_source, index=False)
        args._modeling_rows_source = modeling_rows_source
        args._descriptor_columns = descriptor_columns
        target_data_by_target = {
            target: data
            for target in selected_targets
            if (data := prepare_target_data(target, modeling_rows, descriptor_columns, args)) is not None
        }
        if not target_data_by_target:
            raise ValueError("No requested targets had enough usable rows to train.")

        model_dirs = {
            model_name: train_graph_model(model_name, target_data_by_target, args, torch)
            for model_name in selected_models
        }
        graph_comparison = collect_model_metrics(model_dirs)
        graph_region = combined_experiments.collect_error_by_region(model_dirs)
        if not graph_region.empty:
            graph_region.insert(1, "model_family", "graph_neural")
        graph_benchmark = collect_benchmark_predictions(model_dirs, args, torch)
        similarity_table = compute_similarity_bin_outputs(model_dirs)
        write_outputs(
            args.compare_out,
            args.tree_compare_dir,
            args.neural_compare_dir,
            graph_comparison,
            graph_region,
            graph_benchmark,
            similarity_table,
        )
        print(f"Saved graph comparison to: {args.compare_out / 'graph_model_comparison.csv'}")
        print(f"Saved all-model comparison to: {args.compare_out / 'all_model_comparison.csv'}")
        return 0
    except (FileNotFoundError, ImportError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
