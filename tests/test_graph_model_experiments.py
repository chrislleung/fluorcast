from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.graph_features import ATOM_FEATURE_DIM, BOND_FEATURE_DIM, mol_to_graph


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


graph_script = load_script(
    "run_graph_model_experiments_for_tests",
    "scripts/run_graph_model_experiments.py",
)


def require_torch():
    torch = graph_script.import_torch()
    if torch is None:
        pytest.skip("PyTorch is not installed in this environment.")
    return torch


def test_smiles_to_graph_conversion() -> None:
    graph = mol_to_graph("CCO")

    assert graph is not None
    assert graph.canonical_smiles == "CCO"
    assert graph.num_nodes == 3


def test_graph_contains_node_features_and_edge_indices() -> None:
    graph = mol_to_graph("c1ccccc1")

    assert graph is not None
    assert graph.x.shape == (6, ATOM_FEATURE_DIM)
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == BOND_FEATURE_DIM


def test_invalid_smiles_returns_none() -> None:
    assert mol_to_graph("not-a-smiles") is None


def sample(smiles: str, y: float) -> graph_script.GraphSample:
    graph = mol_to_graph(smiles)
    assert graph is not None
    return graph_script.GraphSample(
        graph=graph,
        solvent=np.asarray([1.0, 2.0], dtype=np.float32),
        y=y,
        row={"canonical_chromophore_smiles": graph.canonical_smiles},
    )


def test_graph_batch_collation_offsets_edge_indices() -> None:
    torch = require_torch()
    first = sample("CCO", 1.0)
    second = sample("CCN", 2.0)

    batch = graph_script.collate_graph_batch([first, second], torch=torch)
    second_edges = batch["edge_index"][:, first.graph.edge_index.shape[1] :]

    assert batch["x"].shape[0] == first.graph.num_nodes + second.graph.num_nodes
    assert second_edges.min().item() >= first.graph.num_nodes
    assert batch["batch"].tolist() == [0] * first.graph.num_nodes + [1] * second.graph.num_nodes


def test_tiny_graph_model_forward_pass() -> None:
    torch = require_torch()
    batch = graph_script.collate_graph_batch([sample("CCO", 1.0), sample("O=C=O", 2.0)], torch=torch)
    GraphRegressor = graph_script.build_model_class(torch)
    model = GraphRegressor(
        model_type="graph_gcn",
        node_dim=ATOM_FEATURE_DIM,
        edge_dim=BOND_FEATURE_DIM,
        solvent_dim=2,
        hidden_dim=16,
        num_layers=2,
        dropout=0.0,
    )

    output = model(batch)

    assert tuple(output.shape) == (2, 1)


def tiny_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "chromophore_smiles": ["CCO", "CCN", "c1ccccc1", "O=C=O"],
            "solvent_original": ["water", "water", "dmso", "dmso"],
            "canonical_chromophore_smiles": ["CCO", "CCN", "c1ccccc1", "O=C=O"],
            "canonical_solvent_smiles": ["O", "O", "CS(C)=O", "CS(C)=O"],
            "source_dataset": ["synthetic"] * 4,
            "emission_nm": [410.0, 430.0, 500.0, 520.0],
            "quantum_yield": [0.1, 0.2, 0.3, 0.4],
            "dielectric_constant": [80.0, 80.0, 47.0, 47.0],
        }
    )


def test_output_comparison_files_created_without_old_comparisons(tmp_path: Path) -> None:
    graph_comparison = pd.DataFrame(
        [
            {
                "model": "graph_gcn",
                "model_family": "graph_neural",
                "target": "emission_nm",
                "mae": 10.0,
                "rmse": 12.0,
                "r2": 0.5,
                "train_rows": 2,
                "test_rows": 2,
            }
        ]
    )

    graph_script.write_outputs(
        compare_out=tmp_path / "compare",
        tree_compare_dir=tmp_path / "missing_tree",
        neural_compare_dir=tmp_path / "missing_neural",
        graph_comparison=graph_comparison,
        graph_region=pd.DataFrame(),
        graph_benchmark=pd.DataFrame(),
        similarity_table=pd.DataFrame(),
    )

    assert (tmp_path / "compare" / "graph_model_comparison.csv").exists()
    assert (tmp_path / "compare" / "all_model_comparison.csv").exists()
    assert (tmp_path / "compare" / "performance_by_similarity_bin.md").exists()


def test_cli_smoke_with_tiny_dataset_and_one_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_torch()
    standardized = tmp_path / "combined.csv"
    solvents = tmp_path / "solvents.csv"
    standardized.write_text("placeholder\n", encoding="utf-8")
    solvents.write_text("placeholder\n", encoding="utf-8")
    rows = tiny_rows()
    monkeypatch.setattr(graph_script, "load_graph_inputs", lambda args: (rows, ["dielectric_constant"]))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_graph_model_experiments.py",
            "--standardized-combined",
            str(standardized),
            "--solvent-descriptors",
            str(solvents),
            "--tree-compare-dir",
            str(tmp_path / "tree"),
            "--neural-compare-dir",
            str(tmp_path / "neural"),
            "--out-root",
            str(tmp_path / "models"),
            "--compare-out",
            str(tmp_path / "compare"),
            "--models",
            "graph_gcn",
            "--targets",
            "emission_nm",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--hidden-dim",
            "16",
            "--num-layers",
            "2",
        ],
    )

    assert graph_script.main() == 0
    assert (tmp_path / "compare" / "graph_model_comparison.csv").exists()
    assert (tmp_path / "compare" / "all_model_comparison.csv").exists()


def test_benchmark_output_schema(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch = require_torch()
    model_dir = tmp_path / "graph_gcn"
    model_dir.mkdir()
    metadata = {
        "model_type": "graph_gcn",
        "model_family": "graph_neural",
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "median_values_used_for_imputation": {
            "emission_nm": {"dielectric_constant": 80.0},
            "quantum_yield": {"dielectric_constant": 80.0},
        },
    }
    (model_dir / "feature_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    solvents = tmp_path / "solvents.csv"
    pd.DataFrame(
        {
            "canonical_solvent_smiles": ["O"],
            "solvent_original": ["water"],
            "dielectric_constant": [80.0],
        }
    ).to_csv(solvents, index=False)
    monkeypatch.setattr(
        graph_script,
        "predict_graph_target",
        lambda *args, **kwargs: 500.0 if args[2] == "emission_nm" else 0.2,
    )
    args = type(
        "Args",
        (),
        {
            "benchmark_smiles": "CCO",
            "benchmark_solvent_smiles": "O",
            "known_emission_nm": 510.0,
            "known_quantum_yield": 0.25,
            "solvent_descriptors": solvents,
            "applicability_threshold": 0.5,
        },
    )()

    result = graph_script.benchmark_prediction_for_model("graph_gcn", model_dir, args, torch)

    assert result["model"] == "graph_gcn"
    assert result["predicted_emission_nm"] == pytest.approx(500.0)
    assert result["emission_absolute_error"] == pytest.approx(10.0)
    assert "nearest_training_similarity" in result
    assert "outside_applicability_domain" in result
