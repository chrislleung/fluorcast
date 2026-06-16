from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


neural = load_script(
    "run_neural_model_experiments_for_tests",
    "scripts/run_neural_model_experiments.py",
)


class MeanRegressor:
    def fit(self, x: np.ndarray, y: np.ndarray) -> "MeanRegressor":
        self.value = float(np.mean(y))
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.value, dtype=float)


class ConstantRegressor:
    def __init__(self, value: float, n_features: int = 33) -> None:
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=float)


def tiny_target_data() -> neural.TargetData:
    rows = pd.DataFrame(
        {
            "canonical_chromophore_smiles": ["CCO", "CCC", "CCN", "C1=CC=CC=C1"],
            "solvent_original": ["water", "water", "dmso", "dmso"],
            "canonical_solvent_smiles": ["O", "O", "CS(C)=O", "CS(C)=O"],
            "source_dataset": ["synthetic"] * 4,
            "emission_nm": [410.0, 520.0, 610.0, 700.0],
        }
    )
    return neural.TargetData(
        target_rows=rows,
        descriptor_values=pd.DataFrame({"dielectric_constant": [80.0, 80.0, 47.0, 47.0]}),
        train_index=np.asarray([0, 1]),
        test_index=np.asarray([2, 3]),
        x_train=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        x_test=np.asarray([[1.0, 1.0], [0.0, 0.0]]),
        y_train=np.asarray([410.0, 520.0]),
        y_test=np.asarray([610.0, 700.0]),
        medians=pd.Series({"dielectric_constant": 80.0}),
    )


def test_requested_model_names_expands_sklearn_alpha_sweep() -> None:
    names = neural.requested_model_names(["mlp_small"], skip_pytorch=False)

    assert names == [
        "mlp_small_alpha_1e-03",
        "mlp_small_alpha_1e-04",
        "mlp_small_alpha_1e-05",
    ]


def test_cli_smoke_creates_neural_outputs_with_stubbed_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standardized = tmp_path / "combined.csv"
    solvents = tmp_path / "solvents.csv"
    tree_dir = tmp_path / "tree"
    out_root = tmp_path / "models"
    compare_out = tmp_path / "compare"
    standardized.write_text("placeholder\n", encoding="utf-8")
    solvents.write_text("placeholder\n", encoding="utf-8")
    tree_dir.mkdir()

    target_data = tiny_target_data()
    modeling_rows = target_data.target_rows.copy()
    modeling_rows["dielectric_constant"] = [80.0, 80.0, 47.0, 47.0]
    monkeypatch.setattr(
        neural,
        "load_training_inputs",
        lambda args: (modeling_rows, ["dielectric_constant"], np.zeros((4, 32))),
    )
    monkeypatch.setattr(
        neural,
        "prepare_target_data",
        lambda **kwargs: target_data if kwargs["target"] == "emission_nm" else None,
    )
    monkeypatch.setattr(neural, "make_sklearn_mlp", lambda *args, **kwargs: MeanRegressor())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_neural_model_experiments.py",
            "--standardized-combined",
            str(standardized),
            "--solvent-descriptors",
            str(solvents),
            "--tree-compare-dir",
            str(tree_dir),
            "--out-root",
            str(out_root),
            "--compare-out",
            str(compare_out),
            "--models",
            "mlp_small",
            "--targets",
            "emission_nm",
            "--max-iter",
            "1",
            "--skip-pytorch",
        ],
    )

    assert neural.main() == 0
    saved = pd.read_csv(compare_out / "neural_model_comparison.csv")

    assert len(saved) == 3
    assert (compare_out / "neural_model_comparison.md").exists()
    assert (compare_out / "all_model_comparison.csv").exists()
    assert (out_root / "mlp_small_alpha_1e-03" / "metrics.json").exists()


def test_missing_torch_skips_pytorch_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standardized = tmp_path / "combined.csv"
    solvents = tmp_path / "solvents.csv"
    standardized.write_text("placeholder\n", encoding="utf-8")
    solvents.write_text("placeholder\n", encoding="utf-8")
    target_data = tiny_target_data()
    monkeypatch.setattr(
        neural,
        "load_training_inputs",
        lambda args: (target_data.target_rows, ["dielectric_constant"], np.zeros((4, 32))),
    )
    monkeypatch.setattr(neural, "prepare_target_data", lambda **kwargs: target_data)
    monkeypatch.setattr(neural, "import_torch", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_neural_model_experiments.py",
            "--standardized-combined",
            str(standardized),
            "--solvent-descriptors",
            str(solvents),
            "--tree-compare-dir",
            str(tmp_path / "missing_tree"),
            "--out-root",
            str(tmp_path / "models"),
            "--compare-out",
            str(tmp_path / "compare"),
            "--models",
            "pytorch_mlp",
            "--targets",
            "emission_nm",
        ],
    )

    assert neural.main() == 0
    saved = pd.read_csv(tmp_path / "compare" / "neural_model_comparison.csv")
    assert saved.empty


def test_tree_comparison_files_merge_into_all_model_outputs(tmp_path: Path) -> None:
    compare_out = tmp_path / "compare"
    tree_dir = tmp_path / "tree"
    tree_dir.mkdir()
    pd.DataFrame(
        [{"model": "rf", "target": "emission_nm", "mae": 20.0, "rmse": 30.0, "r2": 0.5}]
    ).to_csv(tree_dir / "model_comparison.csv", index=False)
    neural_comparison = pd.DataFrame(
        [
            {
                "model": "mlp_small_alpha_1e-04",
                "model_family": "neural",
                "target": "emission_nm",
                "mae": 25.0,
                "rmse": 35.0,
                "r2": 0.4,
            }
        ]
    )

    neural.write_outputs(
        compare_out=compare_out,
        tree_compare_dir=tree_dir,
        comparison=neural_comparison,
        region_comparison=pd.DataFrame(),
        benchmark_comparison=pd.DataFrame(),
    )
    all_models = pd.read_csv(compare_out / "all_model_comparison.csv")

    assert set(all_models["model"]) == {"rf", "mlp_small_alpha_1e-04"}
    assert all_models.loc[all_models["model"] == "rf", "model_family"].iloc[0] == "tree"


def test_benchmark_prediction_output_schema(tmp_path: Path) -> None:
    model_name = "mlp_small_alpha_1e-04"
    model_dir = tmp_path / model_name
    model_dir.mkdir()
    joblib.dump(ConstantRegressor(500.0), model_dir / f"emission_nm_{model_name}.joblib")
    joblib.dump(ConstantRegressor(0.2), model_dir / f"quantum_yield_{model_name}.joblib")
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "target_columns": ["emission_nm", "quantum_yield"],
        "model_type": model_name,
        "median_values_used_for_imputation": {
            "emission_nm": {"dielectric_constant": 80.0},
            "quantum_yield": {"dielectric_constant": 80.0},
        },
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "CCC"]}).to_csv(
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
    args = type(
        "Args",
        (),
        {
            "benchmark_smiles": "CCO",
            "benchmark_solvent_smiles": "O",
            "known_emission_nm": 510.0,
            "known_quantum_yield": 0.25,
            "solvent_descriptors": solvents,
        },
    )()

    result = neural.benchmark_prediction_for_model(model_name, model_dir, args)

    assert result["model"] == model_name
    assert result["predicted_emission_nm"] == pytest.approx(500.0)
    assert result["emission_absolute_error"] == pytest.approx(10.0)
    assert "nearest_training_similarity" in result
    assert "confidence_label" in result
