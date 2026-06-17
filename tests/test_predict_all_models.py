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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "predict_all_models.py"

spec = importlib.util.spec_from_file_location("predict_all_models", SCRIPT_PATH)
predict_script = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(predict_script)


class ConstantRegressor:
    def __init__(self, value: float, n_features: int = 33) -> None:
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features: np.ndarray) -> np.ndarray:
        assert features.shape[1] == self.n_features_in_
        return np.full(features.shape[0], self.value, dtype=float)


def write_solvent_descriptors(tmp_path: Path) -> Path:
    path = tmp_path / "solvent_descriptors.csv"
    pd.DataFrame(
        {
            "canonical_solvent_smiles": ["O", "CCO"],
            "solvent_original": ["water", "ethanol"],
            "dielectric_constant": [80.1, 24.6],
        }
    ).to_csv(path, index=False)
    return path


def write_standardized_combined(tmp_path: Path) -> Path:
    path = tmp_path / "combined.csv"
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "c1ccccc1"]}).to_csv(
        path, index=False
    )
    return path


def write_tree_model_root(
    tmp_path: Path,
    *,
    include_emission: bool = True,
    include_qy: bool = True,
) -> Path:
    root = tmp_path / "tree_models"
    model_dir = root / "rf"
    model_dir.mkdir(parents=True)
    metadata = {
        "fingerprint_radius": 2,
        "fingerprint_n_bits": 32,
        "solvent_descriptor_columns_used": ["dielectric_constant"],
        "median_values_used_for_imputation": {
            "emission_nm": {"dielectric_constant": 1.0},
            "quantum_yield": {"dielectric_constant": 1.0},
        },
        "model_type": "rf",
        "target_columns": ["emission_nm", "quantum_yield"],
    }
    (model_dir / "feature_metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    pd.DataFrame({"canonical_chromophore_smiles": ["CCO", "c1ccccc1"]}).to_csv(
        model_dir / "combined_modeling_rows_after_feature_merge.csv", index=False
    )
    if include_emission:
        joblib.dump(ConstantRegressor(456.0), model_dir / "emission_nm_rf.joblib")
    if include_qy:
        joblib.dump(ConstantRegressor(0.42), model_dir / "quantum_yield_rf.joblib")
    return root


def run_main(monkeypatch, argv: list[str]) -> int:
    monkeypatch.setattr(sys, "argv", ["predict_all_models.py", *argv])
    return int(predict_script.main())


def base_args(tmp_path: Path, tree_root: Path) -> list[str]:
    return [
        "--solvent-descriptors",
        str(write_solvent_descriptors(tmp_path)),
        "--standardized-combined",
        str(write_standardized_combined(tmp_path)),
        "--tree-model-dir",
        str(tree_root),
        "--neural-model-dir",
        str(tmp_path / "missing_neural"),
        "--graph-model-dirs",
    ]


def test_cli_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["predict_all_models.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        predict_script.parse_args()

    assert exc.value.code == 0
    assert "--smiles" in capsys.readouterr().out


def test_invalid_smiles(tmp_path: Path, monkeypatch, capsys) -> None:
    tree_root = write_tree_model_root(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "not_a_smiles",
            "--solvent-smiles",
            "O",
            *base_args(tmp_path, tree_root),
        ],
    )

    assert exit_code == 1
    assert "Invalid molecule SMILES" in capsys.readouterr().err


def test_missing_solvent(tmp_path: Path, monkeypatch, capsys) -> None:
    tree_root = write_tree_model_root(tmp_path)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            *base_args(tmp_path, tree_root),
        ],
    )

    assert exit_code == 1
    assert "Provide either --solvent or --solvent-smiles" in capsys.readouterr().err


def test_output_csv_creation(tmp_path: Path, monkeypatch) -> None:
    tree_root = write_tree_model_root(tmp_path)
    output_csv = tmp_path / "predictions.csv"

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent",
            "water",
            "--out",
            str(output_csv),
            *base_args(tmp_path, tree_root),
        ],
    )

    table = pd.read_csv(output_csv)
    assert exit_code == 0
    assert len(table) == 2
    assert set(table["target"]) == {"emission_nm", "quantum_yield"}
    assert table["predicted_emission_nm"].dropna().unique().tolist() == [456.0]
    assert table["predicted_quantum_yield"].dropna().unique().tolist() == [0.42]


def test_disagreement_summary_correctness() -> None:
    table = pd.DataFrame(
        [
            {
                "model": "a",
                "model_family": "tree",
                "seed": None,
                "predicted_emission_nm": 400.0,
                "predicted_quantum_yield": 0.2,
            },
            {
                "model": "b",
                "model_family": "tree",
                "seed": None,
                "predicted_emission_nm": 500.0,
                "predicted_quantum_yield": 0.6,
            },
        ]
    )

    summaries = predict_script.compute_disagreement_summaries(table)

    assert summaries["emission_nm"]["mean"] == pytest.approx(450.0)
    assert summaries["emission_nm"]["median"] == pytest.approx(450.0)
    assert summaries["emission_nm"]["std"] == pytest.approx(50.0)
    assert summaries["emission_nm"]["min"] == pytest.approx(400.0)
    assert summaries["emission_nm"]["max"] == pytest.approx(500.0)
    assert summaries["emission_nm"]["range"] == pytest.approx(100.0)
    assert summaries["quantum_yield"]["range"] == pytest.approx(0.4)


def test_missing_model_files_are_skipped_with_warning(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    tree_root = write_tree_model_root(tmp_path, include_qy=False)

    exit_code = run_main(
        monkeypatch,
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            *base_args(tmp_path, tree_root),
        ],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Model file not found; skipping" in captured.out
    assert "quantum_yield_rf.joblib" in captured.out
    assert "emission_nm" in captured.out
