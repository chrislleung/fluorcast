from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest

import scripts.predict_conforformer_case as predict_script


class ConstantEstimator:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full((len(x),), self.value, dtype=float)


def _feature_order(solvent_columns: list[str] | None = None) -> list[str]:
    solvent_columns = solvent_columns or ["molecular_weight", "molecular_weight__missing"]
    return (
        [f"conforformer_mean_{idx:03d}" for idx in range(512)]
        + [f"morgan_{idx:04d}" for idx in range(2048)]
        + solvent_columns
    )


def _model_root(tmp_path: Path, *, include_stokes: bool = True, feature_order: list[str] | None = None) -> Path:
    root = tmp_path / "models"
    values = {
        "absorption_nm": 410.0,
        "emission_nm": 490.0,
        "quantum_yield": 0.62,
        "stokes_shift_nm": 77.0,
    }
    for target, value in values.items():
        if target == "stokes_shift_nm" and not include_stokes:
            continue
        target_dir = root / "molecule" / "mean" / "conforformer_morgan_solvent" / target
        target_dir.mkdir(parents=True)
        joblib.dump(ConstantEstimator(value), target_dir / "model.joblib")
        (target_dir / "feature_metadata.json").write_text(
            json.dumps(
                {
                    "feature_order": feature_order or _feature_order(),
                    "pooling_method": "mean",
                    "feature_set": "conforformer_morgan_solvent",
                    "selected_candidate": "constant",
                }
            ),
            encoding="utf-8",
        )
    return root


def _solvent_descriptors(tmp_path: Path) -> Path:
    path = tmp_path / "solvents.csv"
    pd.DataFrame(
        {
            "solvent_original": ["O"],
            "canonical_solvent_smiles": ["O"],
            "is_valid_rdkit": [True],
            "is_environment_label": [False],
            "deep4chem_row_count": [1],
            "molecular_weight": [18.015],
        }
    ).to_csv(path, index=False)
    return path


def _patch_fast_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        predict_script,
        "embed_smiles",
        lambda *args, **kwargs: (np.ones(512, dtype=np.float32), "CCO", "cache-key", True),
    )
    monkeypatch.setattr(
        predict_script,
        "morgan_fingerprint",
        lambda *args, **kwargs: np.ones(2048, dtype=np.float32),
    )


def _common(tmp_path: Path, model_root: Path) -> dict:
    return {
        "models": predict_script.load_target_models(model_root),
        "adapter": SimpleNamespace(),
        "dictionary": SimpleNamespace(sha256="dict-sha"),
        "checkpoint_path": tmp_path / "checkpoint.pt",
        "dictionary_path": tmp_path / "dict.txt",
        "solvent_descriptor_path": _solvent_descriptors(tmp_path),
        "cache_dir": tmp_path / "cache",
    }


def test_cli_parsing_single_csv_and_invalid_combinations(tmp_path: Path) -> None:
    single = predict_script.parse_args(["--smiles", "CCO", "--solvent-smiles", "O"])
    assert single.smiles == "CCO"
    batch = predict_script.parse_args(["--input-csv", str(tmp_path / "cases.csv"), "--output-csv", str(tmp_path / "out.csv")])
    assert batch.input_csv.name == "cases.csv"
    with pytest.raises(SystemExit):
        predict_script.parse_args(["--smiles", "CCO"])
    with pytest.raises(SystemExit):
        predict_script.parse_args(["--input-csv", str(tmp_path / "cases.csv"), "--json"])


def test_invalid_chromophore_and_solvent_smiles() -> None:
    with pytest.raises(ValueError, match="Invalid chromophore SMILES"):
        predict_script.validate_smiles("not a smiles", label="chromophore")
    with pytest.raises(ValueError, match="Invalid solvent SMILES"):
        predict_script.validate_smiles("also bad", label="solvent")


def test_prediction_plumbing_maps_all_four_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_features(monkeypatch)
    common = _common(tmp_path, _model_root(tmp_path, include_stokes=True))
    payload = predict_script.predict_case("CCO", "O", **common)
    assert payload["predictions"]["absorption_nm"] == pytest.approx(410.0)
    assert payload["predictions"]["emission_nm"] == pytest.approx(490.0)
    assert payload["predictions"]["quantum_yield"] == pytest.approx(0.62)
    assert payload["predictions"]["stokes_shift_nm"] == pytest.approx(77.0)
    assert payload["predictions"]["derived_stokes_shift_nm"] == pytest.approx(80.0)
    assert payload["metadata"]["stokes_source"] == "direct"


def test_stokes_falls_back_to_derived_when_direct_model_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_features(monkeypatch)
    common = _common(tmp_path, _model_root(tmp_path, include_stokes=False))
    payload = predict_script.predict_case("CCO", "O", **common)
    assert payload["predictions"]["stokes_shift_nm"] == pytest.approx(80.0)
    assert payload["metadata"]["stokes_source"] == "derived"


def test_feature_schema_mismatch_fails_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_features(monkeypatch)
    bad_order = _feature_order(["molecular_weight", "unexpected_descriptor"])
    common = _common(tmp_path, _model_root(tmp_path, include_stokes=False, feature_order=bad_order))
    with pytest.raises(ValueError, match="Feature schema mismatch"):
        predict_script.predict_case("CCO", "O", **common)


def test_batch_continues_after_failed_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_fast_features(monkeypatch)
    common = _common(tmp_path, _model_root(tmp_path, include_stokes=False))
    input_csv = tmp_path / "cases.csv"
    output_csv = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {"name": "good", "smiles": "CCO", "solvent_smiles": "O"},
            {"name": "bad", "smiles": "not a smiles", "solvent_smiles": "O"},
        ]
    ).to_csv(input_csv, index=False)
    predict_script.batch_predict(input_csv, output_csv, **common)
    rows = pd.read_csv(output_csv)
    assert rows.loc[0, "status"] == "ok"
    assert rows.loc[0, "absorption_nm"] == pytest.approx(410.0)
    assert rows.loc[1, "status"] == "error"
    assert "Invalid" in rows.loc[1, "error"]


def test_json_output_is_parseable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _patch_fast_features(monkeypatch)
    model_root = _model_root(tmp_path, include_stokes=False)
    monkeypatch.setattr(predict_script, "ConforFormerEncoderAdapter", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(predict_script, "load_conforformer_dictionary", lambda path: SimpleNamespace(sha256="dict-sha"))
    checkpoint = tmp_path / "checkpoint.pt"
    dictionary = tmp_path / "dict.txt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    dictionary.write_text("dict", encoding="utf-8")
    assert predict_script.main(
        [
            "--smiles",
            "CCO",
            "--solvent-smiles",
            "O",
            "--model-root",
            str(model_root),
            "--checkpoint",
            str(checkpoint),
            "--dictionary",
            str(dictionary),
            "--solvent-descriptors",
            str(_solvent_descriptors(tmp_path)),
            "--json",
        ]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["predictions"]["derived_stokes_shift_nm"] == pytest.approx(80.0)
