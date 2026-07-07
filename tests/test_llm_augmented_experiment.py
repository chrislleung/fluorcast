"""Tests for deterministic and leakage-safe LLM augmentation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
SCRIPT = PROJECT / "scripts" / "run_llm_augmented_three_way_experiment.py"
SPEC = importlib.util.spec_from_file_location("llm_experiment", SCRIPT)
assert SPEC and SPEC.loader
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)

from chemfluor.llm_prediction.feature_encoding import encode_llm_features
from chemfluor.llm_prediction.schema import DESCRIPTOR_NAMES, LLMOutput
from chemfluor.llm_prediction.template_stub import predict


def test_template_mode_is_deterministic() -> None:
    first = predict("emission_nm", "c1ccccc1N", "O").to_dict()
    assert first == predict("emission_nm", "c1ccccc1N", "O").to_dict()


def test_descriptor_encoding() -> None:
    record = LLMOutput("emission_nm", descriptors={name: "high" for name in DESCRIPTOR_NAMES})
    encoded = encode_llm_features([record])
    assert all(encoded[f"llm_descriptor_{name}"].iloc[0] == 1.0 for name in DESCRIPTOR_NAMES)


def test_numeric_prediction_is_added_as_feature() -> None:
    features = experiment.augmented_features(pd.DataFrame({"rf": [1.0]}),
        [{"target": "emission_nm", "llm_numeric_prediction": 444.0}], "numeric")
    assert features.loc[0, "llm_numeric_prediction"] == 444.0


def test_invalid_outputs_do_not_crash() -> None:
    encoded = encode_llm_features([None, {}, {"descriptors": "bad"}], "emission_nm")
    assert len(encoded) == 3
    assert encoded.isna().any().any()


def test_leakage_check_is_preserved() -> None:
    rows = pd.DataFrame({"row_id": range(12),
        "canonical_chromophore_smiles": [f"mol-{i // 2}" for i in range(12)]})
    rows["split"] = experiment.three_way.assign_three_way_splits(rows, "molecule", (.5, .25, .25), 0)
    assert not experiment.three_way.leakage_report(rows, "molecule")["leakage_detected"]


def test_metrics_compare_baseline_and_llm_variants() -> None:
    truth = np.array([1.0, 2.0, 3.0])
    rows = [experiment.three_way.metrics_row(name, truth, prediction, "emission_nm")
            for name, prediction in (("non_llm_hybrid_baseline", [1.2, 2.2, 3.2]),
                                     ("llm_numeric_only_hybrid", [1.0, 2.0, 3.0]))]
    metrics = pd.DataFrame(rows).set_index("model")
    assert metrics.loc["llm_numeric_only_hybrid", "MAE"] < metrics.loc["non_llm_hybrid_baseline", "MAE"]


def test_cli_help_works() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=PROJECT,
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--llm-mode" in result.stdout
    assert "--llm-feature-mode" in result.stdout
