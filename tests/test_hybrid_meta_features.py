from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chemfluor.hybrid.meta_features import build_meta_features


def test_meta_feature_statistics_and_models() -> None:
    table = pd.DataFrame({
        "model": ["RF", "Graph GIN"],
        "predicted_emission_nm": [500.0, 520.0],
        "predicted_quantum_yield": [0.2, 0.4],
        "nearest_training_similarity": [0.8, 0.8],
    })
    features = build_meta_features(table).iloc[0]
    assert features["emission_nm_mean"] == pytest.approx(510)
    assert features["emission_nm_std"] == pytest.approx(10)
    assert features["emission_nm_range"] == pytest.approx(20)
    assert features["emission_nm_count"] == 2
    assert features["emission_nm_model_graph_gin"] == 520
    assert features["nearest_training_similarity"] == pytest.approx(0.8)


def test_missing_prediction_columns_are_safe() -> None:
    features = build_meta_features(pd.DataFrame({"model": ["rf"]})).iloc[0]
    assert pd.isna(features["emission_nm_mean"])
    assert features["emission_nm_count"] == 0
    assert features["quantum_yield_count"] == 0
