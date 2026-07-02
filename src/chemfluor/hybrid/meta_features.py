"""Numeric second-stage features derived from base-model prediction tables."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMNS = {
    "emission_nm": ("predicted_emission_nm", "emission_nm"),
    "quantum_yield": ("predicted_quantum_yield", "quantum_yield"),
}
APPLICABILITY_COLUMNS = (
    "nearest_training_similarity",
    "molecule_seen_score",
    "solvent_seen_score",
    "pair_seen_score",
    "label_consistency_score",
    "model_agreement_score",
    "overall_confidence_score",
)
STATISTICS = ("mean", "std", "min", "max", "range", "count")


def _prediction_column(table: pd.DataFrame, target: str) -> str | None:
    return next((name for name in TARGET_COLUMNS[target] if name in table), None)


def _safe_name(value: Any) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value)).strip("_").lower()
    return text or "unknown"


def build_meta_feature_row(predictions: pd.DataFrame) -> dict[str, float]:
    """Build one numeric feature row for a single molecule/solvent example."""
    features: dict[str, float] = {}
    for target in TARGET_COLUMNS:
        column = _prediction_column(predictions, target)
        values = (
            pd.Series(dtype=float)
            if column is None
            else pd.to_numeric(predictions[column], errors="coerce").dropna()
        )
        prefix = target
        features[f"{prefix}_mean"] = float(values.mean()) if not values.empty else np.nan
        features[f"{prefix}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
        features[f"{prefix}_min"] = float(values.min()) if not values.empty else np.nan
        features[f"{prefix}_max"] = float(values.max()) if not values.empty else np.nan
        features[f"{prefix}_range"] = (
            float(values.max() - values.min()) if not values.empty else np.nan
        )
        features[f"{prefix}_count"] = float(values.count())

        if column is not None and "model" in predictions:
            working = predictions[["model", column]].copy()
            working[column] = pd.to_numeric(working[column], errors="coerce")
            for model, group in working.groupby("model", dropna=False, sort=True):
                model_values = group[column].dropna()
                features[f"{prefix}_model_{_safe_name(model)}"] = (
                    float(model_values.mean()) if not model_values.empty else np.nan
                )

    for column in APPLICABILITY_COLUMNS:
        if column not in predictions:
            continue
        values = pd.to_numeric(predictions[column], errors="coerce").dropna()
        features[column] = float(values.mean()) if not values.empty else np.nan
    return features


def build_meta_features(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row numeric DataFrame for a prediction table."""
    return pd.DataFrame([build_meta_feature_row(predictions)], dtype=float)


def build_meta_feature_table(tables: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Build an aligned feature matrix from multiple prediction examples."""
    rows = [build_meta_feature_row(table) for table in tables]
    return pd.DataFrame(rows, dtype=float)
