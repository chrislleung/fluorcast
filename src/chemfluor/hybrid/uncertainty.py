"""Split-conformal uncertainty calibration for hybrid regression models."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

CONFIDENCE_BINS = ("high", "medium", "low_medium", "low")


def confidence_bin(confidence: float | None) -> str:
    """Assign confidence to the fixed calibration bins."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return "low"
    if not math.isfinite(value):
        return "low"
    if value >= 0.75:
        return "high"
    if value >= 0.50:
        return "medium"
    if value >= 0.30:
        return "low_medium"
    return "low"


def calibration_residuals(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    confidence: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Create a serializable split-conformal calibration table."""
    true = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if true.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if confidence is None:
        scores = np.full(true.shape, np.nan)
    else:
        scores = np.asarray(list(confidence), dtype=float)
        if scores.shape != true.shape:
            raise ValueError("confidence must have the same shape as y_true")
    result = pd.DataFrame(
        {"absolute_residual": np.abs(true - predicted), "confidence": scores}
    )
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["absolute_residual"])
    result["confidence_bin"] = result["confidence"].map(confidence_bin)
    return result.reset_index(drop=True)


def conformal_quantile(residuals: Iterable[float], coverage: float = 0.90) -> float:
    """Return the finite-sample corrected conformal residual quantile."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be strictly between 0 and 1")
    values = np.asarray(list(residuals), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite calibration residuals are available")
    level = min(1.0, math.ceil((values.size + 1) * coverage) / values.size)
    return float(np.quantile(values, level, method="higher"))


def residual_quantile(
    residuals: pd.DataFrame | Iterable[float],
    confidence: float | None = None,
    coverage: float = 0.90,
) -> float:
    """Use a confidence-bin quantile when available, otherwise the global one."""
    if not isinstance(residuals, pd.DataFrame):
        return conformal_quantile(residuals, coverage)
    if "absolute_residual" not in residuals:
        raise ValueError("Calibration table requires an absolute_residual column")
    selected = residuals
    if confidence is not None and "confidence_bin" in residuals:
        same_bin = residuals[residuals["confidence_bin"] == confidence_bin(confidence)]
        if not same_bin.empty:
            selected = same_bin
    return conformal_quantile(selected["absolute_residual"], coverage)


def prediction_interval(
    prediction: float,
    residuals: pd.DataFrame | Iterable[float],
    confidence: float | None = None,
    coverage: float = 0.90,
) -> tuple[float, float]:
    """Return symmetric lower and upper split-conformal bounds."""
    radius = residual_quantile(residuals, confidence, coverage)
    value = float(prediction)
    return value - radius, value + radius


split_conformal_interval = prediction_interval
