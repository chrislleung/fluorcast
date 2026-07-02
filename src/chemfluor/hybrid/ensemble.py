"""Training, persistence, and inference for FluorCast hybrid ensembles."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REGRESSION_TARGETS = {"emission_nm", "quantum_yield"}
DEFAULT_BRIGHT_THRESHOLD = 0.10


def _regression_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-4, 4, 17))),
        ]
    )


def _classification_pipeline(cv: int) -> Pipeline:
    classifier = CalibratedClassifierCV(
        LogisticRegression(max_iter=2000, class_weight="balanced"), cv=cv
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def train_hybrid_ensemble(
    features: pd.DataFrame,
    labels: pd.Series | np.ndarray,
    target_name: str,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
) -> dict[str, Any]:
    """Fit a regression model and, for QY, an optional calibrated classifier."""
    if target_name not in REGRESSION_TARGETS:
        raise ValueError(f"Unsupported target_name: {target_name}")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(pd.Series(labels), errors="coerce")
    valid = y.notna()
    if valid.sum() < 2:
        raise ValueError("At least two finite labeled examples are required")
    model: dict[str, Any] = {
        "target_name": target_name,
        "regressor": _regression_pipeline().fit(numeric.loc[valid], y.loc[valid]),
        "classifier": None,
        "bright_threshold": float(bright_threshold),
    }
    if target_name == "quantum_yield":
        classes = (y.loc[valid] >= bright_threshold).astype(int)
        counts = classes.value_counts()
        if len(counts) == 2 and int(counts.min()) >= 2:
            model["classifier"] = _classification_pipeline(min(5, int(counts.min()))).fit(
                numeric.loc[valid], classes
            )
    return model


def predict_hybrid_ensemble(model: dict[str, Any], features: pd.DataFrame) -> dict[str, float]:
    """Predict a target and optional bright-class probability."""
    prediction = float(model["regressor"].predict(features)[0])
    result = {"prediction": prediction}
    classifier = model.get("classifier")
    if classifier is not None:
        result["bright_probability"] = float(classifier.predict_proba(features)[0, 1])
        result["qy_bright_dim"] = int(result["bright_probability"] >= 0.5)
    return result


def save_hybrid_ensemble(
    model: dict[str, Any],
    feature_columns: list[str],
    out_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist the fitted model, exact feature order, and training metadata."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / "model.joblib")
    (destination / "feature_columns.json").write_text(
        json.dumps(feature_columns, indent=2) + "\n", encoding="utf-8"
    )
    payload = {
        "target_name": model["target_name"],
        "model_type": "RidgeCV",
        "classification_model_type": (
            "CalibratedClassifierCV(LogisticRegression)" if model.get("classifier") else None
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_data_requirement": (
            "Use only out-of-fold molecule-grouped or scaffold-grouped validation predictions; "
            "never use final test labels."
        ),
    }
    if metadata:
        payload.update(metadata)
    (destination / "metadata.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def load_hybrid_ensemble(model_dir: str | Path) -> tuple[dict[str, Any], list[str], dict]:
    """Load a hybrid ensemble and its inference contract."""
    source = Path(model_dir)
    model = joblib.load(source / "model.joblib")
    columns = json.loads((source / "feature_columns.json").read_text(encoding="utf-8"))
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    return model, list(columns), metadata


def align_features(features: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Align inference features to training order, inserting missing values."""
    return features.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
