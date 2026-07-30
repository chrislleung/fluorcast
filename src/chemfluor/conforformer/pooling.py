"""Pooling utilities for per-conformer ConforFormer embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


POOLING_SCHEMA_VERSION = 1
POOLING_IMPLEMENTATION_VERSION = "fluorcast-conforformer-pooling-v1"
R_KCAL_MOL_K = 0.00198720425864083
DEFAULT_TEMPERATURE_K = 298.15


@dataclass(frozen=True)
class PoolingResult:
    mean: np.ndarray
    lowest_energy: np.ndarray
    boltzmann_298k: np.ndarray
    lowest_energy_index: int
    boltzmann_used: bool
    boltzmann_fallback_reason: str | None
    boltzmann_weights: np.ndarray


def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("embeddings must have shape [num_conformers, embedding_dim]")
    if not np.isfinite(array).all():
        raise ValueError("embeddings contain non-finite values")
    return array


def mean_pool(embeddings: np.ndarray) -> np.ndarray:
    return _validate_embeddings(embeddings).mean(axis=0, dtype=np.float64).astype(np.float32)


def lowest_energy_pool(embeddings: np.ndarray, energies: np.ndarray) -> tuple[np.ndarray, int, str | None]:
    array = _validate_embeddings(embeddings)
    energy = np.asarray(energies, dtype=np.float64)
    if energy.shape != (array.shape[0],):
        raise ValueError("energies must have one value per conformer")
    finite = np.isfinite(energy)
    if finite.any():
        idx = int(np.nanargmin(energy))
        return array[idx].astype(np.float32, copy=True), idx, None
    return mean_pool(array), -1, "no_finite_energies"


def boltzmann_pool(
    embeddings: np.ndarray,
    energies: np.ndarray,
    *,
    temperature_kelvin: float = DEFAULT_TEMPERATURE_K,
) -> tuple[np.ndarray, bool, str | None, np.ndarray]:
    array = _validate_embeddings(embeddings)
    energy = np.asarray(energies, dtype=np.float64)
    if energy.shape != (array.shape[0],):
        raise ValueError("energies must have one value per conformer")
    if temperature_kelvin <= 0:
        raise ValueError("temperature_kelvin must be positive")
    if not np.isfinite(energy).all():
        return mean_pool(array), False, "nonfinite_or_missing_energies", np.full(array.shape[0], np.nan)
    shifted = energy - float(np.min(energy))
    weights = np.exp(-shifted / (R_KCAL_MOL_K * temperature_kelvin))
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return mean_pool(array), False, "invalid_boltzmann_weights", np.full(array.shape[0], np.nan)
    weights = weights / total
    pooled = weights.astype(np.float64) @ array.astype(np.float64)
    return pooled.astype(np.float32), True, None, weights.astype(np.float64)


def pool_all(embeddings: np.ndarray, energies: np.ndarray) -> PoolingResult:
    mean = mean_pool(embeddings)
    lowest, lowest_idx, lowest_reason = lowest_energy_pool(embeddings, energies)
    boltzmann, used, reason, weights = boltzmann_pool(embeddings, energies)
    if lowest_reason and reason is None:
        reason = lowest_reason
    return PoolingResult(
        mean=mean,
        lowest_energy=lowest,
        boltzmann_298k=boltzmann,
        lowest_energy_index=lowest_idx,
        boltzmann_used=used,
        boltzmann_fallback_reason=reason,
        boltzmann_weights=weights,
    )


def pooling_configuration() -> dict[str, Any]:
    return {
        "schema_version": POOLING_SCHEMA_VERSION,
        "implementation_version": POOLING_IMPLEMENTATION_VERSION,
        "methods": ["mean", "lowest_energy", "boltzmann_298k"],
        "boltzmann_temperature_kelvin": DEFAULT_TEMPERATURE_K,
        "boltzmann_R_kcal_mol_K": R_KCAL_MOL_K,
    }

