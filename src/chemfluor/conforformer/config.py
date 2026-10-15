"""Validated configuration for local conformer generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


DEFAULT_CONFORMER_GENERATION_VERSION = "fluorcast-conforformer-conformers-v1"


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class ConformerGenerationConfig:
    """All scientifically relevant knobs for RDKit conformer generation."""

    num_conformers: int = 16
    random_seed: int = 61453
    etkdg_version: str = "ETKDGv3"
    prune_rms_threshold: float = 0.5
    max_attempts: int = 1000
    optimizer: str = "MMFF94"
    fallback_optimizer: str = "UFF"
    max_optimization_iterations: int = 500
    retry_conformer_counts: tuple[int, ...] = field(default_factory=lambda: (8, 4, 1))
    add_hydrogens_for_generation: bool = True
    remove_hydrogens_for_encoder: bool = True
    conformer_generation_version: str = DEFAULT_CONFORMER_GENERATION_VERSION

    def __post_init__(self) -> None:
        if self.num_conformers <= 0:
            raise ValueError("num_conformers must be positive")
        if self.prune_rms_threshold < 0:
            raise ValueError("prune_rms_threshold must be non-negative")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if self.max_optimization_iterations <= 0:
            raise ValueError("max_optimization_iterations must be positive")
        if self.etkdg_version != "ETKDGv3":
            raise ValueError("only ETKDGv3 is supported in this stage")
        if self.optimizer not in {"MMFF94", "MMFF94s"}:
            raise ValueError("optimizer must be MMFF94 or MMFF94s")
        if self.fallback_optimizer != "UFF":
            raise ValueError("fallback_optimizer must be UFF")
        if any(count <= 0 for count in self.retry_conformer_counts):
            raise ValueError("retry_conformer_counts must contain only positive counts")
        if any(count >= self.num_conformers for count in self.retry_conformer_counts):
            raise ValueError("retry_conformer_counts must be smaller than num_conformers")
        if tuple(self.retry_conformer_counts) != tuple(sorted(set(self.retry_conformer_counts), reverse=True)):
            raise ValueError("retry_conformer_counts must be unique and descending")

    def to_payload(self) -> dict[str, Any]:
        return {
            "add_hydrogens_for_generation": self.add_hydrogens_for_generation,
            "conformer_generation_version": self.conformer_generation_version,
            "etkdg_version": self.etkdg_version,
            "fallback_optimizer": self.fallback_optimizer,
            "max_attempts": self.max_attempts,
            "max_optimization_iterations": self.max_optimization_iterations,
            "num_conformers": self.num_conformers,
            "optimizer": self.optimizer,
            "prune_rms_threshold": self.prune_rms_threshold,
            "random_seed": self.random_seed,
            "remove_hydrogens_for_encoder": self.remove_hydrogens_for_encoder,
            "retry_conformer_counts": list(self.retry_conformer_counts),
        }

    def stable_json(self) -> str:
        return _stable_json(self.to_payload())

    @classmethod
    def from_overrides(
        cls,
        *,
        num_conformers: int | None = None,
        random_seed: int | None = None,
        prune_rms_threshold: float | None = None,
    ) -> "ConformerGenerationConfig":
        base = cls()
        effective_num_conformers = base.num_conformers if num_conformers is None else num_conformers
        retry_counts = tuple(
            count for count in base.retry_conformer_counts if count < effective_num_conformers
        )
        if not retry_counts and effective_num_conformers > 1:
            retry_counts = (1,)
        return cls(
            num_conformers=effective_num_conformers,
            random_seed=base.random_seed if random_seed is None else random_seed,
            prune_rms_threshold=(
                base.prune_rms_threshold if prune_rms_threshold is None else prune_rms_threshold
            ),
            etkdg_version=base.etkdg_version,
            max_attempts=base.max_attempts,
            optimizer=base.optimizer,
            fallback_optimizer=base.fallback_optimizer,
            max_optimization_iterations=base.max_optimization_iterations,
            retry_conformer_counts=retry_counts,
            add_hydrogens_for_generation=base.add_hydrogens_for_generation,
            remove_hydrogens_for_encoder=base.remove_hydrogens_for_encoder,
            conformer_generation_version=base.conformer_generation_version,
        )
