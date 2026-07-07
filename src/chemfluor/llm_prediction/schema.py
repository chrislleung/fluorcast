"""Strict, dependency-free schema for LLM prediction responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

DESCRIPTOR_NAMES = (
    "conjugation", "rigidity", "donor_acceptor_character",
    "charge_transfer_likelihood", "solvent_effect_likelihood",
)
CATEGORIES = {"low", "medium", "high", "unknown"}


@dataclass
class LLMOutput:
    target: str
    llm_numeric_prediction: float | None = None
    llm_confidence: float | None = None
    descriptors: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, value: Any, target: str) -> "LLMOutput":
        """Coerce malformed provider output to a safe, entirely usable record."""
        if not isinstance(value, dict):
            return cls.empty(target, "LLM response was not an object")
        warnings = value.get("warnings", [])
        warnings = [str(item) for item in warnings] if isinstance(warnings, list) else [str(warnings)]
        descriptors = value.get("descriptors", {})
        descriptors = descriptors if isinstance(descriptors, dict) else {}
        clean = {name: str(descriptors.get(name, "unknown")).lower() for name in DESCRIPTOR_NAMES}
        for name in DESCRIPTOR_NAMES:
            if clean[name] not in CATEGORIES:
                clean[name] = "unknown"
                warnings.append(f"Invalid descriptor: {name}")
        def number(name: str) -> float | None:
            try:
                result = float(value.get(name))
                return result if result == result and abs(result) != float("inf") else None
            except (TypeError, ValueError):
                return None
        confidence = number("llm_confidence")
        if confidence is not None:
            confidence = min(1.0, max(0.0, confidence))
        return cls(str(value.get("target") or target), number("llm_numeric_prediction"),
                   confidence, clean, warnings)

    @classmethod
    def empty(cls, target: str, warning: str = "Missing LLM output") -> "LLMOutput":
        return cls(target, None, None, {name: "unknown" for name in DESCRIPTOR_NAMES}, [warning])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
