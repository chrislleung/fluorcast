"""Dataclass schemas and validators for ConforFormer conformer caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any

from .config import ConformerGenerationConfig


class GenerationStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


class MoleculeStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class MoleculeConformerRequest:
    chromophore_id: str
    canonical_chromophore_smiles: str | None
    input_smiles: str
    config: ConformerGenerationConfig
    conformer_cache_key: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "canonical_chromophore_smiles": self.canonical_chromophore_smiles,
            "chromophore_id": self.chromophore_id,
            "conformer_cache_key": self.conformer_cache_key,
            "config": self.config.to_payload(),
            "input_smiles": self.input_smiles,
        }


@dataclass(frozen=True)
class ConformerRecord:
    conformer_id: str
    atom_symbols: list[str]
    atomic_numbers: list[int]
    coordinates: list[list[float]]
    energy: float | None
    energy_units: str | None
    optimizer: str
    optimization_convergence_status: str
    generation_status: GenerationStatus
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        atom_count = len(self.atom_symbols)
        if len(self.atomic_numbers) != atom_count:
            raise ValueError("atomic-number count must match atom-symbol count")
        if len(self.coordinates) != atom_count:
            raise ValueError("coordinate rows must match atom count")
        for row in self.coordinates:
            if len(row) != 3:
                raise ValueError("coordinate shape must be [num_atoms, 3]")
        if self.energy is not None and not math.isfinite(self.energy):
            raise ValueError("energy must be finite or None")
        if self.energy is not None and not self.energy_units:
            raise ValueError("energy_units must be explicit when energy is present")
        if self.generation_status == GenerationStatus.OK:
            if self.failure_reason is not None:
                raise ValueError("successful conformers must not have a failure reason")
            for row in self.coordinates:
                if not all(math.isfinite(value) for value in row):
                    raise ValueError("successful conformers must contain finite coordinates")
        if self.generation_status == GenerationStatus.FAILED and not self.failure_reason:
            raise ValueError("failed conformers must preserve their failure reason")

    @property
    def is_successful(self) -> bool:
        return self.generation_status == GenerationStatus.OK

    def to_payload(self) -> dict[str, Any]:
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "atom_symbols": list(self.atom_symbols),
            "conformer_id": self.conformer_id,
            "coordinates": [[float(value) for value in row] for row in self.coordinates],
            "energy": None if self.energy is None else float(self.energy),
            "energy_units": self.energy_units,
            "failure_reason": self.failure_reason,
            "generation_status": self.generation_status.value,
            "optimization_convergence_status": self.optimization_convergence_status,
            "optimizer": self.optimizer,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ConformerRecord":
        return cls(
            conformer_id=str(payload["conformer_id"]),
            atom_symbols=list(payload["atom_symbols"]),
            atomic_numbers=[int(value) for value in payload["atomic_numbers"]],
            coordinates=[[float(value) for value in row] for row in payload["coordinates"]],
            energy=None if payload["energy"] is None else float(payload["energy"]),
            energy_units=payload["energy_units"],
            optimizer=str(payload["optimizer"]),
            optimization_convergence_status=str(payload["optimization_convergence_status"]),
            generation_status=GenerationStatus(payload["generation_status"]),
            failure_reason=payload["failure_reason"],
        )


@dataclass(frozen=True)
class MoleculeConformerCacheRecord:
    chromophore_id: str
    input_smiles: str
    canonical_smiles: str | None
    isomeric_canonical_smiles: str | None
    conformer_cache_key: str
    requested_conformer_count: int
    successful_conformer_count: int
    status: MoleculeStatus
    failure_reason: str | None
    conformer_records: list[ConformerRecord]
    rdkit_version: str
    configuration_payload: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [record.conformer_id for record in self.conformer_records]
        if len(ids) != len(set(ids)):
            raise ValueError("conformer IDs must be unique within a molecule record")
        actual_successes = sum(record.is_successful for record in self.conformer_records)
        if actual_successes != self.successful_conformer_count:
            raise ValueError("successful conformer count must match conformer records")
        if self.status == MoleculeStatus.OK:
            if self.successful_conformer_count <= 0:
                raise ValueError("ok molecule records require at least one successful conformer")
            if self.failure_reason is not None:
                raise ValueError("ok molecule records must not have a failure reason")
        if self.status == MoleculeStatus.FAILED and not self.failure_reason:
            raise ValueError("failed molecule records must preserve their failure reason")

    def to_payload(self) -> dict[str, Any]:
        return {
            "canonical_smiles": self.canonical_smiles,
            "chromophore_id": self.chromophore_id,
            "configuration_payload": self.configuration_payload,
            "conformer_cache_key": self.conformer_cache_key,
            "conformer_records": [record.to_payload() for record in self.conformer_records],
            "failure_reason": self.failure_reason,
            "input_smiles": self.input_smiles,
            "isomeric_canonical_smiles": self.isomeric_canonical_smiles,
            "metadata": self.metadata,
            "rdkit_version": self.rdkit_version,
            "requested_conformer_count": self.requested_conformer_count,
            "status": self.status.value,
            "successful_conformer_count": self.successful_conformer_count,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MoleculeConformerCacheRecord":
        return cls(
            chromophore_id=str(payload["chromophore_id"]),
            input_smiles=str(payload["input_smiles"]),
            canonical_smiles=payload["canonical_smiles"],
            isomeric_canonical_smiles=payload["isomeric_canonical_smiles"],
            conformer_cache_key=str(payload["conformer_cache_key"]),
            requested_conformer_count=int(payload["requested_conformer_count"]),
            successful_conformer_count=int(payload["successful_conformer_count"]),
            status=MoleculeStatus(payload["status"]),
            failure_reason=payload["failure_reason"],
            conformer_records=[
                ConformerRecord.from_payload(record) for record in payload["conformer_records"]
            ],
            rdkit_version=str(payload["rdkit_version"]),
            configuration_payload=dict(payload["configuration_payload"]),
            metadata=dict(payload["metadata"]),
        )
