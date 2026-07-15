"""Deterministic RDKit conformer generation for the first ConforFormer stage."""

from __future__ import annotations

from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from .cache import build_conformer_cache_key
from .config import ConformerGenerationConfig
from .schemas import (
    ConformerRecord,
    GenerationStatus,
    MoleculeConformerCacheRecord,
    MoleculeStatus,
)


def canonicalize_smiles(smiles: str) -> tuple[str | None, str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None, None
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    isomeric = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return canonical, isomeric


def _failed_record(
    *,
    chromophore_id: str,
    input_smiles: str,
    canonical_smiles: str | None,
    isomeric_canonical_smiles: str | None,
    config: ConformerGenerationConfig,
    failure_reason: str,
    metadata: dict[str, Any] | None = None,
) -> MoleculeConformerCacheRecord:
    return MoleculeConformerCacheRecord(
        chromophore_id=chromophore_id,
        input_smiles=input_smiles,
        canonical_smiles=canonical_smiles,
        isomeric_canonical_smiles=isomeric_canonical_smiles,
        conformer_cache_key=build_conformer_cache_key(
            canonical_smiles=canonical_smiles,
            isomeric_canonical_smiles=isomeric_canonical_smiles,
            config=config,
        ),
        requested_conformer_count=config.num_conformers,
        successful_conformer_count=0,
        status=MoleculeStatus.FAILED,
        failure_reason=failure_reason,
        conformer_records=[],
        rdkit_version=rdBase.rdkitVersion,
        configuration_payload=config.to_payload(),
        metadata=metadata or {},
    )


def _embed_params(config: ConformerGenerationConfig) -> AllChem.EmbedParameters:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(config.random_seed)
    params.pruneRmsThresh = float(config.prune_rms_threshold)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(config.max_attempts)
    elif hasattr(params, "maxIterations"):
        params.maxIterations = int(config.max_attempts)
    params.numThreads = 1
    return params


def _coordinates(mol: Chem.Mol, conf_id: int) -> list[list[float]]:
    conf = mol.GetConformer(conf_id)
    return [
        [
            float(conf.GetAtomPosition(idx).x),
            float(conf.GetAtomPosition(idx).y),
            float(conf.GetAtomPosition(idx).z),
        ]
        for idx in range(mol.GetNumAtoms())
    ]


def _atom_symbols(mol: Chem.Mol) -> list[str]:
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def _atomic_numbers(mol: Chem.Mol) -> list[int]:
    return [int(atom.GetAtomicNum()) for atom in mol.GetAtoms()]


def _energy_with_mmff(mol: Chem.Mol, conf_id: int, variant: str) -> float | None:
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    if props is None:
        return None
    force_field = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if force_field is None:
        return None
    return float(force_field.CalcEnergy())


def _energy_with_uff(mol: Chem.Mol, conf_id: int) -> float | None:
    force_field = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if force_field is None:
        return None
    return float(force_field.CalcEnergy())


def _optimize_conformer(
    mol: Chem.Mol,
    conf_id: int,
    config: ConformerGenerationConfig,
) -> tuple[str, str, float | None, str | None, list[str]]:
    notes: list[str] = []
    optimizer = "none"
    status = "not_attempted"
    energy: float | None = None
    energy_units: str | None = None

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            optimizer = config.optimizer
            result = AllChem.MMFFOptimizeMolecule(
                mol,
                confId=conf_id,
                maxIters=config.max_optimization_iterations,
                mmffVariant=config.optimizer,
            )
            status = "converged" if result == 0 else f"not_converged:{result}"
            try:
                energy = _energy_with_mmff(mol, conf_id, config.optimizer)
                energy_units = "kcal/mol" if energy is not None else None
            except Exception as exc:  # pragma: no cover - RDKit-version dependent
                notes.append(f"mmff_energy_failed:{type(exc).__name__}")
        elif AllChem.UFFHasAllMoleculeParams(mol):
            notes.append("mmff_parameters_unavailable")
            optimizer = config.fallback_optimizer
            result = AllChem.UFFOptimizeMolecule(
                mol,
                confId=conf_id,
                maxIters=config.max_optimization_iterations,
            )
            status = "converged" if result == 0 else f"not_converged:{result}"
            try:
                energy = _energy_with_uff(mol, conf_id)
                energy_units = "kcal/mol" if energy is not None else None
            except Exception as exc:  # pragma: no cover - RDKit-version dependent
                notes.append(f"uff_energy_failed:{type(exc).__name__}")
        else:
            notes.extend(["mmff_parameters_unavailable", "uff_parameters_unavailable"])
            status = "parameters_unavailable"
    except Exception as exc:
        status = f"optimization_failed:{type(exc).__name__}"
        notes.append(str(exc))
    return optimizer, status, energy, energy_units, notes


def generate_conformer_cache_record(
    smiles: str,
    *,
    chromophore_id: str | None = None,
    config: ConformerGenerationConfig | None = None,
) -> MoleculeConformerCacheRecord:
    config = config or ConformerGenerationConfig()
    chromophore_id = chromophore_id or smiles
    canonical_smiles, isomeric_canonical_smiles = canonicalize_smiles(smiles)
    if canonical_smiles is None or isomeric_canonical_smiles is None:
        return _failed_record(
            chromophore_id=chromophore_id,
            input_smiles=smiles,
            canonical_smiles=None,
            isomeric_canonical_smiles=None,
            config=config,
            failure_reason="invalid_smiles",
        )

    mol = Chem.MolFromSmiles(isomeric_canonical_smiles)
    if mol is None:
        return _failed_record(
            chromophore_id=chromophore_id,
            input_smiles=smiles,
            canonical_smiles=canonical_smiles,
            isomeric_canonical_smiles=isomeric_canonical_smiles,
            config=config,
            failure_reason="canonical_smiles_failed_to_parse",
        )
    work_mol = Chem.AddHs(mol) if config.add_hydrogens_for_generation else Chem.Mol(mol)

    conformer_counts = (config.num_conformers, *config.retry_conformer_counts)
    embed_attempts: list[dict[str, Any]] = []
    conf_ids: tuple[int, ...] = ()
    for count in conformer_counts:
        attempt_mol = Chem.Mol(work_mol)
        try:
            ids = tuple(
                AllChem.EmbedMultipleConfs(
                    attempt_mol,
                    numConfs=int(count),
                    params=_embed_params(config),
                )
            )
        except Exception as exc:
            embed_attempts.append(
                {"count": count, "failure_reason": f"conformer_generation_failed:{type(exc).__name__}"}
            )
            continue
        embed_attempts.append({"count": count, "generated_conformers": len(ids)})
        if ids:
            work_mol = attempt_mol
            conf_ids = ids
            break

    cache_key = build_conformer_cache_key(
        canonical_smiles=canonical_smiles,
        isomeric_canonical_smiles=isomeric_canonical_smiles,
        config=config,
    )
    if not conf_ids:
        return _failed_record(
            chromophore_id=chromophore_id,
            input_smiles=smiles,
            canonical_smiles=canonical_smiles,
            isomeric_canonical_smiles=isomeric_canonical_smiles,
            config=config,
            failure_reason="no_conformers_generated",
            metadata={"embed_attempts": embed_attempts},
        )

    conformer_records: list[ConformerRecord] = []
    fallback_count = 0
    optimization_notes: dict[str, list[str]] = {}
    for index, conf_id in enumerate(conf_ids):
        optimizer, convergence_status, energy, energy_units, notes = _optimize_conformer(
            work_mol,
            conf_id,
            config,
        )
        if optimizer == config.fallback_optimizer:
            fallback_count += 1
        if notes:
            optimization_notes[f"conf_{index:04d}"] = notes
        conformer_records.append(
            ConformerRecord(
                conformer_id=f"{cache_key}_conf_{index:04d}",
                atom_symbols=_atom_symbols(work_mol),
                atomic_numbers=_atomic_numbers(work_mol),
                coordinates=_coordinates(work_mol, conf_id),
                energy=energy,
                energy_units=energy_units,
                optimizer=optimizer,
                optimization_convergence_status=convergence_status,
                generation_status=GenerationStatus.OK,
                failure_reason=None,
            )
        )

    metadata = {
        "embed_attempts": embed_attempts,
        "force_field_fallback_count": fallback_count,
        "optimization_notes": optimization_notes,
    }
    return MoleculeConformerCacheRecord(
        chromophore_id=chromophore_id,
        input_smiles=smiles,
        canonical_smiles=canonical_smiles,
        isomeric_canonical_smiles=isomeric_canonical_smiles,
        conformer_cache_key=cache_key,
        requested_conformer_count=config.num_conformers,
        successful_conformer_count=len(conformer_records),
        status=MoleculeStatus.OK,
        failure_reason=None,
        conformer_records=conformer_records,
        rdkit_version=rdBase.rdkitVersion,
        configuration_payload=config.to_payload(),
        metadata=metadata,
    )
