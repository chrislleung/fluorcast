from __future__ import annotations

import math

from chemfluor.conforformer.config import ConformerGenerationConfig
from chemfluor.conforformer.conformers import generate_conformer_cache_record
from chemfluor.conforformer.schemas import MoleculeStatus


def _small_config(**kwargs: object) -> ConformerGenerationConfig:
    params = {"num_conformers": 4, "retry_conformer_counts": (2, 1)}
    params.update(kwargs)
    return ConformerGenerationConfig(**params)


def test_invalid_smiles_returns_failed_record() -> None:
    record = generate_conformer_cache_record("not a smiles", chromophore_id="bad", config=_small_config())
    assert record.status == MoleculeStatus.FAILED
    assert record.failure_reason == "invalid_smiles"
    assert record.input_smiles == "not a smiles"


def test_ethanol_generates_successful_conformer() -> None:
    record = generate_conformer_cache_record("CCO", chromophore_id="ethanol", config=_small_config())
    assert record.status == MoleculeStatus.OK
    assert record.successful_conformer_count >= 1


def test_butane_generates_finite_coordinates_and_matching_atoms() -> None:
    record = generate_conformer_cache_record("CCCC", chromophore_id="butane", config=_small_config())
    conformer = record.conformer_records[0]
    assert len(conformer.atom_symbols) == len(conformer.atomic_numbers) == len(conformer.coordinates)
    assert all(len(row) == 3 for row in conformer.coordinates)
    assert all(math.isfinite(value) for row in conformer.coordinates for value in row)


def test_benzene_rigid_pruning_is_handled() -> None:
    record = generate_conformer_cache_record("c1ccccc1", chromophore_id="benzene", config=_small_config())
    assert record.status == MoleculeStatus.OK
    assert 1 <= record.successful_conformer_count <= record.requested_conformer_count


def test_same_smiles_and_seed_reproduce_metadata() -> None:
    config = _small_config(random_seed=123)
    first = generate_conformer_cache_record("CCO", chromophore_id="a", config=config)
    second = generate_conformer_cache_record("CCO", chromophore_id="a", config=config)
    assert first.conformer_cache_key == second.conformer_cache_key
    assert first.successful_conformer_count == second.successful_conformer_count
    assert [r.energy for r in first.conformer_records] == [r.energy for r in second.conformer_records]


def test_conformer_ids_are_unique_and_fields_align() -> None:
    record = generate_conformer_cache_record("CCO", config=_small_config())
    ids = [conformer.conformer_id for conformer in record.conformer_records]
    assert len(ids) == len(set(ids))
    assert all(conformer.optimizer for conformer in record.conformer_records)
    assert all(conformer.optimization_convergence_status for conformer in record.conformer_records)
    assert all(
        conformer.energy is None or conformer.energy_units == "kcal/mol"
        for conformer in record.conformer_records
    )


def test_mmff_is_used_when_available() -> None:
    record = generate_conformer_cache_record("CCO", config=_small_config())
    assert {conformer.optimizer for conformer in record.conformer_records} == {"MMFF94"}


def test_fallback_behavior_is_explicit_for_metal_complex() -> None:
    record = generate_conformer_cache_record("[Na+].[Cl-]", config=_small_config(num_conformers=2, retry_conformer_counts=(1,)))
    assert record.metadata["force_field_fallback_count"] >= 0
    assert all(conformer.optimizer in {"MMFF94", "UFF", "none"} for conformer in record.conformer_records)


def test_no_successful_conformers_is_failed_record() -> None:
    config = ConformerGenerationConfig(num_conformers=1, retry_conformer_counts=())
    record = generate_conformer_cache_record("", config=config)
    assert record.status == MoleculeStatus.FAILED
    assert record.successful_conformer_count == 0


def test_stereochemistry_is_retained_in_isomeric_canonical_smiles() -> None:
    record = generate_conformer_cache_record("F[C@H](Cl)Br", config=_small_config())
    assert "@" in (record.isomeric_canonical_smiles or "")
