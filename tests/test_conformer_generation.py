from __future__ import annotations

import math
from pathlib import Path

import pytest

from chemfluor.conforformer import conformers as conformer_module
from chemfluor.conforformer.cache import load_conformer_cache_record, save_conformer_cache_record
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


def _patch_conformer_record_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        conformer_module,
        "_optimize_conformer",
        lambda mol, conf_id, config: ("MMFF94", "converged", float(conf_id), "kcal/mol", []),
    )
    monkeypatch.setattr(conformer_module, "_atom_symbols", lambda mol: ["C"])
    monkeypatch.setattr(conformer_module, "_atomic_numbers", lambda mol: [6])
    monkeypatch.setattr(
        conformer_module,
        "_coordinates",
        lambda mol, conf_id: [[float(conf_id), 0.0, 0.0]],
    )


def test_normal_etkdg_generation_does_not_invoke_random_coordinate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_conformer_record_extraction(monkeypatch)
    calls: list[dict[str, object]] = []

    def embed_multiple_confs(mol, *, numConfs, params):
        calls.append(
            {
                "numConfs": numConfs,
                "randomSeed": params.randomSeed,
                "useRandomCoords": params.useRandomCoords,
            }
        )
        return (0, 1)

    monkeypatch.setattr(conformer_module.AllChem, "EmbedMultipleConfs", embed_multiple_confs)
    config = _small_config(random_seed=1234)
    record = generate_conformer_cache_record("CCO", config=config)

    assert record.status == MoleculeStatus.OK
    assert calls == [{"numConfs": 4, "randomSeed": 1234, "useRandomCoords": False}]
    assert record.metadata["embed_attempts"] == [{"count": 4, "generated_conformers": 2}]
    assert record.metadata["generation_provenance"] == "normal_etkdg_v3"


def test_random_coordinate_fallback_runs_after_normal_attempts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_conformer_record_extraction(monkeypatch)
    calls: list[dict[str, object]] = []

    def embed_multiple_confs(mol, *, numConfs, params):
        calls.append(
            {
                "numConfs": numConfs,
                "randomSeed": params.randomSeed,
                "useRandomCoords": params.useRandomCoords,
            }
        )
        if params.useRandomCoords and numConfs == 1:
            return (0,)
        return ()

    monkeypatch.setattr(conformer_module.AllChem, "EmbedMultipleConfs", embed_multiple_confs)
    config = ConformerGenerationConfig(num_conformers=16, retry_conformer_counts=(8, 4, 1), random_seed=9876)
    record = generate_conformer_cache_record("CCO", config=config)

    assert record.status == MoleculeStatus.OK
    assert [call["numConfs"] for call in calls] == [16, 8, 4, 1, 8, 4, 1]
    assert [call["useRandomCoords"] for call in calls] == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    assert {call["randomSeed"] for call in calls} == {9876}
    assert record.metadata["generation_provenance"] == "random_coordinate_etkdg_v3_fallback"
    assert record.metadata["embed_attempts"][-3:] == [
        {
            "count": 8,
            "generated_conformers": 0,
            "provenance": "random_coordinate_etkdg_v3_fallback",
            "useRandomCoords": True,
        },
        {
            "count": 4,
            "generated_conformers": 0,
            "provenance": "random_coordinate_etkdg_v3_fallback",
            "useRandomCoords": True,
        },
        {
            "count": 1,
            "generated_conformers": 1,
            "provenance": "random_coordinate_etkdg_v3_fallback",
            "useRandomCoords": True,
        },
    ]


def test_successful_random_coordinate_fallback_is_cached_normally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_conformer_record_extraction(monkeypatch)

    def embed_multiple_confs(mol, *, numConfs, params):
        if params.useRandomCoords:
            return (0,)
        return ()

    monkeypatch.setattr(conformer_module.AllChem, "EmbedMultipleConfs", embed_multiple_confs)
    config = ConformerGenerationConfig(num_conformers=16, retry_conformer_counts=(8, 4, 1), random_seed=7)
    record = generate_conformer_cache_record("CCO", config=config)
    path = save_conformer_cache_record(record, tmp_path)
    loaded = load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key)

    assert loaded.to_payload() == record.to_payload()
    assert loaded.status == MoleculeStatus.OK
    assert loaded.metadata["generation_provenance"] == "random_coordinate_etkdg_v3_fallback"


def test_complete_random_coordinate_fallback_failure_records_no_conformers_generated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, bool]] = []

    def embed_multiple_confs(mol, *, numConfs, params):
        calls.append((numConfs, bool(params.useRandomCoords)))
        return ()

    monkeypatch.setattr(conformer_module.AllChem, "EmbedMultipleConfs", embed_multiple_confs)
    config = ConformerGenerationConfig(num_conformers=16, retry_conformer_counts=(8, 4, 1), random_seed=123)
    record = generate_conformer_cache_record("CCO", config=config)

    assert record.status == MoleculeStatus.FAILED
    assert record.failure_reason == "no_conformers_generated"
    assert record.successful_conformer_count == 0
    assert calls == [
        (16, False),
        (8, False),
        (4, False),
        (1, False),
        (8, True),
        (4, True),
        (1, True),
    ]


def test_random_coordinate_fallback_is_deterministic_with_same_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_conformer_record_extraction(monkeypatch)

    def embed_multiple_confs(mol, *, numConfs, params):
        if params.useRandomCoords and params.randomSeed == 2468 and numConfs == 4:
            return (0, 1)
        return ()

    monkeypatch.setattr(conformer_module.AllChem, "EmbedMultipleConfs", embed_multiple_confs)
    config = ConformerGenerationConfig(
        num_conformers=16,
        retry_conformer_counts=(8, 4, 1),
        random_seed=2468,
    )
    first = generate_conformer_cache_record("CCO", config=config)
    second = generate_conformer_cache_record("CCO", config=config)

    assert first.conformer_cache_key == second.conformer_cache_key
    assert first.successful_conformer_count == second.successful_conformer_count
    assert first.metadata["generation_provenance"] == "random_coordinate_etkdg_v3_fallback"
    assert second.metadata["generation_provenance"] == "random_coordinate_etkdg_v3_fallback"
    assert [record.energy for record in first.conformer_records] == [
        record.energy for record in second.conformer_records
    ]
    assert [
        record.coordinates for record in first.conformer_records
    ] == [record.coordinates for record in second.conformer_records]
