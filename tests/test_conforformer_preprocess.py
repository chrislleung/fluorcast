from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from chemfluor.conforformer.cache import load_conformer_cache_record, save_conformer_cache_record
from chemfluor.conforformer.dictionary import ConforFormerDictionary, load_conforformer_dictionary
from chemfluor.conforformer.preprocess import (
    ConforFormerPreprocessingConfig,
    PreprocessingError,
    collate_preprocessed_conformers,
    preprocess_conformer,
    preprocess_successful_conformers,
)
from chemfluor.conforformer.schemas import (
    ConformerRecord,
    GenerationStatus,
    MoleculeConformerCacheRecord,
    MoleculeStatus,
)


def _dict(tmp_path: Path, extra: list[str] | None = None) -> ConforFormerDictionary:
    tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "C", "O", "N"]
    if extra:
        tokens.extend(extra)
    path = tmp_path / "dict.txt"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return load_conforformer_dictionary(path)


def _conformer(
    *,
    conformer_id: str = "conf-1",
    symbols: list[str],
    atomic_numbers: list[int],
    coordinates: list[list[float]],
    status: GenerationStatus = GenerationStatus.OK,
    failure_reason: str | None = None,
) -> ConformerRecord:
    return ConformerRecord(
        conformer_id=conformer_id,
        atom_symbols=symbols,
        atomic_numbers=atomic_numbers,
        coordinates=coordinates,
        energy=None,
        energy_units=None,
        optimizer="MMFF94",
        optimization_convergence_status="converged",
        generation_status=status,
        failure_reason=failure_reason,
    )


def _molecule(conformers: list[ConformerRecord], *, status: MoleculeStatus = MoleculeStatus.OK) -> MoleculeConformerCacheRecord:
    successes = sum(record.is_successful for record in conformers)
    return MoleculeConformerCacheRecord(
        chromophore_id="mol-1",
        input_smiles="CCO",
        canonical_smiles="CCO",
        isomeric_canonical_smiles="CCO",
        conformer_cache_key="cache-key",
        requested_conformer_count=max(1, len(conformers)),
        successful_conformer_count=successes,
        status=status,
        failure_reason=None if status == MoleculeStatus.OK else "failed",
        conformer_records=conformers,
        rdkit_version="test-rdkit",
        configuration_payload={"test": True},
    )


def _ethanol_conformer() -> ConformerRecord:
    return _conformer(
        symbols=["C", "H", "C", "H", "O", "H"],
        atomic_numbers=[6, 1, 6, 1, 8, 1],
        coordinates=[
            [0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0],
            [2.0, 0.0, 0.0],
            [8.0, 8.0, 8.0],
            [2.0, 2.0, 0.0],
            [7.0, 7.0, 7.0],
        ],
    )


def test_hydrogen_removal_tokenization_and_alignment(tmp_path: Path) -> None:
    dictionary = _dict(tmp_path)
    molecule = _molecule([_ethanol_conformer()])
    record = preprocess_conformer(molecule, molecule.conformer_records[0], dictionary)
    assert record.heavy_atom_symbols == ("C", "C", "O")
    assert record.heavy_atomic_numbers == (6, 6, 8)
    assert record.original_atom_count == 6
    assert record.heavy_atom_count == 3
    assert record.src_tokens.tolist() == [dictionary.cls_id, dictionary.index("C"), dictionary.index("C"), dictionary.index("O"), dictionary.sep_id]
    assert np.allclose(record.src_coord[1:-1], np.asarray([[-4 / 3, -2 / 3, 0], [2 / 3, -2 / 3, 0], [2 / 3, 4 / 3, 0]], dtype=np.float32))


def test_all_hydrogen_input_fails(tmp_path: Path) -> None:
    conformer = _conformer(symbols=["H", "H"], atomic_numbers=[1, 1], coordinates=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(PreprocessingError, match="no heavy atoms"):
        preprocess_conformer(_molecule([conformer]), conformer, _dict(tmp_path))


def test_unknown_atoms_fail_or_use_unk_explicitly(tmp_path: Path) -> None:
    conformer = _conformer(symbols=["Xe"], atomic_numbers=[54], coordinates=[[0, 0, 0]])
    molecule = _molecule([conformer])
    dictionary = _dict(tmp_path)
    with pytest.raises(PreprocessingError, match="Xe"):
        preprocess_conformer(molecule, conformer, dictionary)
    record = preprocess_conformer(
        molecule,
        conformer,
        dictionary,
        ConforFormerPreprocessingConfig(unknown_atom_policy="use_unk"),
    )
    assert record.unknown_atom_symbols == ("Xe",)
    assert record.src_tokens.tolist() == [dictionary.cls_id, dictionary.unk_id, dictionary.sep_id]


def test_coordinate_centering_special_rows_dtype_and_no_mutation(tmp_path: Path) -> None:
    conformer = _conformer(symbols=["C", "O"], atomic_numbers=[6, 8], coordinates=[[0, 0, 0], [2, 0, 0]])
    before = deepcopy(conformer.to_payload())
    record = preprocess_conformer(_molecule([conformer]), conformer, _dict(tmp_path))
    assert np.allclose(record.src_coord, np.asarray([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=np.float32))
    assert np.allclose(record.src_coord[1:-1].mean(axis=0), 0)
    assert record.src_coord.dtype == np.float32
    assert conformer.to_payload() == before


def test_distances_are_hand_calculated_and_include_special_rows(tmp_path: Path) -> None:
    conformer = _conformer(symbols=["C", "O"], atomic_numbers=[6, 8], coordinates=[[0, 0, 0], [2, 0, 0]])
    record = preprocess_conformer(_molecule([conformer]), conformer, _dict(tmp_path))
    expected = np.asarray(
        [
            [0, 1, 1, 0],
            [1, 0, 2, 1],
            [1, 2, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(record.src_distance, expected)
    assert np.allclose(record.src_distance, record.src_distance.T)
    assert np.allclose(np.diag(record.src_distance), 0)
    assert record.src_distance.shape == (4, 4)
    assert record.src_distance.dtype == np.float32


def test_edge_types_use_vocab_size_exactly(tmp_path: Path) -> None:
    small = _dict(tmp_path)
    conformer = _conformer(symbols=["C"], atomic_numbers=[6], coordinates=[[0, 0, 0]])
    record = preprocess_conformer(_molecule([conformer]), conformer, small)
    assert np.array_equal(record.src_edge_type, record.src_tokens[:, None] * len(small) + record.src_tokens[None, :])
    assert record.src_edge_type.shape == (3, 3)
    assert record.src_edge_type.dtype == np.int64
    larger = _dict(tmp_path, extra=["F"])
    changed = preprocess_conformer(_molecule([conformer]), conformer, larger)
    assert not np.array_equal(record.src_edge_type, changed.src_edge_type)


def test_sequence_limits_count_special_tokens(tmp_path: Path) -> None:
    conformer = _conformer(symbols=["C"], atomic_numbers=[6], coordinates=[[0, 0, 0]])
    preprocess_conformer(_molecule([conformer]), conformer, _dict(tmp_path), ConforFormerPreprocessingConfig(max_sequence_length=3))
    with pytest.raises(PreprocessingError, match="sequence length 3 exceeds"):
        preprocess_conformer(_molecule([conformer]), conformer, _dict(tmp_path), ConforFormerPreprocessingConfig(max_sequence_length=2))


def test_batch_collation_right_pads_with_upstream_values(tmp_path: Path) -> None:
    dictionary = _dict(tmp_path)
    one = _conformer(conformer_id="short", symbols=["C"], atomic_numbers=[6], coordinates=[[0, 0, 0]])
    two = _conformer(conformer_id="long", symbols=["C", "O", "N"], atomic_numbers=[6, 8, 7], coordinates=[[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    records = [preprocess_conformer(_molecule([one]), one, dictionary), preprocess_conformer(_molecule([two]), two, dictionary)]
    batch = collate_preprocessed_conformers(records, dictionary)
    assert batch.src_tokens.shape == (2, 8)
    assert batch.src_coord.shape == (2, 8, 3)
    assert batch.src_distance.shape == (2, 8, 8)
    assert batch.src_edge_type.shape == (2, 8, 8)
    assert batch.sequence_lengths.tolist() == [3, 5]
    assert batch.conformer_ids == ("short", "long")
    assert batch.chromophore_ids == ("mol-1", "mol-1")
    assert np.all(batch.src_tokens[0, 3:] == dictionary.pad_id)
    assert np.allclose(batch.src_coord[0, 3:], 0)
    assert np.allclose(batch.src_distance[0, 3:, :], 0)
    assert np.allclose(batch.src_distance[0, :, 3:], 0)
    assert np.all(batch.src_edge_type[0, 3:, :] == 0)
    assert np.all(batch.src_edge_type[0, :, 3:] == 0)
    assert batch.src_tokens.dtype == np.int64
    assert batch.src_coord.dtype == np.float32
    assert batch.src_distance.dtype == np.float32
    assert batch.src_edge_type.dtype == np.int64


def test_stage2_cache_integration_and_failed_conformer_rejection(tmp_path: Path) -> None:
    ok = _ethanol_conformer()
    failed = _conformer(
        conformer_id="failed",
        symbols=["C"],
        atomic_numbers=[6],
        coordinates=[[0, 0, 0]],
        status=GenerationStatus.FAILED,
        failure_reason="simulated",
    )
    molecule = _molecule([ok, failed])
    before = deepcopy(molecule.to_payload())
    path = save_conformer_cache_record(molecule, tmp_path)
    loaded = load_conformer_cache_record(path)
    records = preprocess_successful_conformers(loaded, _dict(tmp_path))
    assert [record.conformer_id for record in records] == [ok.conformer_id]
    assert records[0].conformer_cache_key == molecule.conformer_cache_key
    assert loaded.to_payload() == before
    with pytest.raises(PreprocessingError, match="failed conformer"):
        preprocess_conformer(loaded, failed, _dict(tmp_path))


def test_failed_molecule_record_is_rejected(tmp_path: Path) -> None:
    molecule = _molecule([], status=MoleculeStatus.FAILED)
    with pytest.raises(PreprocessingError, match="failed molecule"):
        preprocess_successful_conformers(molecule, _dict(tmp_path))
