"""NumPy preprocessing for ConforFormer encoder inputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np

from .dictionary import ConforFormerDictionary
from .schemas import ConformerRecord, GenerationStatus, MoleculeConformerCacheRecord, MoleculeStatus


DEFAULT_PREPROCESS_VERSION = "fluorcast-conforformer-preprocess-v2-runtime-mask-vocab"
CENTERING_VALIDATION_ATOL = 1e-5


PADDING_RULE_SOURCES = {
    "task": "third_party/ConforFormer/unimol/unimol/tasks/unimol_contrast.py",
    "tokens": "RightPadDataset(src_dataset, pad_idx=self.dictionary.pad())",
    "coords": (
        "FlattenRightPadDatasetCoord(..., pad_idx=0) and "
        "unimol/data/flatten_coord_pad_dataset.py: pad_to_multiple=8, left_pad=False"
    ),
    "distance": (
        "FlattenRightPadDataset2D(..., pad_idx=0) and "
        "unimol/data/flatten_right_pad_dataset.py: pad_to_multiple=8, left_pad=False"
    ),
    "edge_type": "RightPadDataset2D(edge_type, pad_idx=0)",
}


class PreprocessingError(ValueError):
    """Raised when a conformer cannot be converted to encoder inputs."""


@dataclass(frozen=True)
class ConforFormerPreprocessingConfig:
    hydrogen_policy: str = "remove_all"
    unknown_atom_policy: str = "fail"
    max_sequence_length: int = 512
    preprocess_version: str = DEFAULT_PREPROCESS_VERSION
    pad_to_multiple: int = 8
    coordinate_dtype: str = "float32"
    token_dtype: str = "int64"
    edge_dtype: str = "int64"

    def __post_init__(self) -> None:
        if self.hydrogen_policy != "remove_all":
            raise ValueError("only hydrogen_policy='remove_all' is supported in this stage")
        if self.unknown_atom_policy not in {"fail", "use_unk"}:
            raise ValueError("unknown_atom_policy must be 'fail' or 'use_unk'")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.pad_to_multiple <= 0:
            raise ValueError("pad_to_multiple must be positive")
        if self.coordinate_dtype != "float32":
            raise ValueError("coordinate_dtype must be float32")
        if self.token_dtype != "int64":
            raise ValueError("token_dtype must be int64")
        if self.edge_dtype != "int64":
            raise ValueError("edge_dtype must be int64")

    def to_payload(self) -> dict[str, Any]:
        return {
            "coordinate_dtype": self.coordinate_dtype,
            "edge_dtype": self.edge_dtype,
            "hydrogen_policy": self.hydrogen_policy,
            "max_sequence_length": self.max_sequence_length,
            "pad_to_multiple": self.pad_to_multiple,
            "preprocess_version": self.preprocess_version,
            "token_dtype": self.token_dtype,
            "unknown_atom_policy": self.unknown_atom_policy,
        }

    def stable_json(self) -> str:
        return json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class PreprocessedConformerRecord:
    src_tokens: np.ndarray
    src_coord: np.ndarray
    src_distance: np.ndarray
    src_edge_type: np.ndarray
    chromophore_id: str
    canonical_smiles: str | None
    conformer_id: str
    conformer_cache_key: str
    dictionary_sha256: str
    dictionary_vocab_size: int
    dictionary_pad_id: int
    dictionary_cls_id: int
    dictionary_sep_id: int
    dictionary_unk_id: int
    preprocessing_config: dict[str, Any]
    heavy_atom_symbols: tuple[str, ...]
    heavy_atomic_numbers: tuple[int, ...]
    unknown_atom_symbols: tuple[str, ...]
    original_atom_count: int
    heavy_atom_count: int
    sequence_length: int

    def __post_init__(self) -> None:
        validate_preprocessed_record(self)


@dataclass(frozen=True)
class CollatedConformerBatch:
    src_tokens: np.ndarray
    src_coord: np.ndarray
    src_distance: np.ndarray
    src_edge_type: np.ndarray
    sequence_lengths: np.ndarray
    conformer_ids: tuple[str, ...]
    chromophore_ids: tuple[str, ...]
    padding_sources: dict[str, str]


def _as_coordinate_array(conformer: ConformerRecord) -> np.ndarray:
    coords = np.asarray(conformer.coordinates, dtype=np.float32)
    if coords.shape != (len(conformer.atom_symbols), 3):
        raise PreprocessingError(
            f"coordinates for conformer {conformer.conformer_id} must have shape [num_atoms, 3]"
        )
    if not np.isfinite(coords).all():
        raise PreprocessingError(f"coordinates for conformer {conformer.conformer_id} must be finite")
    return coords


def _unknown_symbols(symbols: list[str], dictionary: ConforFormerDictionary) -> tuple[str, ...]:
    return tuple(sorted({symbol for symbol in symbols if symbol not in dictionary.token_to_index}))


def preprocess_conformer(
    molecule_record: MoleculeConformerCacheRecord,
    conformer: ConformerRecord,
    dictionary: ConforFormerDictionary,
    config: ConforFormerPreprocessingConfig | None = None,
) -> PreprocessedConformerRecord:
    """Convert one successful Stage 2 conformer to ConforFormer NumPy arrays."""

    config = config or ConforFormerPreprocessingConfig()
    if molecule_record.status != MoleculeStatus.OK:
        raise PreprocessingError("failed molecule cache records cannot be preprocessed")
    if conformer.generation_status != GenerationStatus.OK:
        raise PreprocessingError(f"failed conformer records cannot be preprocessed: {conformer.conformer_id}")

    coords = _as_coordinate_array(conformer)
    original_atom_count = len(conformer.atom_symbols)
    heavy_indices = [index for index, symbol in enumerate(conformer.atom_symbols) if symbol != "H"]
    if not heavy_indices:
        raise PreprocessingError(f"conformer {conformer.conformer_id} has no heavy atoms after hydrogen removal")

    heavy_symbols = [conformer.atom_symbols[index] for index in heavy_indices]
    heavy_atomic_numbers = [int(conformer.atomic_numbers[index]) for index in heavy_indices]
    heavy_coords = coords[heavy_indices, :].astype(np.float32, copy=True)
    unknown = _unknown_symbols(heavy_symbols, dictionary)
    if unknown and config.unknown_atom_policy == "fail":
        raise PreprocessingError(
            f"unknown atom symbols for conformer {conformer.conformer_id}: {', '.join(unknown)}"
        )

    centered_heavy = (heavy_coords - heavy_coords.mean(axis=0, keepdims=True)).astype(np.float32)
    token_ids = [
        dictionary.token_to_index.get(symbol, dictionary.unk_id)
        for symbol in heavy_symbols
    ]
    tokens = np.asarray([dictionary.cls_id, *token_ids, dictionary.sep_id], dtype=np.int64)
    sequence_length = int(tokens.shape[0])
    if sequence_length > config.max_sequence_length:
        raise PreprocessingError(
            f"sequence length {sequence_length} exceeds max_sequence_length {config.max_sequence_length}"
        )

    src_coord = np.zeros((sequence_length, 3), dtype=np.float32)
    src_coord[1:-1, :] = centered_heavy
    deltas = src_coord[:, None, :] - src_coord[None, :, :]
    src_distance = np.linalg.norm(deltas, axis=-1).astype(np.float32)
    src_edge_type = (
        tokens[:, None] * np.int64(dictionary.vocab_size) + tokens[None, :]
    ).astype(np.int64)

    return PreprocessedConformerRecord(
        src_tokens=tokens,
        src_coord=src_coord,
        src_distance=src_distance,
        src_edge_type=src_edge_type,
        chromophore_id=molecule_record.chromophore_id,
        canonical_smiles=molecule_record.canonical_smiles,
        conformer_id=conformer.conformer_id,
        conformer_cache_key=molecule_record.conformer_cache_key,
        dictionary_sha256=dictionary.sha256,
        dictionary_vocab_size=dictionary.vocab_size,
        dictionary_pad_id=dictionary.pad_id,
        dictionary_cls_id=dictionary.cls_id,
        dictionary_sep_id=dictionary.sep_id,
        dictionary_unk_id=dictionary.unk_id,
        preprocessing_config=config.to_payload(),
        heavy_atom_symbols=tuple(heavy_symbols),
        heavy_atomic_numbers=tuple(heavy_atomic_numbers),
        unknown_atom_symbols=unknown,
        original_atom_count=original_atom_count,
        heavy_atom_count=len(heavy_symbols),
        sequence_length=sequence_length,
    )


def preprocess_successful_conformers(
    molecule_record: MoleculeConformerCacheRecord,
    dictionary: ConforFormerDictionary,
    config: ConforFormerPreprocessingConfig | None = None,
) -> list[PreprocessedConformerRecord]:
    if molecule_record.status != MoleculeStatus.OK:
        raise PreprocessingError("failed molecule cache records cannot be preprocessed")
    return [
        preprocess_conformer(molecule_record, conformer, dictionary, config)
        for conformer in molecule_record.conformer_records
        if conformer.is_successful
    ]


def validate_preprocessed_record(record: PreprocessedConformerRecord) -> None:
    tokens = record.src_tokens
    coord = record.src_coord
    distance = record.src_distance
    edge_type = record.src_edge_type
    length = record.sequence_length
    if tokens.dtype != np.int64:
        raise ValueError("src_tokens must use int64")
    if edge_type.dtype != np.int64:
        raise ValueError("src_edge_type must use int64")
    if coord.dtype != np.float32:
        raise ValueError("src_coord must use float32")
    if distance.dtype != np.float32:
        raise ValueError("src_distance must use float32")
    if tokens.shape != (length,):
        raise ValueError("token length must equal sequence_length")
    if coord.shape != (length, 3):
        raise ValueError("coordinate rows must match token length")
    if distance.shape != (length, length):
        raise ValueError("distance matrix must be square with token length")
    if edge_type.shape != (length, length):
        raise ValueError("edge-type matrix must be square with token length")
    if not np.isfinite(coord).all() or not np.isfinite(distance).all():
        raise ValueError("coordinates and distances must be finite")
    if record.heavy_atom_count > 0:
        heavy_coords = coord[1:-1]
        heavy_centroid = heavy_coords.astype(np.float64).mean(axis=0)
        if not np.allclose(heavy_centroid, 0.0, atol=CENTERING_VALIDATION_ATOL, rtol=0.0):
            raise ValueError("centered heavy-atom coordinates must have near-zero mean")
    if int(tokens[0]) != record.dictionary_cls_id:
        raise ValueError("first token must be [CLS]")
    if int(tokens[-1]) != record.dictionary_sep_id:
        raise ValueError("last non-padding token must be [SEP]")
    if tokens.min(initial=0) < 0 or tokens.max(initial=0) >= record.dictionary_vocab_size:
        raise ValueError("token IDs must be within dictionary bounds")
    if not np.allclose(coord[0], 0.0) or not np.allclose(coord[-1], 0.0):
        raise ValueError("special-token coordinate rows must be zero")
    if not np.allclose(distance, distance.T, atol=1e-6):
        raise ValueError("distance matrix must be symmetric")
    if not np.allclose(np.diag(distance), 0.0, atol=1e-6):
        raise ValueError("distance diagonal must be zero")
    max_sequence_length = int(record.preprocessing_config["max_sequence_length"])
    if length > max_sequence_length:
        raise ValueError("sequence length exceeds configured maximum")
    expected_edge_type = tokens[:, None] * np.int64(record.dictionary_vocab_size) + tokens[None, :]
    if not np.array_equal(edge_type, expected_edge_type):
        raise ValueError("edge-type formula does not hold")


def _pad_length(max_length: int, multiple: int) -> int:
    if max_length % multiple == 0:
        return max_length
    return int(((max_length - 0.1) // multiple + 1) * multiple)


def collate_preprocessed_conformers(
    records: list[PreprocessedConformerRecord],
    dictionary: ConforFormerDictionary,
    *,
    pad_to_multiple: int | None = None,
) -> CollatedConformerBatch:
    if not records:
        raise ValueError("at least one preprocessed conformer is required")
    multiple = records[0].preprocessing_config.get("pad_to_multiple", 8) if pad_to_multiple is None else pad_to_multiple
    max_length = max(record.sequence_length for record in records)
    padded_length = _pad_length(max_length, int(multiple))
    batch_size = len(records)

    src_tokens = np.full((batch_size, padded_length), dictionary.pad_id, dtype=np.int64)
    src_coord = np.zeros((batch_size, padded_length, 3), dtype=np.float32)
    src_distance = np.zeros((batch_size, padded_length, padded_length), dtype=np.float32)
    src_edge_type = np.zeros((batch_size, padded_length, padded_length), dtype=np.int64)
    sequence_lengths = np.asarray([record.sequence_length for record in records], dtype=np.int64)

    for index, record in enumerate(records):
        length = record.sequence_length
        src_tokens[index, :length] = record.src_tokens
        src_coord[index, :length, :] = record.src_coord
        src_distance[index, :length, :length] = record.src_distance
        src_edge_type[index, :length, :length] = record.src_edge_type

    return CollatedConformerBatch(
        src_tokens=src_tokens,
        src_coord=src_coord,
        src_distance=src_distance,
        src_edge_type=src_edge_type,
        sequence_lengths=sequence_lengths,
        conformer_ids=tuple(record.conformer_id for record in records),
        chromophore_ids=tuple(record.chromophore_id for record in records),
        padding_sources=dict(PADDING_RULE_SOURCES),
    )
