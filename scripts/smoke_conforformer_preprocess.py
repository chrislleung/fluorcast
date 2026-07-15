from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chemfluor.conforformer.cache import CacheError, load_conformer_cache_record
from chemfluor.conforformer.conformers import canonicalize_smiles
from chemfluor.conforformer.dictionary import load_conforformer_dictionary
from chemfluor.conforformer.preprocess import (
    ConforFormerPreprocessingConfig,
    PreprocessingError,
    collate_preprocessed_conformers,
    preprocess_successful_conformers,
)
from chemfluor.conforformer.schemas import MoleculeStatus


EXAMPLE_DICTIONARY = ROOT / "third_party" / "ConforFormer" / "unimol" / "example_data" / "molecule" / "dict.txt"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke test ConforFormer preprocessing without model loading.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--smiles", nargs="*", default=None)
    parser.add_argument("--max-conformers-per-molecule", type=int, default=None)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--unknown-atom-policy", choices=["fail", "use_unk"], default="fail")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-batch", action="store_true")
    return parser


def _requested_canonical_smiles(smiles_values: list[str] | None) -> set[str] | None:
    if not smiles_values:
        return None
    canonical: set[str] = set()
    for smiles in smiles_values:
        canonical_smiles, _ = canonicalize_smiles(smiles)
        if canonical_smiles is None:
            print(f"skipping invalid requested SMILES: {smiles}")
            continue
        canonical.add(canonical_smiles)
    return canonical


def _write_diagnostic(output_dir: Path, index: int, record) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:04d}_{record.conformer_id}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        src_tokens=record.src_tokens,
        src_coord=record.src_coord,
        src_distance=record.src_distance,
        src_edge_type=record.src_edge_type,
    )
    metadata = {
        "canonical_smiles": record.canonical_smiles,
        "chromophore_id": record.chromophore_id,
        "conformer_cache_key": record.conformer_cache_key,
        "conformer_id": record.conformer_id,
        "dictionary_sha256": record.dictionary_sha256,
        "diagnostic_artifact": True,
        "preprocessing_config": record.preprocessing_config,
        "sequence_length": record.sequence_length,
    }
    (output_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dictionary = load_conforformer_dictionary(args.dictionary)
    config = ConforFormerPreprocessingConfig(
        max_sequence_length=args.max_sequence_length,
        unknown_atom_policy=args.unknown_atom_policy,
    )
    if args.dictionary.resolve() == EXAMPLE_DICTIONARY.resolve():
        print(
            "Using the upstream example dictionary for preprocessing validation only; "
            "it is not proven compatible with a future checkpoint."
        )

    requested = _requested_canonical_smiles(args.smiles)
    preprocessed = []
    seen_cache_files = 0
    for path in sorted(args.cache_dir.glob("*.json")):
        try:
            cache_record = load_conformer_cache_record(path)
        except CacheError as exc:
            print(f"skipping invalid cache file {path}: {exc}")
            continue
        seen_cache_files += 1
        if requested is not None and cache_record.canonical_smiles not in requested:
            continue
        if cache_record.status != MoleculeStatus.OK:
            print(f"skipping failed molecule {cache_record.chromophore_id}: {cache_record.failure_reason}")
            continue
        try:
            molecule_records = preprocess_successful_conformers(cache_record, dictionary, config)
        except PreprocessingError as exc:
            print(f"preprocessing failed for {cache_record.chromophore_id}: {exc}")
            continue
        if args.max_conformers_per_molecule is not None:
            molecule_records = molecule_records[: args.max_conformers_per_molecule]
        for record in molecule_records:
            centered_ok = np.allclose(record.src_coord[1:-1].mean(axis=0), 0.0, atol=1e-6)
            symmetry_ok = np.allclose(record.src_distance, record.src_distance.T, atol=1e-6)
            finite_ok = np.isfinite(record.src_coord).all() and np.isfinite(record.src_distance).all()
            print(
                "\n".join(
                    [
                        f"input SMILES: {cache_record.input_smiles}",
                        f"canonical SMILES: {cache_record.canonical_smiles}",
                        f"conformer ID: {record.conformer_id}",
                        f"original atom count: {record.original_atom_count}",
                        f"heavy atom count: {record.heavy_atom_count}",
                        f"sequence length: {record.sequence_length}",
                        f"src_tokens: shape={record.src_tokens.shape} dtype={record.src_tokens.dtype}",
                        f"src_coord: shape={record.src_coord.shape} dtype={record.src_coord.dtype}",
                        f"src_distance: shape={record.src_distance.shape} dtype={record.src_distance.dtype}",
                        f"src_edge_type: shape={record.src_edge_type.shape} dtype={record.src_edge_type.dtype}",
                        f"dictionary sha256: {record.dictionary_sha256}",
                        f"centered-coordinate check: {centered_ok}",
                        f"distance symmetry check: {symmetry_ok}",
                        f"finite-value check: {finite_ok}",
                        "",
                    ]
                )
            )
            preprocessed.append(record)
            if args.output_dir is not None:
                _write_diagnostic(args.output_dir, len(preprocessed), record)

    if not preprocessed:
        print(f"no conformers preprocessed from {seen_cache_files} valid cache file(s)")
        return 1
    if not args.no_batch:
        batch = collate_preprocessed_conformers(preprocessed, dictionary)
        print(f"batch src_tokens: shape={batch.src_tokens.shape} dtype={batch.src_tokens.dtype}")
        print(f"batch src_coord: shape={batch.src_coord.shape} dtype={batch.src_coord.dtype}")
        print(f"batch src_distance: shape={batch.src_distance.shape} dtype={batch.src_distance.dtype}")
        print(f"batch src_edge_type: shape={batch.src_edge_type.shape} dtype={batch.src_edge_type.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
