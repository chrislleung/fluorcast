"""Build local RDKit conformer cache records for ConforFormer Stage 2."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.cache import (  # noqa: E402
    CacheError,
    build_conformer_cache_key,
    conformer_cache_path,
    load_conformer_cache_record,
    save_conformer_cache_record,
)
from chemfluor.conforformer.config import ConformerGenerationConfig  # noqa: E402
from chemfluor.conforformer.conformers import canonicalize_smiles, generate_conformer_cache_record  # noqa: E402
from chemfluor.conforformer.schemas import MoleculeStatus  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--smiles", nargs="+", help="One or more chromophore SMILES.")
    source.add_argument("--input-csv", type=Path, help="CSV containing chromophore SMILES.")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--num-conformers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prune-rms-threshold", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _read_inputs(args: argparse.Namespace) -> list[tuple[str, str]]:
    if args.smiles is not None:
        return [(smiles, smiles) for smiles in args.smiles]
    if args.max_molecules is None:
        raise ValueError("--max-molecules is required for CSV smoke runs")
    rows: list[tuple[str, str]] = []
    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if args.smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"missing SMILES column: {args.smiles_column}")
        if args.id_column is not None and args.id_column not in (reader.fieldnames or []):
            raise ValueError(f"missing ID column: {args.id_column}")
        for index, row in enumerate(reader):
            smiles = row[args.smiles_column]
            chromophore_id = row[args.id_column] if args.id_column else f"row_{index}"
            rows.append((smiles, chromophore_id))
            if args.max_molecules is not None and len(rows) >= args.max_molecules:
                break
    return rows


def _dedupe_rows(rows: list[tuple[str, str]]) -> tuple[list[tuple[str, str, str | None, str | None]], int]:
    deduped: dict[str, tuple[str, str, str | None, str | None]] = {}
    invalid = 0
    for smiles, chromophore_id in rows:
        canonical, isomeric = canonicalize_smiles(smiles)
        if isomeric is None:
            invalid += 1
            key = f"invalid::{chromophore_id}::{smiles}"
            deduped[key] = (smiles, chromophore_id, None, None)
            continue
        if isomeric not in deduped:
            deduped[isomeric] = (smiles, chromophore_id, canonical, isomeric)
    return list(deduped.values()), invalid


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.max_molecules is not None and args.max_molecules <= 0:
            raise ValueError("--max-molecules must be positive")
        rows = _read_inputs(args)
        deduped_rows, invalid_smiles = _dedupe_rows(rows)
        config = ConformerGenerationConfig.from_overrides(
            num_conformers=args.num_conformers,
            random_seed=args.seed,
            prune_rms_threshold=args.prune_rms_threshold,
        )
    except Exception as exc:
        print(f"Command failed: {exc}", file=sys.stderr)
        return 2

    summary = {
        "requested molecules": len(rows),
        "unique canonical molecules": len(deduped_rows),
        "successes": 0,
        "failures": 0,
        "cache hits": 0,
        "cache writes": 0,
        "invalid SMILES": invalid_smiles,
        "optimizer fallback count": 0,
    }

    for smiles, chromophore_id, canonical_smiles, isomeric_smiles in deduped_rows:
        if args.dry_run:
            cache_key = build_conformer_cache_key(
                canonical_smiles=canonical_smiles,
                isomeric_canonical_smiles=isomeric_smiles,
                config=config,
            )
            path = conformer_cache_path(args.output_dir, cache_key)
            print(f"{smiles}\t{cache_key}\t{path}")
            if isomeric_smiles is not None:
                summary["successes"] += 1
            else:
                summary["failures"] += 1
            continue
        record = generate_conformer_cache_record(smiles, chromophore_id=chromophore_id, config=config)
        path = conformer_cache_path(args.output_dir, record.conformer_cache_key)
        print(f"{smiles}\t{record.conformer_cache_key}\t{path}")
        if path.exists() and not args.overwrite:
            try:
                load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key)
                summary["cache hits"] += 1
                continue
            except CacheError as exc:
                print(f"Existing cache record is invalid and will not be overwritten: {path}: {exc}", file=sys.stderr)
                summary["failures"] += 1
                continue
        try:
            save_conformer_cache_record(record, args.output_dir, overwrite=args.overwrite)
            summary["cache writes"] += 1
        except Exception as exc:
            print(f"Failed to write cache for {smiles}: {exc}", file=sys.stderr)
            summary["failures"] += 1
            continue
        if record.status == MoleculeStatus.OK:
            summary["successes"] += 1
        else:
            summary["failures"] += 1
        summary["optimizer fallback count"] += int(record.metadata.get("force_field_fallback_count", 0))

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
