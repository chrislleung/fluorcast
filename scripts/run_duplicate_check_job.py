"""Run one FluorCast dataset duplicate-check job described by JSON."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback as traceback_module
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rdkit import DataStructs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for import_path in (SRC_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import check_molecule_in_dataset as dataset_checker  # noqa: E402
from chemfluor.combined_prediction import (  # noqa: E402
    DEFAULT_N_BITS,
    DEFAULT_RADIUS,
    morgan_bitvect,
)

DEFAULT_DATASET = (
    PROJECT_ROOT / "data" / "processed" / "fluodb_lite" / "combined_deduplicated.csv"
)
REQUIRED_FIELDS = (
    "submission_id",
    "user_id",
    "molecule_smiles",
    "solvent_smiles",
    "submitted_at",
)
SOLVENT_SMILES_CANDIDATES = ("canonical_solvent_smiles", "solvent_smiles")


class JobError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _value(row: pd.Series, candidates: Sequence[str]) -> Any:
    column = dataset_checker.detect_column(row.to_frame().T, candidates)
    if column is None or pd.isna(row[column]):
        return None
    return row[column]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _nullable_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _record(row: pd.Series, similarity: float) -> dict[str, Any]:
    record_id = _value(row, ("record_id", "id", "dataset_row_index"))
    return {
        "record_id": str(record_id),
        "molecule_smiles": str(_value(row, dataset_checker.SMILES_CANDIDATES)),
        "solvent_smiles": _nullable_string(_value(row, SOLVENT_SMILES_CANDIDATES)),
        "similarity": float(similarity),
        "emission_nm": _number(_value(row, ("emission_nm", "emission"))),
        "quantum_yield": _number(_value(row, ("quantum_yield", "plqy"))),
        "source_doi": _nullable_string(
            _value(row, ("source_doi", "reference_doi", "doi"))
        ),
    }


def check_duplicates(
    payload: dict[str, Any], dataset_path: Path, max_matches: int
) -> tuple[bool, str | None, str, str, list[dict[str, Any]], list[str]]:
    canonical_molecule = dataset_checker.canonicalize_smiles(payload["molecule_smiles"])
    canonical_solvent = dataset_checker.canonicalize_smiles(payload["solvent_smiles"])
    if canonical_molecule is None:
        raise JobError("INVALID_SMILES", "Invalid molecule_smiles.")
    if canonical_solvent is None:
        raise JobError("INVALID_SMILES", "Invalid solvent_smiles.")
    if not dataset_path.exists():
        raise JobError("DATASET_NOT_CONFIGURED", f"Dataset not found: {dataset_path}")

    frame = dataset_checker._read_csv(dataset_path)
    molecule_column = dataset_checker.detect_column(frame, dataset_checker.SMILES_CANDIDATES)
    solvent_column = dataset_checker.detect_column(frame, SOLVENT_SMILES_CANDIDATES)
    if molecule_column is None or solvent_column is None:
        raise JobError(
            "INVALID_DATASET",
            "Dataset must contain molecule SMILES and canonical_solvent_smiles/solvent_smiles columns.",
        )
    working = frame.copy()
    working.insert(0, "dataset_row_index", frame.index)
    working["_canonical_molecule"] = working[molecule_column].map(
        dataset_checker.canonicalize_smiles
    )
    working["_canonical_solvent"] = working[solvent_column].map(
        dataset_checker.canonicalize_smiles
    )
    valid = working.loc[working["_canonical_molecule"].notna()].copy()
    exact = valid.loc[
        (valid["_canonical_molecule"] == canonical_molecule)
        & (valid["_canonical_solvent"] == canonical_solvent)
    ]

    reference_smiles = sorted(valid["_canonical_molecule"].drop_duplicates().tolist())
    reference_pairs = [
        (
            smiles,
            morgan_bitvect(smiles, radius=DEFAULT_RADIUS, n_bits=DEFAULT_N_BITS),
        )
        for smiles in reference_smiles
    ]
    reference_pairs = [(smiles, fp) for smiles, fp in reference_pairs if fp is not None]
    query_fp = morgan_bitvect(
        canonical_molecule, radius=DEFAULT_RADIUS, n_bits=DEFAULT_N_BITS
    )
    if query_fp is None:  # Canonicalization above should make this unreachable.
        raise JobError("INVALID_SMILES", "Could not fingerprint molecule_smiles.")
    similarities = DataStructs.BulkTanimotoSimilarity(
        query_fp, [fp for _, fp in reference_pairs]
    )
    similarity_by_smiles = {
        smiles: float(similarity)
        for (smiles, _), similarity in zip(reference_pairs, similarities)
    }
    valid["_similarity"] = valid["_canonical_molecule"].map(similarity_by_smiles)
    nearest_rows = valid.sort_values("_similarity", ascending=False).head(max_matches)
    nearest = [_record(row, row["_similarity"]) for _, row in nearest_rows.iterrows()]
    warnings = []
    invalid_count = int(working["_canonical_molecule"].isna().sum())
    if invalid_count:
        warnings.append(
            f"Skipped {invalid_count} dataset row(s) with invalid molecule SMILES."
        )
    exact_record_id = None if exact.empty else str(exact.iloc[0]["dataset_row_index"])
    if not exact.empty:
        explicit_id = _value(exact.iloc[0], ("record_id", "id"))
        if explicit_id is not None:
            exact_record_id = str(explicit_id)
    return bool(not exact.empty), exact_record_id, canonical_molecule, canonical_solvent, nearest, warnings


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def run_job(
    input_path: Path,
    output_path: Path,
    dataset_path: Path | None,
    max_matches: int = 5,
) -> int:
    payload: dict[str, Any] = {}
    try:
        loaded = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise JobError("INVALID_INPUT", "Input JSON must be an object.")
        payload = loaded
        missing = [field for field in REQUIRED_FIELDS if not payload.get(field)]
        if missing:
            raise JobError("INVALID_INPUT", f"Missing required field(s): {', '.join(missing)}")
        non_strings = [
            field for field in REQUIRED_FIELDS if not isinstance(payload[field], str)
        ]
        if non_strings:
            raise JobError(
                "INVALID_INPUT", f"Field(s) must be strings: {', '.join(non_strings)}"
            )
        if max_matches < 0:
            raise JobError("INVALID_INPUT", "max_matches must be non-negative.")
        if dataset_path is None:
            raise JobError("DATASET_NOT_CONFIGURED", "No dataset path was configured.")
        exact, record_id, molecule, solvent, nearest, warnings = check_duplicates(
            payload, dataset_path, max_matches
        )
        result = {
            "status": "success",
            "submission_id": payload["submission_id"],
            "exact_duplicate_found": exact,
            "exact_duplicate_record_id": record_id,
            "canonical_molecule_smiles": molecule,
            "canonical_solvent_smiles": solvent,
            "nearest_matches": nearest,
            "warnings": warnings,
        }
        exit_code = 0
    except Exception as exc:
        result = {
            "status": "failed",
            "submission_id": payload.get("submission_id"),
            "error_code": getattr(exc, "code", "DUPLICATE_CHECK_FAILED"),
            "error_message": str(exc),
            "traceback": traceback_module.format_exc(),
            "warnings": [],
        }
        exit_code = 1
    write_output(output_path, result)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET if DEFAULT_DATASET.exists() else None,
    )
    parser.add_argument("--max-matches", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_job(args.input, args.output, args.dataset, args.max_matches)


if __name__ == "__main__":
    raise SystemExit(main())
