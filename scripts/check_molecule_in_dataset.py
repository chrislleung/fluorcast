"""Check whether a molecule-solvent pair occurs in one or more CSV datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import pandas as pd
from rdkit import Chem, rdBase


SMILES_CANDIDATES = (
    "smiles",
    "chromophore",
    "molecule_smiles",
    "chromophore_smiles",
    "canonical_smiles",
    "canonical_chromophore_smiles",
)
SOLVENT_CANDIDATES = ("solvent", "solvent_name", "solvent_original")
CANONICAL_COLUMN = "_check_canonical_smiles"
NORMALIZED_SOLVENT_COLUMN = "_check_normalized_solvent"


class PreparedDataset(NamedTuple):
    """A loaded dataset plus the metadata needed to report matches."""

    frame: pd.DataFrame
    path: Path
    smiles_column: str
    solvent_column: str
    rows_checked: int
    invalid_smiles: int


def canonicalize_smiles(smiles: str) -> str | None:
    """Return an RDKit canonical SMILES, or None for missing/invalid input."""
    if pd.isna(smiles) or not str(smiles).strip():
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True)


def normalize_solvent(solvent: str) -> str:
    """Normalize case and runs of whitespace for solvent comparison."""
    if pd.isna(solvent):
        return ""
    return " ".join(str(solvent).split()).casefold()


def detect_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Find the first candidate column, matching names case-insensitively."""
    columns_by_name = {str(column).strip().casefold(): str(column) for column in df.columns}
    for candidate in candidates:
        match = columns_by_name.get(candidate.strip().casefold())
        if match is not None:
            return match
    return None


def _resolve_column(
    df: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    kind: str,
    path: Path,
) -> str:
    if explicit is not None:
        detected = detect_column(df, (explicit,))
        if detected is None:
            raise ValueError(
                f"{path}: requested {kind} column {explicit!r} was not found. "
                f"Available columns: {', '.join(map(str, df.columns))}"
            )
        return detected
    detected = detect_column(df, candidates)
    if detected is None:
        raise ValueError(
            f"{path}: no recognizable {kind} column was found. "
            f"Use --{kind}-column to select one. Available columns: "
            f"{', '.join(map(str, df.columns))}"
        )
    return detected


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        # The checked-in Deep4Chem source contains Latin-1 symbols.
        return pd.read_csv(path, encoding="latin-1")


def load_and_prepare_dataset(
    path: str | Path,
    smiles_column: str | None = None,
    solvent_column: str | None = None,
) -> PreparedDataset:
    """Load a CSV and add canonical molecule and normalized solvent columns."""
    dataset_path = Path(path)
    frame = _read_csv(dataset_path)
    detected_smiles = _resolve_column(
        frame, smiles_column, SMILES_CANDIDATES, "smiles", dataset_path
    )
    detected_solvent = _resolve_column(
        frame, solvent_column, SOLVENT_CANDIDATES, "solvent", dataset_path
    )

    prepared = frame.copy()
    prepared.insert(0, "dataset_row_index", frame.index)
    prepared.insert(0, "source_file", str(dataset_path))
    prepared[CANONICAL_COLUMN] = prepared[detected_smiles].map(canonicalize_smiles)
    prepared[NORMALIZED_SOLVENT_COLUMN] = prepared[detected_solvent].map(normalize_solvent)
    prepared["matched_raw_smiles"] = prepared[detected_smiles]
    prepared["matched_canonical_smiles"] = prepared[CANONICAL_COLUMN]
    prepared["matched_solvent"] = prepared[detected_solvent]
    invalid_count = int(prepared[CANONICAL_COLUMN].isna().sum())
    prepared = prepared.loc[prepared[CANONICAL_COLUMN].notna()].copy()

    return PreparedDataset(
        frame=prepared,
        path=dataset_path,
        smiles_column=detected_smiles,
        solvent_column=detected_solvent,
        rows_checked=len(frame),
        invalid_smiles=invalid_count,
    )


def find_matches(
    prepared_df: pd.DataFrame | PreparedDataset,
    query_canonical_smiles: str,
    query_solvent: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all molecule matches and the subset matching the solvent."""
    frame = prepared_df.frame if isinstance(prepared_df, PreparedDataset) else prepared_df
    molecule_matches = frame.loc[
        frame[CANONICAL_COLUMN] == query_canonical_smiles
    ].copy()
    normalized_query = normalize_solvent(query_solvent)
    pair_matches = molecule_matches.loc[
        molecule_matches[NORMALIZED_SOLVENT_COLUMN] == normalized_query
    ].copy()
    return molecule_matches, pair_matches


def _display_columns(matches: pd.DataFrame, datasets: Sequence[PreparedDataset]) -> list[str]:
    columns = [
        "source_file",
        "dataset_row_index",
        "matched_raw_smiles",
        "matched_canonical_smiles",
        "matched_solvent",
    ]
    property_terms = (
        "absorption",
        "emission",
        "quantum",
        "yield",
        "plqy",
        "lifetime",
        "extinction",
        "reference",
        "source",
    )
    columns.extend(
        column
        for column in matches.columns
        if any(term in str(column).casefold() for term in property_terms)
    )
    return list(dict.fromkeys(column for column in columns if column in matches.columns))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check for a canonical molecule-solvent pair in CSV datasets."
    )
    parser.add_argument("--smiles", required=True, help="Query molecule SMILES.")
    parser.add_argument("--solvent", required=True, help="Query solvent name.")
    parser.add_argument(
        "--dataset", required=True, action="append", type=Path,
        help="Dataset CSV path; repeat this option to check multiple files.",
    )
    parser.add_argument("--smiles-column", help="Explicit SMILES column for all datasets.")
    parser.add_argument("--solvent-column", help="Explicit solvent column for all datasets.")
    parser.add_argument("--out", type=Path, help="Save all molecule matches to this CSV.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    query_canonical = canonicalize_smiles(args.smiles)
    if query_canonical is None:
        raise SystemExit(
            f"Error: RDKit could not parse the input SMILES {args.smiles!r}. "
            "Check the syntax and try again."
        )

    try:
        datasets = [
            load_and_prepare_dataset(path, args.smiles_column, args.solvent_column)
            for path in args.dataset
        ]
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    molecule_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    for dataset in datasets:
        molecule, pair = find_matches(dataset, query_canonical, args.solvent)
        molecule_parts.append(molecule)
        pair_parts.append(pair)
    molecule_matches = pd.concat(molecule_parts, ignore_index=True, sort=False)
    pair_matches = pd.concat(pair_parts, ignore_index=True, sort=False)

    print("Input molecule:")
    print(f"Raw SMILES: {args.smiles}")
    print(f"Canonical SMILES: {query_canonical}")
    print(f"Input solvent: {args.solvent}")
    print("\nDataset summary:")
    print(f"Files checked: {len(datasets)}")
    print(f"Rows checked: {sum(dataset.rows_checked for dataset in datasets)}")
    print(f"Invalid SMILES skipped: {sum(dataset.invalid_smiles for dataset in datasets)}")
    print("\nMolecule match:")
    print(f"Found molecule in dataset: {'YES' if len(molecule_matches) else 'NO'}")
    print(f"Number of molecule-only matches: {len(molecule_matches)}")
    print("\nMolecule-solvent match:")
    print(f"Found exact molecule-solvent pair: {'YES' if len(pair_matches) else 'NO'}")
    print(f"Number of exact pair matches: {len(pair_matches)}")

    shown_matches = pair_matches if len(pair_matches) else molecule_matches
    if len(shown_matches):
        heading = "Matching rows:" if len(pair_matches) else "Molecule-only matching rows:"
        print(f"\n{heading}")
        print(shown_matches[_display_columns(shown_matches, datasets)].to_string(index=False))

    if args.out is not None:
        exported = molecule_matches.copy()
        exported["query_solvent_match"] = (
            exported[NORMALIZED_SOLVENT_COLUMN] == normalize_solvent(args.solvent)
        )
        exported = exported.drop(
            columns=[CANONICAL_COLUMN, NORMALIZED_SOLVENT_COLUMN], errors="ignore"
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        exported.to_csv(args.out, index=False)
        print(f"\nSaved {len(exported)} molecule match(es) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
