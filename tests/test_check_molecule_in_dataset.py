from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_molecule_in_dataset.py"
spec = importlib.util.spec_from_file_location("check_molecule_in_dataset", SCRIPT_PATH)
checker = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(checker)


def test_equivalent_smiles_and_case_insensitive_solvent_match(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"SMILES": ["Cc1ccc(N)cc1"], "Solvent": ["  Ethanol  "]}
    ).to_csv(path, index=False)

    prepared = checker.load_and_prepare_dataset(path)
    query = checker.canonicalize_smiles("CC1=CC=C(C=C1)N")
    molecule, pair = checker.find_matches(prepared, query, "ethanol")

    assert query == checker.canonicalize_smiles("Cc1ccc(N)cc1")
    assert len(molecule) == 1
    assert len(pair) == 1


def test_molecule_only_and_pair_matches_are_distinguished(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "CCO", "CC"], "solvent": ["water", "ethanol", "water"]}
    ).to_csv(path, index=False)

    prepared = checker.load_and_prepare_dataset(path)
    molecule, pair = checker.find_matches(prepared, "CCO", " WATER ")

    assert len(molecule) == 2
    assert list(pair["dataset_row_index"]) == [0]


def test_invalid_dataset_smiles_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "not a smiles", None], "solvent": ["water"] * 3}
    ).to_csv(path, index=False)

    prepared = checker.load_and_prepare_dataset(path)

    assert prepared.rows_checked == 3
    assert prepared.invalid_smiles == 2
    assert len(prepared.frame) == 1


def test_explicit_column_overrides(tmp_path: Path) -> None:
    path = tmp_path / "custom.csv"
    pd.DataFrame({"structure": ["O"], "medium": [" Heavy   Water "]}).to_csv(
        path, index=False
    )

    with pytest.raises(ValueError, match="no recognizable smiles column"):
        checker.load_and_prepare_dataset(path)

    prepared = checker.load_and_prepare_dataset(path, "structure", "medium")
    molecule, pair = checker.find_matches(prepared, "O", "heavy water")
    assert len(molecule) == len(pair) == 1


def test_invalid_query_smiles_returns_none() -> None:
    assert checker.canonicalize_smiles("definitely not smiles") is None


def test_cli_writes_molecule_matches_with_pair_flag(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    output = tmp_path / "results" / "matches.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "CCO"], "solvent": ["water", "ethanol"]}
    ).to_csv(dataset, index=False)

    result = checker.main(
        [
            "--smiles", "OCC", "--solvent", "ETHANOL",
            "--dataset", str(dataset), "--out", str(output),
        ]
    )

    written = pd.read_csv(output)
    assert result == 0
    assert len(written) == 2
    assert list(written["query_solvent_match"]) == [False, True]
    assert set(written["matched_canonical_smiles"]) == {"CCO"}
