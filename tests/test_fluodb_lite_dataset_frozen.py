from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


DATASET = Path("data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv")
EXPECTED_SHA256 = "7de6a3ec74e72985573a098235a82484badff2e8a09678c668a5eb73bbdbfdf7"


def test_original_fluodb_lite_stokes_dataset_is_frozen() -> None:
    rows = pd.read_csv(DATASET, low_memory=False)

    assert hashlib.sha256(DATASET.read_bytes()).hexdigest() == EXPECTED_SHA256
    assert len(rows) == 66820
    assert rows["canonical_chromophore_smiles"].nunique() == 33965
    assert pd.to_numeric(rows["stokes_shift_nm"], errors="coerce").notna().sum() == 37675
