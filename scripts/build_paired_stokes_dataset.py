"""Build a row-aligned absorption/emission dataset for Stokes-shift evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_combined_predictors as base  # noqa: E402

DEFAULT_STANDARDIZED = ROOT / "data" / "processed" / "fluodb_lite" / "combined_deduplicated.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-combined", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def load_source(path: Path | None) -> tuple[pd.DataFrame, str]:
    selected = path or (DEFAULT_STANDARDIZED if DEFAULT_STANDARDIZED.exists() else None)
    if selected is not None:
        if not selected.exists():
            raise FileNotFoundError(f"Standardized combined CSV not found: {selected}")
        return pd.read_csv(selected, low_memory=False), str(selected)
    return base.load_combined_rows(base.DEFAULT_DEEP4CHEM, base.DEFAULT_CHEMFLUOR), "combined defaults"


def _stats(values: pd.Series) -> dict[str, float | None]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {key: (float(value) if len(finite) else None) for key, value in {
        "min": finite.min(), "median": finite.median(), "max": finite.max()
    }.items()}


def build_paired_dataset(rows: pd.DataFrame, max_rows: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    """Filter paired wavelengths, retain identifiers, and return audit statistics."""
    if "absorption_nm" not in rows or "emission_nm" not in rows:
        raise ValueError("Input dataset must contain absorption_nm and emission_nm")
    total = len(rows)
    absorption = pd.to_numeric(rows["absorption_nm"], errors="coerce")
    emission = pd.to_numeric(rows["emission_nm"], errors="coerce")
    missing_abs = ~np.isfinite(absorption)
    missing_em = ~np.isfinite(emission)
    finite_pair = ~missing_abs & ~missing_em
    nonpositive = finite_pair & ((absorption <= 0) | (emission <= 0))
    keep = finite_pair & ~nonpositive

    identifier_columns = [
        "row_id", "canonical_molecule_smiles", "canonical_chromophore_smiles",
        "molecule_smiles", "chromophore_smiles", "canonical_solvent_smiles",
        "solvent_smiles", "solvent", "solvent_name", "solvent_original",
        "quantum_yield", "source_dataset",
    ]
    retained = [column for column in identifier_columns if column in rows]
    paired = rows.loc[keep, retained].copy()
    if "row_id" not in paired:
        paired.insert(0, "row_id", rows.index[keep].to_numpy())
    paired["absorption_nm"] = absorption.loc[keep].to_numpy(dtype=float)
    paired["emission_nm"] = emission.loc[keep].to_numpy(dtype=float)
    paired["stokes_shift_nm"] = paired["emission_nm"] - paired["absorption_nm"]
    paired["stokes_shift_cm^-1"] = 1e7 / paired["absorption_nm"] - 1e7 / paired["emission_nm"]
    paired["physically_valid_stokes"] = paired["emission_nm"] > paired["absorption_nm"]
    pre_sample_count = len(paired)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be positive")
        if len(paired) > max_rows:
            paired = paired.sample(max_rows, random_state=seed).reset_index(drop=True)
    invalid_stokes = int((paired["stokes_shift_nm"] <= 0).sum())
    absorption_stats = _stats(paired["absorption_nm"])
    emission_stats = _stats(paired["emission_nm"])
    stokes_nm_stats = _stats(paired["stokes_shift_nm"])
    stokes_cm_stats = _stats(paired["stokes_shift_cm^-1"])
    summary = {
        "total_rows_loaded": int(total),
        "paired_rows_retained": int(len(paired)),
        "paired_rows_before_sampling": int(pre_sample_count),
        "rows_dropped_for_missing_absorption": int(missing_abs.sum()),
        "rows_dropped_for_missing_emission": int(missing_em.sum()),
        "rows_dropped_for_nonpositive_wavelengths": int(nonpositive.sum()),
        "negative_or_zero_stokes_shift_count": invalid_stokes,
        "negative_or_zero_stokes_shift_fraction": float(invalid_stokes / len(paired)) if len(paired) else 0.0,
        "absorption_nm": absorption_stats,
        "emission_nm": emission_stats,
        "stokes_shift_nm": stokes_nm_stats,
        "stokes_shift_cm^-1": stokes_cm_stats,
        **{f"{stat}_absorption_nm": value for stat, value in absorption_stats.items()},
        **{f"{stat}_emission_nm": value for stat, value in emission_stats.items()},
        **{f"{stat}_stokes_shift_nm": value for stat, value in stokes_nm_stats.items()},
        **{f"{stat}_stokes_shift_cm^-1": value for stat, value in stokes_cm_stats.items()},
    }
    return paired.reset_index(drop=True), summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, source = load_source(args.standardized_combined)
    paired, summary = build_paired_dataset(rows, args.max_rows, args.seed)
    summary["source_dataset"] = source
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.out_csv, index=False)
    args.summary_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
