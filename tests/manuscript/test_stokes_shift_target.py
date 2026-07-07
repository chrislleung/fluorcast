"""Tests for the derived manuscript Stokes-shift target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.manuscript.run_paper_comparison_experiments import (
    add_stokes_shift_target,
)


def test_add_stokes_shift_target_filters_missing_and_negative_rows() -> None:
    rows = pd.DataFrame(
        {
            "absorption_nm": [400.0, np.nan, 500.0, 450.0],
            "emission_nm": [500.0, 550.0, 450.0, np.nan],
            "quantum_yield": [0.1, 0.2, 0.3, 0.4],
        }
    )

    derived = add_stokes_shift_target(rows)

    assert derived["stokes_shift_nm"].tolist()[0] == 100.0
    assert derived["stokes_shift_nm"].isna().tolist() == [False, True, True, True]
    pd.testing.assert_series_equal(derived["quantum_yield"], rows["quantum_yield"])


def test_add_stokes_shift_target_handles_absent_input_column() -> None:
    derived = add_stokes_shift_target(pd.DataFrame({"absorption_nm": [400.0]}))

    assert derived["stokes_shift_nm"].isna().all()
