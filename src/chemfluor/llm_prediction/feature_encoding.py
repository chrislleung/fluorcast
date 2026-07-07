"""Convert validated LLM records into numeric meta-features."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import DESCRIPTOR_NAMES, LLMOutput

CATEGORY_VALUE = {"unknown": np.nan, "low": 0.0, "medium": 0.5, "high": 1.0}


def encode_llm_features(outputs: Iterable[LLMOutput | dict[str, Any]], target: str | None = None) -> pd.DataFrame:
    records = []
    for item in outputs:
        parsed = item if isinstance(item, LLMOutput) else LLMOutput.from_dict(item, target or "unknown")
        row = {"llm_numeric_prediction": parsed.llm_numeric_prediction,
               "llm_confidence": parsed.llm_confidence}
        row.update({f"llm_descriptor_{name}": CATEGORY_VALUE.get(parsed.descriptors.get(name, "unknown"), np.nan)
                    for name in DESCRIPTOR_NAMES})
        records.append(row)
    return pd.DataFrame(records, columns=["llm_numeric_prediction", "llm_confidence",
        *(f"llm_descriptor_{name}" for name in DESCRIPTOR_NAMES)]).apply(pd.to_numeric, errors="coerce")
