"""Prompt construction. Prompts deliberately contain no measured target values."""

from __future__ import annotations

import json

from .schema import DESCRIPTOR_NAMES


def prediction_prompt(target: str, chromophore_smiles: str, solvent_smiles: str | None) -> str:
    schema = {"target": target, "llm_numeric_prediction": "float or null",
              "llm_confidence": "0..1 or null",
              "descriptors": {name: "low|medium|high|unknown" for name in DESCRIPTOR_NAMES},
              "warnings": []}
    return ("Estimate the photophysical target from structure alone. Return JSON only. "
            "Do not assume access to experimental labels.\n"
            f"Target: {target}\nChromophore SMILES: {chromophore_smiles}\n"
            f"Solvent SMILES: {solvent_smiles or 'unknown'}\nSchema: {json.dumps(schema)}")
