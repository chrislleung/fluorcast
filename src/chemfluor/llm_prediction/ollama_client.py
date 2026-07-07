"""Minimal Ollama JSON client (no optional Python dependency)."""

from __future__ import annotations

import json
import urllib.request

from .prompts import prediction_prompt
from .schema import LLMOutput


def predict(target: str, chromophore_smiles: str, solvent_smiles: str | None = None,
            model: str = "llama3.1", base_url: str = "http://localhost:11434") -> LLMOutput:
    body = json.dumps({"model": model, "prompt": prediction_prompt(target, chromophore_smiles, solvent_smiles),
                       "stream": False, "format": "json"}).encode()
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return LLMOutput.from_dict(json.loads(payload.get("response", "{}")), target)
