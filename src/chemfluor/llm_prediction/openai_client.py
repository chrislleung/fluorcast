"""Minimal OpenAI Responses-compatible JSON client."""

from __future__ import annotations

import json
import os
import urllib.request

from .prompts import prediction_prompt
from .schema import LLMOutput


def predict(target: str, chromophore_smiles: str, solvent_smiles: str | None = None,
            model: str = "gpt-4.1-mini", api_key: str | None = None) -> LLMOutput:
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for --llm-mode openai")
    body = json.dumps({"model": model, "input": prediction_prompt(target, chromophore_smiles, solvent_smiles),
                       "text": {"format": {"type": "json_object"}}}).encode()
    request = urllib.request.Request("https://api.openai.com/v1/responses", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    text = payload.get("output_text")
    if not text:
        text = payload["output"][0]["content"][0]["text"]
    return LLMOutput.from_dict(json.loads(text), target)
