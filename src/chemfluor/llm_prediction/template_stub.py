"""Deterministic offline stand-in for an LLM, based only on input structure."""

from __future__ import annotations

from .schema import DESCRIPTOR_NAMES, LLMOutput


def predict(target: str, chromophore_smiles: str, solvent_smiles: str | None = None) -> LLMOutput:
    text = str(chromophore_smiles or "")
    aromatic = sum(text.count(char) for char in "cnosp")
    rings = sum(char.isdigit() for char in text)
    double = text.count("=") + aromatic // 2
    hetero = sum(text.count(char) for char in ("N", "O", "S", "n", "o", "s"))
    def level(score: int, low: int, high: int) -> str:
        return "low" if score <= low else "high" if score >= high else "medium"
    descriptors = {
        "conjugation": level(aromatic + double, 1, 5),
        "rigidity": level(rings, 0, 4),
        "donor_acceptor_character": level(hetero, 0, 3),
        "charge_transfer_likelihood": level(hetero + aromatic, 1, 6),
        "solvent_effect_likelihood": "medium" if solvent_smiles else "unknown",
    }
    # A reproducible, intentionally modest structural estimate; never label-derived.
    if target == "quantum_yield":
        numeric = min(.95, max(.02, .18 + .035 * rings + .015 * aromatic - .02 * hetero))
    elif target == "absorption_nm":
        numeric = 280.0 + 13.0 * aromatic + 9.0 * double + 4.0 * hetero
    else:
        numeric = 320.0 + 15.0 * aromatic + 10.0 * double + 5.0 * hetero
    return LLMOutput(target, float(numeric), .35, descriptors, ["deterministic template estimate"])
