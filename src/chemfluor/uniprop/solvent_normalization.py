"""UniProp-only solvent normalization overlay.

The shared FluorCast processed CSVs are treated as frozen source artifacts.
This module repairs known blank solvent aliases only while building UniProp
manifests, without rewriting the authoritative dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class SolventAlias:
    normalized_name: str
    smiles: str


@dataclass(frozen=True)
class UniPropSolventResolution:
    source_canonical_smiles: str | None
    canonical_smiles: str | None
    normalized_name: str | None
    environment_type: str
    mapping_status: str
    mapping_rule: str


SOLVENT_ALIAS_MAP: dict[str, SolventAlias] = {
    "h2o": SolventAlias("water", "O"),
    "water": SolventAlias("water", "O"),
    "ch2cl2": SolventAlias("dichloromethane", "ClCCl"),
    "dichloromethane": SolventAlias("dichloromethane", "ClCCl"),
    "dmso": SolventAlias("dimethyl sulfoxide", "CS(C)=O"),
    "dimethyl sulfoxide": SolventAlias("dimethyl sulfoxide", "CS(C)=O"),
    "meoh": SolventAlias("methanol", "CO"),
    "methanol": SolventAlias("methanol", "CO"),
    "mecn": SolventAlias("acetonitrile", "CC#N"),
    "acetonitrile": SolventAlias("acetonitrile", "CC#N"),
    "thf": SolventAlias("tetrahydrofuran", "C1CCOC1"),
    "tetrahydrofuran": SolventAlias("tetrahydrofuran", "C1CCOC1"),
    "chcl3": SolventAlias("chloroform", "ClC(Cl)Cl"),
    "chloroform": SolventAlias("chloroform", "ClC(Cl)Cl"),
    "etoh": SolventAlias("ethanol", "CCO"),
    "ethanol": SolventAlias("ethanol", "CCO"),
    "toluene": SolventAlias("toluene", "Cc1ccccc1"),
    "toluene/toluene": SolventAlias("toluene", "Cc1ccccc1"),
    "ethyl acetate": SolventAlias("ethyl acetate", "CCOC(C)=O"),
    "hexane": SolventAlias("n-hexane", "CCCCCC"),
    "n-hexane": SolventAlias("n-hexane", "CCCCCC"),
}

GAS_PHASE_ALIASES = {"gas"}


def solvent_alias_key(value: object) -> str:
    return " ".join(str(value).strip().lower().split())


def canonicalize_smiles(smiles: object) -> str | None:
    if pd.isna(smiles):
        return None
    text = str(smiles).strip()
    if not text:
        return None
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _looks_like_mixture(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in [":", " mixture", "mixture ", "mixed"])


def resolve_uniprop_solvent(source_canonical: object, solvent_original: object) -> UniPropSolventResolution:
    """Resolve one source solvent row for UniProp-only use."""
    source = canonicalize_smiles(source_canonical)
    if source is not None:
        return UniPropSolventResolution(
            source_canonical_smiles=source,
            canonical_smiles=source,
            normalized_name=None if pd.isna(solvent_original) else str(solvent_original).strip(),
            environment_type="molecular_solvent",
            mapping_status="source_canonical",
            mapping_rule="source_canonical_solvent_smiles",
        )

    if pd.isna(solvent_original) or not str(solvent_original).strip():
        return UniPropSolventResolution(
            source_canonical_smiles=None,
            canonical_smiles=None,
            normalized_name=None,
            environment_type="missing_solvent",
            mapping_status="missing",
            mapping_rule="missing_source_solvent",
        )

    original = str(solvent_original).strip()
    key = solvent_alias_key(original)
    if key in GAS_PHASE_ALIASES:
        return UniPropSolventResolution(
            source_canonical_smiles=None,
            canonical_smiles=None,
            normalized_name="gas phase",
            environment_type="gas_phase",
            mapping_status="gas_phase",
            mapping_rule="environment_alias:gas_phase",
        )

    alias = SOLVENT_ALIAS_MAP.get(key)
    if alias is not None:
        canonical = canonicalize_smiles(alias.smiles)
        if canonical is None:
            raise ValueError(f"Invalid UniProp solvent alias mapping: {key} -> {alias.smiles}")
        return UniPropSolventResolution(
            source_canonical_smiles=None,
            canonical_smiles=canonical,
            normalized_name=alias.normalized_name,
            environment_type="molecular_solvent",
            mapping_status="resolved_alias",
            mapping_rule=f"uniprop_alias:{key}->{alias.normalized_name}",
        )

    if _looks_like_mixture(original):
        status = "unresolved_mixture"
        environment = "mixed_solvent"
        rule = "unresolved_mixture"
    else:
        status = "unresolved"
        environment = "unknown_solvent"
        rule = "unresolved_label"
    return UniPropSolventResolution(
        source_canonical_smiles=None,
        canonical_smiles=None,
        normalized_name=key,
        environment_type=environment,
        mapping_status=status,
        mapping_rule=rule,
    )


def apply_uniprop_solvent_overlay(rows: pd.DataFrame) -> pd.DataFrame:
    """Add UniProp solvent columns while preserving source solvent columns."""
    enriched = rows.copy()
    if "solvent_original" not in enriched.columns:
        enriched["solvent_original"] = pd.NA
    resolutions = [
        resolve_uniprop_solvent(row.get("canonical_solvent_smiles"), row.get("solvent_original"))
        for _, row in enriched.iterrows()
    ]
    enriched["source_canonical_solvent_smiles"] = [
        resolution.source_canonical_smiles if resolution.source_canonical_smiles is not None else pd.NA
        for resolution in resolutions
    ]
    enriched["uniprop_canonical_solvent_smiles"] = [
        resolution.canonical_smiles if resolution.canonical_smiles is not None else pd.NA
        for resolution in resolutions
    ]
    enriched["uniprop_solvent_normalized_name"] = [
        resolution.normalized_name if resolution.normalized_name is not None else pd.NA
        for resolution in resolutions
    ]
    enriched["environment_type"] = [resolution.environment_type for resolution in resolutions]
    enriched["uniprop_solvent_mapping_status"] = [
        resolution.mapping_status for resolution in resolutions
    ]
    enriched["uniprop_solvent_mapping_rule"] = [resolution.mapping_rule for resolution in resolutions]
    return enriched


def uniprop_solvent_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Summarize source-vs-UniProp solvent resolution for reports."""
    source = rows["source_canonical_solvent_smiles"]
    resolved = rows["uniprop_canonical_solvent_smiles"]
    environment = rows["environment_type"].astype("string")
    status = rows["uniprop_solvent_mapping_status"].astype("string")
    alias_repaired = source.isna() & resolved.notna() & status.eq("resolved_alias")
    unresolved = resolved.isna() & ~environment.eq("gas_phase")
    return {
        "source_canonical_solvent_rows": int(source.notna().sum()),
        "uniprop_canonical_solvent_rows": int(resolved.notna().sum()),
        "uniprop_alias_repaired_rows": int(alias_repaired.sum()),
        "gas_phase_rows": int(environment.eq("gas_phase").sum()),
        "unresolved_solvent_rows": int(unresolved.sum()),
        "environment_type_counts": environment.fillna("<NA>").value_counts().to_dict(),
        "mapping_status_counts": status.fillna("<NA>").value_counts().to_dict(),
    }
