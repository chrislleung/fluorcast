from __future__ import annotations

import pandas as pd

from chemfluor.uniprop.solvent_normalization import (
    apply_uniprop_solvent_overlay,
    resolve_uniprop_solvent,
)


def test_verified_blank_source_aliases_are_repaired_for_uniprop_only() -> None:
    expected = {
        " H2O ": "O",
        "CH2Cl2": "ClCCl",
        "DMSO": "CS(C)=O",
        "MeOH": "CO",
        "MeCN": "CC#N",
        "THF": "C1CCOC1",
        "CHCl3": "ClC(Cl)Cl",
        "EtOH": "CCO",
        "Toluene/toluene": "Cc1ccccc1",
        "ethyl acetate": "CCOC(C)=O",
        "hexane": "CCCCCC",
    }

    for label, smiles in expected.items():
        resolved = resolve_uniprop_solvent(pd.NA, label)
        assert resolved.canonical_smiles == smiles
        assert resolved.environment_type == "molecular_solvent"
        assert resolved.mapping_status == "resolved_alias"


def test_existing_source_canonical_solvent_is_preserved() -> None:
    resolved = resolve_uniprop_solvent("CCO", "EtOH")

    assert resolved.source_canonical_smiles == "CCO"
    assert resolved.canonical_smiles == "CCO"
    assert resolved.mapping_status == "source_canonical"


def test_gas_phase_has_no_uniprop_solvent_smiles() -> None:
    resolved = resolve_uniprop_solvent(pd.NA, "gas")

    assert resolved.canonical_smiles is None
    assert resolved.environment_type == "gas_phase"
    assert resolved.mapping_status == "gas_phase"


def test_ambiguous_label_remains_unresolved() -> None:
    resolved = resolve_uniprop_solvent(pd.NA, "water:ethanol 1:1")

    assert resolved.canonical_smiles is None
    assert resolved.environment_type == "mixed_solvent"
    assert resolved.mapping_status == "unresolved_mixture"


def test_overlay_does_not_modify_source_canonical_column() -> None:
    rows = pd.DataFrame(
        {
            "solvent_original": ["EtOH", "gas", "unknown"],
            "canonical_solvent_smiles": [pd.NA, pd.NA, pd.NA],
        }
    )

    enriched = apply_uniprop_solvent_overlay(rows)

    assert rows["canonical_solvent_smiles"].isna().all()
    assert enriched["source_canonical_solvent_smiles"].isna().all()
    assert enriched["uniprop_canonical_solvent_smiles"].tolist()[0] == "CCO"
    assert enriched["environment_type"].tolist() == [
        "molecular_solvent",
        "gas_phase",
        "unknown_solvent",
    ]
