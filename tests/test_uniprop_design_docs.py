from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESIGN = PROJECT_ROOT / "docs" / "UNIPROP_3D_DESIGN.md"
ASSET_MAP = PROJECT_ROOT / "docs" / "UNIPROP_ASSET_MAP.md"
LOG = PROJECT_ROOT / "docs" / "UNIPROP_IMPLEMENTATION_LOG.md"


def test_uniprop_stage_one_docs_exist() -> None:
    for path in [DESIGN, ASSET_MAP, LOG]:
        assert path.exists(), f"missing required UniProp stage document: {path}"
        assert path.read_text(encoding="utf-8").strip()


def test_design_documents_core_data_contract() -> None:
    text = DESIGN.read_text(encoding="utf-8")
    required_phrases = [
        "one cached geometry per canonical unique chromophore",
        "one supervised learning record per chromophore-solvent observation",
        "upstream code separate from FluorCast adapters",
        "Leakage prevention",
        "absorption wavelength, emission wavelength, and quantum yield",
        "later extension points for lifetime, extinction coefficient",
    ]
    for phrase in required_phrases:
        assert phrase in text


def test_asset_map_classifies_conforformer_components() -> None:
    text = ASSET_MAP.read_text(encoding="utf-8")
    for heading in [
        "Reusable unchanged",
        "Reusable after refactoring",
        "ConforFormer-specific and not reusable",
    ]:
        assert heading in text
    for component in ["dictionary.py", "preprocess.py", "third_party/ConforFormer/"]:
        assert component in text


def test_implementation_log_records_stage_one_audit() -> None:
    text = LOG.read_text(encoding="utf-8")
    for phrase in [
        "feature/uniprop-3d",
        "No UniProp dependencies were installed",
        "No model was trained or validated",
        "Recommended next stage",
    ]:
        assert phrase in text
