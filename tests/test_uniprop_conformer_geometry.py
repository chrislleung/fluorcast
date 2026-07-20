from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

torch = pytest.importorskip("torch")

from chemfluor.uniprop.conformer_geometry import (  # noqa: E402
    CONFORMER_GEOMETRY_VARIANTS,
    CONFORMER_SET_SCHEMA_VERSION,
    SolventConditionedConformerAttention,
    conformer_set_cache_path,
    detect_xtb_environment,
    energy_weighted_pool,
    equal_pool,
    generate_rdkit_conformer_set,
    migrate_single_geometry_entry,
    validate_conformer_set_entry,
)
from chemfluor.uniprop.experiment_matrix import geometry_cost_profile  # noqa: E402
from chemfluor.uniprop.experiment_matrix import DEFAULT_MODELS  # noqa: E402
from chemfluor.uniprop.geometry_cache import GEOMETRY_SCHEMA_VERSION, generate_geometry_entry  # noqa: E402
from chemfluor.uniprop.manifests import MANIFEST_SCHEMA_VERSION, stable_hash  # noqa: E402


def molecule_id(smiles: str) -> str:
    return stable_hash("mol", MANIFEST_SCHEMA_VERSION, smiles)


def test_conformer_diversity_and_named_cache_paths() -> None:
    mid = molecule_id("CCCC")
    entry = generate_rdkit_conformer_set(mid, "CCCC", geometry_set_name="rdkit_multi_conformer", num_conformers=6)
    coords = [np.asarray(conf["coordinates"], dtype=float) for conf in entry["conformers"]]
    unique_coordinate_sets = {coord.round(4).tobytes() for coord in coords}

    assert entry["schema_version"] == CONFORMER_SET_SCHEMA_VERSION
    assert entry["conformer_count"] >= 2
    assert len(unique_coordinate_sets) >= 2
    assert conformer_set_cache_path(Path("cache"), mid, "rdkit_multi_conformer") != conformer_set_cache_path(Path("cache"), mid, "xtb_single")


def test_deterministic_conformer_ordering() -> None:
    mid = molecule_id("CCCC")
    first = generate_rdkit_conformer_set(mid, "CCCC", num_conformers=6)
    second = generate_rdkit_conformer_set(mid, "CCCC", num_conformers=6)

    assert [conf["conformer_id"] for conf in first["conformers"]] == [conf["conformer_id"] for conf in second["conformers"]]
    assert [conf["energy"] for conf in first["conformers"]] == pytest.approx([conf["energy"] for conf in second["conformers"]])
    assert [conf["energy"] for conf in first["conformers"]] == sorted(conf["energy"] for conf in first["conformers"])


def test_energy_and_geometry_alignment_is_validated() -> None:
    mid = molecule_id("CCO")
    entry = generate_rdkit_conformer_set(mid, "CCO", num_conformers=3)
    validate_conformer_set_entry(entry, molecule_id=mid, canonical_smiles="CCO")
    entry["conformers"][0]["coordinates"] = entry["conformers"][0]["coordinates"][:-1]
    entry["checksum"] = __import__("hashlib").sha256(
        __import__("json").dumps({key: value for key, value in entry.items() if key != "checksum"}, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="alignment"):
        validate_conformer_set_entry(entry, molecule_id=mid, canonical_smiles="CCO")


def test_cache_migration_preserves_single_geometry_baseline() -> None:
    mid = molecule_id("CCO")
    original = generate_geometry_entry(mid, "CCO")
    migrated = migrate_single_geometry_entry(original)

    assert migrated["base_geometry_schema_version"] == GEOMETRY_SCHEMA_VERSION
    assert migrated["geometry_variant"] == "rdkit_mmff_single"
    assert migrated["conformer_count"] == 1
    assert migrated["conformers"][0]["coordinates"] == original["coordinates"]
    assert migrated["conformers"][0]["energy"] == original["energy"]


def test_equal_pooling_is_permutation_invariant() -> None:
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]]])
    permuted = embeddings[:, [2, 0, 1], :]

    torch.testing.assert_close(equal_pool(embeddings), equal_pool(permuted))


def test_energy_weighted_pooling_is_permutation_invariant() -> None:
    embeddings = torch.tensor([[[1.0, 2.0], [3.0, 0.0], [2.0, 4.0]]])
    energies = torch.tensor([[0.2, 1.5, 0.8]])
    order = torch.tensor([2, 0, 1])
    pooled, weights = energy_weighted_pool(embeddings, energies)
    pooled_permuted, weights_permuted = energy_weighted_pool(embeddings[:, order, :], energies[:, order])

    torch.testing.assert_close(pooled, pooled_permuted)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(weights[:, order], weights_permuted)


def test_attention_weights_sum_to_one_and_solvent_changes_weights() -> None:
    module = SolventConditionedConformerAttention.build(torch, conformer_dim=3, solvent_dim=2)
    with torch.no_grad():
        module.conformer_score.weight.copy_(torch.eye(3))
        module.solvent_query.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]))
    conformers = torch.tensor([[[1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [0.5, 0.25, 1.0]]])
    solvent_a = torch.tensor([[1.0, 0.0]])
    solvent_b = torch.tensor([[0.0, 1.0]])

    _, weights_a = module(conformers, solvent_a)
    _, weights_b = module(conformers, solvent_b)

    torch.testing.assert_close(weights_a.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(weights_b.sum(dim=-1), torch.ones(1))
    assert not torch.allclose(weights_a, weights_b)


def test_solvent_changes_weights_but_not_cached_coordinates() -> None:
    mid = molecule_id("CCCC")
    entry = generate_rdkit_conformer_set(mid, "CCCC", num_conformers=4)
    coordinates_before = [[list(row) for row in conf["coordinates"]] for conf in entry["conformers"]]
    module = SolventConditionedConformerAttention.build(torch, conformer_dim=3, solvent_dim=2)
    conformers = torch.randn(len(entry["conformers"]), 3).unsqueeze(0)

    module(conformers, torch.tensor([[1.0, 0.0]]))
    module(conformers, torch.tensor([[0.0, 1.0]]))

    assert [[list(row) for row in conf["coordinates"]] for conf in entry["conformers"]] == coordinates_before


def test_single_conformer_pooling_reproduces_original_baseline() -> None:
    embedding = torch.tensor([[[4.0, -1.0, 2.0]]])
    energies = torch.tensor([[12.0]])
    equal = equal_pool(embedding)
    weighted, weights = energy_weighted_pool(embedding, energies)

    torch.testing.assert_close(equal, embedding[:, 0, :])
    torch.testing.assert_close(weighted, embedding[:, 0, :])
    torch.testing.assert_close(weights, torch.ones_like(weights))


def test_xtb_detection_is_optional_and_structured() -> None:
    report = detect_xtb_environment("definitely_missing_xtb_for_fluorcast_tests")

    assert report.available is False
    assert report.executable is None
    assert "not found" in str(report.detail)


def test_geometry_ablation_variants_and_cost_profiles_are_registered() -> None:
    assert CONFORMER_GEOMETRY_VARIANTS == (
        "rdkit_mmff_single",
        "xtb_single",
        "rdkit_multi_conformer",
        "rdkit_multi_equal_pooling",
        "rdkit_multi_energy_weighted_pooling",
        "rdkit_multi_solvent_conditioned_pooling",
    )
    assert set(CONFORMER_GEOMETRY_VARIANTS).issubset(DEFAULT_MODELS)
    assert geometry_cost_profile("rdkit_mmff_single")["relative_preprocessing_cost"] == 1.0
    assert geometry_cost_profile("xtb_single")["relative_preprocessing_cost"] > 1.0
    assert geometry_cost_profile("rdkit_multi_solvent_conditioned_pooling")["relative_inference_cost"] > 1.0
