from __future__ import annotations

import pytest

from chemfluor.conforformer.config import ConformerGenerationConfig


def test_config_serialization_is_stable() -> None:
    config = ConformerGenerationConfig()
    assert config.stable_json() == ConformerGenerationConfig().stable_json()
    assert list(config.to_payload()) == sorted(config.to_payload())


def test_invalid_conformer_count() -> None:
    with pytest.raises(ValueError, match="num_conformers"):
        ConformerGenerationConfig(num_conformers=0)


def test_invalid_rmsd_threshold() -> None:
    with pytest.raises(ValueError, match="prune"):
        ConformerGenerationConfig(prune_rms_threshold=-0.1)


def test_invalid_retry_configuration() -> None:
    with pytest.raises(ValueError, match="smaller"):
        ConformerGenerationConfig(num_conformers=4, retry_conformer_counts=(4, 2))


def test_defaults_are_deterministic() -> None:
    config = ConformerGenerationConfig()
    assert config.etkdg_version == "ETKDGv3"
    assert config.optimizer == "MMFF94"
    assert config.fallback_optimizer == "UFF"
    assert config.random_seed == ConformerGenerationConfig().random_seed
