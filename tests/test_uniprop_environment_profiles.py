from __future__ import annotations

import pytest

from scripts.audit_uniprop_environment import NIBI_REAL_PROFILE, WINDOWS_SMOKE_PROFILE, readiness


def _availability(available: bool) -> dict[str, object]:
    return {"available": available}


def _report(
    *,
    profile: str,
    system: str,
    python_version: list[int],
    unicore: bool = False,
    unimol_plus: bool = False,
    checkpoints: bool = False,
    cuda: bool = False,
    real_device: str = "cpu",
) -> dict[str, object]:
    return {
        "profile": profile,
        "real_device": real_device,
        "python": {"version_info": python_version},
        "platform": {"system": system},
        "rdkit": _availability(True),
        "lmdb": _availability(True),
        "numpy": _availability(True),
        "pandas": _availability(True),
        "pytorch": {"available": True, "cpu_usable": True, "cuda_available": cuda, "gpu_name": "GPU" if cuda else None},
        "unicore": _availability(unicore),
        "unimol_plus": _availability(unimol_plus),
        "repository_imports": _availability(True),
        "revision": {"commit": "abc"},
        "upstream": {"present": True, "commit": "abc"},
        "checkpoints": {"all_present": checkpoints, "all_hashes_match": checkpoints},
    }


@pytest.mark.windows_smoke
def test_windows_profile_readiness_does_not_require_real_dependencies() -> None:
    report = _report(
        profile=WINDOWS_SMOKE_PROFILE,
        system="Windows",
        python_version=[3, 11, 9],
        unicore=False,
        unimol_plus=False,
        checkpoints=False,
    )

    status = readiness(report)

    assert status["windows_smoke_ready"] is True
    assert status["selected_profile_ready"] is True
    assert status["real_uniprop_cpu_ready"] is False


@pytest.mark.windows_smoke
def test_absent_unicore_and_real_checkpoints_do_not_fail_windows_smoke() -> None:
    report = _report(
        profile=WINDOWS_SMOKE_PROFILE,
        system="Windows",
        python_version=[3, 12, 2],
        unicore=False,
        unimol_plus=False,
        checkpoints=False,
    )

    status = readiness(report)

    assert status["windows_smoke_ready"] is True
    assert status["reasons"]["unicore_available"] is False
    assert status["reasons"]["checkpoints_present"] is False
    assert status["profile_requirements"]["windows_smoke"]["requires_real_checkpoint"] is False


@pytest.mark.real_uniprop
def test_nibi_real_readiness_fails_without_real_dependencies() -> None:
    report = _report(
        profile=NIBI_REAL_PROFILE,
        system="Linux",
        python_version=[3, 10, 14],
        unicore=False,
        unimol_plus=False,
        checkpoints=False,
    )

    status = readiness(report)

    assert status["real_uniprop_cpu_ready"] is False
    assert status["selected_profile_ready"] is False
    assert status["windows_smoke_ready"] is False


@pytest.mark.real_uniprop
def test_nibi_real_cpu_ready_does_not_require_cuda() -> None:
    report = _report(
        profile=NIBI_REAL_PROFILE,
        system="Linux",
        python_version=[3, 10, 14],
        unicore=True,
        unimol_plus=True,
        checkpoints=True,
        cuda=False,
        real_device="cpu",
    )

    status = readiness(report)

    assert status["real_uniprop_cpu_ready"] is True
    assert status["real_uniprop_gpu_ready"] is False
    assert status["selected_profile_ready"] is True


@pytest.mark.cuda
@pytest.mark.real_uniprop
def test_nibi_real_gpu_ready_requires_cuda() -> None:
    report = _report(
        profile=NIBI_REAL_PROFILE,
        system="Linux",
        python_version=[3, 10, 14],
        unicore=True,
        unimol_plus=True,
        checkpoints=True,
        cuda=False,
        real_device="gpu",
    )

    status = readiness(report)

    assert status["real_uniprop_cpu_ready"] is True
    assert status["real_uniprop_gpu_ready"] is False
    assert status["selected_profile_ready"] is False
