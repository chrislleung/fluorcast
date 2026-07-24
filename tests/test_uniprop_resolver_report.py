from __future__ import annotations

import pytest
from packaging.tags import Tag

from chemfluor.uniprop.resolver_report import (
    decoded_artifact_filename,
    parse_selected_wheel,
    validate_lmdb_native_candidate,
    validate_report_item_wheel,
    validate_unicore_runtime_report,
    validate_unicore_runtime_report_item,
)


def _item(name: str, version: str, url: str, requires_dist: list[str] | None = None) -> dict:
    metadata = {"name": name, "version": version}
    if requires_dist is not None:
        metadata["requires_dist"] = requires_dist
    return {"metadata": metadata, "download_info": {"url": url}}


LMDB_CP310_LINUX_TAG = Tag("cp310", "cp310", "linux_x86_64")


@pytest.mark.parametrize("scheme", ["file", "https"])
@pytest.mark.parametrize(
    ("name", "version", "filename"),
    [
        ("requests", "2.34.2+computecanada", "requests-2.34.2%2Bcomputecanada-py3-none-any.whl"),
        ("numpy", "2.2.2+computecanada", "numpy-2.2.2%2Bcomputecanada-cp310-cp310-linux_x86_64.whl"),
        ("torch", "2.6.0+computecanada", "torch-2.6.0%2Bcomputecanada-cp310-cp310-linux_x86_64.whl"),
    ],
)
def test_percent_encoded_alliance_local_version_wheels_are_validated(
    scheme: str,
    name: str,
    version: str,
    filename: str,
) -> None:
    url = f"file:///cvmfs/soft.computecanada.ca/wheelhouse/{filename}"
    if scheme == "https":
        url = f"https://example.invalid/alliancecan/{filename}"

    selected = validate_report_item_wheel(_item(name, version, url))

    assert selected.decoded_filename == filename.replace("%2B", "+")
    assert str(selected.parsed_version) == version
    assert selected.alliance_wheelhouse is True
    assert selected.has_local_version is True


def test_percent_decoding_occurs_before_wheel_parsing() -> None:
    filename, parsed = parse_selected_wheel(
        "file:///cvmfs/wheelhouse/requests-2.34.2%2Bcomputecanada-py3-none-any.whl"
    )

    assert filename == "requests-2.34.2+computecanada-py3-none-any.whl"
    assert str(parsed[1]) == "2.34.2+computecanada"


def test_literal_plus_sign_remains_valid() -> None:
    selected = validate_report_item_wheel(
        _item(
            "requests",
            "2.34.2+computecanada",
            "file:///cvmfs/wheelhouse/requests-2.34.2+computecanada-py3-none-any.whl",
        )
    )

    assert selected.decoded_filename == "requests-2.34.2+computecanada-py3-none-any.whl"
    assert selected.has_local_version is True


def test_query_strings_and_fragments_are_excluded_from_filename() -> None:
    assert (
        decoded_artifact_filename("https://example.invalid/pkg-1.0-py3-none-any.whl?download=1")
        == "pkg-1.0-py3-none-any.whl"
    )
    assert (
        decoded_artifact_filename("https://example.invalid/pkg-1.0-py3-none-any.whl#sha256=abc")
        == "pkg-1.0-py3-none-any.whl"
    )


def test_canonicalized_names_and_local_versions_compare_correctly() -> None:
    selected = validate_report_item_wheel(
        _item(
            "pydantic-core",
            "2.46.4+computecanada",
            "file:///wheelhouse/pydantic_core-2.46.4%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        )
    )

    assert selected.parsed_name == "pydantic-core"
    assert str(selected.parsed_version) == "2.46.4+computecanada"


def test_pypi_wheels_remain_accepted() -> None:
    selected = validate_report_item_wheel(
        _item(
            "wandb",
            "0.17.9",
            "https://files.pythonhosted.org/packages/ab/cd/wandb-0.17.9-py3-none-any.whl",
        )
    )

    assert selected.alliance_wheelhouse is False
    assert selected.has_local_version is False


@pytest.mark.parametrize(
    "url",
    [
        "https://example.invalid/pydantic_core-2.46.4.tar.gz",
        "https://example.invalid/package-1.0.zip",
        "https://example.invalid/package-1.0.tar.bz2",
    ],
)
def test_source_archives_are_rejected(url: str) -> None:
    with pytest.raises(RuntimeError, match="non-wheel artifact"):
        validate_report_item_wheel(_item("package", "1.0", url))


def test_malformed_wheel_filename_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="non-wheel artifact"):
        validate_report_item_wheel(_item("package", "1.0", "https://example.invalid/package-1.0.whl"))


@pytest.mark.parametrize(
    "item",
    [
        {"metadata": {"name": "package", "version": "1.0"}},
        {"metadata": {"name": "package", "version": "1.0"}, "download_info": {}},
    ],
)
def test_missing_urls_fail_clearly(item: dict) -> None:
    with pytest.raises(RuntimeError, match="download_info"):
        validate_report_item_wheel(item)


def test_empty_url_paths_fail_clearly() -> None:
    with pytest.raises(RuntimeError, match="no filename"):
        validate_report_item_wheel(_item("package", "1.0", "https://example.invalid/"))


def test_package_name_mismatch_fails() -> None:
    with pytest.raises(RuntimeError, match="package name does not match"):
        validate_report_item_wheel(
            _item("requests", "1.0", "https://example.invalid/pkg-1.0-py3-none-any.whl")
        )


def test_version_mismatch_fails() -> None:
    with pytest.raises(RuntimeError, match="version does not match"):
        validate_report_item_wheel(
            _item("pkg", "1.1", "https://example.invalid/pkg-1.0-py3-none-any.whl")
        )


def test_runtime_report_rejects_sources_before_installation() -> None:
    payload = {"install": [_item("package", "1.0", "https://example.invalid/package-1.0.tar.gz")]}

    with pytest.raises(RuntimeError, match="non-wheel artifact"):
        validate_unicore_runtime_report(payload)


def test_successful_runtime_validation_returns_selected_wheels() -> None:
    payload = {
        "install": [
            _item(
                "wandb",
                "0.17.9",
                "https://files.pythonhosted.org/packages/ab/cd/wandb-0.17.9-py3-none-any.whl",
            )
        ]
    }

    selected = validate_unicore_runtime_report(payload)

    assert len(selected) == 1
    assert selected[0].report_name == "wandb"


def test_runtime_validation_keeps_forbidden_dependency_guards() -> None:
    with pytest.raises(RuntimeError, match="declares forbidden dependency"):
        validate_unicore_runtime_report_item(
            _item(
                "package",
                "1.0",
                "https://example.invalid/package-1.0-py3-none-any.whl",
                ["pydantic-core; python_version >= '3.10'"],
            )
        )


def test_runtime_validation_rejects_forbidden_selected_packages() -> None:
    with pytest.raises(RuntimeError, match="selected forbidden package"):
        validate_unicore_runtime_report_item(
            _item(
                "maturin",
                "1.0",
                "https://example.invalid/maturin-1.0-py3-none-any.whl",
            )
        )


def test_runtime_validation_rejects_numpy_or_torch_replacement() -> None:
    with pytest.raises(RuntimeError, match="replace protected package"):
        validate_unicore_runtime_report_item(
            _item(
                "numpy",
                "2.2.2+computecanada",
                "file:///wheelhouse/numpy-2.2.2%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
            )
        )


def test_runtime_validation_rejects_unpinned_wandb() -> None:
    with pytest.raises(RuntimeError, match="selected wandb 0.27.2"):
        validate_unicore_runtime_report_item(
            _item(
                "wandb",
                "0.27.2",
                "https://example.invalid/wandb-0.27.2-py3-none-any.whl",
            )
        )


def test_lmdb_computecanada_local_version_satisfies_public_141_policy() -> None:
    selected = validate_lmdb_native_candidate(
        _item(
            "lmdb",
            "1.4.1+computecanada",
            "file:///cvmfs/soft.computecanada.ca/wheelhouse/lmdb-1.4.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        ),
        supported_tags={LMDB_CP310_LINUX_TAG},
    )

    assert selected.parsed_name == "lmdb"
    assert selected.parsed_version.base_version == "1.4.1"
    assert str(selected.parsed_version) == "1.4.1+computecanada"
    assert selected.decoded_filename == "lmdb-1.4.1+computecanada-cp310-cp310-linux_x86_64.whl"
    assert selected.native_candidate is True


@pytest.mark.parametrize(
    ("version", "filename", "match"),
    [
        ("1.7.5", "lmdb-1.7.5-py3-none-any.whl", "public version 1.7.5"),
        ("2.3.0", "lmdb-2.3.0-cp310-cp310-linux_x86_64.whl", "public version 2.3.0"),
        ("1.4.1", "lmdb-1.4.1-py3-none-any.whl", "universal"),
        ("1.4.1", "lmdb-1.4.1-pp310-pypy310_pp73-linux_x86_64.whl", "PyPy"),
        ("1.4.1", "lmdb-1.4.1-cp311-cp311-linux_x86_64.whl", "implementation"),
        ("1.4.1", "lmdb-1.4.1-cp310-abi3-linux_x86_64.whl", "ABI"),
        ("1.4.1", "lmdb-1.4.1-cp310-cp310-win_amd64.whl", "Linux"),
    ],
)
def test_lmdb_native_policy_rejects_unusable_wheels(version: str, filename: str, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_lmdb_native_candidate(
            _item("lmdb", version, f"https://example.invalid/{filename}"),
            supported_tags={LMDB_CP310_LINUX_TAG},
        )


@pytest.mark.parametrize(
    "filename",
    [
        "lmdb-1.4.1.tar.gz",
        "lmdb-1.4.1.zip",
        "lmdb-1.4.1.tar.bz2",
    ],
)
def test_lmdb_source_archives_are_rejected(filename: str) -> None:
    with pytest.raises(RuntimeError, match="non-wheel artifact"):
        validate_lmdb_native_candidate(
            _item("lmdb", "1.4.1", f"https://example.invalid/{filename}"),
            supported_tags={LMDB_CP310_LINUX_TAG},
        )


def test_lmdb_wheel_tags_are_compared_against_supported_tags() -> None:
    with pytest.raises(RuntimeError, match="not compatible"):
        validate_lmdb_native_candidate(
            _item(
                "lmdb",
                "1.4.1+computecanada",
                "file:///wheelhouse/lmdb-1.4.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
            ),
            supported_tags={Tag("cp310", "cp310", "manylinux_2_17_x86_64")},
        )


def test_runtime_report_uses_lmdb_native_policy() -> None:
    payload = {
        "install": [
            _item(
                "lmdb",
                "1.7.5",
                "file:///wheelhouse/lmdb-1.7.5-py3-none-any.whl",
            )
        ]
    }

    with pytest.raises(RuntimeError, match="public version 1.7.5"):
        validate_unicore_runtime_report(payload)
