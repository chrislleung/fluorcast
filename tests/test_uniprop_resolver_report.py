from __future__ import annotations

import pytest
from packaging.tags import Tag

from chemfluor.uniprop.resolver_report import (
    decoded_artifact_filename,
    parse_selected_wheel,
    parse_bootstrap_runtime_constraints,
    public_constraint_allows_distribution,
    validate_protected_package_candidate,
    validate_lmdb_native_candidate,
    validate_native_runtime_candidate,
    validate_report_item_wheel,
    validate_unicore_runtime_report,
    validate_unicore_runtime_report_item,
    validate_unimol_plus_runtime_report,
    validate_unimol_plus_runtime_report_item,
)


def _item(name: str, version: str, url: str, requires_dist: list[str] | None = None) -> dict:
    metadata = {"name": name, "version": version}
    if requires_dist is not None:
        metadata["requires_dist"] = requires_dist
    return {"metadata": metadata, "download_info": {"url": url}}


LMDB_CP310_LINUX_TAG = Tag("cp310", "cp310", "linux_x86_64")
CP310_LINUX_TAG = Tag("cp310", "cp310", "linux_x86_64")


@pytest.mark.parametrize("scheme", ["file", "https"])
@pytest.mark.parametrize(
    ("name", "version", "filename"),
    [
        ("requests", "2.34.2+computecanada", "requests-2.34.2%2Bcomputecanada-py3-none-any.whl"),
        ("numpy", "2.1.1+computecanada", "numpy-2.1.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl"),
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


def test_runtime_validation_allows_protected_packages_in_clean_resolution() -> None:
    selected = validate_unicore_runtime_report_item(
        _item(
            "numpy",
            "2.1.1+computecanada",
            "file:///wheelhouse/numpy-2.1.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        )
    )

    assert selected.report_name == "numpy"


def test_protected_candidate_accepts_exact_installed_selected_equality() -> None:
    decision = validate_protected_package_candidate(
        _item(
            "numpy",
            "2.1.1+computecanada",
            "file:///wheelhouse/numpy-2.1.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        ),
        {"numpy": "2.1.1+computecanada"},
    )

    assert decision is not None
    assert str(decision.installed_version) == "2.1.1+computecanada"
    assert str(decision.selected_version) == "2.1.1+computecanada"
    assert decision.action == "retain"


def test_protected_candidate_rejects_unconstrained_numpy_and_fsspec_upgrades() -> None:
    with pytest.raises(RuntimeError, match="numpy installed=2.1.1\\+computecanada selected=2.2.2\\+computecanada"):
        validate_protected_package_candidate(
            _item(
                "numpy",
                "2.2.2+computecanada",
                "file:///wheelhouse/numpy-2.2.2%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
            ),
            {"numpy": "2.1.1+computecanada"},
        )
    with pytest.raises(RuntimeError, match="fsspec installed=2026.6.0\\+computecanada selected=2026.7.0"):
        validate_protected_package_candidate(
            _item(
                "fsspec",
                "2026.7.0",
                "https://example.invalid/fsspec-2026.7.0-py3-none-any.whl",
            ),
            {"fsspec": "2026.6.0+computecanada"},
        )


def test_bootstrap_runtime_constraints_require_exact_expected_public_pins() -> None:
    parsed = parse_bootstrap_runtime_constraints(
        "\n".join(
            [
                "numpy==2.1.1",
                "torch==2.6.0",
                "filelock==3.32.0",
                "fsspec==2026.6.0",
                "typing-extensions==4.16.0",
                "packaging==26.2",
                "setuptools==83.0.0",
                "wheel==0.47.0",
                "numba==0.61.0",
                "llvmlite==0.44.0",
            ]
        )
    )

    assert sorted(parsed) == [
        "filelock",
        "fsspec",
        "llvmlite",
        "numba",
        "numpy",
        "packaging",
        "setuptools",
        "torch",
        "typing-extensions",
        "wheel",
    ]


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("numpy==2.1.1\n", "Missing bootstrap runtime constraints"),
        (
            "\n".join(
                [
                    "numpy==2.1.1",
                    "numpy==2.1.1",
                    "torch==2.6.0",
                    "filelock==3.32.0",
                    "fsspec==2026.6.0",
                    "typing-extensions==4.16.0",
                    "packaging==26.2",
                    "setuptools==83.0.0",
                    "wheel==0.47.0",
                    "numba==0.61.0",
                    "llvmlite==0.44.0",
                ]
            ),
            "Duplicate bootstrap runtime constraint for numpy",
        ),
        (
            "\n".join(
                [
                    "numpy==2.2.2",
                    "torch==2.6.0",
                    "filelock==3.32.0",
                    "fsspec==2026.6.0",
                    "typing-extensions==4.16.0",
                    "packaging==26.2",
                    "setuptools==83.0.0",
                    "wheel==0.47.0",
                    "numba==0.61.0",
                    "llvmlite==0.44.0",
                ]
            ),
            "Contradictory bootstrap runtime constraint for numpy",
        ),
        (
            "\n".join(
                [
                    "numpy==2.1.1",
                    "torch==2.6.0",
                    "filelock==3.32.0",
                    "fsspec==2026.7.0",
                    "typing-extensions==4.16.0",
                    "packaging==26.2",
                    "setuptools==83.0.0",
                    "wheel==0.47.0",
                    "numba==0.61.0",
                    "llvmlite==0.44.0",
                ]
            ),
            "Contradictory bootstrap runtime constraint for fsspec",
        ),
        (
            "\n".join(
                [
                    "numpy==2.1.1+computecanada",
                    "torch==2.6.0",
                    "filelock==3.32.0",
                    "fsspec==2026.6.0",
                    "typing-extensions==4.16.0",
                    "packaging==26.2",
                    "setuptools==83.0.0",
                    "wheel==0.47.0",
                    "numba==0.61.0",
                    "llvmlite==0.44.0",
                ]
            ),
            "must omit Alliance/local suffixes",
        ),
    ],
)
def test_bootstrap_runtime_constraints_fail_clearly(text: str, match: str) -> None:
    with pytest.raises(RuntimeError, match=match):
        parse_bootstrap_runtime_constraints(text)


def test_public_bootstrap_constraints_accept_alliance_local_distributions() -> None:
    assert public_constraint_allows_distribution("numpy==2.1.1", "2.1.1+computecanada")
    assert public_constraint_allows_distribution("fsspec==2026.6.0", "2026.6.0+computecanada")
    assert public_constraint_allows_distribution("typing-extensions==4.16.0", "4.16.0+computecanada")
    assert public_constraint_allows_distribution("packaging==26.2", "26.2+computecanada")
    assert public_constraint_allows_distribution("numba==0.61.0", "0.61.0+computecanada")
    assert public_constraint_allows_distribution("llvmlite==0.44.0", "0.44.0+computecanada")


@pytest.mark.parametrize(
    ("installed", "selected", "match"),
    [
        ("2.1.1", "2.1.1+computecanada", "installed=2.1.1 selected=2.1.1\\+computecanada"),
        ("2.1.1+other", "2.1.1+computecanada", "installed=2.1.1\\+other selected=2.1.1\\+computecanada"),
        ("2.1.0+computecanada", "2.1.1+computecanada", "installed=2.1.0\\+computecanada selected=2.1.1\\+computecanada"),
    ],
)
def test_protected_candidate_rejects_exact_version_mismatches(
    installed: str,
    selected: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_protected_package_candidate(
            _item(
                "numpy",
                selected,
                f"file:///wheelhouse/numpy-{selected.replace('+', '%2B')}-cp310-cp310-linux_x86_64.whl",
            ),
            {"numpy": installed},
        )


def test_protected_candidate_missing_installed_metadata_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="metadata is missing.*numpy.*selected=2.1.1\\+computecanada"):
        validate_protected_package_candidate(
            _item(
                "numpy",
                "2.1.1+computecanada",
                "file:///wheelhouse/numpy-2.1.1%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
            ),
            {},
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


def test_numba_computecanada_local_version_satisfies_public_policy() -> None:
    selected = validate_unimol_plus_runtime_report_item(
        _item(
            "numba",
            "0.61.0+computecanada",
            "file:///cvmfs/soft.computecanada.ca/wheelhouse/numba-0.61.0%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        ),
        supported_tags={CP310_LINUX_TAG},
    )

    assert selected.parsed_name == "numba"
    assert selected.parsed_version.base_version == "0.61.0"
    assert str(selected.parsed_version) == "0.61.0+computecanada"
    assert selected.native_candidate is True
    assert selected.matching_sys_tag is not None


def test_llvmlite_computecanada_local_version_satisfies_public_policy() -> None:
    selected = validate_native_runtime_candidate(
        _item(
            "llvmlite",
            "0.44.0+computecanada",
            "file:///cvmfs/soft.computecanada.ca/wheelhouse/llvmlite-0.44.0%2Bcomputecanada-cp310-cp310-linux_x86_64.whl",
        ),
        package="llvmlite",
        public_version="0.44.0",
        supported_tags={CP310_LINUX_TAG},
    )

    assert selected.parsed_name == "llvmlite"
    assert selected.parsed_version.base_version == "0.44.0"
    assert str(selected.parsed_version) == "0.44.0+computecanada"
    assert selected.native_candidate is True


@pytest.mark.parametrize(
    ("package", "expected", "version", "filename", "match"),
    [
        ("numba", "0.61.0", "0.61.2", "numba-0.61.2-cp310-cp310-linux_x86_64.whl", "public version 0.61.2"),
        ("llvmlite", "0.44.0", "0.45.0", "llvmlite-0.45.0-cp310-cp310-linux_x86_64.whl", "public version 0.45.0"),
        ("numba", "0.61.0", "0.61.0", "numba-0.61.0-py3-none-any.whl", "universal"),
        ("llvmlite", "0.44.0", "0.44.0", "llvmlite-0.44.0-pp310-pypy310_pp73-linux_x86_64.whl", "PyPy"),
        ("numba", "0.61.0", "0.61.0", "numba-0.61.0-cp311-cp311-linux_x86_64.whl", "not compatible"),
        ("llvmlite", "0.44.0", "0.44.0", "llvmlite-0.44.0-cp310-cp310-win_amd64.whl", "Linux"),
    ],
)
def test_unimol_plus_native_runtime_policy_rejects_unusable_wheels(
    package: str,
    expected: str,
    version: str,
    filename: str,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        validate_native_runtime_candidate(
            _item(package, version, f"https://example.invalid/{filename}"),
            package=package,
            public_version=expected,
            supported_tags={CP310_LINUX_TAG},
        )


@pytest.mark.parametrize(
    ("package", "version", "filename"),
    [
        ("numba", "0.61.0", "numba-0.61.0.tar.gz"),
        ("llvmlite", "0.44.0", "llvmlite-0.44.0.zip"),
    ],
)
def test_unimol_plus_runtime_source_archives_are_rejected(package: str, version: str, filename: str) -> None:
    with pytest.raises(RuntimeError, match="non-wheel artifact"):
        validate_unimol_plus_runtime_report_item(
            _item(package, version, f"https://example.invalid/{filename}"),
            supported_tags={CP310_LINUX_TAG},
        )


def test_unimol_plus_runtime_report_requires_only_numba_stack() -> None:
    payload = {
        "install": [
            _item("numba", "0.61.0", "https://example.invalid/numba-0.61.0-cp310-cp310-linux_x86_64.whl"),
            _item("llvmlite", "0.44.0", "https://example.invalid/llvmlite-0.44.0-cp310-cp310-linux_x86_64.whl"),
        ]
    }

    selected = validate_unimol_plus_runtime_report(payload, supported_tags={CP310_LINUX_TAG})

    assert [wheel.parsed_name for wheel in selected] == ["numba", "llvmlite"]


def test_unimol_plus_runtime_report_rejects_unexpected_dependency() -> None:
    with pytest.raises(RuntimeError, match="unexpected package numpy"):
        validate_unimol_plus_runtime_report_item(
            _item("numpy", "2.1.1", "https://example.invalid/numpy-2.1.1-cp310-cp310-linux_x86_64.whl")
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
