"""Validation helpers for UniProp pip resolver reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag, sys_tags
from packaging.utils import (
    InvalidWheelFilename,
    canonicalize_name,
    parse_wheel_filename,
)
from packaging.version import Version


FORBIDDEN_RUNTIME_PACKAGES = {
    "maturin",
    "pydantic",
    "pydantic-core",
    "pydantic_core",
}
PROTECTED_RUNTIME_PACKAGES = {
    "filelock",
    "fsspec",
    "numpy",
    "packaging",
    "pip",
    "setuptools",
    "torch",
    "typing-extensions",
    "wheel",
}
ALLIANCE_WHEELHOUSE_MARKERS = ("computecanada", "alliancecan", "wheelhouse")
BOOTSTRAP_RUNTIME_CONSTRAINTS = {
    "filelock": "3.32.0",
    "fsspec": "2026.6.0",
    "llvmlite": "0.44.0",
    "numba": "0.61.0",
    "numpy": "2.1.1",
    "packaging": "26.2",
    "setuptools": "83.0.0",
    "torch": "2.6.0",
    "typing-extensions": "4.16.0",
    "wheel": "0.47.0",
}
FORBIDDEN_BOOTSTRAP_RUNTIME_CONSTRAINTS = {
    "fsspec": {"2026.7.0"},
    "llvmlite": {"0.45.0"},
    "numba": {"0.61.2"},
    "numpy": {"2.2.2", "2.2.6"},
}


@dataclass(frozen=True)
class SelectedWheel:
    """A validated wheel selected by a pip installation report."""

    report_name: str
    report_version: str
    original_url: str
    decoded_filename: str
    parsed_name: str
    parsed_version: Version
    wheel_tags: frozenset[Tag]
    alliance_wheelhouse: bool
    has_local_version: bool
    native_candidate: bool = False
    matching_sys_tag: Tag | None = None


@dataclass(frozen=True)
class ProtectedPackageDecision:
    """A protected package selected by clean resolution and retained in place."""

    package_name: str
    installed_version: Version
    selected_version: Version
    action: str = "retain"


@dataclass(frozen=True)
class RuntimeConstraintPolicy:
    """An exact public-version constraint used to keep pip resolution portable."""

    package_name: str
    public_version: Version
    source_line: int


def normalized_package_name(name: str) -> str:
    return str(canonicalize_name(name))


def parse_bootstrap_runtime_constraints(text: str) -> dict[str, RuntimeConstraintPolicy]:
    """Parse and validate the authoritative UniProp bootstrap runtime constraints."""

    constraints: dict[str, RuntimeConstraintPolicy] = {}
    seen_lines: dict[str, int] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        name = normalized_package_name(requirement.name)
        if name not in BOOTSTRAP_RUNTIME_CONSTRAINTS:
            raise RuntimeError(f"Unexpected bootstrap runtime constraint {requirement.name!r} on line {line_number}.")
        if name in constraints:
            first_line = seen_lines[name]
            raise RuntimeError(
                f"Duplicate bootstrap runtime constraint for {name}: lines {first_line} and {line_number}."
            )
        exact_versions = [
            spec.version
            for spec in requirement.specifier
            if spec.operator in {"==", "==="} and "*" not in spec.version
        ]
        if len(exact_versions) != 1 or str(requirement.specifier) != f"=={exact_versions[0]}":
            raise RuntimeError(
                f"Bootstrap runtime constraint for {name} must be one exact public-version pin, got {line!r}."
            )
        version = Version(exact_versions[0])
        if version.local is not None:
            raise RuntimeError(
                f"Bootstrap runtime constraint for {name} must omit Alliance/local suffixes, got {version}."
            )
        expected = Version(BOOTSTRAP_RUNTIME_CONSTRAINTS[name])
        if version != expected:
            raise RuntimeError(
                f"Contradictory bootstrap runtime constraint for {name}: got {version}, expected {expected}."
            )
        forbidden_versions = FORBIDDEN_BOOTSTRAP_RUNTIME_CONSTRAINTS.get(name, set())
        if version.base_version in forbidden_versions:
            raise RuntimeError(f"Forbidden bootstrap runtime constraint for {name}: {version}.")
        constraints[name] = RuntimeConstraintPolicy(name, version, line_number)
        seen_lines[name] = line_number

    missing = sorted(set(BOOTSTRAP_RUNTIME_CONSTRAINTS) - set(constraints))
    if missing:
        raise RuntimeError(f"Missing bootstrap runtime constraints: {', '.join(missing)}.")

    return constraints


def format_bootstrap_runtime_constraints(text: str) -> list[str]:
    constraints = parse_bootstrap_runtime_constraints(text)
    return [f"{name}=={constraints[name].public_version}" for name in sorted(constraints)]


def public_constraint_allows_distribution(constraint: str, distribution_version: str) -> bool:
    """Return whether a public pin accepts a distribution, including local Alliance builds."""

    requirement = Requirement(constraint)
    return SpecifierSet(str(requirement.specifier)).contains(
        Version(distribution_version),
        prereleases=True,
    )


def decoded_artifact_filename(url: str) -> str:
    parsed = urlsplit(url)
    decoded_path = unquote(parsed.path)
    filename = PurePosixPath(decoded_path).name

    if not filename:
        raise RuntimeError(f"Resolver artifact URL has no filename: {url!r}")

    return filename


def parse_selected_wheel(url: str):
    filename = decoded_artifact_filename(url)

    try:
        parsed_wheel = parse_wheel_filename(filename)
    except InvalidWheelFilename as exc:
        raise RuntimeError(f"Resolver selected a non-wheel artifact: {filename}") from exc

    return filename, parsed_wheel


def validate_report_item_wheel(item: dict) -> SelectedWheel:
    metadata = item.get("metadata", {})
    report_name = metadata.get("name", "")
    report_version = metadata.get("version", "")
    download = item.get("download_info")
    if not isinstance(download, dict):
        raise RuntimeError(f"Resolver report item for {report_name or '<unknown>'} is missing download_info.")
    url = download.get("url")
    if not url:
        raise RuntimeError(f"Resolver report item for {report_name or '<unknown>'} is missing download_info.url.")

    decoded_filename = decoded_artifact_filename(url)
    try:
        parsed_wheel = parse_wheel_filename(decoded_filename)
    except InvalidWheelFilename as exc:
        raise RuntimeError(
            "Resolver selected a non-wheel artifact: "
            f"original_url={url} decoded_filename={decoded_filename}"
        ) from exc
    parsed_name, parsed_version, _build, wheel_tags = parsed_wheel
    canonical_report_name = normalized_package_name(report_name)
    canonical_wheel_name = normalized_package_name(str(parsed_name))
    if canonical_wheel_name != canonical_report_name:
        raise RuntimeError(
            "Resolver report package name does not match wheel filename: "
            f"{report_name!r} vs {decoded_filename!r}"
        )
    if Version(report_version) != parsed_version:
        raise RuntimeError(
            "Resolver report package version does not match wheel filename: "
            f"{report_name} {report_version!r} vs {decoded_filename!r}"
        )

    decoded_url_path = unquote(urlsplit(url).path).lower()
    alliance = any(
        marker in decoded_filename.lower() or marker in decoded_url_path
        for marker in ALLIANCE_WHEELHOUSE_MARKERS
    )
    return SelectedWheel(
        report_name=report_name,
        report_version=report_version,
        original_url=url,
        decoded_filename=decoded_filename,
        parsed_name=canonical_wheel_name,
        parsed_version=parsed_version,
        wheel_tags=frozenset(wheel_tags),
        alliance_wheelhouse=alliance,
        has_local_version=parsed_version.local is not None,
    )


def validate_lmdb_native_candidate(
    item: dict,
    *,
    supported_tags: set[Tag] | frozenset[Tag] | None = None,
) -> SelectedWheel:
    """Require the Nibi-capable LMDB CPython 3.10 Linux native wheel policy."""

    selected = validate_report_item_wheel(item)
    selected_version = Version(selected.report_version)
    if selected.parsed_name != "lmdb":
        raise RuntimeError(f"LMDB policy received non-lmdb artifact: {selected.report_name}")
    if selected_version.base_version != "1.4.1":
        raise RuntimeError(
            f"LMDB selected public version {selected_version.base_version}; expected 1.4.1."
        )
    if selected.parsed_version.base_version != "1.4.1":
        raise RuntimeError(
            f"LMDB wheel public version {selected.parsed_version.base_version}; expected 1.4.1."
        )

    tags = selected.wheel_tags
    interpreters = {tag.interpreter for tag in tags}
    abis = {tag.abi for tag in tags}
    platforms = {tag.platform for tag in tags}

    if any(interpreter.startswith("pp") for interpreter in interpreters):
        raise RuntimeError(f"LMDB PyPy wheels are prohibited: {selected.decoded_filename}")
    if any(tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any" for tag in tags):
        raise RuntimeError(f"LMDB universal wheels are prohibited: {selected.decoded_filename}")
    if "cp310" not in interpreters:
        raise RuntimeError(f"LMDB wheel must have a CPython 3.10 implementation tag: {selected.decoded_filename}")
    if "cp310" not in abis:
        raise RuntimeError(f"LMDB wheel must have a CPython 3.10 ABI tag: {selected.decoded_filename}")
    if not any(platform.startswith("linux") or "_linux_" in platform for platform in platforms):
        raise RuntimeError(f"LMDB wheel must be a Linux platform wheel: {selected.decoded_filename}")
    compatible_tags = supported_tags if supported_tags is not None else set(sys_tags())
    if not tags.intersection(compatible_tags):
        raise RuntimeError(f"LMDB wheel tags are not compatible with this Python: {selected.decoded_filename}")

    return SelectedWheel(
        report_name=selected.report_name,
        report_version=selected.report_version,
        original_url=selected.original_url,
        decoded_filename=selected.decoded_filename,
        parsed_name=selected.parsed_name,
        parsed_version=selected.parsed_version,
        wheel_tags=selected.wheel_tags,
        alliance_wheelhouse=selected.alliance_wheelhouse,
        has_local_version=selected.has_local_version,
        native_candidate=True,
        matching_sys_tag=next(iter(tags.intersection(compatible_tags))),
    )


def validate_unimol_plus_runtime_report_item(
    item: dict,
    *,
    supported_tags: set[Tag] | frozenset[Tag] | None = None,
) -> SelectedWheel:
    """Require Nibi-compatible Uni-Mol+ runtime native wheels."""

    metadata = item.get("metadata", {})
    name = normalized_package_name(metadata.get("name", ""))
    expected_versions = {
        "numba": "0.61.0",
        "llvmlite": "0.44.0",
    }
    expected = expected_versions.get(name)
    if expected is None:
        raise RuntimeError(f"Uni-Mol+ runtime resolver selected unexpected package {metadata.get('name', name)}.")
    return validate_native_runtime_candidate(
        item,
        package=name,
        public_version=expected,
        supported_tags=supported_tags,
    )


def validate_native_runtime_candidate(
    item: dict,
    *,
    package: str,
    public_version: str,
    supported_tags: set[Tag] | frozenset[Tag] | None = None,
) -> SelectedWheel:
    """Require a CPython Linux wheel whose tags are supported by this interpreter."""

    selected = validate_report_item_wheel(item)
    selected_version = Version(selected.report_version)
    if selected.parsed_name != package:
        raise RuntimeError(f"{package} policy received non-{package} artifact: {selected.report_name}")
    if selected_version.base_version != public_version:
        raise RuntimeError(
            f"{package} selected public version {selected_version.base_version}; expected {public_version}."
        )
    if selected.parsed_version.base_version != public_version:
        raise RuntimeError(
            f"{package} wheel public version {selected.parsed_version.base_version}; expected {public_version}."
        )

    tags = selected.wheel_tags
    interpreters = {tag.interpreter for tag in tags}
    platforms = {tag.platform for tag in tags}
    if any(interpreter.startswith("pp") for interpreter in interpreters):
        raise RuntimeError(f"{package} PyPy wheels are prohibited: {selected.decoded_filename}")
    if any(tag.interpreter == "py3" and tag.abi == "none" and tag.platform == "any" for tag in tags):
        raise RuntimeError(f"{package} universal wheels are prohibited: {selected.decoded_filename}")
    if not any(interpreter.startswith("cp") for interpreter in interpreters):
        raise RuntimeError(f"{package} wheel must have a CPython implementation tag: {selected.decoded_filename}")
    if not any(platform.startswith("linux") or "linux" in platform for platform in platforms):
        raise RuntimeError(f"{package} wheel must be a Linux platform wheel: {selected.decoded_filename}")
    compatible_tags = supported_tags if supported_tags is not None else set(sys_tags())
    matching_tags = tags.intersection(compatible_tags)
    if not matching_tags:
        raise RuntimeError(f"{package} wheel tags are not compatible with this Python: {selected.decoded_filename}")

    return SelectedWheel(
        report_name=selected.report_name,
        report_version=selected.report_version,
        original_url=selected.original_url,
        decoded_filename=selected.decoded_filename,
        parsed_name=selected.parsed_name,
        parsed_version=selected.parsed_version,
        wheel_tags=selected.wheel_tags,
        alliance_wheelhouse=selected.alliance_wheelhouse,
        has_local_version=selected.has_local_version,
        native_candidate=True,
        matching_sys_tag=next(iter(matching_tags)),
    )


def validate_unicore_runtime_report_item(item: dict, *, required_wandb: str = "0.17.9") -> SelectedWheel:
    metadata = item.get("metadata", {})
    raw_name = metadata.get("name", "")
    name = normalized_package_name(raw_name)
    version = metadata.get("version", "")

    if name in FORBIDDEN_RUNTIME_PACKAGES:
        raise RuntimeError(f"Runtime dependency resolution selected forbidden package {raw_name}.")
    if name == "wandb" and Version(version).base_version != required_wandb:
        raise RuntimeError(f"Runtime dependency resolution selected wandb {version}; expected {required_wandb}.")

    selected = validate_report_item_wheel(item)
    if name == "lmdb":
        selected = validate_lmdb_native_candidate(item)

    requires_dist = metadata.get("requires_dist") or []
    marker_environment = default_environment()
    marker_environment["extra"] = ""
    for requirement_text in requires_dist:
        requirement = Requirement(requirement_text)
        if requirement.marker is not None and not requirement.marker.evaluate(marker_environment):
            continue
        dependency = normalized_package_name(requirement.name)
        if dependency in FORBIDDEN_RUNTIME_PACKAGES:
            raise RuntimeError(f"Runtime dependency {raw_name} declares forbidden dependency {requirement_text}.")

    return selected


def validate_protected_package_candidate(
    item: dict,
    installed_versions: dict[str, str],
) -> ProtectedPackageDecision | None:
    """Validate that a protected clean-resolution candidate matches installed metadata exactly."""

    metadata = item.get("metadata", {})
    raw_name = metadata.get("name", "")
    name = normalized_package_name(raw_name)
    if name not in PROTECTED_RUNTIME_PACKAGES:
        return None

    selected_text = metadata.get("version", "")
    installed_text = installed_versions.get(name)
    if installed_text is None:
        raise RuntimeError(
            "Protected package metadata is missing for clean-resolution candidate "
            f"{raw_name or name} selected={selected_text or '<unknown>'}."
        )

    installed = Version(installed_text)
    selected = Version(selected_text)
    if installed != selected:
        raise RuntimeError(
            "Protected package clean-resolution candidate differs from installed distribution: "
            f"{raw_name or name} installed={installed} selected={selected}."
        )

    return ProtectedPackageDecision(
        package_name=name,
        installed_version=installed,
        selected_version=selected,
    )


def validate_unicore_runtime_report(payload: dict, *, required_wandb: str = "0.17.9") -> list[SelectedWheel]:
    return [
        validate_unicore_runtime_report_item(item, required_wandb=required_wandb)
        for item in payload.get("install", [])
    ]


def validate_unimol_plus_runtime_report(
    payload: dict,
    *,
    supported_tags: set[Tag] | frozenset[Tag] | None = None,
) -> list[SelectedWheel]:
    return [
        validate_unimol_plus_runtime_report_item(item, supported_tags=supported_tags)
        for item in payload.get("install", [])
    ]
