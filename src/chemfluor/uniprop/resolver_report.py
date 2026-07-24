"""Validation helpers for UniProp pip resolver reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit

from packaging.markers import default_environment
from packaging.requirements import Requirement
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
PROTECTED_RUNTIME_PACKAGES = {"numpy", "torch"}
ALLIANCE_WHEELHOUSE_MARKERS = ("computecanada", "alliancecan", "wheelhouse")


@dataclass(frozen=True)
class SelectedWheel:
    """A validated wheel selected by a pip installation report."""

    report_name: str
    report_version: str
    original_url: str
    decoded_filename: str
    parsed_name: str
    parsed_version: Version
    alliance_wheelhouse: bool
    has_local_version: bool


def normalized_package_name(name: str) -> str:
    return str(canonicalize_name(name))


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
    parsed_name, parsed_version, _build, _tags = parsed_wheel
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
        alliance_wheelhouse=alliance,
        has_local_version=parsed_version.local is not None,
    )


def validate_unicore_runtime_report_item(item: dict, *, required_wandb: str = "0.17.9") -> SelectedWheel:
    metadata = item.get("metadata", {})
    raw_name = metadata.get("name", "")
    name = normalized_package_name(raw_name)
    version = metadata.get("version", "")

    if name in PROTECTED_RUNTIME_PACKAGES:
        raise RuntimeError(f"Runtime dependency resolution would replace protected package {raw_name}.")
    if name in FORBIDDEN_RUNTIME_PACKAGES:
        raise RuntimeError(f"Runtime dependency resolution selected forbidden package {raw_name}.")
    if name == "wandb" and Version(version).base_version != required_wandb:
        raise RuntimeError(f"Runtime dependency resolution selected wandb {version}; expected {required_wandb}.")

    selected = validate_report_item_wheel(item)

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


def validate_unicore_runtime_report(payload: dict, *, required_wandb: str = "0.17.9") -> list[SelectedWheel]:
    return [
        validate_unicore_runtime_report_item(item, required_wandb=required_wandb)
        for item in payload.get("install", [])
    ]
