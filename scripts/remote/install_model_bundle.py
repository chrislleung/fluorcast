"""Safely install a versioned FluorCast model bundle archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence


REQUIRED_TOP_LEVEL_DIRS = (
    "tree",
    "neural",
    "hybrid/absorption_nm",
    "hybrid/emission_nm",
    "hybrid/quantum_yield",
)


class BundleError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _safe_extract(archive: Path, destination: Path) -> None:
    suffixes = "".join(archive.suffixes[-2:])
    if suffixes.endswith(".tar.gz") or archive.suffix in {".tgz", ".tar"}:
        mode = "r:gz" if archive.suffix in {".gz", ".tgz"} else "r"
        with tarfile.open(archive, mode) as tar:
            for member in tar.getmembers():
                if not _safe_member(member.name) or member.issym() or member.islnk():
                    raise BundleError("UNSAFE_ARCHIVE_MEMBER", "Archive contains an unsafe member.")
            tar.extractall(destination)
        return
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not _safe_member(info.filename):
                    raise BundleError("UNSAFE_ARCHIVE_MEMBER", "Archive contains an unsafe member.")
            zf.extractall(destination)
        return
    raise BundleError("UNSUPPORTED_ARCHIVE", "Archive must be .tar, .tar.gz, .tgz, or .zip.")


def _load_json(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(code, "JSON file could not be read.") from exc
    if not isinstance(payload, dict):
        raise BundleError(code, "JSON file must contain an object.")
    return payload


def _validate_manifest(manifest: dict[str, Any], extracted: Path) -> None:
    if manifest.get("schema_version") != 1:
        raise BundleError("MANIFEST_SCHEMA_VERSION", "Unsupported manifest schema_version.")
    if not isinstance(manifest.get("artifact_version"), str) or not manifest["artifact_version"]:
        raise BundleError("MANIFEST_INVALID_SCHEMA", "artifact_version is required.")
    for rel in REQUIRED_TOP_LEVEL_DIRS:
        if not (extracted / rel).is_dir():
            raise BundleError("ARTIFACT_DIR_MISSING", f"Required artifact directory is missing: {rel}")
    required_files = manifest.get("required_files")
    if not isinstance(required_files, list):
        raise BundleError("MANIFEST_INVALID_SCHEMA", "required_files must be a list.")
    for entry in required_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise BundleError("MANIFEST_INVALID_SCHEMA", "Each required file needs a path.")
        rel = entry["path"]
        if not _safe_member(rel):
            raise BundleError("MANIFEST_UNSAFE_PATH", "Manifest contains an unsafe required file path.")
        path = extracted / rel
        if not path.is_file():
            raise BundleError("ARTIFACT_FILE_MISSING", f"Required file is missing: {rel}")
        checksum = entry.get("sha256")
        if checksum and _sha256(path) != checksum:
            raise BundleError("ARTIFACT_CHECKSUM_MISMATCH", f"Required file checksum mismatch: {rel}")


def _install_tree(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        current_manifest = destination / "artifact-manifest.json"
        new_manifest = source / "artifact-manifest.json"
        if current_manifest.is_file() and new_manifest.is_file():
            if _sha256(current_manifest) == _sha256(new_manifest):
                shutil.rmtree(source)
                return "already_installed"
        shutil.rmtree(source)
        raise BundleError("DESTINATION_EXISTS", "Destination already exists with a different bundle.")
    temp_final = destination.with_name(f".{destination.name}.installing.{os.getpid()}")
    if temp_final.exists():
        shutil.rmtree(temp_final)
    shutil.move(str(source), str(temp_final))
    os.replace(temp_final, destination)
    return "installed"


def install(args: argparse.Namespace) -> dict[str, Any]:
    archive = args.archive.resolve()
    checksum_payload = _load_json(args.checksum, "CHECKSUM_FILE_INVALID")
    expected = checksum_payload.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise BundleError("CHECKSUM_FILE_INVALID", "Checksum JSON must contain a sha256 value.")
    actual = _sha256(archive)
    if actual.lower() != expected.lower():
        raise BundleError("CHECKSUM_MISMATCH", "Archive SHA-256 does not match the expected checksum.")
    manifest = _load_json(args.manifest, "MANIFEST_INVALID_JSON")

    destination = args.destination.resolve()
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as tmp:
        extract_root = Path(tmp) / "bundle"
        extract_root.mkdir()
        _safe_extract(archive, extract_root)
        children = [child for child in extract_root.iterdir()]
        payload_root = children[0] if len(children) == 1 and children[0].is_dir() else extract_root
        manifest_target = payload_root / "artifact-manifest.json"
        manifest_target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _validate_manifest(manifest, payload_root)
        durable_source = destination.parent / f".{destination.name}.validated.{os.getpid()}"
        if durable_source.exists():
            shutil.rmtree(durable_source)
        shutil.copytree(payload_root, durable_source)
    action = _install_tree(durable_source, destination)
    return {
        "schema_version": 1,
        "status": "success",
        "action": action,
        "artifact_version": manifest["artifact_version"],
        "errors": [],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = install(args)
        code = 0
    except BundleError as exc:
        result = {
            "schema_version": 1,
            "status": "failed",
            "errors": [{"code": exc.code, "message": str(exc)}],
        }
        code = 1
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
