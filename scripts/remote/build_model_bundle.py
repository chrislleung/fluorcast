"""Build a deterministic FluorCast production model bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Sequence


LAYOUT = (
    ("tree", "tree"),
    ("neural", "neural"),
    ("absorption_hybrid", "hybrid/absorption_nm"),
    ("emission_hybrid", "hybrid/emission_nm"),
    ("quantum_yield_hybrid", "hybrid/quantum_yield"),
)
NORMALIZED_MTIME = 0
DIR_MODE = 0o755
FILE_MODE = 0o644
EXEC_FILE_MODE = 0o755


class BuildError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise BuildError("OUTPUT_EXISTS", f"Output already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        if path.exists() and not overwrite:
            raise BuildError("OUTPUT_EXISTS", f"Output already exists: {path.name}")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _atomic_write_archive(path: Path, members: list[tuple[Path, str]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise BuildError("OUTPUT_EXISTS", f"Output already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with temp.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=NORMALIZED_MTIME) as gz:
                with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                    added_dirs: set[str] = set()
                    for source, arcname in members:
                        _add_parent_dirs(tar, arcname, added_dirs)
                        info = tar.gettarinfo(str(source), arcname=arcname)
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = NORMALIZED_MTIME
                        info.mode = EXEC_FILE_MODE if info.mode & stat.S_IXUSR else FILE_MODE
                        with source.open("rb") as handle:
                            tar.addfile(info, handle)
        if path.exists() and not overwrite:
            raise BuildError("OUTPUT_EXISTS", f"Output already exists: {path.name}")
        os.replace(temp, path)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _add_parent_dirs(tar: tarfile.TarFile, arcname: str, added_dirs: set[str]) -> None:
    parts = Path(arcname).parts[:-1]
    current = ""
    for part in parts:
        current = part if not current else f"{current}/{part}"
        if current in added_dirs:
            continue
        info = tarfile.TarInfo(current)
        info.type = tarfile.DIRTYPE
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = NORMALIZED_MTIME
        info.mode = DIR_MODE
        tar.addfile(info)
        added_dirs.add(current)


def _safe_relative(path: Path) -> str:
    rel = path.as_posix()
    if not rel or rel.startswith("/") or "\\" in rel or any(part in {"", ".", ".."} for part in rel.split("/")):
        raise BuildError("UNSAFE_PATH", "Artifact path is not safe for a portable archive.")
    return rel


def _validate_tree(source_dir: Path, dest_root: str) -> list[tuple[Path, str]]:
    if not source_dir.is_dir():
        raise BuildError("SOURCE_DIR_MISSING", f"Source directory is missing: {dest_root}")
    if source_dir.is_symlink():
        raise BuildError("SYMLINK_REJECTED", f"Source directory may not be a symlink: {dest_root}")

    members: list[tuple[Path, str]] = []
    saw_regular = False
    for path in sorted(source_dir.rglob("*"), key=lambda item: item.relative_to(source_dir).as_posix()):
        rel = path.relative_to(source_dir)
        arcname = _safe_relative(Path(dest_root) / rel)
        try:
            info = path.lstat()
        except OSError as exc:
            raise BuildError("SOURCE_UNREADABLE", f"Could not inspect artifact path: {arcname}") from exc
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            raise BuildError("SYMLINK_REJECTED", f"Symlink artifacts are not allowed: {arcname}")
        if stat.S_ISSOCK(mode) or stat.S_ISFIFO(mode) or stat.S_ISCHR(mode) or stat.S_ISBLK(mode):
            raise BuildError("SPECIAL_FILE_REJECTED", f"Special artifact files are not allowed: {arcname}")
        if path.is_dir():
            continue
        if not stat.S_ISREG(mode):
            raise BuildError("SPECIAL_FILE_REJECTED", f"Only regular artifact files are allowed: {arcname}")
        if getattr(info, "st_nlink", 1) > 1:
            raise BuildError("HARDLINK_REJECTED", f"Hard-linked artifact files are not allowed: {arcname}")
        saw_regular = True
        members.append((path, arcname))
    if not saw_regular:
        raise BuildError("SOURCE_DIR_EMPTY", f"Source directory has no regular artifact files: {dest_root}")
    return members


def _git(repo: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise BuildError("GIT_METADATA_UNAVAILABLE", "Could not read repository Git metadata.")
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in sorted(metadata.distributions(), key=lambda item: (item.metadata.get("Name") or "").lower()):
        name = distribution.metadata.get("Name")
        if name:
            versions[name] = distribution.version
    return versions


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = [args.archive, args.manifest, args.checksum]
    if len({path.resolve() for path in output_paths}) != len(output_paths):
        raise BuildError("OUTPUT_PATH_CONFLICT", "Archive, manifest, and checksum outputs must be different files.")
    for path in output_paths:
        if path.exists() and not args.overwrite:
            raise BuildError("OUTPUT_EXISTS", f"Output already exists: {path.name}")

    repo_dir = args.repo_dir.resolve()
    if not (repo_dir / ".git").exists():
        raise BuildError("REPO_NOT_GIT", "Repository directory is not a Git checkout.")
    commit = _git(repo_dir, ["rev-parse", "HEAD"])
    tags = [line for line in _git(repo_dir, ["tag", "--points-at", commit]).splitlines() if line]

    members: list[tuple[Path, str]] = []
    for arg_name, dest_root in LAYOUT:
        members.extend(_validate_tree(getattr(args, f"{arg_name}_dir").resolve(), dest_root))
    members.sort(key=lambda item: item[1])

    required_files = [{"path": arcname, "sha256": _sha256(source)} for source, arcname in members]
    manifest = {
        "schema_version": 1,
        "artifact_version": args.artifact_version,
        "compatible_git": {"commits": [commit], "tags": tags},
        "python_version": "3.11",
        "package_versions": _package_versions(),
        "required_files": required_files,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    _atomic_write_archive(args.archive, members, overwrite=args.overwrite)
    checksum = {"sha256": _sha256(args.archive)}
    _atomic_write_text(args.manifest, manifest_text, overwrite=args.overwrite)
    _atomic_write_text(args.checksum, json.dumps(checksum, sort_keys=True, separators=(",", ":")) + "\n", overwrite=args.overwrite)

    return {
        "schema_version": 1,
        "status": "success",
        "artifact_version": args.artifact_version,
        "archive": args.archive.name,
        "manifest": args.manifest.name,
        "checksum": args.checksum.name,
        "sha256": checksum["sha256"],
        "required_file_count": len(required_files),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree-dir", required=True, type=Path)
    parser.add_argument("--neural-dir", required=True, type=Path)
    parser.add_argument("--absorption-hybrid-dir", required=True, type=Path)
    parser.add_argument("--emission-hybrid-dir", required=True, type=Path)
    parser.add_argument("--quantum-yield-hybrid-dir", required=True, type=Path)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--artifact-version", required=True)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checksum", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = build(parse_args(argv))
        code = 0
    except BuildError as exc:
        print(f"{exc.code}: {exc}", file=sys.stderr)
        result = {"schema_version": 1, "status": "failed", "errors": [{"code": exc.code, "message": str(exc)}]}
        code = 1
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
