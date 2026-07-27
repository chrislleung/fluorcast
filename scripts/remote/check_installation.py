"""Validate a FluorCast remote installation and emit one JSON result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ORIGIN_FRAGMENT = "github.com/chrislleung/fluorcast"
REQUIRED_DATA_FILES = (
    "data/processed/fluodb_lite/combined_deduplicated.csv",
    "data/solvent_descriptors_expanded_deep4chem.csv",
)
REQUIRED_ARTIFACT_DIRS = (
    "tree",
    "neural",
    "hybrid/absorption_nm",
    "hybrid/emission_nm",
    "hybrid/quantum_yield",
)
REQUIRED_IMPORTS = (
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "xgboost",
    "lightgbm",
    "catboost",
    "rdkit",
)
SUCCESS = "OK"
ERROR_EXIT = 1


def _clean_error(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.name


def _run_git(repo: Path, args: Sequence[str]) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _activation_script(env_dir: Path) -> Path | None:
    candidates = (env_dir / "bin" / "activate", env_dir / "Scripts" / "activate")
    return next((path for path in candidates if path.is_file()), None)


def _python_executable(env_dir: Path) -> Path | None:
    candidates = (
        env_dir / "bin" / "python",
        env_dir / "bin" / "python3",
        env_dir / "Scripts" / "python.exe",
        env_dir / "Scripts" / "python",
    )
    return next((path for path in candidates if path.is_file()), None)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    if not path.is_file():
        return None, [_clean_error("MANIFEST_MISSING", "Artifact manifest is missing.")]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, [_clean_error("MANIFEST_INVALID_JSON", "Artifact manifest is not valid JSON.")]
    if not isinstance(payload, dict):
        return None, [_clean_error("MANIFEST_INVALID_SCHEMA", "Artifact manifest must be a JSON object.")]
    return payload, []


def _validate_manifest(
    manifest: dict[str, Any],
    manifest_path: Path,
    artifact_dir: Path,
    expected_version: str,
    repo_commit: str | None,
    repo_tag: str | None,
) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if manifest.get("schema_version") != 1:
        errors.append(_clean_error("MANIFEST_SCHEMA_VERSION", "Unsupported manifest schema_version."))
    if manifest.get("artifact_version") != expected_version:
        errors.append(_clean_error("ARTIFACT_VERSION_MISMATCH", "Artifact version does not match the expected version."))
    compatible = manifest.get("compatible_git") or {}
    if not isinstance(compatible, dict):
        errors.append(_clean_error("MANIFEST_INVALID_SCHEMA", "compatible_git must be an object."))
    else:
        commits = compatible.get("commits") or []
        tags = compatible.get("tags") or []
        if commits and repo_commit not in commits:
            errors.append(_clean_error("GIT_COMMIT_INCOMPATIBLE", "Repository commit is not listed as compatible."))
        if tags and repo_tag not in tags:
            errors.append(_clean_error("GIT_TAG_INCOMPATIBLE", "Repository tag is not listed as compatible."))
    required_files = manifest.get("required_files")
    if not isinstance(required_files, list):
        errors.append(_clean_error("MANIFEST_INVALID_SCHEMA", "required_files must be a list."))
        return errors
    root = manifest_path.parent
    for entry in required_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            errors.append(_clean_error("MANIFEST_INVALID_SCHEMA", "Each required file needs a path."))
            continue
        raw = entry["path"]
        if raw.startswith("/") or ".." in Path(raw).parts:
            errors.append(_clean_error("MANIFEST_UNSAFE_PATH", "Manifest required_files contains an unsafe path."))
            continue
        candidate = root / raw
        if not candidate.is_file():
            candidate = artifact_dir / raw
        if not candidate.is_file():
            errors.append(_clean_error("ARTIFACT_FILE_MISSING", f"Required artifact file is missing: {raw}"))
            continue
        checksum = entry.get("sha256")
        if checksum and _sha256(candidate) != checksum:
            errors.append(_clean_error("ARTIFACT_CHECKSUM_MISMATCH", f"Required artifact checksum mismatch: {raw}"))
    return errors


def _dependency_check(python_path: Path) -> list[dict[str, str]]:
    if os.environ.get("FLUORCAST_SKIP_REMOTE_IMPORTS") == "1":
        return []
    code = "import importlib; " + "; ".join(f"importlib.import_module({name!r})" for name in REQUIRED_IMPORTS)
    completed = subprocess.run(
        [str(python_path), "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return []
    return [_clean_error("DEPENDENCY_IMPORT_FAILED", "One or more runtime imports failed.")]


def _fixture_prediction(repo_dir: Path, python_path: Path, fixture_input: Path) -> list[dict[str, str]]:
    if not fixture_input:
        return []
    with tempfile.TemporaryDirectory(prefix="fluorcast-check-") as tmp:
        output = Path(tmp) / "output.json"
        completed = subprocess.run(
            [
                str(python_path),
                str(repo_dir / "scripts" / "run_prediction_job.py"),
                "--input",
                str(fixture_input),
                "--output",
                str(output),
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return [_clean_error("FIXTURE_PREDICTION_FAILED", "Fixture prediction command failed.")]
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return [_clean_error("FIXTURE_PREDICTION_INVALID_JSON", "Fixture prediction output is not valid JSON.")]
        if payload.get("status") != "success":
            return [_clean_error("FIXTURE_PREDICTION_FAILED", "Fixture prediction did not return success.")]
    return []


def build_result(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    repo_dir = args.repo_dir.resolve()
    env_dir = args.env_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    git_info: dict[str, Any] = {"valid": False, "dirty": None, "commit": None, "tag": None}

    if not (repo_dir / ".git").exists():
        errors.append(_clean_error("REPO_NOT_GIT", "Repository directory is not a Git checkout."))
    else:
        code, origin, _ = _run_git(repo_dir, ["remote", "get-url", "origin"])
        if code != 0:
            errors.append(_clean_error("GIT_ORIGIN_MISSING", "Git origin remote is missing."))
        elif EXPECTED_ORIGIN_FRAGMENT not in origin.lower().replace(":", "/"):
            errors.append(_clean_error("GIT_ORIGIN_UNEXPECTED", "Git origin is not the FluorCast repository."))
        code, commit, _ = _run_git(repo_dir, ["rev-parse", "HEAD"])
        if code == 0:
            git_info["commit"] = commit
        else:
            errors.append(_clean_error("GIT_COMMIT_UNAVAILABLE", "Current Git commit could not be read."))
        code, tag, _ = _run_git(repo_dir, ["describe", "--tags", "--exact-match"])
        git_info["tag"] = tag if code == 0 else None
        code, status, _ = _run_git(repo_dir, ["status", "--porcelain"])
        if code == 0:
            git_info["dirty"] = bool(status)
            if status:
                errors.append(_clean_error("GIT_DIRTY", "Repository has uncommitted changes."))
        else:
            errors.append(_clean_error("GIT_STATUS_UNAVAILABLE", "Git status could not be read."))
        git_info["valid"] = not any(error["code"].startswith("GIT_") for error in errors)

    for rel_path in REQUIRED_DATA_FILES:
        if not (repo_dir / rel_path).is_file():
            errors.append(_clean_error("DATA_FILE_MISSING", f"Required data file is missing: {rel_path}"))

    activation = _activation_script(env_dir)
    if activation is None:
        errors.append(_clean_error("ENV_ACTIVATE_MISSING", "Python activation script is missing."))
    python_path = _python_executable(env_dir)
    if python_path is None:
        errors.append(_clean_error("ENV_PYTHON_MISSING", "Python executable is missing from the environment."))
    else:
        errors.extend(_dependency_check(python_path))

    artifact_status: dict[str, bool] = {}
    for rel_path in REQUIRED_ARTIFACT_DIRS:
        exists = (artifact_dir / rel_path).is_dir()
        artifact_status[rel_path] = exists
        if not exists:
            errors.append(_clean_error("ARTIFACT_DIR_MISSING", f"Required artifact directory is missing: {rel_path}"))

    manifest_path = args.manifest or artifact_dir / "artifact-manifest.json"
    manifest, manifest_errors = _load_manifest(manifest_path)
    errors.extend(manifest_errors)
    if manifest is not None:
        errors.extend(
            _validate_manifest(
                manifest,
                manifest_path,
                artifact_dir,
                args.expected_version,
                git_info["commit"],
                git_info["tag"],
            )
        )

    if args.fixture_input and python_path is not None and not errors:
        errors.extend(_fixture_prediction(repo_dir, python_path, args.fixture_input))
    elif args.fixture_input and errors:
        warnings.append(_clean_error("FIXTURE_PREDICTION_SKIPPED", "Fixture prediction skipped because installation checks failed."))

    result = {
        "schema_version": 1,
        "status": "success" if not errors else "failed",
        "expected_version": args.expected_version,
        "repository": git_info,
        "environment": {
            "activation_script": "present" if activation else "missing",
            "python": "present" if python_path else "missing",
        },
        "artifacts": {"directories": artifact_status},
        "errors": errors,
        "warnings": warnings,
    }
    return result, 0 if not errors else ERROR_EXIT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", required=True, type=Path)
    parser.add_argument("--env-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--expected-version", required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result, exit_code = build_result(args)
    json.dump(result, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
