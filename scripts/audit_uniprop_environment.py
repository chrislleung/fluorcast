"""Audit an isolated UniProp/nablaColors environment.

This script is intentionally import-light and safe to run from the default
FluorCast environment. It reports readiness; it does not install anything.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVISION_FILE = PROJECT_ROOT / "third_party" / "nablacolors.REVISION"
DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "third_party" / "nablacolors"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "uniprop" / "checkpoint_manifest.json"


def read_revision_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def module_report(module_name: str, attr: str = "__version__") -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"available": False, "version": None, "error": None}
    try:
        module = importlib.import_module(module_name)
        return {
            "available": True,
            "version": getattr(module, attr, None),
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on optional installs.
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def rdkit_report() -> dict[str, Any]:
    spec = importlib.util.find_spec("rdkit")
    if spec is None:
        return {"available": False, "version": None, "error": None}
    try:
        from rdkit import rdBase

        return {"available": True, "version": rdBase.rdkitVersion, "error": None}
    except Exception as exc:  # pragma: no cover - depends on optional installs.
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


def torch_report() -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_runtime": None,
            "gpu_name": None,
            "error": None,
        }
    try:
        torch = importlib.import_module("torch")
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        return {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cuda_available": cuda_available,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu_name": gpu_name,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on optional installs.
        return {
            "available": False,
            "version": None,
            "cuda_available": False,
            "cuda_runtime": None,
            "gpu_name": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def git_revision(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False, "commit": None, "error": None}
    if not (path / ".git").exists():
        return {"present": True, "commit": None, "error": "not a Git checkout"}
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        return {"present": True, "commit": commit, "error": None}
    except subprocess.CalledProcessError as exc:
        return {"present": True, "commit": None, "error": exc.output.strip()}


def file_checksum(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": None,
            "checkpoints": [],
            "error": f"manifest not found: {path}",
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"schema_version": None, "checkpoints": [], "error": str(exc)}


def checkpoint_report(manifest_path: Path, checkpoint_dir: Path, *, dry_run: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    rows: list[dict[str, Any]] = []
    for item in manifest.get("checkpoints", []):
        filename = str(item.get("filename", ""))
        path = checkpoint_dir / filename
        checksum_type = str(item.get("checksum_type") or manifest.get("checksum_type") or "md5")
        expected_checksum = item.get("checksum")
        expected_size = item.get("expected_size_bytes")
        present = path.exists()
        actual_size = path.stat().st_size if present else None
        actual_checksum = None if dry_run or not present else file_checksum(path, checksum_type)
        rows.append(
            {
                "filename": filename,
                "path": str(path),
                "present": present,
                "expected_size_bytes": expected_size,
                "actual_size_bytes": actual_size,
                "size_matches": None if not present else actual_size == expected_size,
                "checksum_type": checksum_type,
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
                "checksum_matches": None if not present or dry_run else actual_checksum == expected_checksum,
                "source": item.get("source"),
            }
        )
    return {
        "manifest_path": str(manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "manifest_error": manifest.get("error"),
        "checkpoints": rows,
        "all_present": bool(rows) and all(row["present"] for row in rows),
        "all_hashes_match": bool(rows) and all(row["checksum_matches"] is True for row in rows),
    }


def readiness(report: dict[str, Any]) -> dict[str, Any]:
    python_ok = report["python"]["version_info"][:2] == [3, 10]
    upstream_ok = (
        report["upstream"]["present"]
        and report["upstream"]["commit"] == report["revision"].get("commit")
    )
    preprocessing = python_ok and report["rdkit"]["available"] and report["lmdb"]["available"] and upstream_ok
    cpu_smoke = (
        preprocessing
        and report["pytorch"]["available"]
        and report["unicore"]["available"]
        and report["unimol_plus"]["available"]
        and report["checkpoints"]["all_present"]
        and report["checkpoints"]["all_hashes_match"]
    )
    gpu_training = cpu_smoke and report["pytorch"]["cuda_available"] and bool(report["pytorch"]["gpu_name"])
    return {
        "preprocessing_ready": bool(preprocessing),
        "cpu_smoke_ready": bool(cpu_smoke),
        "gpu_training_ready": bool(gpu_training),
        "reasons": {
            "python_3_10": python_ok,
            "rdkit_available": report["rdkit"]["available"],
            "lmdb_available": report["lmdb"]["available"],
            "upstream_revision_matches": bool(upstream_ok),
            "pytorch_available": report["pytorch"]["available"],
            "unicore_available": report["unicore"]["available"],
            "unimol_plus_available": report["unimol_plus"]["available"],
            "checkpoints_present": report["checkpoints"]["all_present"],
            "checkpoint_hashes_match": report["checkpoints"]["all_hashes_match"],
            "cuda_available": report["pytorch"]["cuda_available"],
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    revision = read_revision_file(args.revision_file)
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = Path(os.environ.get("FLUORCAST_UNIPROP_CHECKPOINT_DIR", "assets/uniprop/checkpoints"))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    report: dict[str, Any] = {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_info": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        },
        "pytorch": torch_report(),
        "rdkit": rdkit_report(),
        "lmdb": module_report("lmdb"),
        "unicore": {
            **module_report("unicore"),
            "unicore_train": shutil.which("unicore-train"),
        },
        "unimol_plus": module_report("unimol_plus"),
        "unimol": module_report("unimol"),
        "chemprop": module_report("chemprop"),
        "revision": revision,
        "upstream": git_revision(args.upstream_dir),
        "checkpoints": checkpoint_report(args.manifest, checkpoint_dir, dry_run=args.dry_run),
    }
    report["readiness"] = readiness(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision-file", type=Path, default=DEFAULT_REVISION_FILE)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip checkpoint hashing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
