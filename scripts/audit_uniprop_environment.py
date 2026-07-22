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
import platform
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
DEFAULT_REVISION_FILE = PROJECT_ROOT / "third_party" / "nablacolors.REVISION"
DEFAULT_UPSTREAM_DIR = PROJECT_ROOT / "third_party" / "nablacolors"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "uniprop" / "checkpoint_manifest.json"
WINDOWS_SMOKE_PROFILE = "windows-smoke"
NIBI_REAL_PROFILE = "nibi-real"


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
        try:
            cpu_usable = bool(torch.isfinite(torch.ones(1, device="cpu")).all().item())
        except Exception:
            cpu_usable = False
        return {
            "available": True,
            "version": getattr(torch, "__version__", None),
            "cpu_usable": cpu_usable,
            "cuda_available": cuda_available,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu_name": gpu_name,
            "error": None,
        }
    except Exception as exc:  # pragma: no cover - depends on optional installs.
        return {
            "available": False,
            "version": None,
            "cpu_usable": False,
            "cuda_available": False,
            "cuda_runtime": None,
            "gpu_name": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def repository_import_report() -> dict[str, Any]:
    modules = [
        "chemfluor.uniprop.manifests",
        "chemfluor.uniprop.geometry_cache",
        "chemfluor.uniprop.lmdb_export",
        "chemfluor.uniprop.windows_smoke",
    ]
    rows = {}
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            rows[module_name] = {"available": True, "error": None}
        except Exception as exc:
            rows[module_name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": all(row["available"] for row in rows.values()), "modules": rows}


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
        size_is_exact = bool(item.get("size_is_exact", True))
        present = path.exists()
        actual_size = path.stat().st_size if present else None
        actual_checksum = None if dry_run or not present else file_checksum(path, checksum_type)
        actual_sha256 = None if dry_run or not present else file_checksum(path, "sha256")
        rows.append(
            {
                "filename": filename,
                "path": str(path),
                "present": present,
                "expected_size_bytes": expected_size,
                "size_is_exact": size_is_exact,
                "actual_size_bytes": actual_size,
                "size_matches": None if not present else actual_size == expected_size,
                "size_accepted": None if not present else (actual_size == expected_size or not size_is_exact),
                "checksum_type": checksum_type,
                "expected_checksum": expected_checksum,
                "actual_checksum": actual_checksum,
                "actual_sha256": actual_sha256,
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
    python_3_10 = report["python"]["version_info"][:2] == [3, 10]
    python_3_11_plus = (
        report["python"]["version_info"][0] == 3
        and report["python"]["version_info"][1] >= 11
    )
    windows_platform = report["platform"]["system"] == "Windows"
    linux_platform = report["platform"]["system"] == "Linux"
    upstream_ok = (
        report["upstream"]["present"]
        and report["upstream"]["commit"] == report["revision"].get("commit")
    )
    windows_smoke = (
        windows_platform
        and python_3_11_plus
        and report["rdkit"]["available"]
        and report["lmdb"]["available"]
        and report["pytorch"]["available"]
        and report["pytorch"]["cpu_usable"]
        and report["numpy"]["available"]
        and report["pandas"]["available"]
        and report["repository_imports"]["available"]
    )
    real_base = (
        linux_platform
        and python_3_10
        and report["rdkit"]["available"]
        and report["lmdb"]["available"]
        and report["pytorch"]["available"]
        and report["pytorch"]["cpu_usable"]
        and report["unicore"]["available"]
        and report["unimol_plus"]["available"]
        and upstream_ok
        and report["checkpoints"]["all_present"]
        and report["checkpoints"]["all_hashes_match"]
    )
    real_gpu = real_base and report["pytorch"]["cuda_available"] and bool(report["pytorch"]["gpu_name"])
    preprocessing = (
        report["rdkit"]["available"]
        and report["lmdb"]["available"]
        and report["repository_imports"]["available"]
    )
    cpu_smoke = windows_smoke if report["profile"] == WINDOWS_SMOKE_PROFILE else real_base
    gpu_training = real_gpu
    return {
        "profile": report["profile"],
        "selected_profile_ready": bool(
            windows_smoke
            if report["profile"] == WINDOWS_SMOKE_PROFILE
            else (
                real_base
                if report["real_device"] == "cpu"
                else real_gpu
                if report["real_device"] == "gpu"
                else real_base and real_gpu
            )
        ),
        "windows_smoke_ready": bool(windows_smoke),
        "real_uniprop_cpu_ready": bool(real_base),
        "real_uniprop_gpu_ready": bool(real_gpu),
        "preprocessing_ready": bool(preprocessing),
        "cpu_smoke_ready": bool(cpu_smoke),
        "gpu_training_ready": bool(gpu_training),
        "reasons": {
            "python_3_10": python_3_10,
            "python_3_11_plus": python_3_11_plus,
            "windows_platform": windows_platform,
            "linux_platform": linux_platform,
            "rdkit_available": report["rdkit"]["available"],
            "lmdb_available": report["lmdb"]["available"],
            "upstream_revision_matches": bool(upstream_ok),
            "pytorch_available": report["pytorch"]["available"],
            "pytorch_cpu_usable": report["pytorch"]["cpu_usable"],
            "numpy_available": report["numpy"]["available"],
            "pandas_available": report["pandas"]["available"],
            "repository_imports_available": report["repository_imports"]["available"],
            "unicore_available": report["unicore"]["available"],
            "unimol_plus_available": report["unimol_plus"]["available"],
            "checkpoints_present": report["checkpoints"]["all_present"],
            "checkpoint_hashes_match": report["checkpoints"]["all_hashes_match"],
            "cuda_available": report["pytorch"]["cuda_available"],
        },
        "profile_requirements": {
            "windows_smoke": {
                "requires_cuda": False,
                "requires_unicore": False,
                "requires_unimol_plus": False,
                "requires_chemprop": False,
                "requires_real_checkpoint": False,
            },
            "nibi_real": {
                "requires_cuda": report["real_device"] in {"gpu", "both"},
                "requires_unicore": True,
                "requires_unimol_plus": True,
                "requires_chemprop": False,
                "requires_real_checkpoint": True,
            },
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
        "profile": args.profile,
        "real_device": args.real_device,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "version_info": [sys.version_info.major, sys.version_info.minor, sys.version_info.micro],
        },
        "platform": {
            "system": platform.system(),
            "platform": platform.platform(),
        },
        "pytorch": torch_report(),
        "rdkit": rdkit_report(),
        "lmdb": module_report("lmdb"),
        "numpy": module_report("numpy"),
        "pandas": module_report("pandas"),
        "unicore": {
            **module_report("unicore"),
            "unicore_train": shutil.which("unicore-train"),
        },
        "unimol_plus": module_report("unimol_plus"),
        "unimol": module_report("unimol"),
        "chemprop": module_report("chemprop"),
        "repository_imports": repository_import_report(),
        "revision": revision,
        "upstream": git_revision(args.upstream_dir),
        "checkpoints": checkpoint_report(args.manifest, checkpoint_dir, dry_run=args.dry_run),
    }
    report["readiness"] = readiness(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=[WINDOWS_SMOKE_PROFILE, NIBI_REAL_PROFILE], default=NIBI_REAL_PROFILE)
    parser.add_argument("--real-device", choices=["cpu", "gpu", "both"], default="cpu")
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
