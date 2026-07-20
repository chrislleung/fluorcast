from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap_uniprop.sh"
AUDIT = PROJECT_ROOT / "scripts" / "audit_uniprop_environment.py"
REVISION = PROJECT_ROOT / "third_party" / "nablacolors.REVISION"
MANIFEST = PROJECT_ROOT / "configs" / "uniprop" / "checkpoint_manifest.json"


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not available")
    return bash


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{rest}"


def _bash_cmd(*args: str | Path) -> list[str]:
    bash = _bash()
    converted = [_bash_path(arg) if isinstance(arg, Path) else arg for arg in args]
    if os.name == "nt" and "WindowsApps" in bash:
        wsl = shutil.which("wsl")
        if wsl is None:
            pytest.skip("WSL bash shim is present but wsl.exe is not available")
        return [wsl, "-e", "bash", *converted]
    return [bash, *converted]


def test_bootstrap_script_shell_syntax() -> None:
    subprocess.run(_bash_cmd("-n", BOOTSTRAP), check=True)


def test_bootstrap_dry_run_does_not_create_clone_or_venv(tmp_path: Path) -> None:
    upstream = tmp_path / "nablacolors"
    venv = tmp_path / "venv"
    report = tmp_path / "bootstrap.json"
    result = subprocess.run(
        [
            *_bash_cmd(BOOTSTRAP),
            "--dry-run",
            "--upstream-dir",
            _bash_path(upstream),
            "--venv",
            _bash_path(venv),
            "--json-output",
            _bash_path(report),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "DRY-RUN:" in result.stdout
    assert not upstream.exists()
    assert not venv.exists()
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "dry-run"


def test_bootstrap_revision_mismatch_detection(tmp_path: Path) -> None:
    upstream = tmp_path / "nablacolors"
    subprocess.run(["git", "init", str(upstream)], check=True, capture_output=True, text=True)
    subprocess.run(["git", "-C", str(upstream), "config", "user.email", "test@example.test"], check=True)
    subprocess.run(["git", "-C", str(upstream), "config", "user.name", "Test User"], check=True)
    (upstream / "README.md").write_text("wrong revision\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(upstream), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(upstream), "commit", "-m", "wrong"], check=True, capture_output=True, text=True)

    result = subprocess.run(
        [
            *_bash_cmd(BOOTSTRAP),
            "--dry-run",
            "--upstream-dir",
            _bash_path(upstream),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    assert "Revision mismatch" in result.stderr


def test_audit_missing_checkpoint_report_json_schema(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(AUDIT),
            "--checkpoint-dir",
            str(tmp_path / "empty_checkpoints"),
            "--json-output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(result.stdout)["schema_version"] == 1
    assert payload["schema_version"] == 1
    assert {"preprocessing_ready", "cpu_smoke_ready", "gpu_training_ready"}.issubset(
        payload["readiness"]
    )
    checkpoints = payload["checkpoints"]["checkpoints"]
    assert checkpoints
    assert all(row["present"] is False for row in checkpoints)
    assert payload["checkpoints"]["all_present"] is False


def test_checkpoint_manifest_schema() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["checkpoint_dir_env"] == "FLUORCAST_UNIPROP_CHECKPOINT_DIR"
    for checkpoint in payload["checkpoints"]:
        assert checkpoint["filename"].endswith(".pt")
        assert checkpoint["expected_size_bytes"] > 0
        assert checkpoint["source"].startswith("https://zenodo.org/records/18061300")
        assert checkpoint["checksum_type"] == "md5"
        assert len(checkpoint["checksum"]) == 32


def test_pinned_revision_file_schema() -> None:
    values = {}
    for line in REVISION.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    assert values["repo"] == "https://github.com/AI4DD/nablaColors.git"
    assert values["ref"] == "v1.0.0"
    assert values["commit"] == "39095389c0a4ecb47872ef74d00b8d13597939c8"


def test_python310_import_smoke_when_available(tmp_path: Path) -> None:
    python310 = shutil.which("python3.10") or shutil.which("python3.10.exe")
    if python310 is None:
        pytest.skip("Python 3.10 executable is not available")
    output = tmp_path / "py310-report.json"
    subprocess.run(
        [
            python310,
            str(AUDIT),
            "--dry-run",
            "--checkpoint-dir",
            str(tmp_path / "empty"),
            "--json-output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["python"]["version_info"][:2] == [3, 10]
