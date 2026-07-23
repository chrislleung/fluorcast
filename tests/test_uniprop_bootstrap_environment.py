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
    assert "install_unicore.sh" not in result.stdout
    assert "conda" not in result.stdout
    assert "unimol_env" not in result.stdout
    assert "python=3.12" not in result.stdout
    assert "torch==2.6.*" in result.stdout


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


def test_cuda_bootstrap_contract_avoids_upstream_conda_installer() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "install_unicore.sh" not in text
    assert "conda " not in text
    assert "conda:" not in text
    assert "unimol_env" not in text
    assert "python=3.12" not in text
    assert "torch==2.6.*" in text
    assert "UNICORE_INSTALL_ARGS=(\"$UNICORE_DIR\")" in text
    assert 'pip install "${UNICORE_INSTALL_ARGS[@]}"' in text
    assert "pip install -e \"$UNIMOL_PLUS_DIR\"" in text


def test_bootstrap_completion_diagnostic_requires_real_imports() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    required = [
        "import torch",
        "import unicore",
        "import unimol_plus",
        "from unimol_plus.models.uniprop import UniPropModel",
        "Feature-schema SHA-256 verified",
        "Pinned upstream Git commit",
    ]
    for snippet in required:
        assert snippet in text


def test_bootstrap_does_not_download_checkpoints_or_train() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    forbidden = [
        "zenodo.org",
        "wget ",
        "curl ",
        "run_uniprop_real_checkpoint_gate.py",
        "train_uniprop",
        "head_pretrain",
    ]
    for snippet in forbidden:
        assert snippet not in text


def test_bootstrap_stages_stop_on_install_failures() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'pip install "$TORCH_SPEC"' in text
    assert 'fail "No compatible PyTorch wheel found' in text
    assert 'fail "PyTorch validation failed."' in text
    assert 'fail "Uni-Core installation failed."' in text
    assert 'fail "Uni-Mol+ installation failed."' in text
    assert 'fail "Final UniProp import diagnostic failed."' in text


def test_bootstrap_supports_partial_env_detection_and_clean_rebuild() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "--clean" in text
    assert 'rm -rf "$VENV_DIR"' in text
    assert "Partial UniProp environment detected" in text
    assert 'if [[ -e "$VENV_DIR" && ! -x "$VENV_PYTHON" ]]' in text


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
        assert checkpoint["size_is_exact"] is False
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
