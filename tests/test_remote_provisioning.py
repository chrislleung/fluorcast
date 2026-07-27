from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECK_SCRIPT = PROJECT_ROOT / "scripts" / "remote" / "check_installation.py"
INSTALL_SCRIPT = PROJECT_ROOT / "scripts" / "remote" / "install_model_bundle.py"
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "remote" / "submit_production_training.sh"
PROVISION_SCRIPT = PROJECT_ROOT / "scripts" / "remote" / "provision_environment.sh"


def _bash() -> str | None:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    if sys.platform == "win32" and git_bash.exists():
        return str(git_bash)
    return shutil.which("bash")


def _run_json(script: Path, *args: str, env: dict[str, str] | None = None) -> tuple[int, dict]:
    completed = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        check=False,
    )
    assert completed.stdout
    return completed.returncode, json.loads(completed.stdout)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _fake_repo(tmp_path: Path, *, dirty: bool = False) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "remote", "add", "origin", "https://github.com/chrislleung/fluorcast.git")
    (repo / "README.md").write_text("FluorCast\n", encoding="utf-8")
    for rel in (
        "data/processed/fluodb_lite/combined_deduplicated.csv",
        "data/solvent_descriptors_expanded_deep4chem.csv",
    ):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n", encoding="utf-8")
    _git(repo, "add", ".")
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test User",
            "commit",
            "-m",
            "init",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(repo, "tag", "vtest")
    if dirty:
        (repo / "README.md").write_text("dirty\n", encoding="utf-8")
    return repo


def _fake_env(tmp_path: Path) -> Path:
    env = tmp_path / "env"
    bin_dir = env / ("Scripts" if sys.platform == "win32" else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if sys.platform == "win32" else "python")).write_text("", encoding="utf-8")
    (bin_dir / "activate").write_text("", encoding="utf-8")
    return env


def _artifact_tree(root: Path) -> Path:
    artifact = root / "artifacts"
    for rel in (
        "tree/rf",
        "neural",
        "hybrid/absorption_nm",
        "hybrid/emission_nm",
        "hybrid/quantum_yield",
    ):
        (artifact / rel).mkdir(parents=True, exist_ok=True)
    files = {
        "tree/rf/model.joblib": "tree",
        "neural/feature_metadata.json": "{}",
        "hybrid/absorption_nm/model.joblib": "abs",
        "hybrid/emission_nm/model.joblib": "em",
        "hybrid/quantum_yield/model.joblib": "qy",
    }
    required = []
    for rel, text in files.items():
        path = artifact / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        required.append({"path": rel, "sha256": hashlib.sha256(text.encode()).hexdigest()})
    manifest = {
        "schema_version": 1,
        "artifact_version": "vtest-artifacts",
        "compatible_git": {"tags": ["vtest"]},
        "python_version": "3.11",
        "package_versions": {"numpy": "1.26.4"},
        "required_files": required,
    }
    (artifact / "artifact-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact


def test_check_installation_success_for_clean_fake_repo(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    env = _fake_env(tmp_path)
    artifact = _artifact_tree(tmp_path)

    code, payload = _run_json(
        CHECK_SCRIPT,
        "--repo-dir",
        str(repo),
        "--env-dir",
        str(env),
        "--artifact-dir",
        str(artifact),
        "--expected-version",
        "vtest-artifacts",
        env={"FLUORCAST_SKIP_REMOTE_IMPORTS": "1"},
    )

    assert code == 0
    assert payload["status"] == "success"
    assert payload["errors"] == []


def test_check_installation_reports_dirty_repo(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path, dirty=True)
    env = _fake_env(tmp_path)
    artifact = _artifact_tree(tmp_path)

    code, payload = _run_json(
        CHECK_SCRIPT,
        "--repo-dir",
        str(repo),
        "--env-dir",
        str(env),
        "--artifact-dir",
        str(artifact),
        "--expected-version",
        "vtest-artifacts",
        env={"FLUORCAST_SKIP_REMOTE_IMPORTS": "1"},
    )

    assert code == 1
    assert {error["code"] for error in payload["errors"]} >= {"GIT_DIRTY"}


def test_check_installation_reports_missing_environment_and_artifacts(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    code, payload = _run_json(
        CHECK_SCRIPT,
        "--repo-dir",
        str(repo),
        "--env-dir",
        str(tmp_path / "missing-env"),
        "--artifact-dir",
        str(tmp_path / "missing-artifacts"),
        "--expected-version",
        "vtest-artifacts",
        env={"FLUORCAST_SKIP_REMOTE_IMPORTS": "1"},
    )

    assert code == 1
    codes = {error["code"] for error in payload["errors"]}
    assert {"ENV_ACTIVATE_MISSING", "ENV_PYTHON_MISSING", "ARTIFACT_DIR_MISSING", "MANIFEST_MISSING"} <= codes


def test_install_model_bundle_success_and_idempotent_rerun(tmp_path: Path) -> None:
    source = _artifact_tree(tmp_path / "source")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="bundle")
    checksum = tmp_path / "checksum.json"
    checksum.write_text(json.dumps({"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}), encoding="utf-8")
    manifest = source / "artifact-manifest.json"
    destination = tmp_path / "installed"

    first_code, first = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        "--manifest",
        str(manifest),
        "--destination",
        str(destination),
    )
    second_code, second = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        "--manifest",
        str(manifest),
        "--destination",
        str(destination),
    )

    assert first_code == 0
    assert first["action"] == "installed"
    assert second_code == 0
    assert second["action"] == "already_installed"


def test_install_model_bundle_rejects_checksum_mismatch(tmp_path: Path) -> None:
    source = _artifact_tree(tmp_path / "source")
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="bundle")
    checksum = tmp_path / "checksum.json"
    checksum.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")

    code, payload = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        "--manifest",
        str(source / "artifact-manifest.json"),
        "--destination",
        str(tmp_path / "installed"),
    )

    assert code == 1
    assert payload["errors"][0]["code"] == "CHECKSUM_MISMATCH"


def test_install_model_bundle_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = tmp_path / "payload.txt"
        payload.write_text("bad", encoding="utf-8")
        tar.add(payload, arcname="../payload.txt")
    checksum = tmp_path / "checksum.json"
    checksum.write_text(json.dumps({"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}), encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_version": "bad",
                "compatible_git": {},
                "python_version": "3.11",
                "package_versions": {},
                "required_files": [],
            }
        ),
        encoding="utf-8",
    )

    code, payload = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(archive),
        "--checksum",
        str(checksum),
        "--manifest",
        str(manifest),
        "--destination",
        str(tmp_path / "installed"),
    )

    assert code == 1
    assert payload["errors"][0]["code"] == "UNSAFE_ARCHIVE_MEMBER"


def test_validate_production_install_marks_ready_after_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.remote import validate_production_install as validator

    repo = tmp_path / "repo"
    repo.mkdir()
    artifact = _artifact_tree(tmp_path)
    state = tmp_path / "install-state.json"

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(args[-1])
        output_path.write_text(
            json.dumps({"status": "success", "predictions": [{"predicted_emission_nm": 500.0}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    result, code = validator.validate(
        argparse_namespace(repo_dir=repo, artifact_dir=artifact, state_file=state)
    )

    assert code == 0
    assert result["status"] == "success"
    assert json.loads(state.read_text(encoding="utf-8"))["status"] == "ready"


def argparse_namespace(**kwargs: object) -> object:
    return type("Args", (), kwargs)()


def test_submit_training_rejects_invalid_account(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    activate = env / "activate"
    activate.write_text("", encoding="utf-8")
    completed = subprocess.run(
        [bash, str(SUBMIT_SCRIPT), "--account", "", "--repo-dir", str(repo), "--env-activate", str(activate)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout.splitlines()[-1])["code"] == "INVALID_SLURM_ACCOUNT"


def test_submit_training_prevents_duplicate_submission_with_state(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo = tmp_path / "repo"
    (repo / "slurm" / "base_models").mkdir(parents=True)
    (repo / "slurm" / "production").mkdir(parents=True)
    (repo / "slurm" / "base_models" / "run_model_experiments_fluodb.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "base_models" / "run_neural_experiments.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "run_hybrid_three_way_experiment.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "production" / "validate_production_install.sbatch").write_text("", encoding="utf-8")
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    activate = env / "activate"
    activate.write_text("", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text("#!/usr/bin/env bash\nprintf '12345\\n' >> \"$SBATCH_LOG\"\nprintf '12345\\n'\n", encoding="utf-8")
    sbatch.chmod(0o755)
    state = tmp_path / "state.json"
    env_vars = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}", "SBATCH_LOG": str(tmp_path / "sbatch.log")}

    first = subprocess.run(
        [bash, str(SUBMIT_SCRIPT), "--account", "def-test", "--repo-dir", str(repo), "--env-activate", str(activate), "--state-file", str(state)],
        env=env_vars,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        [bash, str(SUBMIT_SCRIPT), "--account", "def-test", "--repo-dir", str(repo), "--env-activate", str(activate), "--state-file", str(state)],
        env=env_vars,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0
    assert second.returncode == 0
    assert len((tmp_path / "sbatch.log").read_text(encoding="utf-8").splitlines()) == 6
    assert "TRAINING_ALREADY_SUBMITTED" in second.stdout


def test_new_shell_scripts_pass_syntax_check() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    for script in (PROVISION_SCRIPT, SUBMIT_SCRIPT, PROJECT_ROOT / "slurm" / "production" / "validate_production_install.sbatch"):
        completed = subprocess.run([bash, "-n", str(script)], capture_output=True, text=True, check=False)
        assert completed.returncode == 0, completed.stderr


def test_new_shell_scripts_pass_shellcheck_when_available() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck is not available")
    scripts = [
        str(PROVISION_SCRIPT),
        str(SUBMIT_SCRIPT),
        str(PROJECT_ROOT / "slurm" / "production" / "validate_production_install.sbatch"),
    ]
    completed = subprocess.run([shellcheck, *scripts], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
