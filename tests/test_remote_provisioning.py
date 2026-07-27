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
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "remote" / "build_model_bundle.py"
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


def _bundle_sources(tmp_path: Path) -> dict[str, Path]:
    roots = {
        "tree": tmp_path / "tree-src",
        "neural": tmp_path / "neural-src",
        "absorption": tmp_path / "abs-src",
        "emission": tmp_path / "em-src",
        "qy": tmp_path / "qy-src",
    }
    files = {
        "tree": ("rf/model.joblib", "tree"),
        "neural": ("model.pt", "neural"),
        "absorption": ("model.joblib", "abs"),
        "emission": ("model.joblib", "em"),
        "qy": ("model.joblib", "qy"),
    }
    for key, root in roots.items():
        rel, text = files[key]
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return roots


def _run_builder(tmp_path: Path, repo: Path, sources: dict[str, Path], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--tree-dir",
            str(sources["tree"]),
            "--neural-dir",
            str(sources["neural"]),
            "--absorption-hybrid-dir",
            str(sources["absorption"]),
            "--emission-hybrid-dir",
            str(sources["emission"]),
            "--quantum-yield-hybrid-dir",
            str(sources["qy"]),
            "--repo-dir",
            str(repo),
            "--artifact-version",
            "vtest-artifacts",
            "--archive",
            str(tmp_path / "bundle.tar.gz"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--checksum",
            str(tmp_path / "checksum.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def _minimal_manifest_schema_valid(manifest: dict) -> bool:
    return (
        manifest.get("schema_version") == 1
        and isinstance(manifest.get("artifact_version"), str)
        and isinstance(manifest.get("compatible_git", {}).get("commits"), list)
        and isinstance(manifest.get("compatible_git", {}).get("tags"), list)
        and str(manifest.get("python_version", "")).startswith("3.11")
        and isinstance(manifest.get("package_versions"), dict)
        and all(
            isinstance(entry.get("path"), str)
            and not Path(entry["path"]).is_absolute()
            and ".." not in Path(entry["path"]).parts
            and isinstance(entry.get("sha256"), str)
            and len(entry["sha256"]) == 64
            for entry in manifest.get("required_files", [])
        )
    )


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


def test_build_model_bundle_success_manifest_checksum_and_machine_json(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)

    completed = _run_builder(tmp_path, repo, sources)

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["status"] == "success"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    checksum = json.loads((tmp_path / "checksum.json").read_text(encoding="utf-8"))
    assert _minimal_manifest_schema_valid(manifest)
    assert checksum == {"sha256": hashlib.sha256((tmp_path / "bundle.tar.gz").read_bytes()).hexdigest()}
    assert payload["sha256"] == checksum["sha256"]
    assert manifest["compatible_git"]["commits"] == [_git_output(repo, "rev-parse", "HEAD")]
    assert manifest["compatible_git"]["tags"] == ["vtest"]
    required = {entry["path"]: entry["sha256"] for entry in manifest["required_files"]}
    assert set(required) == {
        "tree/rf/model.joblib",
        "neural/model.pt",
        "hybrid/absorption_nm/model.joblib",
        "hybrid/emission_nm/model.joblib",
        "hybrid/quantum_yield/model.joblib",
    }
    for rel, expected in required.items():
        with tarfile.open(tmp_path / "bundle.tar.gz", "r:gz") as tar:
            extracted = tar.extractfile(rel)
            assert extracted is not None
            assert hashlib.sha256(extracted.read()).hexdigest() == expected
    manifest_text = json.dumps(manifest)
    assert str(tmp_path) not in manifest_text
    for name in (os.environ.get("USER"), os.environ.get("USERNAME")):
        if name:
            assert name not in manifest_text


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_build_model_bundle_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    first = _run_builder(tmp_path, repo, sources)
    assert first.returncode == 0, first.stderr
    first_bytes = (tmp_path / "bundle.tar.gz").read_bytes()
    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _run_builder(second_dir, repo, sources)
    assert second.returncode == 0, second.stderr
    assert first_bytes == (second_dir / "bundle.tar.gz").read_bytes()


def test_build_model_bundle_rejects_missing_and_empty_source_dirs(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    shutil.rmtree(sources["tree"])
    missing = _run_builder(tmp_path, repo, sources)
    assert missing.returncode == 1
    assert json.loads(missing.stdout)["errors"][0]["code"] == "SOURCE_DIR_MISSING"

    sources = _bundle_sources(tmp_path / "empty-case")
    shutil.rmtree(sources["tree"])
    sources["tree"].mkdir(parents=True)
    empty = _run_builder(tmp_path / "empty-output", repo, sources)
    assert empty.returncode == 1
    assert json.loads(empty.stdout)["errors"][0]["code"] == "SOURCE_DIR_EMPTY"


def test_build_model_bundle_rejects_symlink_and_unsafe_paths(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    target = sources["tree"] / "target.txt"
    target.write_text("target", encoding="utf-8")
    try:
        os.symlink(target, sources["tree"] / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not available")
    symlink = _run_builder(tmp_path, repo, sources)
    assert symlink.returncode == 1
    assert json.loads(symlink.stdout)["errors"][0]["code"] == "SYMLINK_REJECTED"

    (sources["tree"] / "link.txt").unlink()
    unsafe_name = sources["tree"] / "bad\\name.txt"
    unsafe_name.write_text("bad", encoding="utf-8")
    unsafe = _run_builder(tmp_path / "unsafe-output", repo, sources)
    if sys.platform == "win32":
        assert unsafe.returncode == 0
    else:
        assert unsafe.returncode == 1
        assert json.loads(unsafe.stdout)["errors"][0]["code"] == "UNSAFE_PATH"


def test_build_model_bundle_rejects_hardlinks_when_supported(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    source = sources["tree"] / "rf" / "model.joblib"
    hardlink = sources["tree"] / "rf" / "copy.joblib"
    try:
        os.link(source, hardlink)
    except (OSError, NotImplementedError):
        pytest.skip("hardlink creation is not available")

    completed = _run_builder(tmp_path, repo, sources)

    assert completed.returncode == 1
    assert json.loads(completed.stdout)["errors"][0]["code"] == "HARDLINK_REJECTED"


def test_build_model_bundle_refuses_existing_outputs_and_overwrites_explicitly(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    first = _run_builder(tmp_path, repo, sources)
    assert first.returncode == 0, first.stderr
    refused = _run_builder(tmp_path, repo, sources)
    assert refused.returncode == 1
    assert json.loads(refused.stdout)["errors"][0]["code"] == "OUTPUT_EXISTS"
    overwritten = _run_builder(tmp_path, repo, sources, "--overwrite")
    assert overwritten.returncode == 0, overwritten.stderr


def test_build_install_check_integration(tmp_path: Path) -> None:
    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    env = _fake_env(tmp_path)
    built = _run_builder(tmp_path, repo, sources)
    assert built.returncode == 0, built.stderr

    install_code, install_payload = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(tmp_path / "bundle.tar.gz"),
        "--checksum",
        str(tmp_path / "checksum.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--destination",
        str(tmp_path / "installed"),
    )
    check_code, check_payload = _run_json(
        CHECK_SCRIPT,
        "--repo-dir",
        str(repo),
        "--env-dir",
        str(env),
        "--artifact-dir",
        str(tmp_path / "installed"),
        "--expected-version",
        "vtest-artifacts",
        env={"FLUORCAST_SKIP_REMOTE_IMPORTS": "1"},
    )

    assert install_code == 0
    assert install_payload["action"] == "installed"
    assert check_code == 0
    assert check_payload["status"] == "success"


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
    written = json.loads(state.read_text(encoding="utf-8"))
    assert written["status"] == "ready"
    assert written["state_file"] == str(state.resolve())


def test_build_install_validate_fixture_integration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from scripts.remote import validate_production_install as validator

    repo = _fake_repo(tmp_path)
    sources = _bundle_sources(tmp_path)
    built = _run_builder(tmp_path, repo, sources)
    assert built.returncode == 0, built.stderr
    install_code, _ = _run_json(
        INSTALL_SCRIPT,
        "--archive",
        str(tmp_path / "bundle.tar.gz"),
        "--checksum",
        str(tmp_path / "checksum.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--destination",
        str(tmp_path / "installed"),
    )
    assert install_code == 0

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        output_path = Path(args[-1])
        output_path.write_text(
            json.dumps({"status": "success", "predictions": [{"predicted_emission_nm": 500.0}]}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(validator.subprocess, "run", fake_run)
    result, code = validator.validate(
        argparse_namespace(repo_dir=repo, artifact_dir=tmp_path / "installed", state_file=tmp_path / "state" / "install-state.json")
    )

    assert code == 0
    assert result["status"] == "success"


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


def _fake_submission_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "submit-repo"
    repo.mkdir()
    _git(repo, "init")
    (repo / "slurm" / "base_models").mkdir(parents=True)
    (repo / "slurm" / "production").mkdir(parents=True)
    (repo / "slurm" / "base_models" / "run_model_experiments_fluodb.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "base_models" / "run_neural_experiments.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "run_hybrid_three_way_experiment.sbatch").write_text("", encoding="utf-8")
    (repo / "slurm" / "production" / "validate_production_install.sbatch").write_text("", encoding="utf-8")
    _git(repo, "add", ".")
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=test@example.invalid", "-c", "user.name=Test User", "commit", "-m", "init"],
        check=True,
        capture_output=True,
        text=True,
    )
    env = tmp_path / "env" / "bin"
    env.mkdir(parents=True)
    activate = env / "activate"
    activate.write_text("", encoding="utf-8")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "count_file=\"$SBATCH_LOG.count\"\n"
        "count=0\n"
        "if [[ -f \"$count_file\" ]]; then count=$(cat \"$count_file\"); fi\n"
        "count=$((count + 1))\n"
        "printf '%s\\n' \"$count\" > \"$count_file\"\n"
        "printf '%s\\n' \"$*\" >> \"$SBATCH_LOG\"\n"
        "printf '1234%s\\n' \"$count\"\n",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    return repo, activate, fake_bin


def _run_submit(tmp_path: Path, repo: Path, activate: Path, fake_bin: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    bash = _bash()
    assert bash is not None
    env_vars = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "XDG_STATE_HOME": str(tmp_path / "xdg-state"),
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "SBATCH_LOG": str(tmp_path / "sbatch.args.log"),
    }
    return subprocess.run(
        [bash, str(SUBMIT_SCRIPT), "--account", "def-test", "--repo-dir", str(repo), "--env-activate", str(activate), *extra],
        env=env_vars,
        capture_output=True,
        text=True,
        check=False,
    )


def test_submit_training_defaults_to_molecule_split_and_external_state(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo, activate, fake_bin = _fake_submission_repo(tmp_path)

    completed = _run_submit(tmp_path, repo, activate, fake_bin)

    assert completed.returncode == 0, completed.stderr
    state = json.loads((tmp_path / "xdg-state" / "fluorcast" / "provisioning-state.json").read_text(encoding="utf-8"))
    assert state["split_type"] == "molecule"
    assert state["state_file"] == str((tmp_path / "xdg-state" / "fluorcast" / "provisioning-state.json").resolve())
    assert state["install_state_file"] == str((tmp_path / "xdg-state" / "fluorcast" / "install-state.json").resolve())
    args_log = (tmp_path / "sbatch.args.log").read_text(encoding="utf-8")
    hybrid_lines = [line for line in args_log.splitlines() if "run_hybrid_three_way_experiment.sbatch" in line]
    assert len(hybrid_lines) == 3
    assert all("FLUORCAST_SPLIT_TYPE=molecule" in line for line in hybrid_lines)
    assert _git_output(repo, "status", "--porcelain") == ""


def test_submit_training_accepts_explicit_scaffold_for_all_hybrid_jobs(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo, activate, fake_bin = _fake_submission_repo(tmp_path)

    completed = _run_submit(tmp_path, repo, activate, fake_bin, "--split-type", "scaffold")

    assert completed.returncode == 0, completed.stderr
    state = json.loads((tmp_path / "xdg-state" / "fluorcast" / "provisioning-state.json").read_text(encoding="utf-8"))
    assert state["split_type"] == "scaffold"
    hybrid_lines = [
        line for line in (tmp_path / "sbatch.args.log").read_text(encoding="utf-8").splitlines()
        if "run_hybrid_three_way_experiment.sbatch" in line
    ]
    assert len(hybrid_lines) == 3
    assert all("FLUORCAST_SPLIT_TYPE=scaffold" in line for line in hybrid_lines)


def test_submit_training_rejects_invalid_split_before_sbatch(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo, activate, fake_bin = _fake_submission_repo(tmp_path)

    completed = _run_submit(tmp_path, repo, activate, fake_bin, "--split-type", "random")

    assert completed.returncode == 2
    assert json.loads(completed.stdout.splitlines()[-1])["code"] == "INVALID_SPLIT_TYPE"
    assert not (tmp_path / "sbatch.args.log").exists()


def test_provision_environment_reports_missing_module_command(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("runtime bash path semantics are platform-specific on Windows")
    bash = _bash()
    if bash is None:
        pytest.skip("bash is not available")
    repo = tmp_path / "repo"
    (repo / "scripts" / "remote").mkdir(parents=True)
    (repo / "requirements.txt").write_text("", encoding="utf-8")
    (repo / "scripts" / "remote" / "nibi-python311-constraints.txt").write_text("", encoding="utf-8")
    completed = subprocess.run(
        [
            bash,
            "-c",
            'unset -f module 2>/dev/null || true; PATH="/usr/bin:/bin" "$1" --repo-dir "$2" --env-dir "$3"',
            "bash",
            str(PROVISION_SCRIPT),
            str(repo),
            str(tmp_path / "env"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout.splitlines()[-1])["code"] == "MODULE_COMMAND_MISSING"
    assert not (tmp_path / "env").exists()


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
