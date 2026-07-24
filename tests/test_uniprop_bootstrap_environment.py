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
RESOLVER_REPORT = PROJECT_ROOT / "src" / "chemfluor" / "uniprop" / "resolver_report.py"
REVISION = PROJECT_ROOT / "third_party" / "nablacolors.REVISION"
MANIFEST = PROJECT_ROOT / "configs" / "uniprop" / "checkpoint_manifest.json"
BUILD_ISOLATION_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_pip_build_isolation_failure.txt"
)
NUMPY_UNAVAILABLE_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_numpy_226_unavailable_failure.txt"
)
DISTUTILS_ASSERTION_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_distutils_hack_assertion_failure.txt"
)
PYDANTIC_CORE_MATURIN_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_pydantic_core_maturin_failure.txt"
)
WANDB_PYDANTIC_RESOLUTION_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_wandb_pydantic_resolution_failure.txt"
)
ENCODED_WHEEL_FALSE_FAILURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "uniprop_encoded_wheel_false_failure.txt"
)
UNICORE_RUNTIME_REQUIREMENTS = (
    PROJECT_ROOT / "configs" / "uniprop" / "unicore_runtime_requirements.txt"
)


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is not None and os.name == "nt" and "WindowsApps" in bash:
        for candidate in _git_bash_candidates():
            if candidate.exists():
                return str(candidate)
    if bash is None:
        pytest.skip("bash is not available")
    return bash


def _git_bash_candidates() -> list[Path]:
    return [
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    ]


def _has_git_bash() -> bool:
    return any(candidate.exists() for candidate in _git_bash_candidates())


def _bash_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    rest = resolved.as_posix().split(":", 1)[1]
    if _has_git_bash():
        return f"/{drive}{rest}"
    return f"/mnt/{drive}{rest}"


def _bash_cmd(*args: str | Path) -> list[str]:
    bash = _bash()
    converted = [_bash_path(arg) if isinstance(arg, Path) else arg for arg in args]
    if os.name == "nt" and "WindowsApps" in bash:
        wsl = shutil.which("wsl")
        if wsl is None:
            pytest.skip("WSL bash shim is present but wsl.exe is not available")
        distro_check = subprocess.run(
            [wsl, "--list", "--quiet"],
            capture_output=True,
            text=True,
        )
        if distro_check.returncode != 0 or not distro_check.stdout.strip():
            pytest.skip("WSL bash shim is present but no WSL distribution is installed")
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
    assert "numpy==2.2.2" in result.stdout
    assert "unicore_runtime_requirements.txt" in result.stdout
    assert "--only-binary=:all:" in result.stdout
    assert "numpy==2.2.6" not in BOOTSTRAP.read_text(encoding="utf-8")


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
    assert "numpy==2.2.2" in text
    assert "numpy==2.2.6" not in text
    assert "FLUORCAST_UNIPROP_NUMPY_REQUIREMENT" in text
    assert "UNICORE_INSTALL_ARGS=(\"$UNICORE_DIR\")" in text
    assert "UNICORE_RUNTIME_REQUIREMENTS=\"configs/uniprop/unicore_runtime_requirements.txt\"" in text
    unicore_install = (
        'env SETUPTOOLS_USE_DISTUTILS=stdlib "$VENV_PYTHON" -m pip install '
        '--no-build-isolation --no-deps "${UNICORE_INSTALL_ARGS[@]}"'
    )
    assert unicore_install in text
    assert "pip install -e \"$UNIMOL_PLUS_DIR\"" in text


def test_bootstrap_completion_diagnostic_requires_real_imports() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    required = [
        "import torch",
        "import numpy",
        "import unicore",
        "import unimol_plus",
        "import wandb",
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
    assert 'fail "No compatible NumPy wheel found' in text
    assert 'fail "NumPy validation failed."' in text
    assert 'pip install "$TORCH_SPEC"' in text
    assert 'fail "Uni-Core build prerequisites are unavailable."' in text
    assert 'fail "No compatible PyTorch wheel found' in text
    assert 'fail "PyTorch validation failed."' in text
    assert 'fail "No compatible binary candidate exists for one or more Uni-Core runtime dependencies."' in text
    assert 'fail "Uni-Core runtime dependency installation failed."' in text
    assert 'fail "Uni-Core installation failed."' in text
    assert 'fail "Uni-Core pip check failed."' in text
    assert 'fail "Uni-Mol+ installation failed."' in text
    assert 'fail "Final UniProp import diagnostic failed."' in text


def test_bootstrap_supports_partial_env_detection_and_clean_rebuild() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "--clean" in text
    assert 'rm -rf "$VENV_DIR"' in text
    assert "Partial UniProp environment detected" in text
    assert 'if [[ -e "$VENV_DIR" && ! -x "$VENV_PYTHON" ]]' in text


def test_cuda_mode_is_preserved_when_explicitly_supplied(tmp_path: Path) -> None:
    upstream = tmp_path / "nablacolors"
    report = tmp_path / "bootstrap.json"
    result = subprocess.run(
        [
            *_bash_cmd(BOOTSTRAP),
            "--dry-run",
            "--mode",
            "cuda",
            "--upstream-dir",
            _bash_path(upstream),
            "--json-output",
            _bash_path(report),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "cuda"
    assert "UniProp bootstrap mode: cuda" in result.stdout


def test_bootstrap_installs_numpy_before_torch_runtime_diagnostic() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    numpy_preflight = text.index('stage "NumPy availability preflight"')
    numpy_install = text.index('pip install "$NUMPY_SPEC"')
    torch_stage = text.index('stage "PyTorch"')
    torch_diagnostic = text.index("PyTorch version: {torch.__version__}")
    assert numpy_preflight < numpy_install < torch_stage < torch_diagnostic


def test_numpy_preflight_uses_structured_pip_report_before_pytorch() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    preflight = text.index('stage "NumPy availability preflight"')
    pytorch = text.index('stage "PyTorch"')
    assert "pip install --dry-run --report" in text[preflight:pytorch]
    assert "Selected NumPy wheel/version" in text[preflight:pytorch]
    assert "metadata" in text[preflight:pytorch]


def test_numpy_local_alliance_suffix_is_accepted_by_base_version() -> None:
    from packaging.version import Version

    assert Version("2.2.2+computecanada").base_version == "2.2.2"
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "Version(selected).base_version" in text
    assert "Version(numpy.__version__)" in text
    assert "installed.base_version != expected_base" in text


def test_unexpected_numpy_public_version_fails_validation() -> None:
    from packaging.version import Version

    assert Version("2.2.6").base_version != "2.2.2"
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "does not match requested" in text


def test_numpy_failures_stop_before_pytorch() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    numpy_preflight_failure = text.index('fail "No compatible NumPy wheel found')
    numpy_validation_failure = text.index('fail "NumPy validation failed."')
    pytorch = text.index('stage "PyTorch"')
    assert numpy_preflight_failure < pytorch
    assert numpy_validation_failure < pytorch


def test_cuda_path_rejects_cpu_only_torch() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "Installed torch is CPU-only; CUDA mode requires an Alliance CUDA-capable torch wheel." in text
    assert "getattr(torch.version, \"cuda\", None) is None" in text


def test_unicore_runtime_requirements_are_explicit_and_do_not_reinstall_torch_numpy() -> None:
    requirements = UNICORE_RUNTIME_REQUIREMENTS.read_text(encoding="utf-8").splitlines()
    assert requirements == [
        "lmdb",
        "tqdm",
        "ml_collections",
        "scipy",
        "tensorboardX",
        "tokenizers",
        "wandb==0.17.9",
    ]
    assert "torch" not in "\n".join(requirements).lower()
    assert "numpy" not in "\n".join(requirements).lower()


def test_unicore_runtime_dependencies_are_installed_before_local_unicore() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    runtime_preflight = text.index('stage "Uni-Core runtime dependency resolver preflight"')
    runtime_install = text.index('stage "Uni-Core runtime dependencies"')
    local_install = text.index('stage "Uni-Core direct install"')
    assert runtime_preflight < runtime_install < local_install


def test_unicore_runtime_dependency_install_is_wheel_only_and_isolated_normally() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    runtime_preflight = text.index('stage "Uni-Core runtime dependency resolver preflight"')
    runtime_install = text.index('stage "Uni-Core runtime dependencies"')
    diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    local_install = text.index('stage "Uni-Core direct install"')
    resolver_block = text[runtime_preflight:runtime_install]
    install_block = text[runtime_install:diagnostic]
    assert "--dry-run --report" in resolver_block
    assert "--only-binary=:all:" in resolver_block
    assert "--only-binary=:all:" in install_block
    assert "--no-build-isolation" not in resolver_block + install_block
    assert "SETUPTOOLS_USE_DISTUTILS" not in resolver_block + install_block


def test_unicore_runtime_report_validation_rejects_bad_resolution() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    helper = RESOLVER_REPORT.read_text(encoding="utf-8")
    preflight = text.index('stage "Uni-Core runtime dependency resolver preflight"')
    install = text.index('stage "Uni-Core runtime dependencies"')
    block = text[preflight:install]
    assert "validate_unicore_runtime_report_item" in block
    assert "decoded_filename=" in block
    assert "original_url=" in block
    assert "artifact_type=wheel" in block
    assert "parsed_wheel_name=" in block
    assert "parsed_wheel_version=" in block
    assert "wheel_has_local_version=" in block
    assert 'str(wandb_requirements[0].specifier) != "==0.17.9"' in block
    assert "selected wandb {version}; expected" in helper
    assert 'FORBIDDEN = {"pydantic", "pydantic-core", "pydantic_core", "maturin"}' in block
    assert 'REPLACE_FORBIDDEN = {"numpy", "torch"}' in block
    assert "would replace protected package" in helper
    assert 'marker_environment["extra"] = ""' in helper
    assert "declares forbidden dependency" in helper
    assert "alliance_wheelhouse=" in block


def test_runtime_dependency_install_remains_wheel_only_after_report_validation() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    runtime_install = text.index('stage "Uni-Core runtime dependencies"')
    build_diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    block = text[runtime_install:build_diagnostic]
    assert 'pip install --only-binary=:all: -r "$UNICORE_RUNTIME_REQUIREMENTS"' in block


def test_local_unicore_still_uses_no_build_isolation_and_no_deps() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    install_stage = text.index('stage "Uni-Core direct install"')
    post_install = text.index('stage "Uni-Core post-install validation"')
    block = text[install_stage:post_install]
    assert "--no-build-isolation --no-deps" in block


def test_encoded_wheel_false_failure_fixture_matches_real_regression() -> None:
    fixture = ENCODED_WHEEL_FALSE_FAILURE.read_text(encoding="utf-8")
    assert "Runtime dependency resolution selected a source distribution for requests:" in fixture
    assert "requests-2.34.2%2Bcomputecanada-py3-none-any.whl" in fixture


def test_wandb_0179_policy_avoids_normal_pydantic_dependency() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    requirements = UNICORE_RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
    assert "wandb==0.17.9" in requirements
    assert "wandb.__version__ != \"0.17.9\"" in text
    assert "launch" not in requirements
    assert "pydantic" not in requirements.lower()


def test_unicore_build_uses_environment_torch_without_isolation() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    install = text.index(
        'env SETUPTOOLS_USE_DISTUTILS=stdlib "$VENV_PYTHON" -m pip install '
        '--no-build-isolation --no-deps "${UNICORE_INSTALL_ARGS[@]}"'
    )
    assert diagnostic < install
    assert "import torch" in text[diagnostic:install]
    assert "Build isolation disabled: yes" in text[diagnostic:install]


def test_unicore_diagnostic_uses_metadata_before_setuptools_import() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    install = text.index('stage "CUDA toolkit compatibility"')
    block = text[diagnostic:install]
    metadata_lookup = block.index('setuptools_version = version("setuptools")')
    setuptools_import = block.index("import setuptools")
    assert "from importlib.metadata import version" in block
    assert "import pip" not in block
    assert metadata_lookup < setuptools_import


def test_unicore_compatibility_probe_receives_stdlib_distutils_env() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    install = text.index('stage "CUDA toolkit compatibility"')
    block = text[diagnostic:install]
    assert 'python_here_env "SETUPTOOLS_USE_DISTUTILS=stdlib"' in block
    assert 'os.environ.get("SETUPTOOLS_USE_DISTUTILS") != "stdlib"' in block
    assert "SETUPTOOLS_USE_DISTUTILS: {os.environ.get('SETUPTOOLS_USE_DISTUTILS')}" in block
    assert "distutils.core path:" in block


def test_unicore_probe_represents_upstream_torch_then_setuptools_order() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    diagnostic = text.index('stage "Uni-Core build prerequisite diagnostic"')
    install = text.index('stage "CUDA toolkit compatibility"')
    block = text[diagnostic:install]
    torch_import = block.index("import torch")
    setuptools_import = block.index("import setuptools")
    distutils_import = block.index("import distutils.core")
    assert torch_import < setuptools_import < distutils_import


def test_unicore_install_receives_scoped_stdlib_distutils_env() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    install_stage = text.index('stage "Uni-Core direct install"')
    import_validation = text.index('stage "Uni-Core post-install validation"')
    block = text[install_stage:import_validation]
    unicore_install = (
        'run_cmd env SETUPTOOLS_USE_DISTUTILS=stdlib "$VENV_PYTHON" '
        '-m pip install --no-build-isolation --no-deps "${UNICORE_INSTALL_ARGS[@]}"'
    )
    assert unicore_install in block


def test_setuptools_distutils_setting_is_scoped_to_unicore_subprocesses() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "export SETUPTOOLS_USE_DISTUTILS" not in text
    assert text.count('python_here_env "SETUPTOOLS_USE_DISTUTILS=stdlib"') == 1
    assert text.count("run_cmd env SETUPTOOLS_USE_DISTUTILS=stdlib") == 1
    assert "SETUPTOOLS_USE_DISTUTILS=stdlib" not in text[
        text.index('stage "Uni-Mol+ direct install"') :
    ]


def test_unicore_failure_prevents_unimol_plus_attempt() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    dependency_resolution = text.index('fail "No compatible binary candidate exists for one or more Uni-Core runtime dependencies."')
    dependency_install = text.index('fail "Uni-Core runtime dependency installation failed."')
    pip_check = text.index('fail "Uni-Core pip check failed."')
    unicore_install = text.index('fail "Uni-Core installation failed."')
    unicore_import = text.index('fail "Uni-Core required imports are unavailable."')
    unimol_install = text.index('stage "Uni-Mol+ direct install"')
    assert dependency_resolution < unimol_install
    assert dependency_install < unimol_install
    assert unicore_install < unimol_install
    assert pip_check < unimol_install
    assert unicore_import < unimol_install


def test_unicore_is_not_attempted_when_pytorch_fails() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    torch_validation = text.index('fail "PyTorch validation failed."')
    unicore_install = text.index('stage "Uni-Core direct install"')
    assert torch_validation < unicore_install


def test_build_isolation_failure_fixture_matches_real_regression() -> None:
    fixture = BUILD_ISOLATION_FAILURE.read_text(encoding="utf-8")
    assert "Getting requirements to build wheel: finished with status 'error'" in fixture
    assert "ModuleNotFoundError: No module named 'torch'" in fixture


def test_distutils_hack_assertion_fixture_matches_real_regression() -> None:
    fixture = DISTUTILS_ASSERTION_FAILURE.read_text(encoding="utf-8")
    assert "AssertionError:" in fixture
    assert "/python/3.10.13/lib/python3.10/distutils/core.py" in fixture


def test_numpy_unavailable_failure_fixture_matches_real_regression() -> None:
    fixture = NUMPY_UNAVAILABLE_FAILURE.read_text(encoding="utf-8")
    assert "Could not find a version that satisfies the requirement numpy==2.2.6" in fixture
    assert "No matching distribution found for numpy==2.2.6" in fixture


def test_pydantic_core_maturin_failure_fixture_matches_real_regression() -> None:
    fixture = PYDANTIC_CORE_MATURIN_FAILURE.read_text(encoding="utf-8")
    assert "pydantic_core-2.46.4.tar.gz" in fixture
    assert "BackendUnavailable: Cannot import 'maturin'" in fixture


def test_wandb_pydantic_resolution_fixture_matches_real_regression() -> None:
    fixture = WANDB_PYDANTIC_RESOLUTION_FAILURE.read_text(encoding="utf-8")
    assert "wandb-0.27.2+computecanada" in fixture
    assert "pydantic-2.13.4+computecanada" in fixture


def test_optional_cuda_extension_toolkit_guard_is_strict_only_when_requested() -> None:
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "Loaded CUDA toolkit version:" in text
    assert "Optional Uni-Core fused CUDA extensions are not being compiled" in text
    assert "does not match PyTorch compiled CUDA" in text


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
