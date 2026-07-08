from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "catboost_info",
    "data",
    "development_results",
    "hybrid predictions",
    "models",
    "outputs",
    "slurm",
}


def _non_slurm_files() -> list[Path]:
    files: list[Path] = []
    for path in PROJECT_ROOT.rglob("*"):
        relative = path.relative_to(PROJECT_ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_no_root_level_files_contain_sbatch() -> None:
    directive = "#" + "SBATCH"
    offenders = [
        path
        for path in _non_slurm_files()
        if "slurm" not in path.relative_to(PROJECT_ROOT).parts
        and directive in path.read_text(encoding="utf-8", errors="ignore")
    ]
    assert offenders == []


def test_active_slurm_scripts_pass_bash_syntax_check() -> None:
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    bash = str(git_bash) if sys.platform == "win32" and git_bash.exists() else shutil.which("bash")
    if bash is None:
        return
    scripts = sorted((PROJECT_ROOT / "slurm").rglob("*.sbatch"))
    assert scripts
    for script in scripts:
        completed = subprocess.run(
            [bash, "-n", str(script)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, f"{script}: {completed.stderr}"
