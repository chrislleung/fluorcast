from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOTS = [
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "NOTEBOOKS_GUIDE.md",
    PROJECT_ROOT / "docs",
    PROJECT_ROOT / "development md",
]


def _docs() -> list[Path]:
    result: list[Path] = []
    for root in DOC_ROOTS:
        if root.is_file():
            result.append(root)
        elif root.exists():
            result.extend(root.glob("*.md"))
    return result


def test_docs_do_not_use_root_sbatch_commands() -> None:
    offenders = []
    for path in _docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "sbatch run_" in text:
            offenders.append(path)
    assert offenders == []


def test_docs_do_not_point_to_root_run_sh_scripts() -> None:
    root_run_script = re.compile(r"(?<![/\\\w-])run_[A-Za-z0-9_ -]+\.sh")
    offenders = []
    for path in _docs():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if root_run_script.search(text):
            offenders.append(path)
        if "test_nibi_gpu.sh" in text:
            offenders.append(path)
    assert offenders == []
