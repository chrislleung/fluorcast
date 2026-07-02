"""Render deterministic JSON and Markdown reports from all-model predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.hybrid.report import (  # noqa: E402
    build_hybrid_report,
    load_prediction_table,
    render_report_markdown,
    write_report_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a hybrid FluorCast prediction report.")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--molecule-smiles")
    parser.add_argument("--solvent-smiles")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    table = load_prediction_table(args.prediction_csv)
    report = build_hybrid_report(table, args.molecule_smiles, args.solvent_smiles)
    write_report_json(report, args.out_json)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_report_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
