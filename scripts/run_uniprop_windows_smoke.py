"""Run the native-Windows UniProp integration smoke profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.windows_smoke import run_windows_smoke  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--json-summary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_windows_smoke(args.output_dir, seed=args.seed, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_summary:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "profile": summary["profile"],
                    "all_stages_passed": summary["all_stages_passed"],
                    "summary": summary["artifacts"]["summary"],
                },
                sort_keys=True,
            )
        )
    return 0 if summary["all_stages_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
