"""Run the native-Windows UniProp smoke over processed FluorCast rows.

By default this writes the full summary to ``<output-dir>/summary.json``.
Passing ``--json-summary`` prints that full summary to stdout; passing
``--json-summary PATH`` also writes a copy to the explicit path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.windows_real_data_smoke import (  # noqa: E402
    DEFAULT_REAL_DATA_SMOKE_OUTPUT_DIR,
    run_windows_real_data_smoke,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Processed FluorCast dataset. Defaults to the authoritative resolver: "
            "combined_deduplicated_with_stokes.csv when present, otherwise combined_deduplicated.csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REAL_DATA_SMOKE_OUTPUT_DIR)
    parser.add_argument("--max-molecules", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--json-summary",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Print the full JSON summary. With PATH, also write that explicit file. "
            "Without PATH, the fixed summary path is <output-dir>/summary.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    max_molecules = args.max_molecules
    max_rows = args.max_rows
    if max_molecules is None and max_rows is None:
        max_molecules = 20
    try:
        summary = run_windows_real_data_smoke(
            args.output_dir,
            dataset=args.dataset,
            max_molecules=max_molecules,
            max_rows=max_rows,
            seed=args.seed,
            workers=max(1, int(args.workers)),
            resume=args.resume,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json_summary is not None:
        if args.json_summary:
            explicit_path = Path(args.json_summary)
            explicit_path.parent.mkdir(parents=True, exist_ok=True)
            explicit_path.write_text(
                json.dumps(summary, indent=2, allow_nan=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(summary, indent=2, allow_nan=False, sort_keys=True))
    else:
        print(
            json.dumps(
                {
                    "profile": summary["profile"],
                    "all_stages_passed": summary["all_stages_passed"],
                    "selected_rows": summary["selected_rows"],
                    "successful_geometry_count": summary["successful_geometry_count"],
                    "failed_geometry_count": summary["failed_geometry_count"],
                    "summary": summary["artifacts"]["summary"],
                },
                sort_keys=True,
            )
        )
    return 0 if summary["all_stages_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
