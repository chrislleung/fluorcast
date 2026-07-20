"""Build the deterministic one-geometry-per-molecule UniProp RDKit cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.geometry_cache import (  # noqa: E402
    DEFAULT_CACHE_DIR,
    DEFAULT_MANIFEST,
    build_geometry_cache,
    load_molecule_manifest,
    status_summary,
    write_failure_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--molecule-id", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-invalid", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--retain-hydrogens", action="store_true")
    parser.add_argument("--mmff-variant", choices=["MMFF94", "MMFF94s"], default="MMFF94s")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--failure-json", type=Path, default=Path("outputs/uniprop_geometry_failures.json"))
    parser.add_argument("--failure-csv", type=Path, default=Path("outputs/uniprop_geometry_failures.csv"))
    parser.add_argument("--status-json", type=Path, default=Path("outputs/uniprop_geometry_status.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_molecule_manifest(args.molecule_manifest)
        results = build_geometry_cache(
            args.molecule_manifest,
            args.cache_dir,
            limit=args.limit,
            molecule_ids=args.molecule_id,
            workers=max(1, int(args.workers)),
            resume=args.resume,
            overwrite_invalid=args.overwrite_invalid,
            fail_fast=args.fail_fast,
            mmff_variant=args.mmff_variant,
            remove_hydrogens=not args.retain_hydrogens,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        expected = len(results) if args.limit or args.molecule_id or args.shard_index is not None else len(manifest)
        summary = status_summary(results, expected_total=expected)
        write_failure_reports(results, args.failure_json, args.failure_csv)
        args.status_json.parent.mkdir(parents=True, exist_ok=True)
        args.status_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    failed = summary["status_counts"].get("failed", 0) + summary["status_counts"].get("invalid_cache", 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
