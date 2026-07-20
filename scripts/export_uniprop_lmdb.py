"""Export FluorCast row manifests plus cached geometries to UniProp LMDB files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.lmdb_export import (  # noqa: E402
    DEFAULT_TARGET_COLUMNS,
    ExportConfig,
    export_uniprop_lmdb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row-manifest", type=Path, default=Path("data/processed/uniprop/row_manifest.csv"))
    parser.add_argument("--molecule-manifest", type=Path, default=Path("data/processed/uniprop/molecule_manifest.csv"))
    parser.add_argument("--split-assignments", type=Path, default=Path("data/processed/uniprop/split_assignments.csv"))
    parser.add_argument("--geometry-cache-dir", type=Path, default=Path("data/processed/uniprop/geometry_cache"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/uniprop/lmdb"))
    parser.add_argument("--split-family", default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--map-size", type=int, default=10 * 1024 * 1024 * 1024)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_columns = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    try:
        metadata = export_uniprop_lmdb(
            ExportConfig(
                row_manifest_path=args.row_manifest,
                molecule_manifest_path=args.molecule_manifest,
                split_assignments_path=args.split_assignments,
                geometry_cache_dir=args.geometry_cache_dir,
                output_dir=args.out_dir,
                split_family=args.split_family,
                seed=args.seed,
                target_columns=target_columns,
                map_size=args.map_size,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
                resume=args.resume,
                valid_size=args.valid_size,
            )
        )
    except (FileNotFoundError, FileExistsError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(metadata["row_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
