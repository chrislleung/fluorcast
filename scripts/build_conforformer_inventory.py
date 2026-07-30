"""Build the full-dataset ConforFormer molecule inventory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.inventory import DEFAULT_DATASET, DEFAULT_SHARD_SIZE, build_inventory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(os.environ.get("FLUORCAST_DATASET", DEFAULT_DATASET)))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=int(os.environ.get("FLUORCAST_SHARD_SIZE", DEFAULT_SHARD_SIZE)))
    parser.add_argument("--max-molecules", type=int, default=(int(os.environ["FLUORCAST_MAX_MOLECULES"]) if os.environ.get("FLUORCAST_MAX_MOLECULES") else None))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_inventory(
        source_csv=args.dataset,
        output_dir=args.run_root / "inventory",
        shard_size=args.shard_size,
        max_molecules=args.max_molecules,
        git_root=PROJECT_ROOT,
    )
    print(f"inventory={result.inventory_path}")
    print(f"manifest={result.manifest_path}")
    print(f"shard_count={result.manifest['shard_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

