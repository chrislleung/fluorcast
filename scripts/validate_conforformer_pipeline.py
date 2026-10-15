"""Validate ConforFormer inventory, shard, finalization, and downstream artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.inventory import load_inventory, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-run-root", type=Path, required=True)
    parser.add_argument("--downstream-run-root", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory, manifest = load_inventory(args.embedding_run_root)
    final_manifest = args.embedding_run_root / "embedding_manifest.json"
    if not final_manifest.exists():
        raise SystemExit(f"missing finalized embedding manifest: {final_manifest}")
    payload = json.loads(final_manifest.read_text(encoding="utf-8"))
    if int(payload["inventory_molecule_count"]) != len(inventory):
        raise SystemExit("final manifest molecule count does not match inventory")
    if args.downstream_run_root is not None:
        required = ["split_assignments.csv", "leakage_check.json", "excluded_rows.csv", "metrics.csv", "training_manifest.json"]
        missing = [name for name in required if not (args.downstream_run_root / name).exists()]
        if missing:
            raise SystemExit(f"downstream run is missing files: {missing}")
    print("pipeline validation OK")
    print(f"inventory_sha256={manifest['inventory_csv_sha256']}")
    print(f"embedding_manifest_sha256={sha256_file(final_manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

