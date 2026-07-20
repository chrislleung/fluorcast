"""Validate FluorCast UniProp LMDB exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.uniprop.lmdb_export import DEFAULT_TARGET_COLUMNS, validate_lmdb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lmdb", type=Path)
    parser.add_argument("--row-manifest", type=Path, default=None)
    parser.add_argument("--split-assignments", type=Path, default=None)
    parser.add_argument("--split-family", default=None)
    parser.add_argument("--partition", choices=["train", "valid", "test"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--json-output", type=Path, default=None)
    return parser.parse_args()


def expected_row_ids(args: argparse.Namespace) -> set[str] | None:
    if args.row_manifest is None:
        return None
    if args.split_assignments is None or args.split_family is None or args.partition is None:
        raise ValueError("--split-assignments, --split-family, and --partition are required with --row-manifest")
    import pandas as pd

    rows = pd.read_csv(args.row_manifest)
    splits = pd.read_csv(args.split_assignments)
    merged = rows[["row_id"]].merge(splits[["row_id", args.split_family]], on="row_id", how="left")
    merged["lmdb_partition"] = merged[args.split_family].map(
        lambda value: "test" if value == "test" else "train"
    )
    train_rows = merged[merged["lmdb_partition"] == "train"].copy()
    if len(train_rows) and args.valid_size > 0:
        import hashlib

        n_valid = max(1, int(round(len(train_rows) * args.valid_size)))
        scored = train_rows[["row_id"]].copy()
        scored["_score"] = scored["row_id"].map(
            lambda row_id: hashlib.sha256(
                f"fluorcast_uniprop_lmdb_v1|valid|{args.seed}|{row_id}".encode("utf-8")
            ).hexdigest()
        )
        valid_ids = set(
            scored.sort_values(["_score", "row_id"], kind="mergesort")
            .head(n_valid)["row_id"]
            .astype(str)
        )
        merged.loc[merged["row_id"].astype(str).isin(valid_ids), "lmdb_partition"] = "valid"
    return set(merged.loc[merged["lmdb_partition"] == args.partition, "row_id"].astype(str))


def main() -> int:
    args = parse_args()
    target_columns = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    try:
        report = validate_lmdb(
            args.lmdb,
            expected_row_ids=expected_row_ids(args),
            target_columns=target_columns,
        )
    except (FileNotFoundError, ImportError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
