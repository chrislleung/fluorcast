"""Build deterministic conformer cache records for one inventory shard."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.cache import CacheError, conformer_cache_path, load_conformer_cache_record, save_conformer_cache_record  # noqa: E402
from chemfluor.conforformer.config import ConformerGenerationConfig  # noqa: E402
from chemfluor.conforformer.conformers import generate_conformer_cache_record  # noqa: E402
from chemfluor.conforformer.embedding_store import status_path  # noqa: E402
from chemfluor.conforformer.inventory import atomic_write_text, load_inventory, sha256_payload  # noqa: E402
from chemfluor.conforformer.schemas import MoleculeStatus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--conformer-cache-dir", type=Path, default=None)
    parser.add_argument("--num-conformers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inventory, _manifest = load_inventory(args.run_root)
    shard_rows = inventory[inventory["shard_index"] == args.shard_index].reset_index(drop=True)
    if shard_rows.empty:
        raise SystemExit(f"empty or unknown shard index: {args.shard_index}")
    cache_dir = args.conformer_cache_dir or (args.run_root / "conformer_cache")
    config = ConformerGenerationConfig.from_overrides(num_conformers=args.num_conformers, random_seed=args.seed)
    records: list[dict[str, object]] = []
    for row in shard_rows.itertuples(index=False):
        smiles = str(row.canonical_chromophore_smiles)
        record = generate_conformer_cache_record(smiles, chromophore_id=str(row.molecule_id), config=config)
        path = conformer_cache_path(cache_dir, record.conformer_cache_key)
        cache_hit = False
        if path.exists():
            try:
                load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key)
                cache_hit = True
            except CacheError:
                save_conformer_cache_record(record, cache_dir, overwrite=True)
        else:
            save_conformer_cache_record(record, cache_dir)
        records.append(
            {
                "molecule_id": row.molecule_id,
                "canonical_chromophore_smiles": smiles,
                "success": record.status == MoleculeStatus.OK,
                "terminal_failure": record.status == MoleculeStatus.FAILED,
                "failure_code": record.failure_reason,
                "failure_message": record.failure_reason,
                "conformer_cache_key": record.conformer_cache_key,
                "conformer_cache_path": str(path),
                "num_conformers": record.successful_conformer_count,
                "optimizer_metadata": record.metadata,
                "configuration_hash": sha256_payload(config.to_payload()),
                "cache_hit": cache_hit,
            }
        )
    payload = {"schema_version": 1, "shard_index": args.shard_index, "records": records}
    atomic_write_text(status_path(args.run_root, args.shard_index), json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {status_path(args.run_root, args.shard_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

