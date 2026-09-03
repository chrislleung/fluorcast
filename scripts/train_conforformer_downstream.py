"""Train downstream FluorCast models from finalized ConforFormer embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.downstream import train_downstream  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/fluodb_lite/combined_deduplicated.csv"))
    parser.add_argument("--embedding-run-root", type=Path, required=True)
    parser.add_argument("--solvent-descriptors", type=Path, default=Path("data/solvent_descriptors_expanded_deep4chem.csv"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-out-dir", type=Path, required=True)
    parser.add_argument("--split-type", choices=["random", "molecule", "scaffold"], default="molecule")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_downstream(
        dataset_csv=args.dataset,
        embedding_run_root=args.embedding_run_root,
        solvent_descriptors=args.solvent_descriptors,
        out_dir=args.out_dir,
        model_out_dir=args.model_out_dir,
        split_type=args.split_type,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    print(f"training outputs written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

