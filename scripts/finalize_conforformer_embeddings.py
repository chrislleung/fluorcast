"""Finalize and validate all ConforFormer embedding shards."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.adapter import inspect_assets  # noqa: E402
from chemfluor.conforformer.config import ConformerGenerationConfig  # noqa: E402
from chemfluor.conforformer.embedding_store import expected_identity, finalize_embeddings  # noqa: E402
from chemfluor.conforformer.inventory import load_inventory  # noqa: E402
from chemfluor.conforformer.preprocess import ConforFormerPreprocessingConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--fake-architecture", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _inventory, inventory_manifest = load_inventory(args.run_root)
    if args.fake_architecture:
        from chemfluor.conforformer.adapter import sha256_file
        from chemfluor.conforformer.dictionary import load_conforformer_dictionary
        dictionary = load_conforformer_dictionary(args.dictionary)
        checkpoint_sha = sha256_file(args.checkpoint)
        upstream = (PROJECT_ROOT / "configs" / "conforformer" / "upstream_commit.txt").read_text(encoding="utf-8").strip()
        arch = {"encoder_embed_dim": 512, "source": "fake_adapter"}
    else:
        dictionary, checkpoint, compatibility = inspect_assets(args.dictionary, args.checkpoint)
        checkpoint_sha = checkpoint.checkpoint_sha256
        upstream = (PROJECT_ROOT / "configs" / "conforformer" / "upstream_commit.txt").read_text(encoding="utf-8").strip()
        arch = compatibility.architecture.with_defaults().__dict__
    identity = expected_identity(
        inventory_manifest=inventory_manifest,
        checkpoint_sha256=checkpoint_sha,
        dictionary_sha256=dictionary.sha256,
        upstream_commit=upstream,
        architecture_payload=arch,
        preprocessing_payload=ConforFormerPreprocessingConfig().to_payload(),
        conformer_config_payload=ConformerGenerationConfig().to_payload(),
    )
    manifest = finalize_embeddings(run_root=args.run_root, identity=identity)
    print(f"finalized {manifest['success_count']} successes and {manifest['terminal_failure_count']} failures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

