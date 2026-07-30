"""Embed one ConforFormer inventory shard into an atomic NPZ artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from chemfluor.conforformer.adapter import ConforFormerEncoderAdapter, inspect_assets, sha256_file  # noqa: E402
from chemfluor.conforformer.cache import load_conformer_cache_record  # noqa: E402
from chemfluor.conforformer.config import ConformerGenerationConfig  # noqa: E402
from chemfluor.conforformer.dictionary import load_conforformer_dictionary  # noqa: E402
from chemfluor.conforformer.embedding_store import expected_identity, shard_is_complete, write_embedding_shard  # noqa: E402
from chemfluor.conforformer.inventory import load_inventory  # noqa: E402
from chemfluor.conforformer.preprocess import ConforFormerPreprocessingConfig, collate_preprocessed_conformers, preprocess_successful_conformers  # noqa: E402
from chemfluor.conforformer.schemas import MoleculeStatus  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path, required=True)
    parser.add_argument("--conformer-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fake-adapter", action="store_true")
    return parser.parse_args()


def _fake_embeddings(ids: list[str]) -> np.ndarray:
    rows = []
    for conformer_id in ids:
        seed = int(hashlib.sha256(conformer_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        rows.append(rng.normal(0, 1, 512).astype(np.float32))
    return np.vstack(rows)


def _read_upstream_commit() -> str:
    return (PROJECT_ROOT / "configs" / "conforformer" / "upstream_commit.txt").read_text(encoding="utf-8").strip()


def build_embedding_identity_and_dictionary(
    *,
    inventory_manifest: dict,
    checkpoint_path: Path,
    dictionary_path: Path,
    fake_adapter: bool,
    preprocess_config: ConforFormerPreprocessingConfig,
    conformer_config: ConformerGenerationConfig,
) -> tuple[dict, object]:
    if fake_adapter:
        dictionary = load_conforformer_dictionary(dictionary_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        dictionary_sha256 = dictionary.sha256
        architecture_payload = {"encoder_embed_dim": 512, "source": "fake_adapter"}
    else:
        dictionary, checkpoint, compatibility = inspect_assets(dictionary_path, checkpoint_path)
        checkpoint_sha256 = checkpoint.checkpoint_sha256
        dictionary_sha256 = dictionary.sha256
        architecture_payload = compatibility.architecture.with_defaults().__dict__
    identity = expected_identity(
        inventory_manifest=inventory_manifest,
        checkpoint_sha256=checkpoint_sha256,
        dictionary_sha256=dictionary_sha256,
        upstream_commit=_read_upstream_commit(),
        architecture_payload=architecture_payload,
        preprocessing_payload=preprocess_config.to_payload(),
        conformer_config_payload=conformer_config.to_payload(),
    )
    return identity, dictionary


def main() -> int:
    args = parse_args()
    inventory, inventory_manifest = load_inventory(args.run_root)
    shard_rows = inventory[inventory["shard_index"] == args.shard_index].reset_index(drop=True)
    if shard_rows.empty:
        raise SystemExit(f"empty or unknown shard index: {args.shard_index}")
    cache_dir = args.conformer_cache_dir or (args.run_root / "conformer_cache")
    conformer_config = ConformerGenerationConfig()
    preprocess_config = ConforFormerPreprocessingConfig()

    identity, dictionary = build_embedding_identity_and_dictionary(
        inventory_manifest=inventory_manifest,
        checkpoint_path=args.checkpoint,
        dictionary_path=args.dictionary,
        fake_adapter=args.fake_adapter,
        preprocess_config=preprocess_config,
        conformer_config=conformer_config,
    )
    if shard_is_complete(args.run_root, args.shard_index, expected_molecule_count=len(shard_rows), identity=identity):
        print(f"shard {args.shard_index} already complete")
        return 0
    adapter = (
        None
        if args.fake_adapter
        else ConforFormerEncoderAdapter(dictionary_path=args.dictionary, checkpoint_path=args.checkpoint, device=args.device, root=PROJECT_ROOT)
    )

    conformer_ids_by_molecule: list[list[str]] = []
    embeddings_by_molecule: list[np.ndarray | None] = []
    energies_by_molecule: list[np.ndarray | None] = []
    failure_codes: list[str | None] = []
    failure_messages: list[str | None] = []
    pending_records: list[tuple[int, object]] = []
    flat_preprocessed = []
    for molecule_idx, row in enumerate(shard_rows.itertuples(index=False)):
        try:
            from chemfluor.conforformer.cache import build_conformer_cache_key, conformer_cache_path
            from chemfluor.conforformer.conformers import canonicalize_smiles
            canonical, isomeric = canonicalize_smiles(str(row.canonical_chromophore_smiles))
            key = build_conformer_cache_key(canonical_smiles=canonical, isomeric_canonical_smiles=isomeric, config=conformer_config)
            record = load_conformer_cache_record(conformer_cache_path(cache_dir, key), expected_cache_key=key)
            if record.status != MoleculeStatus.OK:
                raise ValueError(record.failure_reason or "conformer_generation_failed")
            preprocessed = preprocess_successful_conformers(record, dictionary, preprocess_config)
            pending_records.extend((molecule_idx, p) for p in preprocessed)
            flat_preprocessed.extend(preprocessed)
            conformer_ids_by_molecule.append([p.conformer_id for p in preprocessed])
            energies_by_molecule.append(np.asarray([c.energy if c.energy is not None else np.nan for c in record.conformer_records if c.is_successful], dtype=np.float64))
            embeddings_by_molecule.append(None)
            failure_codes.append(None)
            failure_messages.append(None)
        except Exception as exc:
            conformer_ids_by_molecule.append([])
            embeddings_by_molecule.append(None)
            energies_by_molecule.append(None)
            failure_codes.append(getattr(exc, "reason_code", type(exc).__name__))
            failure_messages.append(str(exc))

    per_molecule: dict[int, list[np.ndarray]] = {idx: [] for idx in range(len(shard_rows))}
    batch_size = args.batch_size
    cursor = 0
    while cursor < len(flat_preprocessed):
        batch = flat_preprocessed[cursor : cursor + batch_size]
        try:
            emb = _fake_embeddings([r.conformer_id for r in batch]) if adapter is None else adapter.encode(collate_preprocessed_conformers(batch, dictionary)).embedding_array.astype(np.float32)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and batch_size > 1:
                batch_size = max(1, batch_size // 2)
                continue
            raise
        for offset, vector in enumerate(emb):
            molecule_idx = pending_records[cursor + offset][0]
            per_molecule[molecule_idx].append(vector)
        cursor += len(batch)
    for idx, vectors in per_molecule.items():
        if vectors:
            embeddings_by_molecule[idx] = np.vstack(vectors).astype(np.float32)
    write_embedding_shard(
        run_root=args.run_root,
        shard_index=args.shard_index,
        rows=shard_rows,
        conformer_ids_by_molecule=conformer_ids_by_molecule,
        embeddings_by_molecule=embeddings_by_molecule,
        energies_by_molecule=energies_by_molecule,
        failure_codes=failure_codes,
        failure_messages=failure_messages,
        identity=identity,
    )
    print(f"embedded shard {args.shard_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
