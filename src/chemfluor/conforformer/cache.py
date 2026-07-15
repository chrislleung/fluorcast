"""Cache-key builders and local conformer-cache serialization."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

from rdkit import rdBase

from .config import ConformerGenerationConfig
from .schemas import MoleculeConformerCacheRecord


CONFORMER_CACHE_SCHEMA_VERSION = 1


class CacheError(RuntimeError):
    """Raised when a conformer cache file cannot be trusted or used."""


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def build_conformer_cache_key(
    *,
    canonical_smiles: str | None,
    isomeric_canonical_smiles: str | None,
    config: ConformerGenerationConfig,
    rdkit_version: str | None = None,
) -> str:
    payload = {
        "canonical_smiles": canonical_smiles,
        "configuration": config.to_payload(),
        "isomeric_canonical_smiles": isomeric_canonical_smiles,
        "rdkit_version": rdkit_version or rdBase.rdkitVersion,
    }
    return sha256_payload(payload)


@dataclass(frozen=True)
class EmbeddingCacheKeyPayload:
    conformer_cache_key: str
    upstream_commit: str
    checkpoint_sha256: str
    dictionary_sha256: str
    architecture_payload: dict[str, Any]
    hydrogen_removal_policy: str
    unknown_atom_policy: str = "fail"
    preprocessing_version: str = "fluorcast-conforformer-preprocess-v1"

    def to_payload(self) -> dict[str, Any]:
        return {
            "architecture_payload": self.architecture_payload,
            "checkpoint_sha256": self.checkpoint_sha256,
            "conformer_cache_key": self.conformer_cache_key,
            "dictionary_sha256": self.dictionary_sha256,
            "hydrogen_removal_policy": self.hydrogen_removal_policy,
            "preprocessing_version": self.preprocessing_version,
            "unknown_atom_policy": self.unknown_atom_policy,
            "upstream_commit": self.upstream_commit,
        }


@dataclass(frozen=True)
class PoolingCacheKeyPayload:
    embedding_cache_key: str
    pooling_method: str
    temperature_kelvin: float | None = None
    pooling_implementation_version: str = "fluorcast-conforformer-pooling-v1"
    pooling_parameters: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "embedding_cache_key": self.embedding_cache_key,
            "pooling_implementation_version": self.pooling_implementation_version,
            "pooling_method": self.pooling_method,
            "pooling_parameters": self.pooling_parameters,
            "temperature_kelvin": self.temperature_kelvin,
        }


def build_embedding_cache_key(payload: EmbeddingCacheKeyPayload) -> str:
    return sha256_payload(payload.to_payload())


def build_pooling_cache_key(payload: PoolingCacheKeyPayload) -> str:
    return sha256_payload(payload.to_payload())


def conformer_cache_path(output_dir: Path, conformer_cache_key: str) -> Path:
    return Path(output_dir) / f"{conformer_cache_key}.json"


def _wrapped_record(record: MoleculeConformerCacheRecord) -> dict[str, Any]:
    record_payload = record.to_payload()
    return {
        "payload_sha256": sha256_payload(record_payload),
        "record": record_payload,
        "schema_version": CONFORMER_CACHE_SCHEMA_VERSION,
    }


def save_conformer_cache_record(
    record: MoleculeConformerCacheRecord,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = conformer_cache_path(output_dir, record.conformer_cache_key)
    if path.exists() and not overwrite:
        load_conformer_cache_record(path, expected_cache_key=record.conformer_cache_key)
        raise FileExistsError(f"valid cache record already exists: {path}")

    wrapped = _wrapped_record(record)
    text = stable_json_dumps(wrapped) + "\n"
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            delete=False,
            prefix=f".{record.conformer_cache_key}.",
            suffix=".tmp",
        ) as handle:
            temp_name = handle.name
            handle.write(text)
            handle.flush()
        Path(temp_name).replace(path)
    except Exception:
        if temp_name is not None:
            temp_path = Path(temp_name)
            if temp_path.exists():
                temp_path.unlink()
        raise
    return path


def load_conformer_cache_record(
    path: Path,
    *,
    expected_cache_key: str | None = None,
) -> MoleculeConformerCacheRecord:
    path = Path(path)
    try:
        wrapped = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CacheError(f"failed to read conformer cache metadata: {path}") from exc

    if wrapped.get("schema_version") != CONFORMER_CACHE_SCHEMA_VERSION:
        raise CacheError(f"unsupported conformer cache schema version in {path}")
    record_payload = wrapped.get("record")
    if not isinstance(record_payload, dict):
        raise CacheError(f"missing conformer cache record payload in {path}")
    expected_hash = wrapped.get("payload_sha256")
    if expected_hash != sha256_payload(record_payload):
        raise CacheError(f"conformer cache payload hash mismatch in {path}")
    record = MoleculeConformerCacheRecord.from_payload(record_payload)
    if expected_cache_key is not None and record.conformer_cache_key != expected_cache_key:
        raise CacheError("loaded conformer cache key does not match expected key")
    return record
