"""Optional ConforFormer encoder adapter.

This module is intentionally import-light: PyTorch, Uni-Core, LMDB, and the
pinned upstream ConforFormer modules are imported only when checkpoint
inspection, model construction, or tensor inference actually needs them.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path
import platform
import sys
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import numpy as np

from .dictionary import ConforFormerDictionary, REQUIRED_SPECIAL_TOKENS, load_conforformer_dictionary
from .preprocess import CollatedConformerBatch, PreprocessedConformerRecord


AUDITED_ARCHITECTURE: dict[str, Any] = {
    "activation_fn": "gelu",
    "architecture_name": "contrast",
    "encoder_attention_heads": 64,
    "encoder_embed_dim": 512,
    "encoder_ffn_embed_dim": 2048,
    "encoder_layers": 15,
    "max_seq_len": 512,
    "model_name": "contrast",
    "mode": "infer",
}
UPSTREAM_RELATIVE_PATH = Path("third_party") / "ConforFormer" / "unimol"
AFFECTED_UPSTREAM_COMMIT = "f3095c5ea0218b6b4b2780cd1f43122410e80a7a"
HMDB_MODULE_NAME = "unimol.data.HugeMDB_dataset"
OMOL_MODULE_NAME = "unimol.data.OMol_dataset"
_COMPAT_PLACEHOLDER_ATTR = "__fluorcast_conforformer_compat_placeholder__"


class AdapterError(RuntimeError):
    """Base error with a machine-readable reason code."""

    reason_code = "adapter_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


class MissingDependencyError(AdapterError):
    reason_code = "missing_dependency"


class AssetUnavailableError(AdapterError):
    reason_code = "missing_asset"


class CompatibilityError(AdapterError):
    reason_code = "compatibility_mismatch"


class InputValidationError(AdapterError):
    reason_code = "input_validation"


class InferenceError(AdapterError):
    reason_code = "inference_failure"


@dataclass(frozen=True)
class DependencyReport:
    pytorch_available: bool
    unicore_available: bool
    upstream_available: bool
    lmdb_available: bool
    native_windows: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpstreamImportCompatibilityDiagnostics:
    hmdb_shim_applied: bool
    hmdb_reason: str
    omol_shim_applied: bool
    omol_reason: str
    upstream_commit: str
    upstream_import_succeeded: bool


@dataclass(frozen=True)
class ArchitectureMetadata:
    architecture_name: str = "contrast"
    model_name: str = "contrast"
    encoder_layers: int | None = None
    encoder_embed_dim: int | None = None
    encoder_ffn_embed_dim: int | None = None
    encoder_attention_heads: int | None = None
    max_seq_len: int | None = None
    gaussian_basis: int | None = None
    token_embedding_vocab_size: int | None = None
    edge_type_vocab_size: int | None = None
    source: str = "audited_defaults"

    def with_defaults(self) -> "ArchitectureMetadata":
        payload = asdict(self)
        for field_name, default_key in [
            ("encoder_layers", "encoder_layers"),
            ("encoder_embed_dim", "encoder_embed_dim"),
            ("encoder_ffn_embed_dim", "encoder_ffn_embed_dim"),
            ("encoder_attention_heads", "encoder_attention_heads"),
            ("max_seq_len", "max_seq_len"),
        ]:
            if payload[field_name] is None:
                payload[field_name] = AUDITED_ARCHITECTURE[default_key]
        return ArchitectureMetadata(**payload)


@dataclass(frozen=True)
class CheckpointInspection:
    checkpoint_path: Path
    checkpoint_sha256: str
    file_size_bytes: int
    top_level_keys: tuple[str, ...]
    has_model: bool
    has_args: bool
    has_cfg: bool
    state_dict_key: str | None
    tensor_shapes: dict[str, tuple[int, ...]]
    architecture: ArchitectureMetadata
    inferred_vocab_size: int | None
    inferred_embedding_dim: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompatibilityReport:
    dictionary_path: Path
    dictionary_sha256: str
    dictionary_source_vocab_size: int
    dictionary_vocab_size: int
    checkpoint_sha256: str
    compatible: bool
    architecture: ArchitectureMetadata
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdapterResult:
    chromophore_id: str
    canonical_smiles: str | None
    conformer_ids: tuple[str, ...]
    conformer_cache_key: str
    dictionary_sha256: str
    checkpoint_sha256: str
    upstream_commit: str
    architecture_metadata: dict[str, Any]
    device: str
    input_shapes: dict[str, tuple[int, ...]]
    embedding_array: np.ndarray
    embedding_shape: tuple[int, ...]
    embedding_dtype: str
    finite_value_result: bool
    deterministic_repeat_result: bool | None
    deterministic_repeat_max_abs_diff: float | None
    warnings: tuple[str, ...]
    status: str
    model_load_missing_keys: tuple[str, ...] = ()
    model_load_unexpected_keys: tuple[str, ...] = ()
    failure_reason: str | None = None


def sha256_file(path: Path | str) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_report(*, upstream_root: Path | None = None) -> DependencyReport:
    warnings: list[str] = []
    native_windows = platform.system().lower() == "windows"
    pytorch_available = importlib.util.find_spec("torch") is not None
    unicore_available = importlib.util.find_spec("unicore") is not None
    lmdb_available = importlib.util.find_spec("lmdb") is not None

    upstream_available = False
    if upstream_root is not None:
        upstream_available = (Path(upstream_root) / "unimol").exists()
    if not upstream_available:
        upstream_available = importlib.util.find_spec("unimol") is not None
    if native_windows:
        warnings.append("native Windows may be incompatible with Uni-Core or upstream compiled dependencies")
    return DependencyReport(
        pytorch_available=pytorch_available,
        unicore_available=unicore_available,
        upstream_available=upstream_available,
        lmdb_available=lmdb_available,
        native_windows=native_windows,
        warnings=tuple(warnings),
    )


def require_torch():
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise MissingDependencyError(
            "PyTorch is required for ConforFormer checkpoint inspection or inference. "
            "Install it in a separate ConforFormer environment, then retry.",
            detail=type(exc).__name__,
        ) from exc


def _torch_load_checkpoint(torch: Any, checkpoint_path: Path) -> Any:
    kwargs: dict[str, Any] = {"map_location": "cpu"}
    signature = inspect.signature(torch.load)
    load_context = nullcontext()

    if "weights_only" in signature.parameters:
        kwargs["weights_only"] = True

        serialization = getattr(torch, "serialization", None)
        safe_globals = getattr(serialization, "safe_globals", None)

        if safe_globals is None:
            raise CompatibilityError(
                "The installed PyTorch version supports weights_only loading "
                "but does not expose torch.serialization.safe_globals."
            )

        # ConforFormer.pt stores its training arguments as
        # argparse.Namespace. Allowlist only this known metadata type while
        # retaining the restricted weights-only unpickler.
        load_context = safe_globals([argparse.Namespace])

    try:
        with load_context:
            return torch.load(checkpoint_path, **kwargs)
    except Exception as exc:
        if kwargs.get("weights_only") is True:
            raise CompatibilityError(
                "checkpoint could not be loaded with "
                "torch.load(..., weights_only=True) using the approved "
                "ConforFormer metadata allowlist. This adapter does not "
                "fall back to arbitrary pickle execution.",
                detail=str(exc),
            ) from exc

        raise CompatibilityError(
            "checkpoint could not be loaded safely with the installed "
            "PyTorch version",
            detail=str(exc),
        ) from exc


def _state_dict_from_checkpoint(state: Any) -> tuple[str | None, dict[str, Any] | None]:
    if isinstance(state, dict):
        for key in ("model", "state_dict", "module"):
            value = state.get(key)
            if isinstance(value, dict):
                return key, value
        if state and all(hasattr(value, "shape") for value in state.values()):
            return None, state
    return None, None


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return tuple(int(dim) for dim in shape)


def _get_attr_or_key(source: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        if isinstance(source, dict) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return None


def _metadata_architecture(state: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not isinstance(state, dict):
        return payload
    for source_name in ("args", "cfg"):
        source = state.get(source_name)
        if source is None:
            continue
        for key, names in {
            "architecture_name": ("arch", "architecture_name"),
            "encoder_layers": ("encoder_layers",),
            "encoder_embed_dim": ("encoder_embed_dim",),
            "encoder_ffn_embed_dim": ("encoder_ffn_embed_dim",),
            "encoder_attention_heads": ("encoder_attention_heads",),
            "max_seq_len": ("max_seq_len",),
        }.items():
            value = _get_attr_or_key(source, names)
            if value is not None:
                payload[key] = value
    return payload


def _infer_architecture(state: Any, state_dict: dict[str, Any] | None) -> ArchitectureMetadata:
    payload = _metadata_architecture(state)
    tensor_shapes = {name: _shape(value) for name, value in (state_dict or {}).items()}
    embed_shape = tensor_shapes.get("embed_tokens.weight")
    if embed_shape is not None and len(embed_shape) == 2:
        payload.setdefault("token_embedding_vocab_size", embed_shape[0])
        payload.setdefault("encoder_embed_dim", embed_shape[1])
    gbf_mul = tensor_shapes.get("gbf.mul.weight")
    if gbf_mul is not None and len(gbf_mul) >= 1:
        payload.setdefault("edge_type_vocab_size", gbf_mul[0])
    gbf_means = tensor_shapes.get("gbf.means.weight")
    if gbf_means is not None and len(gbf_means) == 2:
        payload.setdefault("gaussian_basis", gbf_means[1])
    gbf_heads = tensor_shapes.get("gbf_proj.linear2.weight")
    if gbf_heads is not None and len(gbf_heads) == 2:
        payload.setdefault("encoder_attention_heads", gbf_heads[0])
    layers = {
        int(name.split(".")[2])
        for name in tensor_shapes
        if name.startswith("encoder.layers.") and len(name.split(".")) > 3 and name.split(".")[2].isdigit()
    }
    if layers:
        payload.setdefault("encoder_layers", max(layers) + 1)
    if "architecture_name" not in payload:
        payload["architecture_name"] = "contrast"
    payload.setdefault("model_name", "contrast")
    payload.setdefault("source", "checkpoint_metadata_and_state_dict")
    return ArchitectureMetadata(**{key: payload.get(key) for key in ArchitectureMetadata.__dataclass_fields__})


def inspect_checkpoint(checkpoint_path: Path | str) -> CheckpointInspection:
    path = Path(checkpoint_path)
    if not path.exists():
        raise AssetUnavailableError(f"checkpoint unavailable: {path}")
    torch = require_torch()
    checkpoint_sha = sha256_file(path)
    state = _torch_load_checkpoint(torch, path)
    state_dict_key, state_dict = _state_dict_from_checkpoint(state)
    if state_dict is None:
        raise CompatibilityError("checkpoint does not contain a model/state_dict tensor mapping")
    tensor_shapes = {
        name: shape
        for name, value in state_dict.items()
        if (shape := _shape(value)) is not None
    }
    top_keys = tuple(str(key) for key in state.keys()) if isinstance(state, dict) else ()
    architecture = _infer_architecture(state, state_dict)
    warnings: list[str] = []
    if state_dict_key is None:
        warnings.append("checkpoint appears to be a bare tensor state dict")
    if not isinstance(state, dict) or "model" not in state:
        warnings.append("checkpoint does not expose upstream state['model']; direct loading may require explicit handling")
    if architecture.max_seq_len is None:
        warnings.append("max_seq_len was not inferable; audited default 512 will be used unless overridden")
    return CheckpointInspection(
        checkpoint_path=path,
        checkpoint_sha256=checkpoint_sha,
        file_size_bytes=path.stat().st_size,
        top_level_keys=top_keys,
        has_model=isinstance(state, dict) and "model" in state,
        has_args=isinstance(state, dict) and "args" in state,
        has_cfg=isinstance(state, dict) and "cfg" in state,
        state_dict_key=state_dict_key,
        tensor_shapes=tensor_shapes,
        architecture=architecture,
        inferred_vocab_size=architecture.token_embedding_vocab_size,
        inferred_embedding_dim=architecture.encoder_embed_dim,
        warnings=tuple(warnings),
    )


def validate_dictionary_checkpoint_compatibility(
    dictionary: ConforFormerDictionary,
    checkpoint: CheckpointInspection,
) -> CompatibilityReport:
    errors: list[str] = []
    warnings: list[str] = list(checkpoint.warnings)
    arch = checkpoint.architecture.with_defaults()

    if dictionary.source_vocab_size != dictionary.vocab_size:
        warnings.append(
            "runtime dictionary adds [MASK] as required by "
            "unimol_contrast: "
            f"source_vocab_size={dictionary.source_vocab_size}, "
            f"runtime_vocab_size={dictionary.vocab_size}"
        )
    if checkpoint.inferred_vocab_size is not None and checkpoint.inferred_vocab_size != dictionary.vocab_size:
        errors.append(
            f"dictionary vocabulary size {dictionary.vocab_size} does not match checkpoint token embedding "
            f"size {checkpoint.inferred_vocab_size}"
        )
    expected_edge = dictionary.vocab_size * dictionary.vocab_size
    if arch.edge_type_vocab_size is not None and arch.edge_type_vocab_size != expected_edge:
        errors.append(
            f"dictionary edge-type vocabulary {expected_edge} does not match checkpoint gbf edge types "
            f"{arch.edge_type_vocab_size}"
        )
    for token in REQUIRED_SPECIAL_TOKENS:
        if token not in dictionary.token_to_index:
            errors.append(f"required special token missing from dictionary: {token}")
    if arch.encoder_embed_dim is None:
        errors.append("checkpoint embedding dimension could not be inferred")
    if arch.encoder_attention_heads is None:
        warnings.append("attention-head count could not be inferred")
    if arch.encoder_layers is None:
        warnings.append("layer count could not be inferred")
    if arch.max_seq_len is None:
        warnings.append("maximum sequence length could not be inferred")
    if arch.architecture_name not in {"contrast", "unimol_contrast"}:
        errors.append(f"unsupported checkpoint architecture: {arch.architecture_name}")
    if errors:
        raise CompatibilityError("; ".join(errors))
    return CompatibilityReport(
        dictionary_path=dictionary.path,
        dictionary_sha256=dictionary.sha256,
        dictionary_source_vocab_size=dictionary.source_vocab_size,
        dictionary_vocab_size=dictionary.vocab_size,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        compatible=True,
        architecture=arch,
        warnings=tuple(warnings),
    )


def select_device(device: str):
    torch = require_torch()
    if device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise MissingDependencyError("CUDA was requested but torch.cuda.is_available() is false")
        selected = "cuda"
    elif device == "cpu":
        selected = "cpu"
    else:
        raise ValueError("device must be one of: cpu, cuda, auto")
    return torch.device(selected)


def tensors_from_preprocessed(
    record_or_batch: PreprocessedConformerRecord | CollatedConformerBatch,
    *,
    device: str = "cpu",
    max_sequence_length: int | None = None,
) -> dict[str, Any]:
    torch = require_torch()
    torch_device = select_device(device)
    arrays = {
        "src_tokens": np.asarray(record_or_batch.src_tokens, dtype=np.int64).copy(),
        "src_coord": np.asarray(record_or_batch.src_coord, dtype=np.float32).copy(),
        "src_distance": np.asarray(record_or_batch.src_distance, dtype=np.float32).copy(),
        "src_edge_type": np.asarray(record_or_batch.src_edge_type, dtype=np.int64).copy(),
    }
    if arrays["src_tokens"].ndim == 1:
        arrays["src_tokens"] = arrays["src_tokens"][None, :]
        arrays["src_coord"] = arrays["src_coord"][None, :, :]
        arrays["src_distance"] = arrays["src_distance"][None, :, :]
        arrays["src_edge_type"] = arrays["src_edge_type"][None, :, :]
    validate_input_arrays(arrays, max_sequence_length=max_sequence_length)
    return {
        "src_tokens": torch.as_tensor(arrays["src_tokens"], dtype=torch.int64, device=torch_device),
        "src_coord": torch.as_tensor(arrays["src_coord"], dtype=torch.float32, device=torch_device),
        "src_distance": torch.as_tensor(arrays["src_distance"], dtype=torch.float32, device=torch_device),
        "src_edge_type": torch.as_tensor(arrays["src_edge_type"], dtype=torch.int64, device=torch_device),
    }


def validate_input_arrays(arrays: dict[str, np.ndarray], *, max_sequence_length: int | None = None) -> None:
    tokens = arrays["src_tokens"]
    coord = arrays["src_coord"]
    distance = arrays["src_distance"]
    edge_type = arrays["src_edge_type"]
    if tokens.ndim != 2:
        raise InputValidationError("src_tokens must have shape [batch, length]")
    batch, length = tokens.shape
    if coord.shape != (batch, length, 3):
        raise InputValidationError("src_coord must have shape [batch, length, 3]")
    if distance.shape != (batch, length, length):
        raise InputValidationError("src_distance must have shape [batch, length, length]")
    if edge_type.shape != (batch, length, length):
        raise InputValidationError("src_edge_type must have shape [batch, length, length]")
    if max_sequence_length is not None and length > max_sequence_length:
        raise InputValidationError(f"sequence length {length} exceeds model maximum {max_sequence_length}")
    if not np.isfinite(coord).all() or not np.isfinite(distance).all():
        raise InputValidationError("src_coord and src_distance must contain only finite values")


def extract_cls_embedding(encoder_output: Any, *, expected_batch_size: int, expected_dim: int | None = None):
    tensor = encoder_output[0] if isinstance(encoder_output, (tuple, list)) else encoder_output
    if getattr(tensor, "ndim", None) != 3:
        raise InferenceError("encoder_rep must have rank 3 before CLS extraction")
    cls = tensor[:, 0, :]
    validate_embedding_tensor(cls, expected_batch_size=expected_batch_size, expected_dim=expected_dim)
    return cls


def validate_embedding_tensor(embedding: Any, *, expected_batch_size: int, expected_dim: int | None = None) -> None:
    if getattr(embedding, "ndim", None) != 2:
        raise InferenceError("CLS embedding must have rank 2")
    if int(embedding.shape[0]) != int(expected_batch_size):
        raise InferenceError("CLS embedding batch size does not match input batch size")
    if expected_dim is not None and int(embedding.shape[1]) != int(expected_dim):
        raise InferenceError("CLS embedding dimension does not match checkpoint architecture")
    if hasattr(embedding, "isfinite"):
        finite = bool(embedding.isfinite().all().item())
    else:
        finite = bool(np.isfinite(np.asarray(embedding)).all())
    if not finite:
        raise InferenceError("CLS embedding contains non-finite values")


def embeddings_close(first: np.ndarray, second: np.ndarray, *, rtol: float = 1e-5, atol: float = 1e-6) -> bool:
    return bool(np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=False))


def _read_upstream_commit(root: Path) -> str:
    path = root / "configs" / "conforformer" / "upstream_commit.txt"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "unknown"


def _upstream_data_file(root: Path, file_name: str) -> Path:
    return root / UPSTREAM_RELATIVE_PATH / "unimol" / "data" / file_name


def _is_own_placeholder(module_name: str) -> bool:
    module = sys.modules.get(module_name)
    return bool(module is not None and getattr(module, _COMPAT_PLACEHOLDER_ATTR, False))


def _placeholder_dataset_class(class_name: str, message: str) -> type:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError(message)

    return type(class_name, (), {"__init__": __init__, "__module__": __name__})


def _register_dataset_placeholder(module_name: str, class_name: str, message: str) -> None:
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(existing, _COMPAT_PLACEHOLDER_ATTR, False):
        return
    module = ModuleType(module_name)
    setattr(module, class_name, _placeholder_dataset_class(class_name, message))
    setattr(module, _COMPAT_PLACEHOLDER_ATTR, True)
    sys.modules[module_name] = module


def _import_real_or_raise(module_name: str) -> ModuleType:
    if _is_own_placeholder(module_name):
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def _ensure_hmdb_import_compatibility(root: Path, upstream_commit: str) -> tuple[bool, str]:
    source_path = _upstream_data_file(root, "HugeMDB_dataset.py")
    if upstream_commit != AFFECTED_UPSTREAM_COMMIT:
        return False, f"pinned commit {upstream_commit} is not the documented affected commit"
    if source_path.exists():
        _import_real_or_raise(HMDB_MODULE_NAME)
        return False, f"real source file exists at {source_path}"
    try:
        _import_real_or_raise(HMDB_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != HMDB_MODULE_NAME:
            raise
    else:
        return False, "real module imported successfully"
    _register_dataset_placeholder(
        HMDB_MODULE_NAME,
        "HMDBDataset",
        "unimol.data.HugeMDB_dataset.HMDBDataset is unavailable because the pinned "
        "ConforFormer source is missing unimol/data/HugeMDB_dataset.py. This dataset "
        "is not supported or needed by the direct FluorCast contrast-encoder pathway.",
    )
    return True, f"missing upstream source file: {source_path}"


def _ensure_omol_import_compatibility() -> tuple[bool, str]:
    try:
        _import_real_or_raise(OMOL_MODULE_NAME)
    except ModuleNotFoundError as exc:
        if exc.name != "fairchem":
            raise
    else:
        return False, "real module imported successfully"
    _register_dataset_placeholder(
        OMOL_MODULE_NAME,
        "OMolDataset",
        "unimol.data.OMol_dataset.OMolDataset could not be imported because fairchem "
        "is not installed. This dataset is not supported or needed by the direct "
        "FluorCast contrast-encoder pathway.",
    )
    return True, "real OMol_dataset import failed because fairchem is not installed"


def ensure_upstream_import_compatibility(root: Path | str) -> UpstreamImportCompatibilityDiagnostics:
    root = Path(root)
    upstream_commit = _read_upstream_commit(root)
    hmdb_applied, hmdb_reason = _ensure_hmdb_import_compatibility(root, upstream_commit)
    omol_applied, omol_reason = _ensure_omol_import_compatibility()
    upstream_import_succeeded = False
    try:
        importlib.import_module("unimol.tasks.unimol_contrast")
        importlib.import_module("unimol.models.unimol_contrast")
    except Exception:
        upstream_import_succeeded = False
    else:
        upstream_import_succeeded = True
    return UpstreamImportCompatibilityDiagnostics(
        hmdb_shim_applied=hmdb_applied,
        hmdb_reason=hmdb_reason,
        omol_shim_applied=omol_applied,
        omol_reason=omol_reason,
        upstream_commit=upstream_commit,
        upstream_import_succeeded=upstream_import_succeeded,
    )

def _ensure_upstream_imports(root: Path) -> None:
    upstream_path = root / UPSTREAM_RELATIVE_PATH
    if not upstream_path.exists():
        raise AssetUnavailableError(f"pinned upstream ConforFormer path unavailable: {upstream_path}")
    if str(upstream_path) not in sys.path:
        sys.path.insert(0, str(upstream_path))
    try:
        importlib.import_module("unicore")
    except Exception as exc:
        raise MissingDependencyError(
            "Uni-Core is required to construct the upstream ConforFormer model.",
            detail=type(exc).__name__,
        ) from exc
    ensure_upstream_import_compatibility(root)
    try:
        importlib.import_module("unimol.tasks.unimol_contrast")
        importlib.import_module("unimol.models.unimol_contrast")
    except Exception as exc:
        raise MissingDependencyError(
            "Pinned upstream ConforFormer modules could not be imported.",
            detail=type(exc).__name__,
        ) from exc


class _DictionaryShim:
    def __init__(self, dictionary: ConforFormerDictionary) -> None:
        self._dictionary = dictionary

    def __len__(self) -> int:
        return self._dictionary.vocab_size

    def pad(self) -> int:
        return self._dictionary.pad_id

    def bos(self) -> int:
        return self._dictionary.cls_id

    def eos(self) -> int:
        return self._dictionary.sep_id


def _model_args(architecture: ArchitectureMetadata, dictionary_path: Path) -> SimpleNamespace:
    arch = architecture.with_defaults()
    return SimpleNamespace(
        activation_dropout=0.0,
        activation_fn="gelu",
        arch="contrast",
        contrast_temperature=0.1,
        data=str(dictionary_path.parent),
        delta_pair_repr_norm_loss=-1.0,
        dict_name=dictionary_path.name,
        dropout=0.1,
        emb_dropout=0.1,
        encoder_attention_heads=arch.encoder_attention_heads,
        encoder_embed_dim=arch.encoder_embed_dim,
        encoder_ffn_embed_dim=arch.encoder_ffn_embed_dim,
        encoder_layers=arch.encoder_layers,
        max_seq_len=arch.max_seq_len,
        masked_coord_loss=-1.0,
        masked_dist_loss=-1.0,
        masked_token_loss=-1.0,
        mode="infer",
        model_name="contrast",
        only_polar=0,
        pooler_activation_fn="tanh",
        pooler_dropout=0.0,
        post_ln=False,
        task="unimol_contrast",
        update_freq=[1],
        x_norm_loss=-1.0,
    )

def _checkpoint_args_payload(state: Any) -> dict[str, Any]:
    """Return a mutable copy of checkpoint training arguments."""

    if not isinstance(state, dict):
        return {}

    source = state.get("args")

    if isinstance(source, dict):
        return dict(source)

    if source is not None and hasattr(source, "__dict__"):
        return dict(vars(source))

    return {}


def _state_dict_has_prefix(
    state_dict: dict[str, Any],
    prefix: str,
) -> bool:
    return any(key.startswith(prefix) for key in state_dict)


def _is_positive_loss(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _is_nonnegative_loss(value: Any) -> bool:
    try:
        return float(value) >= 0
    except (TypeError, ValueError):
        return False


def _model_args_from_checkpoint(
    state: Any,
    state_dict: dict[str, Any],
    architecture: ArchitectureMetadata,
    dictionary_path: Path,
) -> SimpleNamespace:
    """Reconstruct the checkpoint's exact model module graph.

    ConforFormer conditionally creates its masked-token, coordinate,
    distance, and final pair-normalization modules from loss arguments.
    The checkpoint state dictionary is treated as authoritative about
    whether each conditional module must exist.
    """

    payload = vars(
        _model_args(
            architecture,
            dictionary_path,
        )
    ).copy()

    # Restore training-time architectural settings such as dropout,
    # normalization, and conditional pretraining-head flags.
    payload.update(_checkpoint_args_payload(state))

    arch = architecture.with_defaults()

    # Force values controlled by inspected checkpoint dimensions or by
    # FluorCast's direct inference pathway.
    payload.update(
        {
            "arch": "contrast",
            "model_name": "contrast",
            "task": "unimol_contrast",
            "mode": "infer",
            "data": str(dictionary_path.parent),
            "dict_name": dictionary_path.name,
            "only_polar": 0,
            "encoder_layers": arch.encoder_layers,
            "encoder_embed_dim": arch.encoder_embed_dim,
            "encoder_ffn_embed_dim": arch.encoder_ffn_embed_dim,
            "encoder_attention_heads": arch.encoder_attention_heads,
            "max_seq_len": arch.max_seq_len,
        }
    )

    # Reproduce the precise conditional module graph represented in the
    # checkpoint. This prevents both unexpected and missing state keys.
    conditional_heads = (
        ("masked_token_loss", "lm_head."),
        ("masked_coord_loss", "pair2coord_proj."),
        ("masked_dist_loss", "dist_head."),
    )

    for argument_name, state_prefix in conditional_heads:
        if _state_dict_has_prefix(state_dict, state_prefix):
            if not _is_positive_loss(payload.get(argument_name)):
                payload[argument_name] = 1.0
        else:
            payload[argument_name] = -1.0

    if _state_dict_has_prefix(
        state_dict,
        "encoder.final_head_layer_norm.",
    ):
        if not _is_nonnegative_loss(
            payload.get("delta_pair_repr_norm_loss")
        ):
            payload["delta_pair_repr_norm_loss"] = 1.0
    else:
        payload["delta_pair_repr_norm_loss"] = -1.0

    return SimpleNamespace(**payload)


@dataclass
class ConforFormerEncoderAdapter:
    dictionary_path: Path
    checkpoint_path: Path
    device: str = "cpu"
    allow_nonstrict: bool = False
    root: Path = field(default_factory=lambda: Path.cwd())
    dictionary: ConforFormerDictionary = field(init=False)
    checkpoint: CheckpointInspection = field(init=False)
    compatibility: CompatibilityReport = field(init=False)
    model: Any = field(init=False)
    torch_device: Any = field(init=False)
    upstream_commit: str = field(init=False)
    load_missing_keys: tuple[str, ...] = field(init=False, default=())
    load_unexpected_keys: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.dictionary_path = Path(self.dictionary_path)
        self.checkpoint_path = Path(self.checkpoint_path)
        if not self.dictionary_path.exists():
            raise AssetUnavailableError(f"dictionary unavailable: {self.dictionary_path}")
        if not self.checkpoint_path.exists():
            raise AssetUnavailableError(f"checkpoint unavailable: {self.checkpoint_path}")
        torch = require_torch()
        self.torch_device = select_device(self.device)
        self.dictionary = load_conforformer_dictionary(self.dictionary_path)
        self.checkpoint = inspect_checkpoint(self.checkpoint_path)
        self.compatibility = validate_dictionary_checkpoint_compatibility(self.dictionary, self.checkpoint)
        _ensure_upstream_imports(self.root)
        self.upstream_commit = _read_upstream_commit(self.root)

        from unicore import models

        # Load checkpoint metadata and tensors before constructing the model.
        # Its state dictionary determines which conditional pretraining heads
        # must exist for strict checkpoint loading.
        state = _torch_load_checkpoint(
            torch,
            self.checkpoint_path,
        )
        state_dict_key, state_dict = _state_dict_from_checkpoint(
            state
        )

        if state_dict is None or state_dict_key != "model":
            raise CompatibilityError(
                "upstream loading requires checkpoint state['model']"
            )

        args = _model_args_from_checkpoint(
            state,
            state_dict,
            self.compatibility.architecture,
            self.dictionary_path,
        )
        task = SimpleNamespace(
            dictionary=_DictionaryShim(self.dictionary)
        )
        model = models.build_model(args, task)

        load_result = model.load_state_dict(
            state_dict,
            strict=not self.allow_nonstrict,
        )
        self.load_missing_keys = tuple(getattr(load_result, "missing_keys", ()))
        self.load_unexpected_keys = tuple(getattr(load_result, "unexpected_keys", ()))
        if (self.load_missing_keys or self.load_unexpected_keys) and not self.allow_nonstrict:
            raise CompatibilityError("strict checkpoint loading reported missing or unexpected keys")
        model.to(self.torch_device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model = model

    def encode(
        self,
        record_or_batch: PreprocessedConformerRecord | CollatedConformerBatch,
        *,
        repeat_check: bool = False,
    ) -> AdapterResult:
        torch = require_torch()
        arch = self.compatibility.architecture.with_defaults()
        tensors = tensors_from_preprocessed(
            record_or_batch,
            device=str(self.torch_device),
            max_sequence_length=arch.max_seq_len,
        )
        batch_size = int(tensors["src_tokens"].shape[0])
        with torch.inference_mode():
            output = self.model(
                src_tokens=tensors["src_tokens"],
                src_distance=tensors["src_distance"],
                src_coord=tensors["src_coord"],
                src_edge_type=tensors["src_edge_type"],
                features_only=True,
            )
            embedding = extract_cls_embedding(output, expected_batch_size=batch_size, expected_dim=arch.encoder_embed_dim)
            repeat_ok: bool | None = None
            repeat_max_abs_diff: float | None = None
            if repeat_check:
                repeat_output = self.model(
                    src_tokens=tensors["src_tokens"],
                    src_distance=tensors["src_distance"],
                    src_coord=tensors["src_coord"],
                    src_edge_type=tensors["src_edge_type"],
                    features_only=True,
                )
                repeat_embedding = extract_cls_embedding(
                    repeat_output,
                    expected_batch_size=batch_size,
                    expected_dim=arch.encoder_embed_dim,
                )
                embedding_np = embedding.detach().cpu().numpy()
                repeat_np = repeat_embedding.detach().cpu().numpy()
                repeat_max_abs_diff = float(np.max(np.abs(embedding_np - repeat_np)))
                repeat_ok = embeddings_close(embedding_np, repeat_np)
        array = embedding.detach().cpu().numpy()
        finite = bool(np.isfinite(array).all())
        conformer_ids = getattr(record_or_batch, "conformer_ids", None)
        if conformer_ids is None:
            conformer_ids = (getattr(record_or_batch, "conformer_id"),)
        chromophore_id = getattr(record_or_batch, "chromophore_id", None)
        if chromophore_id is None:
            chromophore_ids = getattr(record_or_batch, "chromophore_ids", ("unknown",))
            chromophore_id = chromophore_ids[0]
        canonical_smiles = getattr(record_or_batch, "canonical_smiles", None)
        conformer_cache_key = getattr(record_or_batch, "conformer_cache_key", "batch")
        return AdapterResult(
            chromophore_id=str(chromophore_id),
            canonical_smiles=canonical_smiles,
            conformer_ids=tuple(str(value) for value in conformer_ids),
            conformer_cache_key=str(conformer_cache_key),
            dictionary_sha256=self.dictionary.sha256,
            checkpoint_sha256=self.checkpoint.checkpoint_sha256,
            upstream_commit=self.upstream_commit,
            architecture_metadata=asdict(arch),
            device=str(self.torch_device),
            input_shapes={key: tuple(int(dim) for dim in value.shape) for key, value in tensors.items()},
            embedding_array=array,
            embedding_shape=tuple(int(dim) for dim in array.shape),
            embedding_dtype=str(array.dtype),
            finite_value_result=finite,
            deterministic_repeat_result=repeat_ok,
            deterministic_repeat_max_abs_diff=repeat_max_abs_diff,
            model_load_missing_keys=self.load_missing_keys,
            model_load_unexpected_keys=self.load_unexpected_keys,
            warnings=tuple(self.compatibility.warnings),
            status="ok" if finite and (repeat_ok is not False) else "failed",
            failure_reason=None if finite and (repeat_ok is not False) else "embedding_validation_failed",
        )


def inspect_assets(dictionary_path: Path | str, checkpoint_path: Path | str) -> tuple[ConforFormerDictionary, CheckpointInspection, CompatibilityReport]:
    dictionary_path = Path(dictionary_path)
    checkpoint_path = Path(checkpoint_path)
    if not dictionary_path.exists():
        raise AssetUnavailableError(f"dictionary unavailable: {dictionary_path}")
    dictionary = load_conforformer_dictionary(dictionary_path)
    checkpoint = inspect_checkpoint(checkpoint_path)
    compatibility = validate_dictionary_checkpoint_compatibility(dictionary, checkpoint)
    return dictionary, checkpoint, compatibility
