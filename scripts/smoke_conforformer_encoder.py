from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chemfluor.conforformer.adapter import (
    AdapterError,
    AssetUnavailableError,
    ConforFormerEncoderAdapter,
    dependency_report,
    ensure_upstream_import_compatibility,
    inspect_assets,
)
from chemfluor.conforformer.cache import CacheError, load_conformer_cache_record
from chemfluor.conforformer.conformers import canonicalize_smiles
from chemfluor.conforformer.dictionary import load_conforformer_dictionary
from chemfluor.conforformer.preprocess import (
    ConforFormerPreprocessingConfig,
    PreprocessingError,
    collate_preprocessed_conformers,
    preprocess_successful_conformers,
)
from chemfluor.conforformer.schemas import MoleculeStatus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Checkpoint-gated ConforFormer encoder smoke test.")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--smiles", default=None)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--max-conformers", type=int, default=1)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--env-report", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-nonstrict", action="store_true")
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _print_report(report: dict[str, Any], output: Path | None = None) -> None:
    text = json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n"
    print(text)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def _import_status(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "version": getattr(module, "__version__", None)}


def _environment_report(args: argparse.Namespace) -> dict[str, Any]:
    torch_status = _import_status("torch")
    cuda_available = None
    if torch_status["available"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
    upstream_status: dict[str, Any]
    compatibility_shims: dict[str, Any]
    upstream_path = ROOT / "third_party" / "ConforFormer" / "unimol"
    inserted = False
    if str(upstream_path) not in sys.path:
        sys.path.insert(0, str(upstream_path))
        inserted = True
    try:
        diagnostics = ensure_upstream_import_compatibility(ROOT)
        compatibility_shims = asdict(diagnostics)
        upstream_status = {"available": diagnostics.upstream_import_succeeded}
        if not diagnostics.upstream_import_succeeded:
            upstream_status["error"] = "upstream import failed after compatibility shim registration"
    except Exception as exc:
        compatibility_shims = {"error": f"{type(exc).__name__}: {exc}"}
        upstream_status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if inserted:
            try:
                sys.path.remove(str(upstream_path))
            except ValueError:
                pass
    commit_path = ROOT / "configs" / "conforformer" / "upstream_commit.txt"
    return {
        "asset_availability": {
            "checkpoint": None if args.checkpoint is None else {"path": str(args.checkpoint), "exists": args.checkpoint.exists()},
            "dictionary": None if args.dictionary is None else {"path": str(args.dictionary), "exists": args.dictionary.exists()},
        },
        "cuda_available": cuda_available,
        "dependency_availability": asdict(dependency_report(upstream_root=upstream_path)),
        "lmdb": _import_status("lmdb"),
        "operating_system": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "pytorch": torch_status,
        "unicore": _import_status("unicore"),
        "upstream_commit": commit_path.read_text(encoding="utf-8").strip() if commit_path.exists() else "unknown",
        "upstream_conforformer": upstream_status,
        "upstream_import_status": upstream_status,
        "applied_compatibility_shims": compatibility_shims,
    }


def _requested_canonical(smiles: str | None) -> str | None:
    if smiles is None:
        return None
    canonical, _ = canonicalize_smiles(smiles)
    if canonical is None:
        raise PreprocessingError(f"invalid requested SMILES: {smiles}")
    return canonical


def _load_preprocessed(args: argparse.Namespace, max_sequence_length: int) -> list[Any]:
    if args.cache_dir is None:
        raise AssetUnavailableError("--cache-dir is required for normal smoke mode")
    if args.max_conformers <= 0:
        raise ValueError("--max-conformers must be positive")
    dictionary = load_conforformer_dictionary(args.dictionary)
    requested = _requested_canonical(args.smiles)
    config = ConforFormerPreprocessingConfig(max_sequence_length=max_sequence_length)
    selected: list[Any] = []
    for path in sorted(args.cache_dir.glob("*.json")):
        try:
            cache_record = load_conformer_cache_record(path)
        except CacheError as exc:
            print(f"skipping invalid cache file {path}: {exc}", file=sys.stderr)
            continue
        if requested is not None and cache_record.canonical_smiles != requested:
            continue
        if cache_record.status != MoleculeStatus.OK:
            continue
        try:
            records = preprocess_successful_conformers(cache_record, dictionary, config)
        except PreprocessingError as exc:
            raise PreprocessingError(f"preprocessing failed for {cache_record.chromophore_id}: {exc}") from exc
        selected.extend(records)
        if len(selected) >= args.max_conformers:
            return selected[: args.max_conformers]
    raise PreprocessingError("no matching successful conformers were available for smoke inference")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report: dict[str, Any] = {
        "dependency_availability": asdict(dependency_report(upstream_root=ROOT / "third_party" / "ConforFormer" / "unimol")),
        "mode": "env-report" if args.env_report else "inspect-only" if args.inspect_only else "encoder-smoke",
        "status": "failed",
    }
    try:
        if args.env_report:
            report.update(_environment_report(args))
            report["status"] = "env_report_ok"
            _print_report(report, args.output)
            return 0
        if args.dictionary is None:
            raise AssetUnavailableError("--dictionary is required unless --env-report is used")
        if args.checkpoint is None:
            raise AssetUnavailableError("--checkpoint is required unless --env-report is used")
        _, checkpoint, compatibility = inspect_assets(args.dictionary, args.checkpoint)
        report.update(
            {
                "checkpoint": checkpoint,
                "compatibility": compatibility,
                "status": "inspection_ok",
            }
        )
        if args.inspect_only:
            _print_report(report, args.output)
            return 0

        records = _load_preprocessed(args, compatibility.architecture.with_defaults().max_seq_len or 512)
        batch = collate_preprocessed_conformers(records, load_conforformer_dictionary(args.dictionary))
        adapter = ConforFormerEncoderAdapter(
            dictionary_path=args.dictionary,
            checkpoint_path=args.checkpoint,
            device=args.device,
            allow_nonstrict=args.allow_nonstrict,
            root=ROOT,
        )
        result = adapter.encode(batch, repeat_check=args.repeat_check)
        result_payload = asdict(result)
        result_payload["embedding_array"] = result.embedding_array.tolist()
        report.update(
            {
                "adapter_result": result_payload,
                "embedding_shape": result.embedding_shape,
                "status": result.status,
            }
        )
        _print_report(report, args.output)
        return 0 if result.status == "ok" else 1
    except AdapterError as exc:
        report.update(
            {
                "failure_category": exc.reason_code,
                "failure_reason": str(exc),
                "failure_detail": exc.detail,
            }
        )
    except PreprocessingError as exc:
        report.update({"failure_category": "preprocessing_failure", "failure_reason": str(exc)})
    except Exception as exc:
        report.update({"failure_category": type(exc).__name__, "failure_reason": str(exc)})
    _print_report(report, args.output)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
