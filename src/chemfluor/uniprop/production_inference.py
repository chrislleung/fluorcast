"""Production UniProp model-bundle loading and JSON prediction inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from rdkit import Chem, DataStructs, RDLogger

from .geometry_cache import (
    GEOMETRY_SCHEMA_VERSION,
    atomic_write_json,
    cache_path,
    generate_geometry_entry,
    read_valid_cache,
    validate_geometry_entry,
)
from .manifests import MANIFEST_SCHEMA_VERSION, stable_hash
from .physics_constraints import PHYSICS_SCHEMA_VERSION, PhysicsConstrainedOutputHead, physics_consistency_metrics

RDLogger.DisableLog("rdApp.*")

BUNDLE_SCHEMA_VERSION = "fluorcast_uniprop_model_bundle_v1"
PREDICTION_SCHEMA_VERSION = "fluorcast_uniprop_prediction_v1"
BACKEND_ADAPTER_SCHEMA_VERSION = "fluorcast_backend_prediction_adapter_v1"
SUPPORTED_BUNDLE_SCHEMAS = {BUNDLE_SCHEMA_VERSION}
TINY_3D_SMOKE_MODEL_KIND = "tiny_3d_smoke_backbone"
FORBIDDEN_PRODUCTION_MODEL_KINDS = {TINY_3D_SMOKE_MODEL_KIND}


class BundleError(Exception):
    """A model bundle is missing, corrupt, or incompatible."""


class PredictionInputError(Exception):
    """A request cannot be predicted because its input is invalid."""


@dataclass(frozen=True)
class ProductionBundle:
    bundle_dir: Path
    metadata: dict[str, Any]
    architecture_config: dict[str, Any]
    target_definitions: dict[str, Any]
    scalers: dict[str, Any]
    solvent_assets: dict[str, Any]
    model: Any
    device: str


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for UniProp production inference.") from exc
    return torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"Could not read bundle asset: {path}") from exc
    if not isinstance(payload, dict):
        raise BundleError(f"Bundle asset must be a JSON object: {path}")
    return payload


def canonicalize_smiles(smiles: str, *, field_name: str = "SMILES") -> str:
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        raise PredictionInputError(f"Invalid {field_name}: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def molecule_id_for_canonical_smiles(canonical_smiles: str) -> str:
    return stable_hash("mol", MANIFEST_SCHEMA_VERSION, canonical_smiles)


def resolve_solvent_smiles(solvent: str | None, solvent_smiles: str | None, assets: dict[str, Any]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    name_map = {str(key).lower(): value for key, value in assets.get("solvent_name_to_smiles", {}).items()}
    supported = set(str(value) for value in assets.get("supported_solvent_smiles", []))
    if solvent_smiles:
        canonical = canonicalize_smiles(solvent_smiles, field_name="solvent_smiles")
    elif solvent:
        value = name_map.get(str(solvent).strip().lower())
        if value is None:
            raise PredictionInputError(f"Unsupported solvent: {solvent}")
        canonical = canonicalize_smiles(str(value), field_name="solvent")
    else:
        raise PredictionInputError("Either solvent or solvent_smiles is required.")
    if supported and canonical not in supported:
        raise PredictionInputError(f"Unsupported solvent: {canonical}")
    if not supported:
        warnings.append("Bundle does not declare supported solvents; accepted canonical solvent without lookup.")
    return canonical, warnings


def load_bundle(bundle_dir: Path, *, device: str = "cpu") -> ProductionBundle:
    torch = _require_torch()
    if not bundle_dir.exists():
        raise BundleError(f"Model bundle directory does not exist: {bundle_dir}")
    metadata = _read_json(bundle_dir / "metadata.json")
    if metadata.get("schema_version") not in SUPPORTED_BUNDLE_SCHEMAS:
        raise BundleError(f"Unsupported model bundle schema: {metadata.get('schema_version')}")
    if (
        metadata.get("model_kind") in FORBIDDEN_PRODUCTION_MODEL_KINDS
        or metadata.get("profile") == "windows-smoke"
        or metadata.get("real_uniprop_used") is False
    ):
        raise BundleError("Windows smoke artifacts cannot be loaded as a real production UniProp bundle.")
    if metadata.get("physics_schema_version") != PHYSICS_SCHEMA_VERSION:
        raise BundleError(f"Model physics schema mismatch: {metadata.get('physics_schema_version')}")
    if metadata.get("supported_geometry_schema") != GEOMETRY_SCHEMA_VERSION:
        raise BundleError(f"Unsupported geometry schema: {metadata.get('supported_geometry_schema')}")

    asset_names = {
        "architecture_config": "architecture_config.json",
        "target_definitions": "target_definitions.json",
        "scalers": "scalers.json",
        "solvent_encoder_assets": "solvent_encoder_assets.json",
        "model_weights": "model_weights.pt",
    }
    expected_hashes = metadata.get("asset_sha256", {})
    for asset_key, filename in asset_names.items():
        path = bundle_dir / filename
        if not path.exists():
            raise BundleError(f"Bundle asset missing: {filename}")
        expected = expected_hashes.get(asset_key)
        if expected and file_sha256(path) != expected:
            raise BundleError(f"Bundle asset checksum mismatch: {filename}")

    architecture = _read_json(bundle_dir / asset_names["architecture_config"])
    targets = _read_json(bundle_dir / asset_names["target_definitions"])
    scalers = _read_json(bundle_dir / asset_names["scalers"])
    solvent_assets = _read_json(bundle_dir / asset_names["solvent_encoder_assets"])
    if set(targets.get("targets", [])) - {"absorption_nm", "emission_nm", "quantum_yield", "lifetime_ns", "log_extinction"}:
        raise BundleError("Bundle declares unsupported target definitions.")

    requested_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    model = PhysicsConstrainedOutputHead.build(
        torch,
        input_dim=int(architecture["input_dim"]),
        variant=str(architecture.get("physics_variant", "complete")),
    ).to(requested_device)
    try:
        state = torch.load(bundle_dir / asset_names["model_weights"], map_location=requested_device, weights_only=False)
        if isinstance(state, dict) and state.get("model_kind") in FORBIDDEN_PRODUCTION_MODEL_KINDS:
            raise BundleError("Tiny smoke checkpoint cannot be loaded in real UniProp model mode.")
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
    except BundleError:
        raise
    except Exception as exc:
        raise BundleError("Could not load model weights from bundle.") from exc
    model.eval()
    return ProductionBundle(bundle_dir, metadata, architecture, targets, scalers, solvent_assets, model, requested_device)


def _stable_float_features(text: str, dim: int) -> np.ndarray:
    values = []
    for index in range(dim):
        digest = hashlib.sha256(f"{index}|{text}".encode("utf-8")).digest()
        values.append((int.from_bytes(digest[:4], "big") / 2**32) * 2.0 - 1.0)
    return np.asarray(values, dtype=np.float32)


def _geometry_text(entry: dict[str, Any]) -> str:
    parts = [entry["canonical_smiles"], ",".join(entry["atom_symbols"])]
    parts.extend(",".join(f"{float(value):.8f}" for value in xyz) for xyz in entry["coordinates"])
    return "|".join(parts)


def _solvent_vector(canonical_solvent: str, assets: dict[str, Any], dim: int) -> np.ndarray:
    vectors = assets.get("solvent_vectors", {})
    if canonical_solvent in vectors:
        values = np.asarray(vectors[canonical_solvent], dtype=np.float32)
        if values.shape != (dim,):
            raise BundleError(f"Solvent vector has wrong size for {canonical_solvent}.")
        return values
    return _stable_float_features(canonical_solvent, dim)


def _feature_vector(canonical_smiles: str, canonical_solvent: str, geometry: dict[str, Any], bundle: ProductionBundle) -> np.ndarray:
    mol_dim = int(bundle.architecture_config.get("molecule_feature_dim", 16))
    solvent_dim = int(bundle.architecture_config.get("solvent_feature_dim", 8))
    molecule = _stable_float_features(f"{canonical_smiles}|{_geometry_text(geometry)}", mol_dim)
    solvent = _solvent_vector(canonical_solvent, bundle.solvent_assets, solvent_dim)
    features = np.concatenate([molecule, solvent]).astype(np.float32)
    expected = int(bundle.architecture_config["input_dim"])
    if len(features) != expected:
        raise BundleError(f"Feature vector dimension {len(features)} does not match architecture input_dim {expected}.")
    return features


def geometry_for_request(
    canonical_smiles: str,
    molecule_id: str,
    *,
    cache_dir: Path | None,
    precomputed_geometry: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    if precomputed_geometry is not None:
        validate_geometry_entry(precomputed_geometry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
        return precomputed_geometry, "provided", warnings
    if cache_dir is not None:
        path = cache_path(cache_dir, molecule_id)
        if path.exists():
            try:
                return read_valid_cache(path, molecule_id, canonical_smiles), "cache_hit", warnings
            except (OSError, json.JSONDecodeError, ValueError) as exc:
                warnings.append(f"Ignored invalid cached geometry: {exc}")
    entry = generate_geometry_entry(molecule_id, canonical_smiles)
    validate_geometry_entry(entry, molecule_id=molecule_id, canonical_smiles=canonical_smiles)
    if cache_dir is not None:
        atomic_write_json(cache_path(cache_dir, molecule_id), entry)
    return entry, "generated", warnings


def _fingerprint(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.RDKFingerprint(mol)


def applicability_domain(canonical_smiles: str, bundle: ProductionBundle) -> dict[str, Any]:
    references = bundle.metadata.get("training_data_fingerprint", {}).get("reference_smiles", [])
    query_fp = _fingerprint(canonical_smiles)
    best_similarity = None
    best_smiles = None
    if query_fp is not None:
        for smiles in references:
            ref_fp = _fingerprint(str(smiles))
            if ref_fp is None:
                continue
            similarity = float(DataStructs.TanimotoSimilarity(query_fp, ref_fp))
            if best_similarity is None or similarity > best_similarity:
                best_similarity = similarity
                best_smiles = str(smiles)
    threshold = float(bundle.metadata.get("applicability_domain", {}).get("similarity_threshold", 0.35))
    outside = best_similarity is not None and best_similarity < threshold
    label = "unknown" if best_similarity is None else ("high" if best_similarity >= 0.75 else ("medium" if best_similarity >= threshold else "low"))
    return {
        "outside_applicability_domain": bool(outside),
        "nearest_training_similarity": best_similarity,
        "nearest_training_smiles": best_smiles,
        "confidence_label": label,
    }


def _nullable_number(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def predict_one(request: dict[str, Any], bundle: ProductionBundle, *, cache_dir: Path | None = None) -> dict[str, Any]:
    torch = _require_torch()
    request_id = request.get("request_id")
    warnings: list[str] = []
    canonical_molecule = canonicalize_smiles(str(request.get("chromophore_smiles") or request.get("molecule_smiles") or ""), field_name="chromophore_smiles")
    canonical_solvent, solvent_warnings = resolve_solvent_smiles(request.get("solvent"), request.get("solvent_smiles"), bundle.solvent_assets)
    warnings.extend(solvent_warnings)
    molecule_id = molecule_id_for_canonical_smiles(canonical_molecule)
    geometry, geometry_source, geometry_warnings = geometry_for_request(
        canonical_molecule,
        molecule_id,
        cache_dir=cache_dir,
        precomputed_geometry=request.get("precomputed_geometry"),
    )
    warnings.extend(geometry_warnings)
    features = _feature_vector(canonical_molecule, canonical_solvent, geometry, bundle)
    with torch.no_grad():
        outputs = bundle.model(torch.tensor(features[None, :], dtype=torch.float32, device=bundle.device))
    consistency = physics_consistency_metrics(outputs)
    prediction: dict[str, Any] = {
        "model_name": str(bundle.metadata["model_name"]),
        "model_version": str(bundle.metadata["model_version"]),
        "predicted_absorption_nm": _nullable_number(outputs["absorption_nm"][0].detach().cpu().item()),
        "predicted_emission_nm": _nullable_number(outputs["emission_nm"][0].detach().cpu().item()),
        "predicted_quantum_yield": _nullable_number(outputs["quantum_yield"][0].detach().cpu().item()),
        "predicted_lifetime_ns": _nullable_number(outputs["lifetime_ns"][0].detach().cpu().item())
        if "lifetime_ns" in bundle.target_definitions.get("targets", [])
        else None,
        "predicted_log_extinction": _nullable_number(outputs["log_extinction"][0].detach().cpu().item())
        if "log_extinction" in bundle.target_definitions.get("targets", [])
        else None,
        "physically_valid_stokes": bool(outputs["stokes_energy_ev"][0].detach().cpu().item() >= 0.0),
        "physical_consistency": consistency,
        "warnings": [],
    }
    prediction["predicted_stokes_shift_nm"] = prediction["predicted_emission_nm"] - prediction["predicted_absorption_nm"]
    prediction["predicted_stokes_shift_cm^-1"] = 1.0e7 / prediction["predicted_absorption_nm"] - 1.0e7 / prediction["predicted_emission_nm"]
    ad = applicability_domain(canonical_molecule, bundle)
    if ad["outside_applicability_domain"]:
        warnings.append("Prediction is outside the applicability domain.")
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "status": "success",
        "request_id": request_id,
        "canonical_molecule_smiles": canonical_molecule,
        "canonical_solvent_smiles": canonical_solvent,
        "molecule_id": molecule_id,
        "geometry_source": geometry_source,
        "predictions": [prediction],
        "applicability_domain": ad,
        "provenance": {
            "model_name": bundle.metadata["model_name"],
            "model_version": bundle.metadata["model_version"],
            "bundle_schema_version": bundle.metadata["schema_version"],
            "upstream_revision": bundle.metadata.get("upstream_revision"),
            "checkpoint_hashes": bundle.metadata.get("checkpoint_hashes", {}),
            "supported_geometry_schema": bundle.metadata.get("supported_geometry_schema"),
            "metrics_summary": bundle.metadata.get("metrics_summary", {}),
        },
        "warnings": warnings,
    }


def _failure_record(request: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "status": "failed",
        "request_id": request.get("request_id"),
        "canonical_molecule_smiles": None,
        "canonical_solvent_smiles": None,
        "predictions": [],
        "applicability_domain": None,
        "provenance": None,
        "warnings": [],
        "error": {"code": exc.__class__.__name__, "message": str(exc)},
    }


def predict_json(payload: dict[str, Any], *, bundle_dir: Path, cache_dir: Path | None = None, device: str = "cpu") -> dict[str, Any]:
    bundle = load_bundle(bundle_dir, device=device)
    requests = payload.get("batch")
    if requests is None:
        requests = [payload]
        batch_mode = False
    else:
        if not isinstance(requests, list):
            raise PredictionInputError("batch must be a list of prediction requests.")
        batch_mode = True
    results = []
    for request in requests:
        try:
            results.append(predict_one(request, bundle, cache_dir=cache_dir))
        except Exception as exc:
            results.append(_failure_record(request if isinstance(request, dict) else {}, exc))
    if batch_mode:
        return {"schema_version": PREDICTION_SCHEMA_VERSION, "status": "success", "results": results}
    return results[0]


def validate_prediction_output_schema(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("Prediction output schema_version is missing or unsupported.")
    if payload.get("status") not in {"success", "failed"}:
        raise ValueError("Prediction output status is invalid.")
    if payload["status"] == "success":
        for field in ["canonical_molecule_smiles", "canonical_solvent_smiles", "predictions", "provenance", "warnings"]:
            if field not in payload:
                raise ValueError(f"Prediction output missing field: {field}")
        if not payload["predictions"]:
            raise ValueError("Successful prediction must include at least one prediction record.")
        for prediction in payload["predictions"]:
            if "model_version" not in prediction or "model_name" not in prediction:
                raise ValueError("Every prediction must include model provenance.")
    else:
        if "error" not in payload:
            raise ValueError("Failed prediction must include an error object.")


def to_backend_prediction_contract(result: dict[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
    """Adapt one production prediction result to the existing FluorCast backend contract."""
    if result.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ValueError("Cannot adapt unsupported prediction schema.")
    if result.get("status") == "success":
        return {
            "schema_version": BACKEND_ADAPTER_SCHEMA_VERSION,
            "job_id": job_id if job_id is not None else result.get("request_id"),
            "status": "success",
            "canonical_molecule_smiles": result.get("canonical_molecule_smiles"),
            "canonical_solvent_smiles": result.get("canonical_solvent_smiles"),
            "predictions": result.get("predictions", []),
            "applicability_domain": result.get("applicability_domain"),
            "warnings": result.get("warnings", []),
        }
    return {
        "schema_version": BACKEND_ADAPTER_SCHEMA_VERSION,
        "job_id": job_id if job_id is not None else result.get("request_id"),
        "status": "failed",
        "canonical_molecule_smiles": None,
        "canonical_solvent_smiles": None,
        "predictions": [],
        "applicability_domain": None,
        "warnings": result.get("warnings", []),
        "error": result.get("error", {"code": "PREDICTION_FAILED", "message": "Prediction failed."}),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-cache-dir", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = predict_json(payload, bundle_dir=args.bundle_dir, cache_dir=args.geometry_cache_dir, device=args.device)
        if "results" in result:
            for item in result["results"]:
                if item.get("status") == "success":
                    validate_prediction_output_schema(item)
        else:
            validate_prediction_output_schema(result)
        write_json(args.output, result)
        return 0 if (result.get("status") == "success") else 1
    except Exception as exc:
        fallback = _failure_record({}, exc)
        write_json(args.output, fallback)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
