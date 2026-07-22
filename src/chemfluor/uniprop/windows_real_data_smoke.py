"""Native-Windows UniProp smoke over the processed FluorCast dataset.

This profile exercises the real FluorCast processed data path with the same
local tiny 3D backbone used by the synthetic Windows smoke. It deliberately
does not import Uni-Core, Uni-Mol+, CUDA-only code, or real UniProp
checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .geometry_cache import (
    GEOMETRY_SCHEMA_VERSION,
    GeometryResult,
    build_geometry_cache,
    cache_path,
    read_valid_cache,
)
from .lmdb_export import (
    DEFAULT_TARGET_COLUMNS,
    ExportConfig,
    export_uniprop_lmdb,
    file_sha256,
    read_lmdb_records,
    validate_lmdb,
)
from .manifests import (
    MANIFEST_SCHEMA_VERSION,
    ManifestBundle,
    audit_split_leakage,
    build_manifests,
    make_split_assignments,
    resolve_authoritative_dataset,
    split_statistics,
    validate_manifest_reconciliation,
)
from .windows_smoke import (
    TINY_3D_SMOKE_MODEL_KIND,
    Tiny3DSmokeConfig,
    Tiny3DSmokeModel,
    FluorCastUniPropSmokeDataset,
    _coordinates_sha256,
    _require_torch,
    build_tiny_smoke_model,
    changed_parameter_names,
    finite_forward_report,
    inverse_scaled_predictions,
    parameter_state,
    predictions_numpy,
    smoke_train_step,
    tensor_shape_report,
    windows_smoke_environment_report,
)

WINDOWS_REAL_DATA_SMOKE_PROFILE = "windows-real-data-smoke"
WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION = "fluorcast_uniprop_windows_real_data_smoke_v1"
DEFAULT_REAL_DATA_SMOKE_OUTPUT_DIR = Path("artifacts/uniprop_windows_real_data_smoke")
REAL_DATA_SMOKE_TARGETS = tuple(DEFAULT_TARGET_COLUMNS)
REAL_DATA_SMOKE_PARTITIONS = ("train", "valid", "test")
PREDICTION_BATCH_SIZE = 64
TRAINING_BATCH_SIZE = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key)) for key in fieldnames} for row in rows])


def source_dataset_sha256(path: Path | None = None) -> tuple[Path, str]:
    """Return the authoritative processed dataset path and SHA-256 digest."""
    dataset = resolve_authoritative_dataset(path)
    return dataset, file_sha256(dataset)


def _selection_score(molecule_id: str, seed: int) -> str:
    payload = f"{WINDOWS_REAL_DATA_SMOKE_PROFILE}|subset|{seed}|{molecule_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_real_data_subset(
    row_manifest: pd.DataFrame,
    molecule_manifest: pd.DataFrame,
    *,
    max_molecules: int | None,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Select a deterministic bounded subset by stable molecule IDs."""
    if max_molecules is not None and max_molecules < 1:
        raise ValueError("--max-molecules must be positive when provided.")
    if max_rows is not None and max_rows < 1:
        raise ValueError("--max-rows must be positive when provided.")

    scored_molecules = molecule_manifest.copy()
    scored_molecules["_selection_score"] = scored_molecules["molecule_id"].astype(str).map(
        lambda molecule_id: _selection_score(molecule_id, seed)
    )
    scored_molecules = scored_molecules.sort_values(
        ["_selection_score", "molecule_id"], kind="mergesort"
    ).reset_index(drop=True)
    if max_molecules is not None:
        scored_molecules = scored_molecules.head(max_molecules).copy()

    source_rows = row_manifest.copy()
    source_rows["_selection_score"] = source_rows["molecule_id"].astype(str).map(
        lambda molecule_id: _selection_score(molecule_id, seed)
    )
    source_rows = source_rows[source_rows["molecule_id"].isin(scored_molecules["molecule_id"])].copy()

    selected_parts: list[pd.DataFrame] = []
    selected_count = 0
    for _, molecule in scored_molecules.iterrows():
        molecule_id = str(molecule["molecule_id"])
        rows_for_molecule = source_rows[source_rows["molecule_id"].astype(str) == molecule_id].sort_values(
            ["source_row_number", "row_id"], kind="mergesort"
        )
        if rows_for_molecule.empty:
            continue
        if max_rows is None:
            selected_parts.append(rows_for_molecule)
            selected_count += len(rows_for_molecule)
            continue
        remaining = max_rows - selected_count
        if remaining <= 0:
            break
        if len(rows_for_molecule) <= remaining:
            selected_parts.append(rows_for_molecule)
            selected_count += len(rows_for_molecule)
        elif selected_count == 0:
            selected_parts.append(rows_for_molecule.head(remaining))
            selected_count += remaining
            break
        else:
            break

    if not selected_parts:
        raise ValueError("Real-data smoke subset selection produced no rows.")

    selected_rows = pd.concat(selected_parts, ignore_index=True)
    selected_rows = selected_rows.sort_values(
        ["_selection_score", "molecule_id", "source_row_number", "row_id"],
        kind="mergesort",
    ).drop(columns=["_selection_score"]).reset_index(drop=True)

    selected_molecule_ids = set(selected_rows["molecule_id"].astype(str))
    selected_molecules = molecule_manifest[
        molecule_manifest["molecule_id"].astype(str).isin(selected_molecule_ids)
    ].copy()
    selected_counts = selected_rows.groupby("molecule_id").size()
    selected_molecules["source_row_count"] = selected_molecules["molecule_id"].map(selected_counts).astype(int)
    selected_molecules = selected_molecules.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)

    report = {
        "selection_seed": int(seed),
        "selection_method": (
            "molecules ordered by SHA-256 of stable molecule_id and seed; "
            "rows retained in source_row_number order within each selected molecule"
        ),
        "requested_max_molecules": max_molecules,
        "requested_max_rows": max_rows,
        "selected_source_rows": int(len(selected_rows)),
        "selected_unique_molecules": int(selected_rows["molecule_id"].nunique()),
        "selected_source_row_numbers": selected_rows["source_row_number"].astype(int).tolist(),
        "selected_molecule_ids": sorted(selected_molecule_ids),
        "repeated_molecule_count": int((selected_rows.groupby("molecule_id").size() > 1).sum()),
    }
    return selected_rows, selected_molecules, report


def _target_coverage(rows: pd.DataFrame, targets: Sequence[str]) -> dict[str, dict[str, Any]]:
    total = int(len(rows))
    return {
        target: {
            "available": int(pd.to_numeric(rows[target], errors="coerce").notna().sum()) if target in rows.columns else 0,
            "missing": int(pd.to_numeric(rows[target], errors="coerce").isna().sum()) if target in rows.columns else total,
            "coverage_fraction": (
                float(pd.to_numeric(rows[target], errors="coerce").notna().mean())
                if target in rows.columns and total
                else 0.0
            ),
        }
        for target in targets
    }


def _solvent_coverage(rows: pd.DataFrame) -> dict[str, Any]:
    solvent = rows["canonical_solvent_smiles"]
    missing = int(solvent.isna().sum())
    report = {
        "selected_rows": int(len(rows)),
        "missing_solvent_rows": missing,
        "non_missing_solvent_rows": int(len(rows) - missing),
        "unique_solvents_including_missing": int(rows["solvent_id"].nunique()),
        "unique_non_missing_solvents": int(solvent.dropna().nunique()),
    }
    if {
        "source_canonical_solvent_smiles",
        "uniprop_canonical_solvent_smiles",
        "uniprop_solvent_mapping_status",
        "environment_type",
    }.issubset(rows.columns):
        source = rows["source_canonical_solvent_smiles"]
        resolved = rows["uniprop_canonical_solvent_smiles"]
        status = rows["uniprop_solvent_mapping_status"].astype("string")
        environment = rows["environment_type"].astype("string")
        report.update(
            {
                "source_canonical_solvent_rows": int(source.notna().sum()),
                "uniprop_canonical_solvent_rows": int(resolved.notna().sum()),
                "uniprop_alias_repaired_rows": int((source.isna() & resolved.notna() & status.eq("resolved_alias")).sum()),
                "gas_phase_rows": int(environment.eq("gas_phase").sum()),
                "unresolved_solvent_rows": int((resolved.isna() & ~environment.eq("gas_phase")).sum()),
            }
        )
    return report


def build_selected_real_manifests(
    output_dir: Path,
    *,
    dataset: Path | None,
    max_molecules: int | None,
    max_rows: int | None,
    seed: int,
) -> tuple[ManifestBundle, dict[str, Any]]:
    """Build and persist the selected source-row manifests before geometry."""
    dataset_path, dataset_hash = source_dataset_sha256(dataset)
    full_bundle = build_manifests(
        dataset_path,
        target_columns=REAL_DATA_SMOKE_TARGETS,
        compute_inchikey=False,
        compute_rdkit_properties=False,
        compute_nonisomeric=False,
    )
    validate_manifest_reconciliation(full_bundle)
    selected_rows, selected_molecules, selection_report = select_real_data_subset(
        full_bundle.row_manifest,
        full_bundle.molecule_manifest,
        max_molecules=max_molecules,
        max_rows=max_rows,
        seed=seed,
    )
    targets = tuple(full_bundle.metadata["target_columns"])
    metadata = {
        **full_bundle.metadata,
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "source_dataset_path": str(dataset_path),
        "source_dataset_sha256": dataset_hash,
        "full_source_rows": int(full_bundle.metadata["source_rows"]),
        "full_unique_molecules": int(full_bundle.metadata["unique_molecules"]),
        "source_rows": int(len(selected_rows)),
        "manifest_rows": int(len(selected_rows)),
        "unique_molecules": int(selected_rows["molecule_id"].nunique()),
        "unique_solvents": int(selected_rows["solvent_id"].nunique()),
        "target_columns": list(targets),
        "selection": selection_report,
        "solvent_coverage": _solvent_coverage(selected_rows),
        "target_coverage": _target_coverage(selected_rows, targets),
    }
    selected_bundle = ManifestBundle(dataset_path, selected_molecules, selected_rows, metadata)
    validate_manifest_reconciliation(selected_bundle)

    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected_rows.to_csv(manifest_dir / "selected_row_manifest.csv", index=False)
    selected_molecules.to_csv(manifest_dir / "selected_molecule_manifest.csv", index=False)
    _write_json(manifest_dir / "selected_manifest_metadata.json", metadata)
    return selected_bundle, metadata


def _subset_bundle_for_rows(bundle: ManifestBundle, rows: pd.DataFrame) -> ManifestBundle:
    """Return a manifest bundle restricted to row-selected molecules."""
    selected_rows = rows.copy().reset_index(drop=True)
    molecule_ids = set(selected_rows["molecule_id"].astype(str))
    molecules = bundle.molecule_manifest[
        bundle.molecule_manifest["molecule_id"].astype(str).isin(molecule_ids)
    ].copy()
    if len(molecules):
        counts = selected_rows.groupby("molecule_id").size()
        molecules["source_row_count"] = molecules["molecule_id"].map(counts).astype(int)
    molecules = molecules.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    metadata = {
        **bundle.metadata,
        "source_rows": int(len(selected_rows)),
        "manifest_rows": int(len(selected_rows)),
        "unique_molecules": int(selected_rows["molecule_id"].nunique()) if len(selected_rows) else 0,
        "unique_solvents": int(selected_rows["solvent_id"].nunique()) if len(selected_rows) else 0,
    }
    restricted = ManifestBundle(bundle.source_path, molecules, selected_rows, metadata)
    if len(selected_rows):
        validate_manifest_reconciliation(restricted)
    return restricted


def _solvent_terminal_masks(rows: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    environment = rows.get("environment_type", pd.Series("molecular_solvent", index=rows.index)).astype("string")
    canonical = rows["canonical_solvent_smiles"]
    gas = environment == "gas_phase"
    molecular = (environment == "molecular_solvent") & canonical.notna()
    unresolved = ~molecular & ~gas
    return molecular.fillna(False), gas.fillna(False), unresolved.fillna(False)


def _status_counts(results: Sequence[GeometryResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _classify_geometry_failure(status: str, detail: str | None) -> str:
    text = (detail or "").lower()
    if status == "invalid_cache":
        return "invalid_cache"
    if status == "optimization_failed":
        return "optimization_failure"
    if status == "non_molecular_solvent":
        return "non_molecular_solvent"
    if status == "unresolved_solvent":
        return "unresolved_solvent"
    if "invalid canonical smiles" in text or "smiles parse" in text:
        return "invalid_smiles"
    if "embedding failed" in text or "etkdgv3" in text:
        return "embedding_failure"
    if "no mmff or uff" in text or "mmff parameters unavailable" in text:
        return "unsupported_molecule"
    if "optimiz" in text or "converge" in text:
        return "optimization_failure"
    return "geometry_generation_failure"


def _geometry_failure(
    *,
    molecule_id: str,
    canonical_smiles: str,
    status: str,
    cache_path_value: Path | None,
    detail: str | None,
) -> dict[str, Any]:
    category = _classify_geometry_failure(status, detail)
    return {
        "molecule_id": molecule_id,
        "canonical_smiles": canonical_smiles,
        "status": status,
        "failure_category": category,
        "failure_reason": category,
        "detail": detail,
        "cache_path": str(cache_path_value) if cache_path_value is not None else None,
    }


def _geometry_success_entry(entry: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "molecule_id": str(entry["molecule_id"]),
        "canonical_smiles": str(entry["canonical_smiles"]),
        "cache_path": str(path),
        "atom_count": int(len(entry["atom_symbols"])),
        "coordinate_sha256": _coordinates_sha256(entry),
        "geometry_status": str(entry.get("geometry_status", "success")),
        "geometry_quality": str(entry.get("geometry_quality")),
        "geometry_support_status": str(entry.get("geometry_support_status", "supported")),
        "force_field_support_status": str(entry.get("force_field_support_status")),
        "model_vocabulary_status": str(entry.get("model_vocabulary_status", "not_evaluated")),
        "optimization_method": str(entry.get("optimization_method")),
        "mmff_available": bool(entry.get("mmff_available", str(entry.get("optimization_method", "")).startswith("MMFF"))),
        "converged": bool(entry.get("convergence_status", {}).get("converged")),
    }


def _failure_rows(
    selected_rows: pd.DataFrame,
    failures: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures_by_molecule = {str(item["molecule_id"]): item for item in failures}
    rows: list[dict[str, Any]] = []
    for _, row in selected_rows.iterrows():
        failure = failures_by_molecule.get(str(row["molecule_id"]))
        if failure is None:
            continue
        rows.append(
            {
                "row_id": str(row["row_id"]),
                "source_row_number": int(row["source_row_number"]),
                "source_dataset": str(row["source_dataset"]),
                "molecule_id": str(row["molecule_id"]),
                "canonical_smiles": failure["canonical_smiles"],
                "solvent_id": str(row["solvent_id"]),
                "canonical_solvent_smiles": None
                if pd.isna(row["canonical_solvent_smiles"])
                else str(row["canonical_solvent_smiles"]),
                "source_canonical_solvent_smiles": None
                if "source_canonical_solvent_smiles" not in row.index or pd.isna(row["source_canonical_solvent_smiles"])
                else str(row["source_canonical_solvent_smiles"]),
                "uniprop_canonical_solvent_smiles": None
                if "uniprop_canonical_solvent_smiles" not in row.index or pd.isna(row["uniprop_canonical_solvent_smiles"])
                else str(row["uniprop_canonical_solvent_smiles"]),
                "failure_category": failure["failure_category"],
                "failure_reason": failure["failure_reason"],
                "detail": failure["detail"],
            }
        )
    return rows


def run_real_geometry_stage(
    selected_bundle: ManifestBundle,
    output_dir: Path,
    *,
    workers: int,
    resume: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    """Generate, validate, and re-hit the selected real-data geometry cache."""
    manifest_dir = output_dir / "manifests"
    geometry_dir = output_dir / "geometry_cache"
    selected_molecule_manifest = manifest_dir / "selected_molecule_manifest.csv"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    selected_bundle.molecule_manifest.to_csv(selected_molecule_manifest, index=False)
    selected_bundle.row_manifest.to_csv(manifest_dir / "selected_row_manifest.csv", index=False)
    first_results = build_geometry_cache(
        selected_molecule_manifest,
        geometry_dir,
        workers=max(1, int(workers)),
        resume=bool(resume),
        overwrite_invalid=False,
        mmff_variant="MMFF94s",
        remove_hydrogens=True,
    )
    molecule_index = selected_bundle.molecule_manifest.set_index("molecule_id")
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for result in first_results:
        molecule_id = str(result.molecule_id)
        canonical_smiles = str(molecule_index.loc[molecule_id, "canonical_isomeric_smiles"])
        path = cache_path(geometry_dir, molecule_id)
        if result.status in {"failed", "invalid_cache"}:
            failures.append(
                _geometry_failure(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    status=result.status,
                    cache_path_value=result.cache_path,
                    detail=result.detail,
                )
            )
            continue
        try:
            entry = read_valid_cache(path, molecule_id, canonical_smiles)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            failures.append(
                _geometry_failure(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    status="invalid_cache",
                    cache_path_value=path,
                    detail=str(exc),
                )
            )
            continue
        method = str(entry.get("optimization_method"))
        converged = bool(entry.get("convergence_status", {}).get("converged"))
        if not converged:
            failures.append(
                _geometry_failure(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    status="optimization_failed",
                    cache_path_value=path,
                    detail=f"{method} did not converge with code {entry.get('convergence_status', {}).get('optimizer_code')}.",
                )
            )
            continue
        if method == "UFF":
            entry = {
                **entry,
                "geometry_status": entry.get("geometry_status", "success"),
                "geometry_quality": entry.get("geometry_quality", "uff_converged"),
                "mmff_available": bool(entry.get("mmff_available", False)),
                "force_field_support_status": entry.get("force_field_support_status", "uff_only"),
                "model_vocabulary_status": entry.get("model_vocabulary_status", "not_evaluated"),
            }
        elif not method.startswith("MMFF"):
            failures.append(
                _geometry_failure(
                    molecule_id=molecule_id,
                    canonical_smiles=canonical_smiles,
                    status="optimization_failed",
                    cache_path_value=path,
                    detail=f"Unsupported optimization method {method}.",
                )
            )
            continue
        successes.append(_geometry_success_entry(entry, path))

    success_ids = {item["molecule_id"] for item in successes}
    success_molecules = selected_bundle.molecule_manifest[
        selected_bundle.molecule_manifest["molecule_id"].astype(str).isin(success_ids)
    ].copy()
    success_rows = selected_bundle.row_manifest[
        selected_bundle.row_manifest["molecule_id"].astype(str).isin(success_ids)
    ].copy()
    success_counts = success_rows.groupby("molecule_id").size()
    if len(success_molecules):
        success_molecules["source_row_count"] = success_molecules["molecule_id"].map(success_counts).astype(int)
    success_molecules = success_molecules.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    success_rows = success_rows.sort_values(["source_row_number", "row_id"], kind="mergesort").reset_index(drop=True)
    success_molecule_manifest = manifest_dir / "successful_molecule_manifest.csv"
    success_molecules.to_csv(success_molecule_manifest, index=False)

    second_results = build_geometry_cache(
        success_molecule_manifest,
        geometry_dir,
        workers=max(1, int(workers)),
        resume=True,
        overwrite_invalid=False,
        mmff_variant="MMFF94s",
        remove_hydrogens=True,
    )
    second_counts = _status_counts(second_results)

    row_geometry: list[dict[str, Any]] = []
    for _, row in success_rows.iterrows():
        molecule_id = str(row["molecule_id"])
        canonical_smiles = str(molecule_index.loc[molecule_id, "canonical_isomeric_smiles"])
        entry = read_valid_cache(cache_path(geometry_dir, molecule_id), molecule_id, canonical_smiles)
        row_geometry.append(
            {
                "row_id": str(row["row_id"]),
                "source_row_number": int(row["source_row_number"]),
                "molecule_id": molecule_id,
                "coordinate_sha256": _coordinates_sha256(entry),
            }
        )

    repeated_groups = []
    row_geometry_frame = pd.DataFrame(row_geometry)
    if len(row_geometry_frame):
        for molecule_id, group in row_geometry_frame.groupby("molecule_id"):
            if len(group) > 1:
                repeated_groups.append(
                    {
                        "molecule_id": str(molecule_id),
                        "row_ids": sorted(group["row_id"].astype(str).tolist()),
                        "source_row_numbers": sorted(group["source_row_number"].astype(int).tolist()),
                        "unique_coordinate_hashes": sorted(group["coordinate_sha256"].unique().tolist()),
                    }
                )

    failed_rows = _failure_rows(selected_bundle.row_manifest, failures)
    geometry_failure_fields = [
        "molecule_id",
        "canonical_smiles",
        "status",
        "failure_category",
        "failure_reason",
        "detail",
        "cache_path",
    ]
    failed_row_fields = [
        "row_id",
        "source_row_number",
        "source_dataset",
        "molecule_id",
        "canonical_smiles",
        "solvent_id",
        "canonical_solvent_smiles",
        "source_canonical_solvent_smiles",
        "uniprop_canonical_solvent_smiles",
        "failure_category",
        "failure_reason",
        "detail",
    ]
    _write_csv(output_dir / "geometry_failures.csv", failures, geometry_failure_fields)
    _write_csv(output_dir / "failed_rows.csv", failed_rows, failed_row_fields)
    _write_json(
        output_dir / "geometry_failures.json",
        {
            "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
            "failures": failures,
        },
    )
    _write_json(
        output_dir / "failed_rows.json",
        {
            "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
            "failed_rows": failed_rows,
        },
    )

    second_cache_hits_ok = (
        len(successes) == 0
        or (
            second_counts.get("hit", 0) == len(successes)
            and second_counts.get("generated", 0) == 0
            and second_counts.get("failed", 0) == 0
            and second_counts.get("invalid_cache", 0) == 0
        )
    )
    repeated_identity_ok = all(len(item["unique_coordinate_hashes"]) == 1 for item in repeated_groups)
    optimization_method_counts: dict[str, int] = {}
    geometry_quality_counts: dict[str, int] = {}
    mmff_available_counts: dict[str, int] = {}
    for success in successes:
        method = str(success.get("optimization_method"))
        quality = str(success.get("geometry_quality"))
        mmff_key = str(bool(success.get("mmff_available")))
        optimization_method_counts[method] = optimization_method_counts.get(method, 0) + 1
        geometry_quality_counts[quality] = geometry_quality_counts.get(quality, 0) + 1
        mmff_available_counts[mmff_key] = mmff_available_counts.get(mmff_key, 0) + 1
    report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "geometry_schema_version": GEOMETRY_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "selected_unique_molecules": int(selected_bundle.row_manifest["molecule_id"].nunique()),
        "successful_geometry_count": int(len(successes)),
        "failed_geometry_count": int(len(failures)),
        "optimization_method_counts": optimization_method_counts,
        "geometry_quality_counts": geometry_quality_counts,
        "mmff_available_counts": mmff_available_counts,
        "first_run_status_counts": _status_counts(first_results),
        "first_run_writes": int(_status_counts(first_results).get("generated", 0)),
        "first_run_existing_hits": int(_status_counts(first_results).get("hit", 0)),
        "second_run_status_counts": second_counts,
        "second_run_cache_hits": int(second_counts.get("hit", 0)),
        "second_run_regenerations": int(second_counts.get("generated", 0)),
        "second_run_cache_hits_only": bool(second_cache_hits_ok),
        "one_geometry_per_successful_unique_chromophore": int(len(successes)) == int(len(success_molecules)),
        "repeated_chromophores_reuse_geometry": bool(repeated_identity_ok),
        "successes": successes,
        "repeated_chromophore_groups": repeated_groups,
        "failure_reports": {
            "geometry_json": str(output_dir / "geometry_failures.json"),
            "geometry_csv": str(output_dir / "geometry_failures.csv"),
            "failed_rows_json": str(output_dir / "failed_rows.json"),
            "failed_rows_csv": str(output_dir / "failed_rows.csv"),
        },
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "geometry_validation_report.json", report)
    return success_rows, success_molecules, failed_rows, report


def build_success_manifests(
    selected_bundle: ManifestBundle,
    success_rows: pd.DataFrame,
    success_molecules: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
) -> tuple[ManifestBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if len(success_rows) < 2 or success_rows["molecule_id"].nunique() < 2:
        raise ValueError("At least two successfully embedded molecules are required for split and LMDB smoke validation.")
    targets = tuple(selected_bundle.metadata["target_columns"])
    metadata = {
        **selected_bundle.metadata,
        "source_rows": int(len(success_rows)),
        "manifest_rows": int(len(success_rows)),
        "unique_molecules": int(success_rows["molecule_id"].nunique()),
        "unique_solvents": int(success_rows["solvent_id"].nunique()),
        "selected_source_rows": int(selected_bundle.metadata["selection"]["selected_source_rows"]),
        "successfully_exportable_rows": int(len(success_rows)),
        "target_coverage": _target_coverage(success_rows, targets),
        "solvent_coverage": _solvent_coverage(success_rows),
    }
    success_bundle = ManifestBundle(selected_bundle.source_path, success_molecules, success_rows, metadata)
    validate_manifest_reconciliation(success_bundle)

    split_assignments = make_split_assignments(
        success_bundle.row_manifest,
        success_bundle.molecule_manifest,
        seed=int(seed),
        test_size=0.2,
        compute_rdkit_scaffolds=False,
    )
    leakage = audit_split_leakage(success_bundle.row_manifest, success_bundle.molecule_manifest, split_assignments)
    stats = split_statistics(success_bundle.row_manifest, success_bundle.molecule_manifest, split_assignments, targets)

    manifest_dir = output_dir / "manifests"
    success_bundle.row_manifest.to_csv(manifest_dir / "row_manifest.csv", index=False)
    success_bundle.molecule_manifest.to_csv(manifest_dir / "molecule_manifest.csv", index=False)
    split_assignments.to_csv(manifest_dir / "split_assignments.csv", index=False)
    leakage.to_csv(manifest_dir / "split_leakage_audit.csv", index=False)
    stats.to_csv(manifest_dir / "split_statistics.csv", index=False)
    _write_json(manifest_dir / "manifest_metadata.json", {**metadata, "all_leakage_audits_passed": bool(leakage["passed"].all())})
    return success_bundle, split_assignments, leakage, stats, metadata


def export_real_lmdb(
    output_dir: Path,
    targets: Sequence[str],
    *,
    seed: int,
    resume: bool,
    overwrite: bool,
) -> dict[str, Any]:
    manifest_dir = output_dir / "manifests"
    lmdb_dir = output_dir / "lmdb"
    config = ExportConfig(
        row_manifest_path=manifest_dir / "row_manifest.csv",
        molecule_manifest_path=manifest_dir / "molecule_manifest.csv",
        split_assignments_path=manifest_dir / "split_assignments.csv",
        geometry_cache_dir=output_dir / "geometry_cache",
        output_dir=lmdb_dir,
        split_family="random",
        seed=int(seed),
        target_columns=tuple(targets),
        map_size=512 * 1024 * 1024,
        batch_size=128,
        overwrite=bool(overwrite),
        resume=bool(resume),
        valid_size=0.1,
    )
    metadata = export_uniprop_lmdb(config)
    validation = {
        partition: validate_lmdb(lmdb_dir / f"{partition}.lmdb", target_columns=tuple(targets))
        for partition in REAL_DATA_SMOKE_PARTITIONS
    }
    report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "metadata": metadata,
        "partitions": validation,
        "all_valid": all(item["valid"] for item in validation.values()),
        "lmdb_counts": {partition: int(validation[partition]["rows"]) for partition in REAL_DATA_SMOKE_PARTITIONS},
        "total_rows": int(sum(validation[partition]["rows"] for partition in REAL_DATA_SMOKE_PARTITIONS)),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "lmdb_validation_report.json", report)
    return report


def _all_lmdb_records(lmdb_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for partition in REAL_DATA_SMOKE_PARTITIONS:
        for _, record in read_lmdb_records(lmdb_dir / f"{partition}.lmdb"):
            copied = dict(record)
            copied["lmdb_partition"] = partition
            records.append(copied)
    return records


def validate_lmdb_source_identity(
    row_manifest: pd.DataFrame,
    lmdb_dir: Path,
    targets: Sequence[str],
) -> dict[str, Any]:
    source = row_manifest.set_index("row_id", drop=False)
    records = _all_lmdb_records(lmdb_dir)
    errors: list[str] = []
    exported_row_ids = [str(record["row_id"]) for record in records]
    if len(exported_row_ids) != len(set(exported_row_ids)):
        errors.append("duplicate exported row_id values")
    if set(exported_row_ids) != set(source.index.astype(str)):
        errors.append("exported row_ids do not match successful row manifest")

    solvent_mismatches = 0
    source_row_mismatches = 0
    target_mismatches = 0
    mask_mismatches = 0
    for record in records:
        row_id = str(record["row_id"])
        if row_id not in source.index:
            continue
        row = source.loc[row_id]
        expected_solvent = "" if pd.isna(row["canonical_solvent_smiles"]) else str(row["canonical_solvent_smiles"])
        if str(record.get("solvent_smi", "")) != expected_solvent:
            solvent_mismatches += 1
        if "source_row_number" not in record:
            source_row_mismatches += 1
        elif int(record["source_row_number"]) != int(row["source_row_number"]):
            source_row_mismatches += 1
        target_columns = [str(item) for item in np.asarray(record.get("target_columns", targets)).tolist()]
        target_values = np.asarray(record["target"])
        target_mask = np.asarray(record["target_mask"], dtype=np.bool_)
        for target in targets:
            index = target_columns.index(target)
            expected_available = not pd.isna(row[target]) if target in row.index else False
            if bool(target_mask[index]) != bool(expected_available):
                mask_mismatches += 1
            if expected_available:
                expected = np.asarray(row[target], dtype=np.float32)
                actual = np.asarray(target_values[index], dtype=np.float32)
                if not np.array_equal(actual, expected):
                    target_mismatches += 1
            elif not pd.isna(target_values[index]):
                target_mismatches += 1

    repeated_groups = []
    for molecule_id, group_records in pd.DataFrame(
        [
            {
                "row_id": str(record["row_id"]),
                "molecule_id": str(record["molecule_id"]),
                "coordinate_sha256": hashlib.sha256(
                    np.asarray(record["label_pos"], dtype=np.float32).tobytes()
                ).hexdigest(),
            }
            for record in records
        ]
    ).groupby("molecule_id"):
        if len(group_records) > 1:
            repeated_groups.append(
                {
                    "molecule_id": str(molecule_id),
                    "row_ids": sorted(group_records["row_id"].astype(str).tolist()),
                    "unique_coordinate_hashes": sorted(group_records["coordinate_sha256"].unique().tolist()),
                }
            )
    repeated_identity = all(len(item["unique_coordinate_hashes"]) == 1 for item in repeated_groups)

    passed = (
        not errors
        and solvent_mismatches == 0
        and source_row_mismatches == 0
        and target_mismatches == 0
        and mask_mismatches == 0
        and repeated_identity
    )
    return {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "row_identity_passed": bool(passed),
        "exported_rows": int(len(records)),
        "source_rows": int(len(row_manifest)),
        "errors": errors,
        "solvent_mismatches": int(solvent_mismatches),
        "source_row_number_mismatches": int(source_row_mismatches),
        "target_value_mismatches": int(target_mismatches),
        "mask_mismatches": int(mask_mismatches),
        "target_values_unchanged": int(target_mismatches) == 0,
        "masks_match_missingness": int(mask_mismatches) == 0,
        "repeated_molecule_coordinate_identity": bool(repeated_identity),
        "repeated_molecule_groups": repeated_groups,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


class _RecordDataset(FluorCastUniPropSmokeDataset):
    def __init__(self, records: list[dict[str, Any]], *, targets: Sequence[str], solvent_feature_dim: int) -> None:
        self.lmdb_path = Path("<in-memory-real-data-smoke>")
        self.targets = tuple(targets)
        self.solvent_feature_dim = int(solvent_feature_dim)
        self.records = records


def fit_real_smoke_target_normalizer(samples: list[dict[str, Any]], targets: Sequence[str]) -> dict[str, Any]:
    values = np.stack([sample["target"] for sample in samples]).astype(np.float32)
    masks = np.stack([sample["target_mask"] for sample in samples]).astype(bool)
    means: list[float] = []
    scales: list[float] = []
    counts: list[int] = []
    missing_targets: list[str] = []
    for index, target in enumerate(targets):
        available = values[masks[:, index], index]
        if available.size == 0:
            means.append(0.0)
            scales.append(1.0)
            counts.append(0)
            missing_targets.append(str(target))
            continue
        mean = float(available.mean())
        std = float(available.std())
        means.append(mean)
        scales.append(std if std > 1.0e-8 else 1.0)
        counts.append(int(available.size))
    return {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "targets": list(targets),
        "mean": means,
        "scale": scales,
        "available_counts": counts,
        "targets_without_training_labels": missing_targets,
        "fit_partition": "train",
        "fit_row_ids": [sample["row_id"] for sample in samples],
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }


def _select_labeled_batch_samples(dataset: FluorCastUniPropSmokeDataset, batch_size: int) -> list[dict[str, Any]]:
    samples = [dataset[index] for index in range(len(dataset))]
    labeled = [sample for sample in samples if bool(np.asarray(sample["target_mask"]).any())]
    if not labeled:
        raise ValueError("Training partition has no available labels for the masked smoke loss.")
    return labeled[: max(1, min(batch_size, len(labeled)))]


def save_real_smoke_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    config: Tiny3DSmokeConfig,
    normalizer: dict[str, Any],
    update_index: int,
    loss: float,
) -> None:
    torch = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
            "model_kind": TINY_3D_SMOKE_MODEL_KIND,
            "real_uniprop_used": False,
            "real_checkpoint_loaded": False,
            "tiny_backbone_used": True,
            "checkpoint_kind": "windows_real_data_smoke",
            "update_index": int(update_index),
            "loss": float(loss),
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer": normalizer,
            "torch_random_state": torch.get_rng_state(),
        },
        path,
    )


def load_real_smoke_checkpoint(path: Path) -> tuple[Any, Any, Tiny3DSmokeConfig, dict[str, Any], dict[str, Any]]:
    torch = _require_torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported real-data smoke checkpoint schema: {checkpoint.get('schema_version')}")
    if checkpoint.get("model_kind") != TINY_3D_SMOKE_MODEL_KIND:
        raise ValueError("Checkpoint is not a tiny 3D smoke backbone checkpoint.")
    config_payload = dict(checkpoint["model_config"])
    config_payload["targets"] = tuple(config_payload["targets"])
    config = Tiny3DSmokeConfig(**config_payload)
    model = Tiny3DSmokeModel.build(torch, config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, config, checkpoint["normalizer"], checkpoint


def run_training_stage(output_dir: Path, targets: Sequence[str], seed: int) -> dict[str, Any]:
    torch = _require_torch()
    config = Tiny3DSmokeConfig(targets=tuple(targets), seed=int(seed))
    lmdb_dir = output_dir / "lmdb"
    train_dataset = FluorCastUniPropSmokeDataset(
        lmdb_dir / "train.lmdb",
        targets=config.targets,
        solvent_feature_dim=config.solvent_feature_dim,
    )
    if len(train_dataset) == 0:
        raise ValueError("Train LMDB is empty; cannot run real-data smoke training.")
    train_samples = [train_dataset[index] for index in range(len(train_dataset))]
    train_batch_samples = _select_labeled_batch_samples(train_dataset, TRAINING_BATCH_SIZE)
    train_batch = train_dataset.collater(train_batch_samples)
    shape_report = tensor_shape_report(train_batch)
    shape_report["schema_version"] = WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION
    shape_report["profile"] = WINDOWS_REAL_DATA_SMOKE_PROFILE
    _write_json(output_dir / "tensor_shape_report.json", shape_report)

    normalizer = fit_real_smoke_target_normalizer(train_samples, config.targets)
    _write_json(output_dir / "training_normalization.json", normalizer)

    model = build_tiny_smoke_model(torch, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    forward_report = finite_forward_report(torch, model, train_batch)
    forward_report["schema_version"] = WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION
    forward_report["profile"] = WINDOWS_REAL_DATA_SMOKE_PROFILE
    before = parameter_state(model)
    loss, grad_report = smoke_train_step(torch, model, optimizer, train_batch, normalizer)
    grad_report["schema_version"] = WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION
    grad_report["profile"] = WINDOWS_REAL_DATA_SMOKE_PROFILE
    changed = changed_parameter_names(torch, before, model)
    loss_report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "loss_values": [{"update_index": 0, "masked_multitask_mse": float(loss)}],
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    changed_report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "changed_parameter_names": changed,
        "changed_parameter_count": int(len(changed)),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "loss_values.json", loss_report)
    _write_json(output_dir / "gradient_statistics.json", grad_report)
    _write_json(output_dir / "changed_parameter_names.json", changed_report)

    checkpoint_path = output_dir / "checkpoints" / "checkpoint.pt"
    save_real_smoke_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        normalizer=normalizer,
        update_index=0,
        loss=loss,
    )
    before_reload = predictions_numpy(torch, model, train_batch, normalizer)
    reloaded_model, reloaded_optimizer, reloaded_config, reloaded_normalizer, checkpoint = load_real_smoke_checkpoint(checkpoint_path)
    after_reload = predictions_numpy(torch, reloaded_model, train_batch, reloaded_normalizer)
    reload_identical = bool(np.allclose(before_reload, after_reload, rtol=1e-6, atol=1e-6))
    checkpoint_report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "checkpoint": str(checkpoint_path),
        "reload_predictions_identical": reload_identical,
        "checkpoint_update_index": int(checkpoint["update_index"]),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "checkpoint_reload_report.json", checkpoint_report)
    return {
        "config": reloaded_config,
        "model": reloaded_model,
        "optimizer": reloaded_optimizer,
        "normalizer": reloaded_normalizer,
        "shape_report": shape_report,
        "forward_report": forward_report,
        "gradient_report": grad_report,
        "loss_report": loss_report,
        "changed_report": changed_report,
        "checkpoint_report": checkpoint_report,
    }


def _predict_records(
    records: list[dict[str, Any]],
    *,
    model: Any,
    normalizer: dict[str, Any],
    config: Tiny3DSmokeConfig,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    dataset = _RecordDataset(records, targets=config.targets, solvent_feature_dim=config.solvent_feature_dim)
    prediction_rows: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(dataset), PREDICTION_BATCH_SIZE):
        items = [dataset[index] for index in range(start, min(start + PREDICTION_BATCH_SIZE, len(dataset)))]
        batch = dataset.collater(items)
        with torch.no_grad():
            values = inverse_scaled_predictions(model(batch).detach().cpu().numpy(), normalizer)
        if not bool(torch.isfinite(torch.as_tensor(values)).all().item()):
            raise FloatingPointError("Real-data smoke produced non-finite predictions.")
        for item, prediction in zip(items, values, strict=True):
            row = {
                "row_id": item["row_id"],
                "molecule_id": item["molecule_id"],
                "solvent_id": item["solvent_id"],
            }
            for index, target in enumerate(config.targets):
                row[f"predicted_{target}"] = float(prediction[index])
            prediction_rows.append(row)
    return prediction_rows


def write_predictions_and_join_report(
    output_dir: Path,
    row_manifest: pd.DataFrame,
    *,
    model: Any,
    normalizer: dict[str, Any],
    config: Tiny3DSmokeConfig,
) -> dict[str, Any]:
    records = _all_lmdb_records(output_dir / "lmdb")
    prediction_rows = _predict_records(records, model=model, normalizer=normalizer, config=config)
    predictions = pd.DataFrame(prediction_rows)
    source = row_manifest[
        ["row_id", "source_row_number", "source_dataset", "molecule_id", "solvent_id", "canonical_solvent_smiles"]
    ].copy()
    joined = predictions.merge(source, on=["row_id", "molecule_id", "solvent_id"], how="left", validate="one_to_one")
    missing_source_rows = int(joined["source_row_number"].isna().sum())
    duplicate_predictions = int(predictions["row_id"].duplicated().sum()) if len(predictions) else 0
    row_ids_match = set(predictions["row_id"].astype(str)) == set(row_manifest["row_id"].astype(str))
    joined = joined.sort_values("source_row_number", kind="mergesort").reset_index(drop=True)
    joined.to_csv(output_dir / "predictions.csv", index=False)
    _write_json(
        output_dir / "predictions.json",
        {
            "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
            "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
            "model_kind": TINY_3D_SMOKE_MODEL_KIND,
            "prediction_rows": joined.to_dict("records"),
            "real_uniprop_used": False,
            "real_checkpoint_loaded": False,
            "tiny_backbone_used": True,
            "warning": "Tiny-backbone outputs are smoke-test values, not scientific UniProp performance.",
        },
    )
    report = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "prediction_rows": int(len(predictions)),
        "source_rows": int(len(row_manifest)),
        "row_ids_match": bool(row_ids_match),
        "missing_source_rows": missing_source_rows,
        "duplicate_prediction_rows": duplicate_predictions,
        "join_passed": bool(row_ids_match and missing_source_rows == 0 and duplicate_predictions == 0),
        "predictions": str(output_dir / "predictions.csv"),
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
    }
    _write_json(output_dir / "prediction_join_report.json", report)
    return report


def _leakage_summary(leakage: pd.DataFrame) -> dict[str, Any]:
    return {
        "passed": bool(leakage["passed"].all()),
        "families": [
            {
                key: _jsonable(value)
                for key, value in row.items()
            }
            for row in leakage.to_dict("records")
        ],
    }


def _stage(passed: bool, **extra: Any) -> dict[str, Any]:
    return {"status": "passed" if passed else "failed", **extra}


def run_windows_real_data_smoke(
    output_dir: Path,
    *,
    dataset: Path | None = None,
    max_molecules: int | None = 20,
    max_rows: int | None = None,
    seed: int = 42,
    workers: int = 1,
    resume: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if overwrite:
            shutil.rmtree(output_dir)
        elif not resume:
            raise FileExistsError(f"Output directory is not empty; pass --resume or --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stages: dict[str, dict[str, Any]] = {}
    environment = windows_smoke_environment_report()
    environment["schema_version"] = WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION
    environment["profile"] = WINDOWS_REAL_DATA_SMOKE_PROFILE
    _write_json(output_dir / "environment_report.json", environment)
    stages["environment_report"] = _stage(True)

    selected_bundle, selected_metadata = build_selected_real_manifests(
        output_dir,
        dataset=dataset,
        max_molecules=max_molecules,
        max_rows=max_rows,
        seed=seed,
    )
    molecular_mask, gas_mask, unresolved_mask = _solvent_terminal_masks(selected_bundle.row_manifest)
    gas_rows = selected_bundle.row_manifest[gas_mask].copy()
    unresolved_rows = selected_bundle.row_manifest[unresolved_mask].copy()
    geometry_input_rows = selected_bundle.row_manifest[molecular_mask].copy()
    geometry_bundle = _subset_bundle_for_rows(selected_bundle, geometry_input_rows)
    solvent_terminal_report = {
        "selected_source_rows": int(len(selected_bundle.row_manifest)),
        "molecular_solvent_rows": int(len(geometry_input_rows)),
        "gas_phase_rows": int(len(gas_rows)),
        "unresolved_solvent_rows": int(len(unresolved_rows)),
    }
    _write_json(output_dir / "solvent_terminal_rows.json", solvent_terminal_report)
    stages["selected_manifests"] = _stage(
        True,
        selected_unique_molecules=int(selected_metadata["selection"]["selected_unique_molecules"]),
        **solvent_terminal_report,
    )

    success_rows, success_molecules, failed_rows, geometry_report = run_real_geometry_stage(
        geometry_bundle,
        output_dir,
        workers=workers,
        resume=resume,
    )
    geometry_passed = (
        bool(geometry_report["second_run_cache_hits_only"])
        and bool(geometry_report["one_geometry_per_successful_unique_chromophore"])
        and bool(geometry_report["repeated_chromophores_reuse_geometry"])
        and len(success_rows) > 0
    )
    stages["geometry"] = _stage(
        geometry_passed,
        successful_geometry_count=int(geometry_report["successful_geometry_count"]),
        failed_geometry_count=int(geometry_report["failed_geometry_count"]),
        first_run_writes=int(geometry_report["first_run_writes"]),
        second_run_cache_hits=int(geometry_report["second_run_cache_hits"]),
    )

    success_bundle, split_assignments, leakage, stats, success_metadata = build_success_manifests(
        geometry_bundle,
        success_rows,
        success_molecules,
        output_dir,
        seed=seed,
    )
    leakage_summary = _leakage_summary(leakage)
    stages["split_leakage"] = _stage(bool(leakage_summary["passed"]))

    lmdb_report = export_real_lmdb(
        output_dir,
        success_bundle.metadata["target_columns"],
        seed=seed,
        resume=resume,
        overwrite=overwrite,
    )
    stages["lmdb"] = _stage(bool(lmdb_report["all_valid"]), **lmdb_report["lmdb_counts"])

    identity_report = validate_lmdb_source_identity(
        success_bundle.row_manifest,
        output_dir / "lmdb",
        success_bundle.metadata["target_columns"],
    )
    _write_json(output_dir / "source_row_reconciliation.json", identity_report)
    stages["source_row_reconciliation"] = _stage(bool(identity_report["row_identity_passed"]))

    selected_source_rows = int(selected_bundle.metadata["selection"]["selected_source_rows"])
    exported_rows = int(lmdb_report["total_rows"])
    explicitly_failed_rows = int(len(failed_rows))
    gas_phase_rows = int(len(gas_rows))
    unresolved_solvent_rows = int(len(unresolved_rows))
    accounting = {
        "selected_source_rows": selected_source_rows,
        "exported_molecular_solvent_rows": exported_rows,
        "geometry_failed_rows": explicitly_failed_rows,
        "gas_phase_rows": gas_phase_rows,
        "genuinely_unresolved_solvent_rows": unresolved_solvent_rows,
        "identity_holds": selected_source_rows
        == exported_rows + explicitly_failed_rows + gas_phase_rows + unresolved_solvent_rows,
    }
    stages["selected_rows_accounting"] = _stage(bool(accounting["identity_holds"]), **accounting)
    if not accounting["identity_holds"]:
        raise ValueError(f"Selected/exported/failed row accounting does not reconcile: {accounting}")

    training = run_training_stage(output_dir, success_bundle.metadata["target_columns"], seed)
    stages["dataset_adapter_batch"] = _stage(bool(training["shape_report"]["target_mask_true"] > 0))
    stages["forward"] = _stage(bool(training["forward_report"]["all_finite"]), shape=training["forward_report"]["shape"])
    stages["backward"] = _stage(bool(training["gradient_report"]["all_finite"]))
    stages["optimizer_step"] = _stage(
        bool(training["changed_report"]["changed_parameter_names"]),
        changed_parameter_count=int(training["changed_report"]["changed_parameter_count"]),
    )
    stages["checkpoint_reload"] = _stage(bool(training["checkpoint_report"]["reload_predictions_identical"]))

    prediction_join = write_predictions_and_join_report(
        output_dir,
        success_bundle.row_manifest,
        model=training["model"],
        normalizer=training["normalizer"],
        config=training["config"],
    )
    stages["prediction_source_join"] = _stage(bool(prediction_join["join_passed"]))

    all_passed = all(item["status"] == "passed" for item in stages.values())
    summary = {
        "schema_version": WINDOWS_REAL_DATA_SMOKE_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "profile": WINDOWS_REAL_DATA_SMOKE_PROFILE,
        "model_kind": TINY_3D_SMOKE_MODEL_KIND,
        "real_uniprop_used": False,
        "real_checkpoint_loaded": False,
        "tiny_backbone_used": True,
        "all_stages_passed": bool(all_passed),
        "source_dataset": {
            "path": str(selected_bundle.source_path),
            "sha256": selected_metadata["source_dataset_sha256"],
            "full_source_rows": int(selected_metadata["full_source_rows"]),
            "full_unique_molecules": int(selected_metadata["full_unique_molecules"]),
        },
        "selected_rows": int(selected_source_rows),
        "unique_molecules": int(selected_metadata["selection"]["selected_unique_molecules"]),
        "solvent_coverage": selected_metadata["solvent_coverage"],
        "target_coverage": selected_metadata["target_coverage"],
        "successful_geometry_count": int(geometry_report["successful_geometry_count"]),
        "failed_geometry_count": int(geometry_report["failed_geometry_count"]),
        "first_run_writes": int(geometry_report["first_run_writes"]),
        "second_run_cache_hits": int(geometry_report["second_run_cache_hits"]),
        "lmdb_counts": lmdb_report["lmdb_counts"],
        "leakage_audit": leakage_summary,
        "tensor_shapes": training["shape_report"]["shapes"],
        "gradient_results": {
            "all_finite": bool(training["gradient_report"]["all_finite"]),
            "parameter_count": int(len(training["gradient_report"]["parameters"])),
        },
        "optimizer_results": {
            "changed_parameter_count": int(training["changed_report"]["changed_parameter_count"]),
            "changed_parameter_names": training["changed_report"]["changed_parameter_names"],
        },
        "source_row_reconciliation": {
            **accounting,
            "row_identity_passed": bool(identity_report["row_identity_passed"]),
            "target_values_unchanged": bool(identity_report["target_values_unchanged"]),
            "masks_match_missingness": bool(identity_report["masks_match_missingness"]),
            "predictions_join_back_to_source": bool(prediction_join["join_passed"]),
        },
        "stages": stages,
        "artifacts": {
            "summary": str(output_dir / "summary.json"),
            "environment_report": str(output_dir / "environment_report.json"),
            "selected_row_manifest": str(output_dir / "manifests" / "selected_row_manifest.csv"),
            "selected_molecule_manifest": str(output_dir / "manifests" / "selected_molecule_manifest.csv"),
            "row_manifest": str(output_dir / "manifests" / "row_manifest.csv"),
            "molecule_manifest": str(output_dir / "manifests" / "molecule_manifest.csv"),
            "split_assignments": str(output_dir / "manifests" / "split_assignments.csv"),
            "split_leakage_audit": str(output_dir / "manifests" / "split_leakage_audit.csv"),
            "geometry_validation_report": str(output_dir / "geometry_validation_report.json"),
            "geometry_failures_json": str(output_dir / "geometry_failures.json"),
            "geometry_failures_csv": str(output_dir / "geometry_failures.csv"),
            "failed_rows_json": str(output_dir / "failed_rows.json"),
            "failed_rows_csv": str(output_dir / "failed_rows.csv"),
            "lmdb_validation_report": str(output_dir / "lmdb_validation_report.json"),
            "source_row_reconciliation": str(output_dir / "source_row_reconciliation.json"),
            "tensor_shape_report": str(output_dir / "tensor_shape_report.json"),
            "training_normalization": str(output_dir / "training_normalization.json"),
            "loss_values": str(output_dir / "loss_values.json"),
            "gradient_statistics": str(output_dir / "gradient_statistics.json"),
            "changed_parameter_names": str(output_dir / "changed_parameter_names.json"),
            "checkpoint": str(output_dir / "checkpoints" / "checkpoint.pt"),
            "checkpoint_reload_report": str(output_dir / "checkpoint_reload_report.json"),
            "predictions": str(output_dir / "predictions.csv"),
            "prediction_join_report": str(output_dir / "prediction_join_report.json"),
        },
        "warnings": [
            "This profile uses Tiny3DSmokeBackbone only; do not report predictions or losses as scientific UniProp model performance."
        ],
        "nibi_only_gate": [
            "Audit nibi-real readiness with Uni-Core, Uni-Mol+, staged checkpoint hashes, and the intended CPU/GPU device.",
            "Then run a tiny real UniProp checkpoint load plus forward/backward smoke on Nibi before full training.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary
