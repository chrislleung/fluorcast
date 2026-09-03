"""Predict custom molecule cases with trained FluorCast-ConforFormer models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chemfluor.conforformer.adapter import ConforFormerEncoderAdapter, sha256_file  # noqa: E402
from chemfluor.conforformer.cache import sha256_payload, stable_json_dumps  # noqa: E402
from chemfluor.conforformer.config import ConformerGenerationConfig  # noqa: E402
from chemfluor.conforformer.conformers import canonicalize_smiles, generate_conformer_cache_record  # noqa: E402
from chemfluor.conforformer.dictionary import load_conforformer_dictionary  # noqa: E402
from chemfluor.conforformer.downstream import (  # noqa: E402
    TARGETS,
    feature_names,
    load_solvent_descriptors,
    merge_solvent_descriptors,
    morgan_fingerprint,
)
from chemfluor.conforformer.embedding_store import EXPECTED_EMBEDDING_DIM  # noqa: E402
from chemfluor.conforformer.pooling import pool_all, pooling_configuration  # noqa: E402
from chemfluor.conforformer.preprocess import (  # noqa: E402
    ConforFormerPreprocessingConfig,
    collate_preprocessed_conformers,
    preprocess_successful_conformers,
)
from chemfluor.conforformer.schemas import MoleculeStatus  # noqa: E402
from scripts.make_deep4chem_solvent_descriptors import (  # noqa: E402
    DESCRIPTOR_COLUMNS as RDKIT_SOLVENT_DESCRIPTOR_COLUMNS,
    compute_rdkit_descriptors,
    import_rdkit,
)


DEFAULT_MODEL_PARENT = PROJECT_ROOT / "models" / "conforformer_downstream"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "conforformer" / "ConforFormer.pt"
DEFAULT_DICTIONARY = PROJECT_ROOT / "configs" / "conforformer" / "OMOL_full_dict.txt"
DEFAULT_SOLVENT_DESCRIPTORS = PROJECT_ROOT / "data" / "solvent_descriptors_expanded_deep4chem.csv"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "conforformer_inference"
DEFAULT_SPLIT = "molecule"
DEFAULT_POOLING = "mean"
DEFAULT_FEATURE_SET = "conforformer_morgan_solvent"
MODEL_FILE = "model.joblib"
METADATA_FILE = "feature_metadata.json"
OUTPUT_COLUMNS = [
    "name",
    "smiles",
    "canonical_smiles",
    "solvent_smiles",
    "canonical_solvent_smiles",
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "stokes_shift_nm",
    "derived_stokes_shift_nm",
    "stokes_source",
    "status",
    "error",
]


@dataclass(frozen=True)
class TargetModel:
    target: str
    model_dir: Path
    model_path: Path
    metadata_path: Path
    estimator: Any
    feature_order: list[str]
    pooling_method: str
    feature_set: str
    selected_candidate: str | None


@dataclass(frozen=True)
class CaseFeatures:
    canonical_smiles: str
    canonical_solvent_smiles: str
    frame: pd.DataFrame
    embedding_cache_key: str | None
    embedding_cache_hit: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smiles", help="Chromophore/molecule SMILES for single-case prediction.")
    mode.add_argument("--input-csv", type=Path, help="Batch CSV with smiles and solvent_smiles columns.")
    parser.add_argument("--solvent-smiles", help="Solvent SMILES for single-case prediction.")
    parser.add_argument("--output-csv", type=Path, help="Batch output CSV path.")
    parser.add_argument("--model-root", type=Path, default=None, help="Downstream model root or run root.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--solvent-descriptors", type=Path, default=DEFAULT_SOLVENT_DESCRIPTORS)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--json", action="store_true", help="Emit JSON for single-case prediction.")
    parser.add_argument("--split-type", choices=["molecule", "scaffold"], default=DEFAULT_SPLIT)
    parser.add_argument("--pooling", choices=["mean", "lowest_energy", "boltzmann_298k"], default=DEFAULT_POOLING)
    parser.add_argument("--feature-set", choices=["conforformer_solvent", "morgan_solvent", "conforformer_morgan_solvent"], default=DEFAULT_FEATURE_SET)
    args = parser.parse_args(argv)
    if args.smiles and not args.solvent_smiles:
        parser.error("--solvent-smiles is required with --smiles")
    if args.input_csv and args.solvent_smiles:
        parser.error("--solvent-smiles is only valid with --smiles")
    if args.input_csv and args.json:
        parser.error("--json is only valid with single-case --smiles mode")
    return args


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_root(model_root: Path | None) -> Path:
    if model_root is not None:
        return Path(model_root)
    parent = DEFAULT_MODEL_PARENT
    if not parent.exists():
        raise FileNotFoundError(
            f"No default downstream model directory exists at {parent}. "
            "Pass --model-root pointing to a trained ConforFormer downstream run."
        )
    candidates: list[tuple[str, Path]] = []
    for run_dir in parent.iterdir():
        if not run_dir.is_dir():
            continue
        model_dir = run_dir / DEFAULT_SPLIT / DEFAULT_POOLING / DEFAULT_FEATURE_SET
        if any((model_dir / target / MODEL_FILE).exists() for target in TARGETS):
            manifest_path = PROJECT_ROOT / "outputs" / "conforformer" / "downstream" / run_dir.name / "training_manifest.json"
            created_at = _read_json(manifest_path).get("created_at", "") if manifest_path.exists() else ""
            candidates.append((str(created_at), run_dir))
    if not candidates:
        raise FileNotFoundError(
            f"No trained downstream model runs found under {parent}. "
            "Pass --model-root pointing to a run containing model.joblib artifacts."
        )
    if len(candidates) == 1:
        return candidates[0][1]
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates[0][0]:
        raise ValueError(
            "Multiple downstream model runs were found and at least one lacks "
            "training_manifest.json created_at metadata. Pass --model-root explicitly."
        )
    return candidates[0][1]


def target_model_dir(model_root: Path, *, split_type: str, pooling: str, feature_set: str, target: str) -> Path:
    root = Path(model_root)
    direct = root / split_type / pooling / feature_set / target
    if direct.exists():
        return direct
    if root.name == target and (root / MODEL_FILE).exists():
        return root
    return direct


def load_target_models(
    model_root: Path,
    *,
    split_type: str = DEFAULT_SPLIT,
    pooling: str = DEFAULT_POOLING,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> dict[str, TargetModel]:
    models: dict[str, TargetModel] = {}
    for target in TARGETS:
        model_dir = target_model_dir(model_root, split_type=split_type, pooling=pooling, feature_set=feature_set, target=target)
        model_path = model_dir / MODEL_FILE
        metadata_path = model_dir / METADATA_FILE
        if not model_path.exists():
            continue
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing feature metadata for {target}: {metadata_path}")
        metadata = _read_json(metadata_path)
        feature_order = metadata.get("feature_order")
        if not isinstance(feature_order, list) or not feature_order:
            raise ValueError(f"Invalid or missing feature_order in {metadata_path}")
        models[target] = TargetModel(
            target=target,
            model_dir=model_dir,
            model_path=model_path,
            metadata_path=metadata_path,
            estimator=joblib.load(model_path),
            feature_order=[str(name) for name in feature_order],
            pooling_method=str(metadata.get("pooling_method", pooling)),
            feature_set=str(metadata.get("feature_set", feature_set)),
            selected_candidate=metadata.get("selected_candidate"),
        )
    required = {"absorption_nm", "emission_nm", "quantum_yield"}
    missing = sorted(required.difference(models))
    if missing:
        raise FileNotFoundError(f"Missing required trained target model(s) under {model_root}: {', '.join(missing)}")
    return models


def validate_smiles(smiles: str, *, label: str) -> str:
    canonical, _isomeric = canonicalize_smiles(smiles)
    if canonical is None:
        raise ValueError(f"Invalid {label} SMILES: {smiles}")
    return canonical


def _read_upstream_commit() -> str:
    path = PROJECT_ROOT / "configs" / "conforformer" / "upstream_commit.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else "unknown"


def _embedding_cache_key(
    *,
    canonical_smiles: str,
    checkpoint_path: Path,
    dictionary_path: Path,
    dictionary_sha256: str,
    conformer_config: ConformerGenerationConfig,
    preprocess_config: ConforFormerPreprocessingConfig,
    pooling_method: str,
) -> str:
    payload = {
        "canonical_smiles": canonical_smiles,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "dictionary_path": str(dictionary_path),
        "dictionary_sha256": dictionary_sha256,
        "conformer_configuration": conformer_config.to_payload(),
        "preprocessing_configuration": preprocess_config.to_payload(),
        "pooling_method": pooling_method,
        "pooling_configuration": pooling_configuration(),
        "upstream_commit": _read_upstream_commit(),
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
    }
    return sha256_payload(payload)


def _load_cached_embedding(cache_dir: Path, cache_key: str) -> np.ndarray | None:
    path = cache_dir / f"{cache_key}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    try:
        if str(data["cache_key"]) != cache_key:
            return None
        embedding = data["pooled_embedding"].astype(np.float32)
    finally:
        data.close()
    if embedding.shape != (EXPECTED_EMBEDDING_DIM,) or not np.isfinite(embedding).all():
        return None
    return embedding


def _write_cached_embedding(cache_dir: Path, cache_key: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cache_key}.npz"
    np.savez_compressed(
        path,
        cache_key=np.asarray(cache_key),
        pooled_embedding=np.asarray(embedding, dtype=np.float32),
        metadata_json=np.asarray(stable_json_dumps(metadata)),
    )


def embed_smiles(
    smiles: str,
    *,
    adapter: ConforFormerEncoderAdapter,
    dictionary: Any,
    checkpoint_path: Path,
    dictionary_path: Path,
    cache_dir: Path,
    pooling_method: str,
    conformer_config: ConformerGenerationConfig | None = None,
    preprocess_config: ConforFormerPreprocessingConfig | None = None,
) -> tuple[np.ndarray, str, str, bool]:
    conformer_config = conformer_config or ConformerGenerationConfig()
    preprocess_config = preprocess_config or ConforFormerPreprocessingConfig()
    canonical = validate_smiles(smiles, label="chromophore")
    cache_key = _embedding_cache_key(
        canonical_smiles=canonical,
        checkpoint_path=checkpoint_path,
        dictionary_path=dictionary_path,
        dictionary_sha256=dictionary.sha256,
        conformer_config=conformer_config,
        preprocess_config=preprocess_config,
        pooling_method=pooling_method,
    )
    cached = _load_cached_embedding(cache_dir, cache_key)
    if cached is not None:
        return cached, canonical, cache_key, True

    record = generate_conformer_cache_record(smiles, chromophore_id=canonical, config=conformer_config)
    if record.status != MoleculeStatus.OK:
        raise ValueError(f"Conformer generation failed for chromophore SMILES {smiles}: {record.failure_reason}")
    preprocessed = preprocess_successful_conformers(record, dictionary, preprocess_config)
    if not preprocessed:
        raise ValueError(f"No successful conformers available for chromophore SMILES {smiles}")
    batch = collate_preprocessed_conformers(preprocessed, dictionary)
    encoded = adapter.encode(batch).embedding_array.astype(np.float32)
    energies = np.asarray(
        [conf.energy if conf.energy is not None else np.nan for conf in record.conformer_records if conf.is_successful],
        dtype=np.float64,
    )
    pooled = pool_all(encoded, energies)
    pooled_by_name = {
        "mean": pooled.mean,
        "lowest_energy": pooled.lowest_energy,
        "boltzmann_298k": pooled.boltzmann_298k,
    }
    embedding = pooled_by_name[pooling_method].astype(np.float32)
    _write_cached_embedding(
        cache_dir,
        cache_key,
        embedding,
        {
            "canonical_smiles": canonical,
            "pooling_method": pooling_method,
            "conformer_cache_key": record.conformer_cache_key,
            "successful_conformer_count": record.successful_conformer_count,
            "conformer_configuration": conformer_config.to_payload(),
            "preprocessing_configuration": preprocess_config.to_payload(),
        },
    )
    return embedding, canonical, cache_key, False


def solvent_descriptor_frame(solvent_smiles: str, descriptor_path: Path) -> tuple[pd.DataFrame, str]:
    canonical = validate_smiles(solvent_smiles, label="solvent")
    descriptors = load_solvent_descriptors(descriptor_path)
    rows = pd.DataFrame({"canonical_solvent_smiles": [canonical], "solvent_original": [solvent_smiles]})
    merged, descriptor_columns = merge_solvent_descriptors(rows, descriptors)
    solvent_values = merged[descriptor_columns].apply(pd.to_numeric, errors="coerce") if descriptor_columns else pd.DataFrame(index=[0])

    if descriptor_columns and solvent_values.isna().all(axis=None):
        rdkit = import_rdkit()
        mol = rdkit["Chem"].MolFromSmiles(canonical)
        rdkit_values = compute_rdkit_descriptors(mol, rdkit)
        for column in RDKIT_SOLVENT_DESCRIPTOR_COLUMNS:
            if column in solvent_values.columns:
                solvent_values.loc[0, column] = rdkit_values[column]

    if descriptor_columns:
        base = solvent_values.copy()
        indicators = base.isna().astype(np.float32)
        indicators.columns = [f"{column}__missing" for column in base.columns]
        solvent_values = pd.concat([base, indicators], axis=1)
    return solvent_values.astype(np.float32), canonical


def feature_dimensions(feature_order: list[str]) -> dict[str, int]:
    return {
        "conforformer": sum(name.startswith("conforformer_") for name in feature_order),
        "morgan": sum(name.startswith("morgan_") for name in feature_order),
        "solvent": sum(not name.startswith("conforformer_") and not name.startswith("morgan_") for name in feature_order),
    }


def build_case_features(
    smiles: str,
    solvent_smiles: str,
    *,
    expected_feature_order: list[str],
    adapter: ConforFormerEncoderAdapter,
    dictionary: Any,
    checkpoint_path: Path,
    dictionary_path: Path,
    solvent_descriptor_path: Path,
    cache_dir: Path,
    pooling_method: str,
    feature_set: str,
) -> CaseFeatures:
    validate_smiles(smiles, label="chromophore")
    validate_smiles(solvent_smiles, label="solvent")
    embedding, canonical, cache_key, cache_hit = embed_smiles(
        smiles,
        adapter=adapter,
        dictionary=dictionary,
        checkpoint_path=checkpoint_path,
        dictionary_path=dictionary_path,
        cache_dir=cache_dir,
        pooling_method=pooling_method,
    )
    canonical_solvent = validate_smiles(solvent_smiles, label="solvent")
    fp = morgan_fingerprint(canonical, radius=2, n_bits=2048)
    if fp is None:
        raise ValueError(f"Morgan fingerprint generation failed for chromophore SMILES: {smiles}")
    solvent_values, canonical_solvent = solvent_descriptor_frame(solvent_smiles, solvent_descriptor_path)
    solvent_columns = list(solvent_values.columns)
    names = feature_names(pooling=pooling_method, feature_set=feature_set, solvent_columns=solvent_columns, n_bits=2048)
    values: list[float] = []
    if feature_set in {"conforformer_solvent", "conforformer_morgan_solvent"}:
        values.extend(float(value) for value in embedding)
    if feature_set in {"morgan_solvent", "conforformer_morgan_solvent"}:
        values.extend(float(value) for value in fp)
    values.extend(float(value) if pd.notna(value) else np.nan for value in solvent_values.iloc[0].to_list())
    frame = pd.DataFrame([values], columns=names)
    validate_feature_schema(frame, expected_feature_order)
    return CaseFeatures(
        canonical_smiles=canonical,
        canonical_solvent_smiles=canonical_solvent,
        frame=frame[expected_feature_order],
        embedding_cache_key=cache_key,
        embedding_cache_hit=cache_hit,
    )


def validate_feature_schema(frame: pd.DataFrame, expected_feature_order: list[str]) -> None:
    actual = list(frame.columns)
    if actual != expected_feature_order:
        missing = [name for name in expected_feature_order if name not in actual][:10]
        extra = [name for name in actual if name not in expected_feature_order][:10]
        first_mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected_feature_order)) if a != b), None)
        raise ValueError(
            "Feature schema mismatch before prediction: "
            f"actual_dim={len(actual)}, expected_dim={len(expected_feature_order)}, "
            f"first_mismatch={first_mismatch}, missing={missing}, extra={extra}"
        )
    dims = feature_dimensions(expected_feature_order)
    if dims["conforformer"] not in {0, EXPECTED_EMBEDDING_DIM}:
        raise ValueError(f"Unexpected ConforFormer embedding feature count: {dims['conforformer']}")
    if dims["morgan"] not in {0, 2048}:
        raise ValueError(f"Unexpected Morgan feature count: {dims['morgan']}")


def predict_case(
    smiles: str,
    solvent_smiles: str,
    *,
    models: dict[str, TargetModel],
    adapter: ConforFormerEncoderAdapter,
    dictionary: Any,
    checkpoint_path: Path,
    dictionary_path: Path,
    solvent_descriptor_path: Path,
    cache_dir: Path,
) -> dict[str, Any]:
    reference = models["absorption_nm"]
    incompatible = [
        target
        for target, target_model in models.items()
        if target_model.feature_order != reference.feature_order
        or target_model.pooling_method != reference.pooling_method
        or target_model.feature_set != reference.feature_set
    ]
    if incompatible:
        raise ValueError(
            "Target models use incompatible feature schemas or representations: "
            + ", ".join(sorted(incompatible))
        )
    features = build_case_features(
        smiles,
        solvent_smiles,
        expected_feature_order=reference.feature_order,
        adapter=adapter,
        dictionary=dictionary,
        checkpoint_path=checkpoint_path,
        dictionary_path=dictionary_path,
        solvent_descriptor_path=solvent_descriptor_path,
        cache_dir=cache_dir,
        pooling_method=reference.pooling_method,
        feature_set=reference.feature_set,
    )
    predictions: dict[str, float] = {}
    for target, target_model in models.items():
        frame = features.frame[target_model.feature_order]
        predictions[target] = float(np.asarray(target_model.estimator.predict(frame.to_numpy(dtype=np.float32)))[0])
    derived = predictions["emission_nm"] - predictions["absorption_nm"]
    if "stokes_shift_nm" in predictions:
        stokes_source = "direct"
        stokes = predictions["stokes_shift_nm"]
    else:
        stokes_source = "derived"
        stokes = derived
        predictions["stokes_shift_nm"] = stokes
    return {
        "smiles": smiles,
        "canonical_smiles": features.canonical_smiles,
        "solvent_smiles": solvent_smiles,
        "canonical_solvent_smiles": features.canonical_solvent_smiles,
        "predictions": {
            "absorption_nm": predictions["absorption_nm"],
            "emission_nm": predictions["emission_nm"],
            "quantum_yield": predictions["quantum_yield"],
            "stokes_shift_nm": stokes,
            "derived_stokes_shift_nm": derived,
        },
        "metadata": {
            "stokes_source": stokes_source,
            "model_paths": {target: str(model.model_path) for target, model in models.items()},
            "feature_metadata_paths": {target: str(model.metadata_path) for target, model in models.items()},
            "feature_dimensions": feature_dimensions(reference.feature_order),
            "pooling_method": reference.pooling_method,
            "feature_set": reference.feature_set,
            "checkpoint_path": str(checkpoint_path),
            "dictionary_path": str(dictionary_path),
            "solvent_descriptor_path": str(solvent_descriptor_path),
            "embedding_cache_key": features.embedding_cache_key,
            "embedding_cache_hit": features.embedding_cache_hit,
        },
    }


def format_text(payload: dict[str, Any]) -> str:
    predictions = payload["predictions"]
    stokes_label = "Stokes Shift" if payload["metadata"]["stokes_source"] == "direct" else "Stokes Shift (derived)"
    lines = [
        "=" * 80,
        "FLUORCAST-CONFORFORMER PREDICTION",
        "=" * 80,
        f"Chromophore SMILES : {payload['smiles']}",
        f"Canonical SMILES   : {payload['canonical_smiles']}",
        f"Solvent SMILES     : {payload['solvent_smiles']}",
        f"Canonical Solvent  : {payload['canonical_solvent_smiles']}",
        "",
        "Prediction",
        "-" * 80,
        f"Absorption         : {predictions['absorption_nm']:.1f} nm",
        f"Emission           : {predictions['emission_nm']:.1f} nm",
        f"Quantum Yield      : {predictions['quantum_yield']:.3f}",
        f"{stokes_label:<19}: {predictions['stokes_shift_nm']:.1f} nm",
        "-" * 80,
        f"Derived Em-Abs     : {predictions['derived_stokes_shift_nm']:.1f} nm",
        "=" * 80,
    ]
    return "\n".join(lines)


def batch_rows(input_csv: Path) -> list[dict[str, str]]:
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = {"smiles", "solvent_smiles"}.difference(fields)
        if missing:
            raise ValueError(f"Input CSV missing required column(s): {', '.join(sorted(missing))}")
        return [dict(row) for row in reader]


def batch_predict(input_csv: Path, output_csv: Path, **kwargs: Any) -> None:
    rows = []
    for index, row in enumerate(batch_rows(input_csv)):
        out = {column: "" for column in OUTPUT_COLUMNS}
        out["name"] = row.get("name", "") or f"case_{index + 1}"
        out["smiles"] = row.get("smiles", "")
        out["solvent_smiles"] = row.get("solvent_smiles", "")
        try:
            payload = predict_case(out["smiles"], out["solvent_smiles"], **kwargs)
            preds = payload["predictions"]
            out.update(
                {
                    "canonical_smiles": payload["canonical_smiles"],
                    "canonical_solvent_smiles": payload["canonical_solvent_smiles"],
                    "absorption_nm": preds["absorption_nm"],
                    "emission_nm": preds["emission_nm"],
                    "quantum_yield": preds["quantum_yield"],
                    "stokes_shift_nm": preds["stokes_shift_nm"],
                    "derived_stokes_shift_nm": preds["derived_stokes_shift_nm"],
                    "stokes_source": payload["metadata"]["stokes_source"],
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            out["status"] = "error"
            out["error"] = str(exc)
        rows.append(out)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model_root = resolve_model_root(args.model_root)
        models = load_target_models(model_root, split_type=args.split_type, pooling=args.pooling, feature_set=args.feature_set)
        dictionary = load_conforformer_dictionary(args.dictionary)
        adapter = ConforFormerEncoderAdapter(
            dictionary_path=args.dictionary,
            checkpoint_path=args.checkpoint,
            device=args.device,
            root=PROJECT_ROOT,
        )
        common = {
            "models": models,
            "adapter": adapter,
            "dictionary": dictionary,
            "checkpoint_path": args.checkpoint,
            "dictionary_path": args.dictionary,
            "solvent_descriptor_path": args.solvent_descriptors,
            "cache_dir": args.cache_dir,
        }
        if args.input_csv:
            output_csv = args.output_csv or args.input_csv.with_name(args.input_csv.stem + "_predictions.csv")
            batch_predict(args.input_csv, output_csv, **common)
            print(f"Wrote predictions to {output_csv}")
            return 0
        payload = predict_case(args.smiles, args.solvent_smiles, **common)
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(format_text(payload))
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
