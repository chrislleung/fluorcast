# UniProp 3D Integration Design

Status: Stage 1 repository audit and design. No production behavior is changed in
this stage.

## Scope

The integration target is the main FluorCast machine-learning repository on
branch `feature/uniprop-3d`. The desktop application is out of scope. The old
ConforFormer experiment was reviewed read-only as a source of generic adapter
ideas, not as a dependency.

The first production-capable UniProp path should support:

- one cached geometry per canonical unique chromophore;
- one supervised learning record per chromophore-solvent observation;
- absorption wavelength, emission wavelength, and quantum yield as initial
  supervised targets;
- later extension points for lifetime, extinction coefficient, and
  physics-aware constraints such as non-negative Stokes shift.

## Current FluorCast Audit

Repository state:

- Branch confirmed: `feature/uniprop-3d`.
- Working tree before this stage: clean.
- Local audit Python: `Python 3.14.0`; Slurm workflows currently load
  `python/3.11`.
- Package layout: top-level scripts under `scripts/`, reusable code under
  `src/` and `src/chemfluor/`, tests under `tests/`, batch jobs under `slurm/`.
- Runtime requirements are lightweight ML dependencies in `requirements.txt`:
  NumPy `<2.0`, pandas, RDKit, scikit-learn, SciPy, matplotlib, LightGBM,
  XGBoost, CatBoost, Jupyter, and pytest.

Data entry points:

- Original ChemFluor workflow: `src.train` via `python -m src.train`, using
  `src/data.py`, `src/features.py`, `src/models.py`, `src/splitting.py`, and
  `src/config.py`.
- Combined training workflow: `scripts/train_combined_predictors.py`.
- Manuscript comparison workflow:
  `scripts/manuscript/run_paper_comparison_experiments.py`.
- Standardized data layer: `src/chemfluor/data_standardization.py`.
- Prediction entry points: `scripts/predict_combined_molecule.py`,
  `scripts/predict_all_models.py`, `scripts/predict_full_fluorcast.py`, and
  `scripts/run_prediction_job.py`.

Split generation:

- Legacy workflow has random and scaffold splits in `src/splitting.py`.
- Manuscript workflow has random, molecule-grouped, and scaffold-grouped splits
  in `scripts/manuscript/manuscript_splits.py`.
- Combined and graph workflows use group splits by
  `canonical_chromophore_smiles`.
- Leakage checks already exist for molecule and scaffold group overlap.

Targets:

- Standardized target columns are `absorption_nm`, `emission_nm`,
  `lifetime_ns`, `quantum_yield`, and `log_extinction`.
- Current paper comparison trains `absorption_nm`, `emission_nm`, and
  `quantum_yield`, with historical support for derived `stokes_shift_nm`.
- Full prediction records expose predicted absorption, emission, Stokes shift,
  quantum yield, brightness class, intervals, applicability domain, and warnings.

Model interfaces and result formats:

- Tree and neural artifacts are target-specific files named
  `{target}_{model}.joblib` with `feature_metadata.json` and `metrics.json`.
- Graph artifacts are target-specific `{target}_{model}.pt` checkpoints with
  scalers, metadata, metrics, and prediction CSVs.
- Comparison outputs are CSV and Markdown tables under `outputs/...`.
- Paper comparison writes `metrics_by_split_model_target.csv`,
  `metrics_with_bootstrap_ci.csv`, `split_leakage_report.csv`,
  target/model/split/seed prediction CSVs, figures, and summary Markdown.

Slurm conventions:

- Jobs live under `slurm/`, `slurm/base_models/`, `slurm/manuscript/`,
  `slurm/util/`, and `slurm/legacy/`.
- Current jobs resolve the repository through environment variables and submit
  directory fallbacks, create `outputs/slurm`, and keep generated artifacts under
  `outputs/` or `models/`.
- CPU workflows commonly load Python, GCC, and RDKit modules. GPU graph jobs
  isolate GPU requirements.
- New UniProp Slurm jobs should follow the same pattern and make account,
  environment activation, data path, asset path, output path, seed, and device
  configurable.

## Upstream nablaColors Audit

Official sources reviewed:

- https://github.com/AI4DD/nablaColors
- https://zenodo.org/records/18061300
- https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus

Required environment:

- Recommended Python is `3.10`.
- Install Uni-Core from the repository root with `bash install_unicore.sh`, or
  manually install the local `Uni-Core` package and ensure `unicore-train` is
  available.
- Install Uni-Mol+ components with `cd unimol_plus && pip install -e .`.
- The repository vendors Chemprop v1.3.0 for solvent embedding and warns that
  Chemprop v2.x is not directly compatible with v1.x models.

Expected upstream structure:

- `Uni-Core/`
- `unimol_plus/`
- `models/chemprop/fold_0/model_1/model.pt`
- `examples/conformation_generation/`
- `install_unicore.sh`

LMDB records:

- LMDB values are gzip-compressed pickled dictionaries.
- LMDB keys are byte strings from which integer identifiers can be derived.
- The official dataset contains `absorption_conformations.zip` with geometries
  optimized at xTB, DFT vacuum, and DFT implicit-solvent levels.
- FluorCast adapters should write LMDB records through a small upstream-facing
  serializer and keep FluorCast supervision rows in CSV or Parquet outside the
  LMDB. The LMDB should not duplicate targets except where required by upstream
  UniProp tasks.

Training paths:

- Multitarget head-only path:
  `bash unimol_plus/head_pretrain_uniprop_multitarget.sh --data-path <dataset> --pretrained-model <unimol_plus_pcq_small.pt>`.
- Multitarget unfrozen-backbone path:
  `bash unimol_plus/finetune_unfreeze_backbone_multitarget.sh --data-path <dataset> --pretrained-model <head_checkpoint.pt>`.
- Single-target absorption head-only path:
  `bash unimol_plus/head_pretrain_uniprop_singletarget.sh --data-path <dataset> --pretrained-model <unimol_plus_pcq_small.pt>`.
- Single-target absorption unfrozen-backbone path:
  `bash unimol_plus/finetune_unfreeze_backbone_singletarget.sh --data-path <dataset> --pretrained-model <head_checkpoint.pt>`.

Validation and inference entry points:

- Multitarget validation:
  `bash unimol_plus/validate_multitarget.sh --data-path <dataset> --weight-path <checkpoint.pt> --subset valid --results-path <eval_dir>`.
- Single-target absorption validation:
  `bash unimol_plus/validate_singletarget_absorption.sh --data-path <dataset> --weight-path <checkpoint.pt> --subset valid --results-path <eval_dir>`.
- Validation outputs are `${results_path}/${subset}.metrics.json` and
  `${results_path}/${subset}.preds.pkl`.
- Screening LMDB creation example:
  `python examples/conformation_generation/04_csv_to_lmdb_rdkit.py --csv screening.csv --out-dir screening_lmdb --split test --smiles-col smiles --solvent-col solvent_smi`.

Pretrained assets:

- Uni-Mol+ small checkpoint from the official Uni-Mol+ repository.
- Chemprop solvent embedding model at
  `models/chemprop/fold_0/model_1/model.pt`.
- Zenodo UniProp checkpoints:
  `uniprop_rdkit_to_dft_implicit.pt`,
  `uniprop_rdkit_to_xtb.pt`,
  `uniprop_xtb_to_dft_implicit.pt`, and
  `uniprop_xtb_to_dft_vacuum.pt`.

## FluorCast Data Model

Geometry cache:

- Keyed by canonical chromophore identity, geometry generation config, upstream
  geometry method, RDKit or external optimizer version, and input level
  (`rdkit`, `xtb`, or future external optimized geometry).
- Stores one selected geometry per canonical unique chromophore for the initial
  UniProp path.
- Does not include solvent-specific target values.
- Failed geometry attempts are cached with reason codes so resumed jobs skip or
  retry deterministically.

Supervised records:

- One row per chromophore-solvent observation.
- Required identity columns:
  `canonical_chromophore_smiles`, `canonical_solvent_smiles`,
  `solvent_original`, `source_dataset`, and stable observation ID.
- Target columns: `absorption_nm`, `emission_nm`, `quantum_yield`, with nullable
  `lifetime_ns` and `log_extinction` retained for later stages.
- Geometry is joined by canonical chromophore key only. Solvent embedding and
  target labels are joined per observation.

Leakage prevention:

- Geometry generation must run before or independently of target splits and
  must not inspect target values.
- Train-only imputers, scalers, calibration, and target normalization must be
  fitted within each split.
- Molecule and scaffold splits must group by canonical chromophore or scaffold
  before expanding to chromophore-solvent observations.
- Any pretrained UniProp checkpoint is allowed only as an external prior; no
  FluorCast test labels may be used during head pretraining, finetuning, early
  stopping, or model selection.
- Duplicate exact measurements and replicate molecule-solvent observations must
  retain their existing standardized treatment and be reported in metrics.

## Proposed Module Structure

Keep upstream code separate from FluorCast adapters:

```text
src/chemfluor/uniprop/
  __init__.py
  schemas.py              # dataclasses for assets, geometries, records, results
  assets.py               # path validation, checksums, asset manifest loading
  geometry.py             # canonical chromophore geometry requests and cache keys
  lmdb_io.py              # gzip/pickle LMDB read/write adapter
  datasets.py             # supervised record builders from standardized FluorCast rows
  splits.py               # wrappers around existing split logic with UniProp IDs
  commands.py             # typed command builders for upstream shell entry points
  results.py              # metrics/prediction conversion into FluorCast formats
  validation.py           # leakage and schema checks

scripts/
  build_uniprop_geometry_cache.py
  build_uniprop_lmdb.py
  run_uniprop_training.py
  validate_uniprop_checkpoint.py
  predict_uniprop_records.py

slurm/uniprop/
  run_uniprop_head_pretrain.sbatch
  run_uniprop_finetune.sbatch
  run_uniprop_validate.sbatch
  run_uniprop_screening.sbatch

configs/uniprop/
  asset_manifest.example.json
  training.example.json
```

Third-party source should be supplied outside Git, for example through a
configurable `FLUORCAST_UNIPROP_UPSTREAM_DIR`, or as a pinned submodule only if
the project explicitly chooses to vendor it later. Heavy dependencies,
checkpoints, LMDB files, generated geometries, and logs must remain ignored.

## Migration Decisions From ConforFormer

The old ConforFormer experiment contains useful adapter patterns but should not
be imported directly.

Reusable unchanged:

- Stable JSON payload hashing pattern.
- Atomic cache write plus hash-verified cache read pattern.
- Explicit asset path validation and checksum recording pattern.
- Error taxonomy with clear reason codes.

Reusable after refactoring:

- Conformer-generation request/config schema. Refactor names away from
  ConforFormer and support UniProp geometry levels.
- Cache-key payloads. Refactor embedding and pooling keys into UniProp geometry,
  LMDB, checkpoint, and prediction keys.
- CLI dry-run, resumability, and cache-hit behavior from
  `scripts/build_conformer_cache.py`.
- Test style for cache determinism, corruption detection, dry-run behavior, and
  validation errors.
- Environment/asset inspection pattern from the ConforFormer adapter.

ConforFormer-specific and not reusable:

- ConforFormer dictionary loading.
- Uni-Mol contrast token preprocessing, edge-type formulas, hydrogen policy, and
  CLS embedding extraction.
- ConforFormer checkpoint architecture inspection and upstream model-building
  shim.
- Vendored `third_party/ConforFormer` source, checkpoint asset map, and
  ConforFormer Slurm/setup docs.
- Smoke scripts that construct or inspect ConforFormer encoders.

## Stage Boundaries

This stage creates documentation and a test guard only. It does not install
Uni-Core, Uni-Mol+, Chemprop, LMDB, PyTorch, CUDA packages, or nablaColors
source; it does not train or validate a model.

Recommended next stage: implement `src/chemfluor/uniprop/schemas.py`,
`assets.py`, and `lmdb_io.py` with unit tests and tiny synthetic LMDB fixtures,
still without training.
