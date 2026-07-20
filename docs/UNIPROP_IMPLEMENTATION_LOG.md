# UniProp Implementation Log

## 2026-07-20 - Stage 1 Audit And Design

Branch:

- Confirmed current branch: `feature/uniprop-3d`.
- Initial working tree: clean.

Repository audit:

- Inspected package layout, requirements, top-level scripts, standardized data
  layer, splitting code, training scripts, prediction scripts, result formats,
  Slurm wrappers, and tests.
- Confirmed current FluorCast standardized targets include
  `absorption_nm`, `emission_nm`, `lifetime_ns`, `quantum_yield`, and
  `log_extinction`.
- Confirmed current paper-comparison targets are absorption, emission, and
  quantum yield, with historical derived Stokes-shift support.
- Confirmed existing leakage-safe split implementations for random, molecule,
  and scaffold splits.

Upstream audit:

- Reviewed official `AI4DD/nablaColors` README and Zenodo record.
- Recorded Python 3.10 recommendation, Uni-Core install, editable Uni-Mol+
  install, vendored Chemprop v1.3.0, LMDB gzip/pickle storage, pretrained
  checkpoints, training scripts, validation scripts, and screening LMDB builder.

ConforFormer read-only audit:

- Located the old ConforFormer experiment as a sibling repository.
- Reviewed only generic adapter/config/cache/schema/docs/tests from that
  experiment.
- Classified reusable components in `docs/UNIPROP_ASSET_MAP.md` and
  `docs/UNIPROP_3D_DESIGN.md`.
- Decision: reuse design patterns only; do not depend on ConforFormer-specific
  preprocessing, encoder loading, dictionary handling, or vendored source.

Files added:

- `docs/UNIPROP_3D_DESIGN.md`
- `docs/UNIPROP_ASSET_MAP.md`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
- `tests/test_uniprop_design_docs.py`

Commands run:

- `git branch --show-current`
- `git status --short`
- `rg --files`
- `Get-ChildItem -Force`
- `rg -n "target|absorption|emission|quantum|split|scaffold|random|model|registry|Slurm|SBATCH|seed|train" README.md src scripts tests slurm docs`
- `Get-Content` audits for `requirements.txt`, `src/config.py`,
  `src/data.py`, `src/splitting.py`, `src/models.py`, `src/train.py`,
  `src/chemfluor/data_standardization.py`,
  `scripts/train_combined_predictors.py`,
  `scripts/manuscript/manuscript_splits.py`,
  `scripts/manuscript/run_paper_comparison_experiments.py`,
  `scripts/run_graph_model_experiments.py`,
  `scripts/predict_all_models.py`, and
  `slurm/run_paper_comparison_experiments.sbatch`
- Web review of official `AI4DD/nablaColors` GitHub README and Zenodo record.
- Read-only sibling ConforFormer audit with `rg --files`, `Get-ChildItem`, and
  `Get-Content` for relevant adapter/cache/schema/config/docs/tests.

Test results:

- `python -m pytest tests\test_uniprop_design_docs.py`: 4 passed in 0.03s.
- `python -m pytest`: 215 passed, 4 warnings in 62.29s.

Remaining limitations:

- No UniProp dependencies were installed.
- No upstream source was cloned into FluorCast.
- No model was trained or validated.
- Upstream LMDB record keys beyond README-level inspection remain to be
  confirmed with real tiny fixture reads in Stage 2.
- Exact UniProp task internals should be audited from the installed/pinned
  upstream source before command wrappers are finalized.

Recommended next stage:

- Implement the adapter foundation: `src/chemfluor/uniprop/schemas.py`,
  `assets.py`, and `lmdb_io.py`, plus unit tests and a tiny synthetic LMDB
  integration test. Do not train yet.

## 2026-07-20 - Stage 2 Reproducible Dependency Bootstrap

Scope:

- Added an isolated UniProp/nablaColors bootstrap without changing the default
  FluorCast Python environment.
- Target environment is Python 3.10 only.
- No dependencies, upstream source, checkpoints, LMDBs, or model artifacts were
  downloaded or installed during this stage.

Pinned upstream:

- Added `third_party/nablacolors.REVISION`.
- Pinned repo: `https://github.com/AI4DD/nablaColors.git`.
- Pinned ref: `v1.0.0`.
- Pinned commit: `39095389c0a4ecb47872ef74d00b8d13597939c8`.

Files added or updated:

- `.gitignore`
- `third_party/nablacolors.REVISION`
- `configs/uniprop/checkpoint_manifest.json`
- `scripts/bootstrap_uniprop.sh`
- `scripts/audit_uniprop_environment.py`
- `docs/UNIPROP_DEPENDENCIES.md`
- `tests/test_uniprop_bootstrap_environment.py`

Bootstrap behavior:

- Clones `third_party/nablacolors/` only when absent.
- Verifies an existing checkout is a Git checkout and exactly matches the
  pinned commit before installing.
- Refuses revision mismatches.
- Creates/reuses `.venv-uniprop/`.
- Verifies Python 3.10 and virtualenv isolation before non-dry-run installs.
- Runs upstream `install_unicore.sh`, then editable installs Uni-Core when
  present and Uni-Mol+ from `unimol_plus`.
- Supports `--mode cpu`, `--mode cuda`, `--dry-run`, and `--json-output`.

Audit behavior:

- Reports Python, PyTorch, CUDA availability, CUDA runtime, GPU name, RDKit,
  LMDB, Uni-Core, Uni-Mol+, Uni-Mol, Chemprop, pinned upstream revision,
  upstream checkout revision, checkpoint presence, sizes, and checksums.
- Emits readiness booleans for preprocessing, CPU smoke testing, and GPU
  training.
- Supports `--dry-run` and `--json-output`.

Checkpoint manifest:

- Added all four Zenodo UniProp checkpoint filenames, source URLs, MD5
  checksums, and expected size fields.
- Checkpoint binaries remain ignored and untracked.

Commands run:

- `git ls-remote https://github.com/AI4DD/nablaColors.git refs/tags/v1.0.0 refs/heads/main`
- `python -m pytest tests\test_uniprop_bootstrap_environment.py`
- `python -m pytest tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py`
- `python -m pytest`

Test results:

- Initial focused run exposed Windows/WSL bash-path handling and an
  `importlib.util` import issue; both were fixed.
- `python -m pytest tests\test_uniprop_bootstrap_environment.py`: 6 passed,
  1 skipped in 3.03s. The skipped test is the Python 3.10 import smoke because
  `python3.10` is not available in the current Windows environment.
- `python -m pytest tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py`:
  13 passed, 1 skipped in 2.80s.
- `python -m pytest`: 221 passed, 1 skipped, 4 warnings in 43.78s.

Remaining limitations:

- Bootstrap was syntax-checked and dry-run tested; non-dry-run dependency
  installation was intentionally not performed.
- The current local Python remains 3.14, so readiness is expected to be false
  until run from the isolated Python 3.10 environment.
- Manifest expected sizes are recorded from Zenodo's published MB display and
  should be tightened to exact byte counts when staging real checkpoint files.

Recommended next stage:

- Run the bootstrap on WSL or Nibi with Python 3.10, stage checkpoints outside
  Git, run `scripts/audit_uniprop_environment.py` from `.venv-uniprop/`, and
  then implement `src/chemfluor/uniprop/schemas.py`, `assets.py`, and
  `lmdb_io.py`.

## 2026-07-20 - Stage 3 Stable Manifests And Leakage-Safe Splits

Scope:

- Added the UniProp data-contract layer before geometry generation.
- The authoritative processed FluorCast source is resolved from
  `data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv` when
  present, otherwise `combined_deduplicated.csv`.
- No geometry, LMDB, training, or model code is needed to inspect the emitted
  manifests.

Files added:

- `src/chemfluor/uniprop/__init__.py`
- `src/chemfluor/uniprop/manifests.py`
- `scripts/build_uniprop_manifests.py`
- `tests/test_uniprop_manifests.py`

Generated local artifacts:

- `data/processed/uniprop/molecule_manifest.csv`
- `data/processed/uniprop/row_manifest.csv`
- `data/processed/uniprop/split_assignments.csv`
- `data/processed/uniprop/split_leakage_audit.csv`
- `data/processed/uniprop/split_statistics.csv`
- `data/processed/uniprop/training_normalization_statistics.csv`
- `data/processed/uniprop/manifest_metadata.json`

Manifest behavior:

- Molecule IDs are stable SHA-256 based IDs derived from RDKit canonical
  isomeric SMILES and the manifest schema version.
- Row IDs are stable SHA-256 based IDs derived from molecule ID, solvent ID,
  source dataset, and target values, independent of dataframe order.
- Missing target values remain missing. Availability masks are emitted as
  `{target}_available` Boolean columns.
- The molecule manifest includes original SMILES, canonical isomeric SMILES,
  canonical non-isomeric SMILES, InChIKey, formal charge, atom counts,
  canonicalization status, source row count, and deterministic molecule seed.
- InChIKey, atom-count, formal-charge, and non-isomeric bulk RDKit passes are
  explicit CLI opt-ins for full-dataset runs; columns remain present when not
  computed.

Split behavior:

- Preserved random, molecule-held-out, and scaffold-held-out split families.
- Added solvent-held-out and double-cold-start split families.
- The emitted scaffold split was generated with RDKit Bemis-Murcko scaffold
  groups via `--compute-rdkit-scaffolds`.
- Double-cold-start emits `train`, `test`, and `heldout_boundary` partitions;
  boundary rows are excluded from train/test because they contain exactly one
  held-out axis and would otherwise leak either molecule or solvent identity.
- Leakage audit checks all five split families. Random is recorded as
  not-applicable/pass; molecule, scaffold, solvent, and double-cold-start prove
  no train/test overlap for their constrained IDs.
- Target normalization statistics are computed from training rows only for each
  split family.

Real manifest reconciliation:

- Authoritative source rows: 66,820.
- Row manifest rows: 66,820.
- Molecule manifest rows: 33,965.
- Unique solvents: 1,369.
- Leakage audit result: all five split families passed.

Commands run:

- `python -m pytest tests\test_uniprop_manifests.py`
- `python scripts\build_uniprop_manifests.py --out-dir data\processed\uniprop --seed 42 --test-size 0.2 --compute-rdkit-scaffolds`

Test results:

- `python -m pytest tests\test_uniprop_manifests.py`: 9 passed in 0.81s.
- Manifest build result: all five leakage audits passed; wrote 33,965 molecule rows
  and 66,820 row records to `data/processed/uniprop`.

Remaining limitations:

- Full-dataset InChIKey and molecule atom-property extraction are opt-in
  because they can be slow on difficult structures in the current Windows
  RDKit environment.
- The default `canonical_nonisomeric_smiles` mirrors isomeric canonical SMILES;
  use `--compute-nonisomeric` when a full non-isomeric grouping/report is
  needed.

Recommended next stage:

- Use the manifest outputs as the only input to geometry-cache generation.
- Geometry generation should consume one row per `molecule_id` from
  `molecule_manifest.csv` and must not read target labels.

## 2026-07-20 - Stage 4 Deterministic RDKit Geometry Cache

Scope:

- Added the initial deterministic one-geometry-per-molecule RDKit cache.
- Geometry generation consumes only `molecule_manifest.csv`, never
  `row_manifest.csv` or experimental target columns.
- One JSON cache entry is written per stable `molecule_id`.

Files added:

- `src/chemfluor/uniprop/geometry_cache.py`
- `scripts/build_uniprop_geometry_cache.py`
- `slurm/uniprop/run_uniprop_geometry_cache_array.sbatch`
- `tests/test_uniprop_geometry_cache.py`

Cache behavior:

- Uses RDKit ETKDGv3 embedding with explicit hydrogens.
- Uses explicit hydrogens during force-field optimization.
- Defaults to MMFF94s when MMFF parameters are available; supports MMFF94 via
  `--mmff-variant`.
- Falls back to UFF only when MMFF parameters are unavailable, and records
  `optimization_method` as `UFF`.
- Removes hydrogens after optimization by default for heavy-atom UniProp-style
  geometry payloads; `--retain-hydrogens` is available.
- Derives the RDKit random seed from stable `molecule_id`, not row order.
- Verifies heavy-atom graph, bond topology, formal charge, and atom order
  against the canonical input after optimization.
- Stores molecule ID, canonical SMILES, atom symbols, atomic numbers,
  coordinates, optimization method, energy, convergence status, RDKit version,
  seed, timestamps, schema version, hydrogen policy, topology signature, and
  checksum.
- Uses atomic temp-file writes followed by `os.replace`.
- Existing cache files must pass JSON parsing, schema, checksum, shape, and
  topology validation before counting as hits.

CLI behavior:

- `scripts/build_uniprop_geometry_cache.py` supports `--limit`,
  repeatable `--molecule-id`, `--workers`, `--resume/--no-resume`,
  `--overwrite-invalid`, `--fail-fast`, `--shard-index`, `--shard-count`,
  JSON status output, and JSON/CSV failure reports.
- Status totals reconcile against the selected molecule-manifest subset.
- Invalid SMILES, corrupt JSON, checksum mismatches, topology mismatches, and
  embedding/optimization failures are reported as structured failures.

Slurm behavior:

- Added a resumable job-array wrapper at
  `slurm/uniprop/run_uniprop_geometry_cache_array.sbatch`.
- Shards by stable sorted molecule-manifest ranges using
  `--shard-index`/`--shard-count`.
- Defaults to `--resume --overwrite-invalid` for safe reruns.
- Paths, shard count, cache directory, repository root, and environment
  activation are configurable through environment variables.

Smoke runs:

- First unrestricted `--limit 3` smoke against the real molecule manifest
  generated 2 entries and reported 1 structured RDKit embedding failure,
  proving failure-report emission.
- A targeted three-molecule smoke generated 3 entries, then reran as 3 cache
  hits.
- After intentionally corrupting one JSON entry, validation reported
  `invalid_cache`; rerunning with `--overwrite-invalid` regenerated 1 entry and
  reused 2 hits.

Commands run:

- `python -m pytest tests\test_uniprop_geometry_cache.py`
- `python scripts\build_uniprop_geometry_cache.py --molecule-manifest data\processed\uniprop\molecule_manifest.csv --cache-dir data\processed\uniprop\geometry_cache_smoke --limit 3 --workers 1 --status-json outputs\uniprop_geometry_smoke_status.json --failure-json outputs\uniprop_geometry_smoke_failures.json --failure-csv outputs\uniprop_geometry_smoke_failures.csv`
- `python scripts\build_uniprop_geometry_cache.py --molecule-manifest data\processed\uniprop\molecule_manifest.csv --cache-dir data\processed\uniprop\geometry_cache_smoke_good --molecule-id mol_0000485e9d6fab52 --molecule-id mol_000049a8f03a85dc --molecule-id mol_0000b7567836b095 --workers 1 --status-json outputs\uniprop_geometry_smoke_good_status.json --failure-json outputs\uniprop_geometry_smoke_good_failures.json --failure-csv outputs\uniprop_geometry_smoke_good_failures.csv`
- Same targeted command rerun for resume validation.
- Same targeted command after intentional corruption.
- Same targeted command with `--overwrite-invalid` for repair validation.
- `python -m pytest tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py`

Test results:

- `python -m pytest tests\test_uniprop_geometry_cache.py`: 13 passed in 2.37s.
- Focused UniProp/slurm guard suite: 37 passed, 1 skipped in 8.44s.

Remaining limitations:

- Full cache generation for all 33,965 manifest molecules was not run locally.
- RDKit can fail embedding or parameterization for some manifest structures;
  those failures are now structured and resumable.

Recommended next stage:

- Run the Slurm array on Nibi or a suitable Linux RDKit environment for the
  full molecule manifest.
- Use the validated JSON cache as the geometry source for the later UniProp
  LMDB adapter.

## 2026-07-20 - Stage 5 UniProp-Compatible LMDB Exporter

Scope:

- Added a FluorCast-to-UniProp LMDB adapter that consumes row manifests,
  split assignments, and existing molecule geometry cache entries.
- The exporter does not generate conformers or call geometry generation code.
- Each LMDB record corresponds to one experimental row and preserves both
  `row_id` and `molecule_id` for later prediction reconciliation.

Upstream inspection:

- Cloned ignored pinned source under `third_party/nablacolors/`.
- Verified pinned commit:
  `39095389c0a4ecb47872ef74d00b8d13597939c8`.
- Inspected:
  - `examples/conformation_generation/04_csv_to_lmdb_rdkit.py`
  - `unimol_plus/unimol_plus/data/pcq_dataset.py`
  - `unimol_plus/unimol_plus/data/lmdb_dataset.py`
  - `unimol_plus/unimol_plus/data/conformer_sample_dataset.py`
  - `unimol_plus/unimol_plus/tasks/pcq.py`
  - `unimol_plus/scripts/get_3d_lmdb.py`
- Confirmed upstream LMDB values are gzip-compressed pickles.
- Confirmed upstream keys are integer byte keys; FluorCast uses deterministic
  8-byte big-endian keys in sorted row order.
- Confirmed required record fields:
  `atoms`, `input_pos`, `label_pos`, `smi`, `solvent_smi`, `node_attr`,
  `edge_index`, `edge_attr`, and `target`.
- Confirmed `PCQDataset` forwards `target`, integer `id`, and `solvent_smi`.
  It does not collate missing-label masks, so a thin FluorCast wrapper was
  added for mask-preserving training use.

Files added or updated:

- `pytest.ini`
- `requirements.txt`
- `src/chemfluor/uniprop/lmdb_export.py`
- `src/chemfluor/uniprop/upstream_compat.py`
- `scripts/export_uniprop_lmdb.py`
- `scripts/validate_uniprop_lmdb.py`
- `tests/test_uniprop_lmdb_export.py`

LMDB record behavior:

- Loads cached geometry by `molecule_id`; missing, invalid, corrupt, or
  mismatched geometry entries fail validation.
- Attaches row-level `solvent_smi`, target vector, Boolean `target_mask`,
  target column names, `row_id`, `molecule_id`, and `solvent_id`.
- Uses cached atom symbols and coordinates as `atoms`, `input_pos`, and
  `label_pos`.
- Computes `node_attr`, `edge_index`, and `edge_attr` with the exact feature
  encoding from pinned nablaColors `get_3d_lmdb.py`.
- Missing targets are stored as `NaN` with mask `False`; no missing target is
  replaced by a real-looking zero.

Split/export behavior:

- Exports separate `train.lmdb`, `valid.lmdb`, and `test.lmdb`.
- Existing train/test manifest splits are preserved; `valid` is carved
  deterministically from training rows by stable `row_id`, seed, and
  `valid_size`.
- Supports transaction batching, configurable map size, completion markers,
  resume validation, and overwrite rebuilds.
- Writes `metadata.json` with schema version, row-manifest hash,
  molecule-manifest hash, split hash, geometry-cache schema, upstream revision,
  row counts, target counts, and creation configuration.

Validation behavior:

- Added `scripts/validate_uniprop_lmdb.py`.
- Validates required keys, atom/coordinate shapes, graph dtypes, target and mask
  shapes, deterministic row IDs, completion marker presence, and row-manifest
  reconciliation.
- Reads LMDB records in key order and reports target coverage.

Commands run:

- `python -m pip install lmdb`
- `python -m pytest tests\test_uniprop_lmdb_export.py`
- `python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py`
- `python -m pytest`

Test results:

- `python -m pytest tests\test_uniprop_lmdb_export.py`: 12 passed in 4.41s.
- Focused UniProp suite: 47 passed, 1 skipped in 13.18s.
- Initial full-suite run collected ignored upstream tests under
  `third_party/nablacolors/`; added `pytest.ini` with `testpaths = tests`.
- Final `python -m pytest`: 255 passed, 1 skipped, 4 warnings in 51.60s.

Smoke coverage:

- The tests build LMDBs from cached geometry fixtures without invoking
  conformer generation.
- A 100-row smoke LMDB loads through the actual pinned upstream
  `LMDBDataset` class loaded from
  `third_party/nablacolors/unimol_plus/unimol_plus/data/lmdb_dataset.py`.
- Train/valid/test row IDs jointly reconcile exactly to the row manifest.

Remaining limitations:

- Full real-manifest LMDB export was not run because full geometry cache
  generation has not been completed locally.
- The pinned upstream full package import still requires its optional Uni-Core
  dependencies; the test loads the exact upstream LMDBDataset source file
  directly to avoid unrelated tokenizer imports.

Recommended next stage:

- Run full geometry-cache generation first.
- Export split-specific LMDBs from the complete cache.
- Wire `TargetMaskDataset` or an equivalent upstream task extension into
  training before fitting multitarget heads with missing labels.

## 2026-07-20 - Stage 6 Head-Only Smoke Training

Scope:

- Added a deterministic FluorCast-owned head-only smoke trainer for exported
  UniProp LMDBs.
- The molecular backbone projection is frozen. Only `solvent_adapter`,
  `fusion`, and `heads` parameters are trainable.
- The smoke path trains on a deterministic subset that must include at least
  one non-missing example for absorption, emission, and quantum yield.

Files added:

- `src/chemfluor/uniprop/head_smoke_training.py`
- `scripts/train_uniprop_head_smoke.py`
- `configs/uniprop/head_smoke.example.json`
- `configs/uniprop/head_smoke.overfit_fixture.json`
- `slurm/uniprop/run_uniprop_head_smoke.sbatch`
- `tests/test_uniprop_head_smoke_training.py`

Training behavior:

- Writes resolved configuration, environment report, dataset/split hashes,
  target scalers, metrics history, validation predictions, best checkpoint,
  and last checkpoint.
- Checkpoints include model state, optimizer state, scheduler state, scaler,
  metrics history, and RNG states.
- Resume loads `last_checkpoint.pt` and continues at the exact next update.
- Refuses validation on the `test` partition to avoid accidental test-set
  evaluation.
- Detects empty target batches, non-finite training loss, and NaN or infinite
  gradients.
- Saves per-row validation predictions and recomputes metrics from that CSV.

Commands run:

- `python -m pytest tests\test_uniprop_head_smoke_training.py`
- `python -m pytest tests\test_slurm_layout.py`
- `python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_uniprop_head_smoke_training.py`

Test results:

- `tests\test_uniprop_head_smoke_training.py`: 10 passed in 22.12s.
- `tests\test_slurm_layout.py`: 2 passed in 8.03s.
- Focused UniProp suite: 57 passed, 1 skipped in 26.20s.

Remaining limitations:

- The local run proves the smoke trainer and CLI on synthetic LMDB fixtures.
- A real Nibi GPU submission still requires staged full LMDBs and a configured
  UniProp/PyTorch environment on Nibi.

## 2026-07-20 - Stage 7 Backbone Fine-Tuning Smoke

Scope:

- Added a second-stage fine-tuning trainer initialized from the best head-only
  checkpoint.
- The trainer loads the head-only checkpoint, verifies the prior schema, then
  unfreezes the backbone for optimization.
- Optimizer groups use a lower learning rate for `backbone` than for
  `solvent_adapter`, `fusion`, and `heads`.

Files added:

- `src/chemfluor/uniprop/backbone_finetune.py`
- `scripts/train_uniprop_backbone_finetune.py`
- `configs/uniprop/backbone_finetune.example.json`
- `slurm/uniprop/run_uniprop_backbone_finetune.sbatch`
- `tests/test_uniprop_backbone_finetune.py`

Training behavior:

- Supports gradient clipping, gradient accumulation, CUDA AMP when available,
  early stopping, best-checkpoint selection, and exact resume.
- EMA is optional and tested; checkpoint state includes EMA only when enabled.
- Checkpoints include model, optimizer, AMP scaler, scheduler, scaler, metric
  history, best-metric state, and update position.
- Writes parameter counts by component, optimizer group learning rates,
  environment report, memory report, dataset/split hashes, metrics history,
  best and last checkpoints, and per-row validation predictions.
- Refuses `validation_partition = test` so test metrics cannot influence
  checkpoint selection.
- CUDA out-of-memory errors are rewritten with a recovery message recommending
  lower micro-batch size, lower accumulation, AMP, or more GPU memory.
- The Nibi Slurm script requests one CUDA GPU generically; H100 can be selected
  at submit time without making it a hard-coded requirement.

Commands run:

- `python -m pytest tests\test_uniprop_backbone_finetune.py`
- `python -m pytest tests\test_slurm_layout.py`

Test results:

- `tests\test_uniprop_backbone_finetune.py`: 10 passed in 11.34s.
- `tests\test_slurm_layout.py`: 2 passed in 4.66s.

Remaining limitations:

- The local tests prove checkpoint transition and deterministic resume on
  synthetic LMDB fixtures. A real Nibi fine-tuning job still requires staged
  LMDBs, a head-only checkpoint, and a configured UniProp/PyTorch environment.

## 2026-07-20 - Stage 8 Experiment Matrix And Benchmark Runner

Scope:

- Added a reproducible experiment-matrix runner for current FluorCast-style
  baselines and UniProp variants under identical manifest splits.
- Default matrix covers random, molecule, scaffold, solvent held out, and
  double-cold-start split families; three controlled seeds; absorption,
  emission, and quantum-yield targets; and six initial model variants.
- Test-set evaluation is a separate explicit command.

Files added:

- `src/chemfluor/uniprop/experiment_matrix.py`
- `scripts/run_uniprop_experiment_matrix.py`
- `configs/uniprop/experiment_matrix.example.json`
- `slurm/uniprop/run_uniprop_experiment_matrix.sbatch`
- `tests/test_uniprop_experiment_matrix.py`

Runner behavior:

- Runs split leakage audit before training and refuses to train when selected
  split families fail.
- Uses identical target rows for model comparisons sharing a split, target,
  and seed.
- Writes per-run `metrics.json`, validation/test prediction CSVs, checkpoint
  artifacts, manifest hashes, config hashes, checkpoint hashes, target
  coverage, and train/valid/test counts.
- Reports MAE, RMSE, and R2 for regression targets, with bright-class F1 and
  accuracy for quantum yield.
- Includes metrics by emission region and similarity bin in per-run metrics.
- Adds `validate` for either a single run or a whole experiment directory.
- Adds `summarize` to exclude failed or partial runs, reject duplicate run IDs,
  and write `aggregate_summary.csv`, `per_run_summary.csv`, and
  `excluded_runs.json`.
- Aggregation reports mean, standard deviation, and bootstrap confidence
  intervals for MAE.

Commands run:

- `python -m pytest tests\test_uniprop_experiment_matrix.py`
- `python -m pytest tests\test_slurm_layout.py`
- `python -m pytest tests\test_uniprop_experiment_matrix.py tests\test_uniprop_backbone_finetune.py tests\test_uniprop_head_smoke_training.py tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py`

Test results:

- `tests\test_uniprop_experiment_matrix.py`: 10 passed in 25.92s.
- `tests\test_slurm_layout.py`: 2 passed in 4.18s.
- Focused UniProp/matrix suite: 79 passed, 1 skipped in 51.80s.

Remaining limitations:

- The local matrix smoke uses fixture-scale manifests. Running the full matrix
  on Nibi requires the full manifests and completed UniProp artifacts staged in
  the configured locations.

## 2026-07-20 - Stage 9 Physics-Constrained Multitask Extension

Scope:

- Added a differentiable FluorCast physics output module for UniProp heads.
- Centralized photon wavelength/energy conversion in one tested module.
- Added ordinary-fluorescence Stokes energy, latent radiative and
  nonradiative log rates, derived quantum yield, derived lifetime, and separate
  log-extinction output support.
- Added explicit verified anti-Stokes masking so negative measured Stokes rows
  can be excluded from ordinary-Stokes supervision only when flagged.
- Registered ablation variants for independent heads, wavelength-constrained
  heads, rate-constrained heads, and the complete physics-constrained model in
  the experiment matrix.

Files added or updated:

- `src/chemfluor/uniprop/physics_constraints.py`
- `tests/test_uniprop_physics_constraints.py`
- `src/chemfluor/uniprop/experiment_matrix.py`
- `configs/uniprop/experiment_matrix.example.json`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`

Physics behavior:

- Absorption energy is derived as emission energy plus a nonnegative
  ordinary-fluorescence Stokes energy for constrained variants.
- Absorption and emission wavelengths are derived from energies using
  `HC_EV_NM / energy_ev`; no intermediate rounding is performed.
- Quantum yield is derived from rates as `k_r / (k_r + k_nr)`.
- Lifetime is derived from rates as `1 / (k_r + k_nr)` and reported in ns.
- Missing target masks gate each loss term, so absent labels do not create
  zero-valued artificial targets.
- Consistency diagnostics report Stokes equation error, quantum-yield rate
  equation error, lifetime rate equation error, Stokes sign violations,
  quantum-yield bound violations, and lifetime positivity violations.
- Three-target baseline checkpoint migration is explicit and reports migrated
  and skipped tensors.

Commands run:

- `python -m pytest tests\test_uniprop_physics_constraints.py`
- `python -m pytest tests\test_uniprop_experiment_matrix.py`

Test results:

- `tests\test_uniprop_physics_constraints.py`: 9 passed in 2.36s.
- `tests\test_uniprop_experiment_matrix.py`: 11 passed in 24.16s.

Remaining limitations:

- The local tests validate the constrained equations, masks, gradients, and
  matrix wiring on fixtures. Full training stability still needs a staged Nibi
  run with completed UniProp LMDBs and checkpoints.

## 2026-07-20 - Stage 10 Reproducible Production Inference

Scope:

- Added a versioned UniProp production model-bundle loader and JSON prediction
  runner without integrating it into the desktop application.
- The inference module deliberately avoids importing the head-smoke,
  fine-tuning, or experiment-matrix training modules.
- Added a CLI wrapper for end-to-end JSON prediction from one bundle.

Files added or updated:

- `src/chemfluor/uniprop/production_inference.py`
- `scripts/predict_uniprop_bundle.py`
- `tests/test_uniprop_production_inference.py`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`

Bundle contract:

- Bundle schema: `fluorcast_uniprop_model_bundle_v1`.
- Required files:
  - `metadata.json`
  - `model_weights.pt`
  - `architecture_config.json`
  - `target_definitions.json`
  - `scalers.json`
  - `solvent_encoder_assets.json`
- `metadata.json` records model/version, physics schema, upstream revision,
  checkpoint hashes, supported geometry schema, training-data fingerprint,
  metrics summary, applicability-domain settings, and SHA-256 hashes for all
  required assets.

Prediction behavior:

- Accepts chromophore SMILES, solvent name or solvent SMILES, optional
  precomputed geometry, and optional request ID.
- Canonicalizes molecule and solvent SMILES.
- Reuses valid cached geometry when available.
- Generates one deterministic RDKit geometry and writes it to cache when no
  valid entry exists.
- Returns absorption, emission, quantum yield, lifetime, and log extinction
  when the bundle declares those targets.
- Includes applicability-domain information from bundle reference
  fingerprints, model/version provenance, checkpoint provenance, physical
  consistency metrics, and structured warnings.
- Supports batch prediction with mixed success/failure records.
- Provides `to_backend_prediction_contract` as a versioned adapter to the
  existing FluorCast backend prediction contract.

Commands run:

- `python -m pytest tests\test_uniprop_production_inference.py`

Test results:

- `tests\test_uniprop_production_inference.py`: 13 passed in 6.02s.

Remaining limitations:

- Tests use a fixture bundle with deterministic weights. Packaging the real
  stable Nibi-trained checkpoint requires the finalized production checkpoint,
  scalers, solvent assets, metrics summary, and training fingerprint.

## 2026-07-20 - Stage 11 Later Conformer And xTB Ablation Preparation

Scope:

- Added additive named conformer-set cache support without invalidating the
  original `uniprop_geometry_cache_v1` one-geometry files.
- Added geometry-ablation variant names for later controlled matrix runs:
  RDKit/MMFF single geometry, xTB single geometry, multiple RDKit conformers,
  equal pooling, energy-weighted pooling, and solvent-conditioned conformer
  weighting.
- Added optional xTB environment detection. xTB remains an external optional
  dependency and is not required for local tests.
- Added isolated pooling/attention components for conformer ablations.
- Registered geometry-ablation variants in the experiment matrix and recorded
  relative preprocessing/inference cost metadata in per-run and aggregate
  outputs.

Files added or updated:

- `src/chemfluor/uniprop/conformer_geometry.py`
- `tests/test_uniprop_conformer_geometry.py`
- `src/chemfluor/uniprop/experiment_matrix.py`
- `configs/uniprop/experiment_matrix.example.json`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`

Cache behavior:

- New schema: `uniprop_named_conformer_set_v1`.
- A molecule can have multiple named geometry sets under
  `<cache>/<molecule_id>/<geometry_set_name>.json`.
- Each conformer stores conformer ID, rank, source RDKit conformer ID, method,
  energy, coordinates, and convergence status.
- Conformers are sorted deterministically by energy and source conformer ID.
- Original single-geometry entries can be migrated into a named
  `rdkit_mmff_single` conformer set while preserving coordinates, energy,
  method, and convergence.

Model behavior:

- `equal_pool` and `energy_weighted_pool` are permutation-invariant.
- `SolventConditionedConformerAttention` is isolated from the main model so
  solvent-dependent conformer weighting can be ablated independently.
- Solvent attention changes weights only; cached conformer coordinates remain
  immutable input artifacts.

Commands run:

- `python -m pytest tests\test_uniprop_conformer_geometry.py`
- `python -m pytest tests\test_uniprop_experiment_matrix.py`

Test results:

- `tests\test_uniprop_conformer_geometry.py`: 11 passed in 5.65s.
- `tests\test_uniprop_experiment_matrix.py`: 11 passed in 25.64s.

Remaining limitations:

- Per the prompt guard, no conformer/xTB ablation jobs were run. The additional
  geometry complexity is not yet justified by measured generalization gains.
- xTB optimization itself is not invoked locally; only optional dependency
  detection and schema readiness are implemented.
- The original one-RDKit-geometry model remains the reproducible baseline.

## 2026-07-20 - Stage 12 UniProp Stage-Gate Audit

Scope:

- Audited the existing UniProp implementation on `feature/uniprop-3d` without
  adding new features or changing runtime behavior.
- Inspected the handoff context, every module under `src/chemfluor/uniprop`,
  UniProp scripts, configs, Slurm wrappers, tests, dependency docs, asset docs,
  and implementation log.
- Added `docs/UNIPROP_STAGE_GATE_REPORT.md` with the required component table
  and explicit real-versus-simulated determinations.

Repository state before edits:

- Branch: `feature/uniprop-3d`.
- Latest commit: `322e1c3 feat: add validated UniProp 3D integration scaffold`.
- Pre-existing untracked paths included `artifacts/`,
  `development md/UniProp Handoff/`, and the active-file scratch path whose
  name begins with `owords`.

Gate commands:

- `python -m pytest -q`: 323 passed, 1 skipped, 4 warnings in 91.83s.
- `python -m compileall -q src scripts`: passed.
- `git diff --check`: passed.

Verified current state:

- nablaColors revision pinning is real. `third_party/nablacolors.REVISION`
  pins `https://github.com/AI4DD/nablaColors.git` at
  `39095389c0a4ecb47872ef74d00b8d13597939c8`.
- The ignored local `third_party/nablacolors/` checkout is present and clean at
  the pinned commit when inspected with a one-off `safe.directory` Git flag.
- The local audit environment is Python 3.14.0 with RDKit 2026.03.2, LMDB
  2.3.0, PyTorch 2.10.0 CPU-only, no CUDA, no Uni-Core, no Uni-Mol+, no
  Chemprop, and no staged UniProp checkpoints.
- Manifests, split auditing, RDKit geometry cache, LMDB export/validation,
  target-mask wrapper, physics equations, conformer-cache prep, and
  production JSON/bundle contracts are real local FluorCast code.
- Head-only training, backbone fine-tuning, UniProp matrix variants, Chemprop
  solvent-encoder variant, and production fixture inference are dummy or
  lightweight local stand-ins and must not be treated as proof that upstream
  UniProp works.
- The pinned upstream `LMDBDataset` source-file smoke is partial upstream
  validation only; full Uni-Core/Uni-Mol+ package/task loading remains
  unverified.

Dependency and Git hygiene:

- `requirements.txt` keeps heavy UniProp dependencies out of the base
  environment. It includes `lmdb`, but not Uni-Core, Uni-Mol+, Chemprop,
  PyTorch, checkpoints, or nablaColors source.
- `.gitignore` excludes the downloaded nablaColors checkout, UniProp assets,
  checkpoints, LMDBs, generated UniProp data, output directories, model
  directories, Python caches, and virtual environments.
- `compileall` produced ignored `__pycache__/` files under
  `src/chemfluor/uniprop/`; these remain ignored and untracked.
- No UniProp scripts/configs contain credentials or tokens. UniProp tracked
  files do not hard-code a Windows username path.
- UniProp Slurm wrappers use configurable `$HOME/scratch/...` defaults.
  `run_uniprop_head_smoke.sbatch` currently hard-codes an H100 GPU request by
  default; fine-tuning requests a generic single GPU.

Components requiring Python 3.10:

- Non-dry-run `scripts/bootstrap_uniprop.sh`.
- Uni-Core import and `unicore-train`.
- Uni-Mol+ import and upstream task/loader validation.
- Chemprop v1.3 solvent-encoder validation.
- Real pretrained UniProp checkpoint loading and upstream training/validation
  entry points.

Components requiring Nibi or CUDA:

- Real GPU head-only upstream UniProp training.
- Real backbone fine-tuning.
- Practical full-matrix execution with trained artifacts.
- The current head-smoke Slurm default requests an H100; the fine-tune wrapper
  requests one generic CUDA GPU.

Single next stage:

- Python 3.10 upstream environment validation and checkpoint staging. Run the
  isolated bootstrap/audit, stage Zenodo checkpoints outside Git, verify
  Uni-Core/Uni-Mol+/Chemprop imports, and run a tiny upstream loader/task
  smoke. Do not generate the full FluorCast geometry cache or start model
  training in that stage.
