# UniProp Handoff 2

Date: 2026-07-20

Repository: `C:\Users\CL\OneDrive\Desktop\python\FluorCast`

Branch/context: continuing the UniProp/nablaColors integration after
`development md/UniProp Handoff/handoff1.md`.

This handoff captures all work completed in this chat after Prompt 2:

- Prompt 3: stable molecule manifest and leakage-safe splits.
- Prompt 4: deterministic one-geometry-per-molecule RDKit cache.
- Prompt 5: UniProp-compatible LMDB exporter.
- A later pasted FluorCast Desktop/NIBI runtime redesign request was read, but
  the turn was interrupted before implementation. No desktop repository edits
  were made.

## General Constraints Carried Forward

- Work stayed in the main FluorCast ML repository:
  `C:\Users\CL\OneDrive\Desktop\python\FluorCast`.
- The FluorCast desktop app was not modified.
- Generated data, upstream clones, checkpoints, LMDB files, logs, and model
  artifacts are kept out of Git.
- New code uses stable IDs and deterministic seeds.
- Geometry generation does not inspect experimental targets.
- LMDB export does not generate geometries.
- Full test suite was run after each major stage.

## Git Status At End Of Chat

Current short status:

```text
 M .gitignore
 M requirements.txt
?? configs/
?? "development md/UniProp Handoff/"
?? docs/UNIPROP_3D_DESIGN.md
?? docs/UNIPROP_ASSET_MAP.md
?? docs/UNIPROP_DEPENDENCIES.md
?? docs/UNIPROP_IMPLEMENTATION_LOG.md
?? pytest.ini
?? scripts/audit_uniprop_environment.py
?? scripts/bootstrap_uniprop.sh
?? scripts/build_uniprop_geometry_cache.py
?? scripts/build_uniprop_manifests.py
?? scripts/export_uniprop_lmdb.py
?? scripts/validate_uniprop_lmdb.py
?? slurm/uniprop/
?? src/chemfluor/uniprop/
?? tests/test_uniprop_bootstrap_environment.py
?? tests/test_uniprop_design_docs.py
?? tests/test_uniprop_geometry_cache.py
?? tests/test_uniprop_lmdb_export.py
?? tests/test_uniprop_manifests.py
?? third_party/
```

Important note:

- Many untracked files above are from Prompts 1 and 2, documented in
  `handoff1.md`.
- Prompt 3-5 added more files under `src/chemfluor/uniprop/`, `scripts/`,
  `slurm/uniprop/`, `tests/`, `pytest.ini`, and updated `requirements.txt`,
  `.gitignore`, and `docs/UNIPROP_IMPLEMENTATION_LOG.md`.
- `third_party/nablacolors/` is intentionally ignored and was cloned locally
  during Prompt 5 for exact upstream source inspection.

## Prompt 3 - Stable Molecule Manifest And Leakage-Safe Splits

### User Request

Build a stable data-contract layer before generating geometry:

- identify the authoritative processed FluorCast dataset;
- create molecule and row manifests;
- preserve missing labels;
- add solvent-held-out and double-cold-start splits;
- audit leakage;
- emit statistics and train-only normalization reports;
- ensure IDs/splits are deterministic and independent of row order.

### Files Added

- `src/chemfluor/uniprop/__init__.py`
- `src/chemfluor/uniprop/manifests.py`
- `scripts/build_uniprop_manifests.py`
- `tests/test_uniprop_manifests.py`

### Files Updated

- `.gitignore`
  - Added `data/processed/uniprop/` so generated manifest/cache artifacts stay
    local.
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
  - Added Stage 3 section.

### Authoritative Dataset Decision

The manifest builder resolves the authoritative processed dataset as:

1. `data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv`
2. fallback: `data/processed/fluodb_lite/combined_deduplicated.csv`

No duplicate source of truth is created.

### Manifest Outputs

Generated local artifacts under ignored `data/processed/uniprop/`:

- `manifest_metadata.json`
- `molecule_manifest.csv`
- `row_manifest.csv`
- `split_assignments.csv`
- `split_leakage_audit.csv`
- `split_statistics.csv`
- `training_normalization_statistics.csv`

Metadata from the real generated manifest:

```json
{
  "authoritative_dataset": "data\\processed\\fluodb_lite\\combined_deduplicated_with_stokes.csv",
  "source_rows": 66820,
  "manifest_rows": 66820,
  "unique_molecules": 33965,
  "unique_solvents": 1369,
  "all_leakage_audits_passed": true
}
```

Targets in the manifest:

- `absorption_nm`
- `emission_nm`
- `lifetime_ns`
- `quantum_yield`
- `log_extinction`
- `stokes_shift_nm`

Missing target values are preserved as missing; availability masks are emitted
as `{target}_available`.

### Molecule ID Policy

- `molecule_id` is derived from manifest schema version plus RDKit canonical
  isomeric SMILES.
- Stereochemical policy is documented in metadata:
  molecule IDs use canonical isomeric SMILES; non-isomeric SMILES are retained
  only as an auxiliary grouping/reporting field.
- Deterministic molecule seed is derived from stable `molecule_id`.

### Optional Heavy RDKit Passes

Full-dataset InChIKey, atom counts, formal charge, and non-isomeric
canonicalization can be slow or fragile on difficult structures in this Windows
RDKit environment, so they are opt-in:

- `--compute-inchikey`
- `--compute-rdkit-properties`
- `--compute-nonisomeric`

The schema columns remain present when not computed.

### Split Families

Five split families are emitted:

- `random`
- `molecule`
- `scaffold`
- `solvent`
- `double_cold_start`

The persisted scaffold split was generated with RDKit Bemis-Murcko scaffold
groups via `--compute-rdkit-scaffolds`.

Double-cold-start uses:

- `train`
- `test`
- `heldout_boundary`

Boundary rows contain exactly one held-out axis and are excluded from
train/test to avoid molecule or solvent leakage.

Final split leakage audit from generated real artifacts:

```text
random             passed
molecule           overlapping_molecule_ids = 0
scaffold           overlapping_scaffold_groups = 0
solvent            overlapping_solvent_ids = 0
double_cold_start  overlapping_molecule_ids = 0, overlapping_solvent_ids = 0
```

### Key Implementation Details

Main functions in `src/chemfluor/uniprop/manifests.py`:

- `resolve_authoritative_dataset`
- `build_manifests`
- `make_split_assignments`
- `audit_split_leakage`
- `split_statistics`
- `training_normalization_statistics`
- `validate_manifest_reconciliation`
- `write_manifest_outputs`

CLI:

```bash
python scripts/build_uniprop_manifests.py \
  --out-dir data/processed/uniprop \
  --seed 42 \
  --test-size 0.2 \
  --compute-rdkit-scaffolds
```

### Prompt 3 Tests

Focused:

```text
python -m pytest tests\test_uniprop_manifests.py
9 passed
```

Full suite after Prompt 3:

```text
python -m pytest
230 passed, 1 skipped, 4 warnings
```

Warnings were existing invalid scaffold/SMILES warning-path tests.

## Prompt 4 - Deterministic One-Geometry-Per-Molecule Cache

### User Request

Implement an initial RDKit geometry cache:

- one geometry per unique molecule ID, never per experimental row;
- ETKDGv3 embedding;
- explicit hydrogens for embedding and optimization;
- MMFF94/MMFF94s with documented UFF fallback;
- seed derived from stable molecule ID;
- validate topology/charge/atom order;
- atomic writes and resumable validation;
- support `--limit`, `--molecule-id`, `--workers`, `--resume`,
  `--overwrite-invalid`, `--fail-fast`;
- JSON and CSV failures;
- Slurm job array.

### Files Added

- `src/chemfluor/uniprop/geometry_cache.py`
- `scripts/build_uniprop_geometry_cache.py`
- `slurm/uniprop/run_uniprop_geometry_cache_array.sbatch`
- `tests/test_uniprop_geometry_cache.py`

### Files Updated

- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
  - Added Stage 4 section.

### Cache Record Schema

Schema version:

```text
uniprop_geometry_cache_v1
```

Each cache JSON stores:

- `schema_version`
- `molecule_id`
- `canonical_smiles`
- `atom_symbols`
- `atomic_numbers`
- `coordinates`
- `optimization_method`
- `energy`
- `convergence_status`
- `rdkit_version`
- `seed`
- `created_at`
- `updated_at`
- `hydrogen_policy`
- `topology_signature`
- `checksum`

### Geometry Behavior

- Uses RDKit ETKDGv3.
- Adds explicit hydrogens before embedding.
- Optimizes with MMFF94s by default when possible.
- Supports `--mmff-variant MMFF94`.
- Falls back to UFF only when MMFF properties are unavailable.
- Removes hydrogens after optimization by default.
- Validates heavy-atom graph, bond topology, formal charge, and atom ordering
  against canonical input.
- Existing JSON entries must pass schema, checksum, atom/coordinate shape, and
  topology validation before counting as cache hits.
- Writes are atomic temp-file writes followed by `os.replace`.

### CLI

```bash
python scripts/build_uniprop_geometry_cache.py \
  --molecule-manifest data/processed/uniprop/molecule_manifest.csv \
  --cache-dir data/processed/uniprop/geometry_cache \
  --workers 4 \
  --resume \
  --overwrite-invalid
```

Supported options:

- `--limit`
- repeatable `--molecule-id`
- `--workers`
- `--resume` / `--no-resume`
- `--overwrite-invalid`
- `--fail-fast`
- `--retain-hydrogens`
- `--mmff-variant MMFF94|MMFF94s`
- `--shard-index`
- `--shard-count`
- `--failure-json`
- `--failure-csv`
- `--status-json`

### Slurm Array

Added:

```text
slurm/uniprop/run_uniprop_geometry_cache_array.sbatch
```

Behavior:

- Uses stable sorted manifest ranges through `--shard-index` and
  `--shard-count`.
- Defaults to resume and overwrite invalid cache entries.
- Configurable environment variables include:
  - `FLUORCAST_REPO`
  - `FLUORCAST_ACTIVATE`
  - `FLUORCAST_UNIPROP_MOLECULE_MANIFEST`
  - `FLUORCAST_UNIPROP_GEOMETRY_CACHE`
  - `FLUORCAST_UNIPROP_GEOMETRY_SHARDS`

### Smoke Runs

Real manifest unrestricted first-three smoke:

```text
expected_total: 3
processed_total: 3
status_counts:
  generated: 2
  failed: 1
```

The failure was structured:

```text
molecule_id: mol_000002cd9fedb8f2
detail: RDKit ETKDGv3 embedding failed with code -1.
```

Targeted known-good real-manifest smoke:

```text
molecule IDs:
mol_0000485e9d6fab52
mol_000049a8f03a85dc
mol_0000b7567836b095

first run: generated 3
resume run: hit 3
after intentional corruption: invalid_cache 1, hit 2
repair with --overwrite-invalid: generated 1, hit 2
```

Generated local smoke cache directories under ignored
`data/processed/uniprop/`:

- `geometry_cache_smoke/`
- `geometry_cache_smoke_good/`

### Prompt 4 Tests

Focused:

```text
python -m pytest tests\test_uniprop_geometry_cache.py
13 passed
```

Focused UniProp/slurm guard:

```text
python -m pytest tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py
37 passed, 1 skipped
```

Full suite after Prompt 4:

```text
python -m pytest
243 passed, 1 skipped, 4 warnings
```

### Remaining Prompt 4 Limitation

Full cache generation for all 33,965 manifest molecules was not run locally.
The Slurm array is ready for that.

## Prompt 5 - UniProp-Compatible LMDB Exporter

### User Request

Implement a FluorCast-to-UniProp LMDB adapter:

- inspect exact upstream task/dataset code before finalizing field names;
- do not generate geometries in the exporter;
- one LMDB record per experimental row;
- attach cached geometry, solvent, targets, masks, row ID, molecule ID;
- match upstream UniProp fields for atoms, positions, graph features, SMILES,
  solvent SMILES;
- determine multitarget fields and add thin loader extension if required;
- build `train.lmdb`, `valid.lmdb`, `test.lmdb`;
- add metadata sidecar;
- transaction batching, map size, completion markers, resume;
- add `scripts/validate_uniprop_lmdb.py`;
- tests including upstream loader smoke.

### Upstream Source Inspection

The pinned upstream source was not present at first. Actions:

1. Tried to inspect `third_party/nablacolors/`; only
   `third_party/nablacolors.REVISION` existed.
2. Network-restricted clone failed.
3. Approval was requested and granted for:

```text
git clone https://github.com/AI4DD/nablaColors.git third_party/nablacolors
```

4. Verified commit:

```text
39095389c0a4ecb47872ef74d00b8d13597939c8
```

Important: `third_party/nablacolors/` is ignored and should remain untracked.

Inspected upstream files:

- `third_party/nablacolors/examples/conformation_generation/04_csv_to_lmdb_rdkit.py`
- `third_party/nablacolors/unimol_plus/unimol_plus/data/pcq_dataset.py`
- `third_party/nablacolors/unimol_plus/unimol_plus/data/lmdb_dataset.py`
- `third_party/nablacolors/unimol_plus/unimol_plus/data/conformer_sample_dataset.py`
- `third_party/nablacolors/unimol_plus/unimol_plus/tasks/pcq.py`
- `third_party/nablacolors/unimol_plus/scripts/get_3d_lmdb.py`
- `third_party/nablacolors/unimol_plus/scripts/get_label3d_lmdb.py`

Confirmed upstream LMDB details:

- Values are gzip-compressed pickles.
- Keys are integer byte keys.
- Upstream `LMDBDataset` injects integer `id` from the key.
- Upstream required fields:
  - `atoms`
  - `input_pos`
  - `label_pos`
  - `smi`
  - `solvent_smi`
  - `node_attr`
  - `edge_index`
  - `edge_attr`
  - `target`
- Upstream `PCQDataset` forwards `target`, `id`, and `solvent_smi`.
- Upstream `PCQDataset` does not collate target masks, so a tiny FluorCast
  wrapper was added.

### Files Added

- `src/chemfluor/uniprop/lmdb_export.py`
- `src/chemfluor/uniprop/upstream_compat.py`
- `scripts/export_uniprop_lmdb.py`
- `scripts/validate_uniprop_lmdb.py`
- `tests/test_uniprop_lmdb_export.py`
- `pytest.ini`

### Files Updated

- `requirements.txt`
  - Added `lmdb`.
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
  - Added Stage 5 section.

### Dependency Install

`lmdb` was initially missing:

```text
python -c "import importlib.util; print(importlib.util.find_spec('lmdb'))"
None
```

Sandboxed install failed due network restriction. Approval was requested and
granted for:

```text
python -m pip install lmdb
```

Installed:

```text
lmdb-2.3.0
```

### Pytest Collection Config

After cloning ignored upstream source, full `pytest` began collecting
`third_party/nablacolors/Uni-Core/tests/test_softmax.py`.

Added `pytest.ini`:

```ini
[pytest]
testpaths = tests
```

This scopes test collection to the FluorCast test suite and prevents ignored
upstream tests from being collected.

### LMDB Export Schema

Schema version:

```text
fluorcast_uniprop_lmdb_v1
```

Each LMDB record contains upstream fields:

- `atoms`
- `input_pos`
- `label_pos`
- `smi`
- `solvent_smi`
- `node_attr`
- `edge_index`
- `edge_attr`
- `target`

Plus FluorCast fields:

- `target_mask`
- `target_columns`
- `row_id`
- `molecule_id`
- `solvent_id`
- `id`
- `geometry_cache_schema`

### Missing Target Policy

- Missing targets are stored as `NaN`.
- Missing target masks are `False`.
- Real target values have mask `True`.
- Zero is never used as an unmasked substitute for missing labels.

### Split Export Behavior

Exporter writes:

- `train.lmdb`
- `valid.lmdb`
- `test.lmdb`
- completion markers such as `train.lmdb.complete`
- `metadata.json`

Existing manifest split assignments are train/test. The exporter preserves the
held-out `test` partition and deterministically carves `valid` out of training
rows using:

- stable `row_id`
- `seed`
- `valid_size`

Default `valid_size` is `0.1`.

### Metadata Sidecar

`metadata.json` includes:

- schema version;
- created timestamp;
- row-manifest hash;
- molecule-manifest hash;
- split-assignment hash;
- geometry-cache schema;
- upstream revision;
- row counts;
- target counts;
- creation configuration;
- partition validation reports.

### Main LMDB Functions

In `src/chemfluor/uniprop/lmdb_export.py`:

- `encode_int_key`
- `decode_int_key`
- `get_graph`
- `build_lmdb_record`
- `load_export_inputs`
- `export_uniprop_lmdb`
- `read_lmdb_records`
- `validate_record`
- `validate_lmdb`

Graph feature encoding was copied to match pinned nablaColors
`get_3d_lmdb.py`:

- atom feature size: 9 integer fields;
- edge feature size: 3 integer fields;
- bidirectional edges;
- dtypes: `np.int32`.

### CLI

Export:

```bash
python scripts/export_uniprop_lmdb.py \
  --row-manifest data/processed/uniprop/row_manifest.csv \
  --molecule-manifest data/processed/uniprop/molecule_manifest.csv \
  --split-assignments data/processed/uniprop/split_assignments.csv \
  --geometry-cache-dir data/processed/uniprop/geometry_cache \
  --out-dir data/processed/uniprop/lmdb \
  --split-family random \
  --seed 42 \
  --targets absorption_nm,emission_nm,lifetime_ns,quantum_yield,log_extinction,stokes_shift_nm \
  --map-size 10737418240 \
  --batch-size 1000 \
  --resume
```

Validate:

```bash
python scripts/validate_uniprop_lmdb.py \
  data/processed/uniprop/lmdb/test.lmdb \
  --row-manifest data/processed/uniprop/row_manifest.csv \
  --split-assignments data/processed/uniprop/split_assignments.csv \
  --split-family random \
  --partition test
```

### Thin Upstream-Compatible Loader Extension

Added `TargetMaskDataset` in:

```text
src/chemfluor/uniprop/upstream_compat.py
```

Purpose:

- The pinned upstream LMDB loader preserves extra record fields.
- The pinned upstream `PCQDataset.collater` batches `target` but not
  `target_mask`.
- `TargetMaskDataset` wraps an upstream PCQ-style dataset and adds batched
  `target_mask` when present.

This is intentionally small and does not modify upstream code.

### Prompt 5 Tests

Added `tests/test_uniprop_lmdb_export.py`.

Coverage includes:

- exact required record keys, shapes, and dtypes;
- graph features agree with direct RDKit calculation;
- repeated chromophore rows have identical coordinates;
- repeated chromophore rows keep distinct solvent/target/row fields;
- masks exactly match missingness;
- missing targets are `NaN`, not zero;
- LMDB key order deterministic;
- all row IDs occur exactly once across train/valid/test;
- corrupt/incomplete LMDBs rejected;
- validator reconciles with row manifest;
- export CLI and validate CLI smoke;
- exporter does not call geometry generation;
- 100-row smoke loads through the exact pinned upstream `LMDBDataset` source
  file loaded directly from:

```text
third_party/nablacolors/unimol_plus/unimol_plus/data/lmdb_dataset.py
```

The full upstream package import requires optional Uni-Core dependencies such
as `tokenizers`, so the test loads the exact upstream LMDBDataset source file
directly to avoid unrelated imports.

### Prompt 5 Test Results

Focused:

```text
python -m pytest tests\test_uniprop_lmdb_export.py
12 passed in 4.41s
```

Focused UniProp suite:

```text
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py
47 passed, 1 skipped in 13.18s
```

Final full suite:

```text
python -m pytest
255 passed, 1 skipped, 4 warnings in 51.60s
```

The 4 warnings are existing invalid scaffold/SMILES warning-path tests in
hybrid/paired workflows.

### Remaining Prompt 5 Limitation

Full real-manifest LMDB export was not run because full geometry cache
generation has not yet been completed.

## Interrupted FluorCast Desktop/NIBI Runtime Request

After Prompt 5, the user attached a long request:

```text
Redesign FluorCast Desktop's remote NIBI runtime deployment.
```

Key architecture from that request:

- FluorCast ML repo owns prediction code and artifacts.
- `fluorcast-desktop` owns remote environment setup, validation, runtime
  initialization, Slurm orchestration, SSH/WSL integration, uploads, polling,
  cancellation, and diagnostics.
- Desktop-owned helper scripts should be bundled under something like
  `src-tauri/resources/nibi-runtime/` in `fluorcast-desktop`.
- Desktop should deploy those runtime assets to:

```text
$HOME/.fluorcast/runtime/<runtime-bundle-version>/
```

- Remote ML repo validation should require ML assets only, especially:

```text
scripts/run_prediction_job.py
```

- It should not require desktop-owned helpers in the remote ML repo.
- The ML repo should expose:

```bash
python scripts/run_prediction_job.py --protocol-info
```

returning structured prediction protocol JSON.

### What Happened In This Chat

The request was read from:

```text
C:\Users\CL\.codex\attachments\11a92247-fc86-48c0-a62b-2c19215745a1\pasted-text.txt
```

The desktop repository was located at:

```text
C:\Users\CL\OneDrive\Desktop\projects\fluorcast-desktop
```

The current writable sandbox root was still the FluorCast ML repo:

```text
C:\Users\CL\OneDrive\Desktop\python\FluorCast
```

I stated I would first add the companion ML-side `--protocol-info` endpoint,
then inspect/touch the desktop repo only with approval because it is outside
the writable workspace.

The user interrupted the turn before any implementation edits for this desktop
runtime request were made.

### Important: Not Yet Done

No changes were made to:

- `scripts/run_prediction_job.py` for `--protocol-info`;
- `fluorcast-desktop`;
- desktop Tauri resource bundling;
- desktop runtime manifest;
- desktop SSH/SCP synchronization logic;
- desktop UI status rows;
- desktop tests.

The attached request should be treated as the next major task if the user asks
to continue it.

## Generated/Ignored Local Artifacts

These local generated/ignored paths now exist or may exist:

- `third_party/nablacolors/`
  - cloned pinned upstream source;
  - ignored by `.gitignore`;
  - used for inspection and tests.
- `data/processed/uniprop/`
  - ignored by `.gitignore`;
  - contains Prompt 3 manifests and Prompt 4 smoke geometry caches.
- `outputs/uniprop_geometry*.json`
- `outputs/uniprop_geometry*.csv`
  - ignored under `outputs/`.

Do not commit generated data/cache/outputs or downloaded upstream source.

## Commands Run In This Chat

Prompt 3:

```text
Get-Content development md/UniProp Handoff/handoff1.md
Get-Content src/chemfluor/data_standardization.py
Get-Content src/splitting.py
Get-Content src/data.py
Get-Content scripts/manuscript/manuscript_splits.py
Get-Content scripts/run_graph_model_experiments.py
Get-Content tests/test_standardized_deduplication.py
Get-Content docs/UNIPROP_IMPLEMENTATION_LOG.md
python -c "... inspect data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv ..."
python -m pytest tests\test_uniprop_manifests.py
python scripts\build_uniprop_manifests.py --out-dir data\processed\uniprop --seed 42 --test-size 0.2
python scripts\build_uniprop_manifests.py --out-dir data\processed\uniprop --seed 42 --test-size 0.2 --compute-rdkit-scaffolds
python -m pytest
```

Prompt 4:

```text
Get-Content slurm/run_duplicate_check_job.sbatch
Get-Content slurm/run_paper_comparison_experiments.sbatch
python -c "... probe RDKit MMFF/UFF availability ..."
python -m pytest tests\test_uniprop_geometry_cache.py
python scripts\build_uniprop_geometry_cache.py --molecule-manifest data\processed\uniprop\molecule_manifest.csv --cache-dir data\processed\uniprop\geometry_cache_smoke --limit 3 --workers 1 --status-json outputs\uniprop_geometry_smoke_status.json --failure-json outputs\uniprop_geometry_smoke_failures.json --failure-csv outputs\uniprop_geometry_smoke_failures.csv
python scripts\build_uniprop_geometry_cache.py --molecule-manifest data\processed\uniprop\molecule_manifest.csv --cache-dir data\processed\uniprop\geometry_cache_smoke_good --molecule-id mol_0000485e9d6fab52 --molecule-id mol_000049a8f03a85dc --molecule-id mol_0000b7567836b095 --workers 1 --status-json outputs\uniprop_geometry_smoke_good_status.json --failure-json outputs\uniprop_geometry_smoke_good_failures.json --failure-csv outputs\uniprop_geometry_smoke_good_failures.csv
python -m pytest tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py
python -m pytest
```

Prompt 5:

```text
git clone --filter=blob:none --no-checkout https://github.com/AI4DD/nablaColors.git third_party/nablacolors
git clone https://github.com/AI4DD/nablaColors.git third_party/nablacolors
git -c safe.directory=C:/Users/CL/OneDrive/Desktop/python/FluorCast/third_party/nablacolors -C third_party/nablacolors rev-parse HEAD
rg -n "input_pos|label_pos|atoms|input_atoms|label_atoms|solvent|smi|smiles|target|mask|Lmdb|LMDB|lmdb|edge" third_party/nablacolors/unimol_plus third_party/nablacolors/examples third_party/nablacolors/Uni-Core/unicore/data
Get-Content third_party/nablacolors/examples/conformation_generation/04_csv_to_lmdb_rdkit.py
Get-Content third_party/nablacolors/unimol_plus/unimol_plus/data/pcq_dataset.py
Get-Content third_party/nablacolors/unimol_plus/unimol_plus/data/lmdb_dataset.py
Get-Content third_party/nablacolors/unimol_plus/unimol_plus/tasks/pcq.py
Get-Content third_party/nablacolors/unimol_plus/unimol_plus/data/conformer_sample_dataset.py
Get-Content third_party/nablacolors/unimol_plus/scripts/get_3d_lmdb.py
Get-Content third_party/nablacolors/unimol_plus/scripts/get_label3d_lmdb.py
python -c "import importlib.util; print(importlib.util.find_spec('lmdb'))"
python -m pip install lmdb
python -m pytest tests\test_uniprop_lmdb_export.py
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py
python -m pytest
```

Interrupted desktop request:

```text
Get-Content C:\Users\CL\.codex\attachments\11a92247-fc86-48c0-a62b-2c19215745a1\pasted-text.txt
Get-Content scripts/run_prediction_job.py
Get-Content tests/test_prediction_job_hybrid.py
Get-ChildItem C:\Users\CL\OneDrive\Desktop\projects
```

## Test Results Summary

Final known test state:

```text
python -m pytest
255 passed, 1 skipped, 4 warnings in 51.60s
```

Skipped test:

- Python 3.10 UniProp bootstrap smoke because no `python3.10` executable is
  available in the current Windows environment.

Warnings:

- Existing invalid scaffold/SMILES warning-path tests from hybrid and paired
  workflows.

## Recommended Next Steps

For UniProp:

1. Run full geometry cache generation on Nibi/Linux:

```bash
sbatch slurm/uniprop/run_uniprop_geometry_cache_array.sbatch
```

2. Inspect/merge per-shard geometry failure reports.
3. Export full LMDBs after the cache is complete:

```bash
python scripts/export_uniprop_lmdb.py \
  --row-manifest data/processed/uniprop/row_manifest.csv \
  --molecule-manifest data/processed/uniprop/molecule_manifest.csv \
  --split-assignments data/processed/uniprop/split_assignments.csv \
  --geometry-cache-dir data/processed/uniprop/geometry_cache \
  --out-dir data/processed/uniprop/lmdb \
  --split-family molecule \
  --overwrite
```

4. Validate exported LMDBs:

```bash
python scripts/validate_uniprop_lmdb.py data/processed/uniprop/lmdb/train.lmdb
python scripts/validate_uniprop_lmdb.py data/processed/uniprop/lmdb/valid.lmdb
python scripts/validate_uniprop_lmdb.py data/processed/uniprop/lmdb/test.lmdb
```

5. Wire `TargetMaskDataset` or an equivalent upstream task extension into
   training before fitting multitarget heads with missing labels.

For the interrupted desktop/NIBI runtime request:

1. Get explicit permission to work outside this workspace in:

```text
C:\Users\CL\OneDrive\Desktop\projects\fluorcast-desktop
```

2. Add the ML companion protocol endpoint here:

```bash
python scripts/run_prediction_job.py --protocol-info
```

3. Then implement the desktop-owned runtime bundle architecture in the desktop
   repository as described in the pasted request.

