# UniProp Handoff

Date: 2026-07-20

Repository: `C:\Users\CL\OneDrive\Desktop\python\FluorCast`

Branch: `feature/uniprop-3d`

This handoff captures the context, decisions, files, commands, tests, and
remaining work from the UniProp/nablaColors integration chat through Prompt 2.

## User Constraints From The Chat

Shared instructions for every stage:

- Work only in the main FluorCast machine-learning repository on branch
  `feature/uniprop-3d`.
- Inspect the repository before changing anything.
- Do not modify the FluorCast desktop application.
- Do not modify or depend directly on the old ConforFormer experimental
  directory unless reviewing a generic component for reuse.
- Implement the requested stage, not only a proposal.
- Preserve existing interfaces and tests unless a documented migration is
  necessary.
- Keep generated data, third-party source trees, checkpoints, LMDB files, and
  logs out of Git.
- Make paths configurable and never hard-code a local username, Windows path,
  Nibi account, scratch path, or CUDA device.
- Add type hints, validation, clear errors, deterministic seeds, and resumable
  behavior where implementation code is added.
- Add unit and integration tests.
- Run all new tests and the relevant existing test suite.
- Update `docs/UNIPROP_IMPLEMENTATION_LOG.md`.
- Finish each stage with files changed, commands run, exact test results,
  remaining limitations, and recommended next stage.
- Do not claim success unless tests were actually run.

IDE context at the last user request:

- Active file: `slurm/run_duplicate_check_job.sbatch`
- Open tabs:
  - `slurm/run_duplicate_check_job.sbatch`
  - `slurm/run_prediction_job.sbatch`
  - `slurm/run_predict_full_fluorcast.sbatch`
  - `slurm/run_paper_comparison_experiments.sbatch`
  - `slurm/run_paired_spectral_three_way_experiment.sbatch`

## Stage 1 Prompt

Prompt: Repository audit and design for beginning UniProp/nablaColors
integration.

Acceptance criteria:

- No production behavior changed.
- Existing tests still pass.
- Required upstream assets and interfaces documented.
- Migration decision for every reusable ConforFormer component explicit.

## Stage 1 Work Completed

Created:

- `docs/UNIPROP_3D_DESIGN.md`
- `docs/UNIPROP_ASSET_MAP.md`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
- `tests/test_uniprop_design_docs.py`

Key repository facts captured:

- Branch confirmed as `feature/uniprop-3d`.
- Initial working tree was clean.
- Local Python used by tests was `Python 3.14.0`.
- Existing Slurm workflows generally target Python 3.11.
- Package layout:
  - reusable code: `src/`, `src/chemfluor/`
  - scripts: `scripts/`
  - tests: `tests/`
  - jobs: `slurm/`
  - docs: `docs/`
- Existing requirements are lightweight FluorCast ML dependencies:
  `numpy<2.0`, pandas, RDKit, scikit-learn, SciPy, matplotlib, LightGBM,
  XGBoost, CatBoost, Jupyter, pytest.

Data/training entry points audited:

- Original workflow: `src.train`
- Combined predictors: `scripts/train_combined_predictors.py`
- Manuscript comparison:
  `scripts/manuscript/run_paper_comparison_experiments.py`
- Standardized data:
  `src/chemfluor/data_standardization.py`
- Prediction:
  `scripts/predict_combined_molecule.py`
  `scripts/predict_all_models.py`
  `scripts/predict_full_fluorcast.py`
  `scripts/run_prediction_job.py`

Split code audited:

- `src/splitting.py`
- `scripts/manuscript/manuscript_splits.py`
- Combined and graph workflows using grouped splits by
  `canonical_chromophore_smiles`.

Target columns:

- Existing standardized targets:
  - `absorption_nm`
  - `emission_nm`
  - `lifetime_ns`
  - `quantum_yield`
  - `log_extinction`
- Initial UniProp targets:
  - `absorption_nm`
  - `emission_nm`
  - `quantum_yield`
- Later extension:
  - `lifetime_ns`
  - `log_extinction`
  - physics constraints such as non-negative Stokes shift.

Result formats:

- Tree/neural artifacts:
  `{target}_{model}.joblib`, `feature_metadata.json`, `metrics.json`
- Graph artifacts:
  `{target}_{model}.pt`, scalers, metadata, metrics, prediction CSVs
- Paper comparison:
  `metrics_by_split_model_target.csv`,
  `metrics_with_bootstrap_ci.csv`, `split_leakage_report.csv`, prediction CSVs,
  figures, Markdown summaries.

Slurm conventions:

- Jobs live under `slurm/`, `slurm/base_models/`, `slurm/manuscript/`,
  `slurm/util/`, and `slurm/legacy/`.
- Jobs resolve repo paths through environment variables and submit-directory
  fallbacks.
- Generated artifacts go under `outputs/` and `models/`.
- New UniProp Slurm scripts should keep account, environment activation, data
  paths, asset paths, output paths, seed, and device configurable.

## Stage 1 Upstream nablaColors Findings

Official sources reviewed:

- `https://github.com/AI4DD/nablaColors`
- `https://zenodo.org/records/18061300`
- `https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus`

Important upstream requirements:

- Recommended Python: `3.10`
- Install Uni-Core from repo root with `bash install_unicore.sh`, or manually
  install local `Uni-Core` and ensure `unicore-train` is available.
- Install Uni-Mol+ with `cd unimol_plus && pip install -e .`.
- Chemprop v1.3.0 is vendored by nablaColors for solvent embedding.
- Chemprop v2.x is not directly compatible with v1.x models.

Expected upstream layout:

- `Uni-Core/`
- `unimol_plus/`
- `models/chemprop/fold_0/model_1/model.pt`
- `examples/conformation_generation/`
- `install_unicore.sh`

LMDB facts:

- LMDB values are gzip-compressed pickled dictionaries.
- Keys are byte strings.
- Dataset includes `absorption_conformations.zip` with xTB, DFT vacuum, and
  DFT implicit-solvent geometries.

Training entry points:

- Multitarget head-only:
  `bash unimol_plus/head_pretrain_uniprop_multitarget.sh --data-path <dataset> --pretrained-model <unimol_plus_pcq_small.pt>`
- Multitarget unfrozen backbone:
  `bash unimol_plus/finetune_unfreeze_backbone_multitarget.sh --data-path <dataset> --pretrained-model <head_checkpoint.pt>`
- Single-target absorption head-only:
  `bash unimol_plus/head_pretrain_uniprop_singletarget.sh --data-path <dataset> --pretrained-model <unimol_plus_pcq_small.pt>`
- Single-target absorption unfrozen backbone:
  `bash unimol_plus/finetune_unfreeze_backbone_singletarget.sh --data-path <dataset> --pretrained-model <head_checkpoint.pt>`

Validation entry points:

- Multitarget:
  `bash unimol_plus/validate_multitarget.sh --data-path <dataset> --weight-path <checkpoint.pt> --subset valid --results-path <eval_dir>`
- Single-target absorption:
  `bash unimol_plus/validate_singletarget_absorption.sh --data-path <dataset> --weight-path <checkpoint.pt> --subset valid --results-path <eval_dir>`

Validation outputs:

- `${results_path}/${subset}.metrics.json`
- `${results_path}/${subset}.preds.pkl`

Screening LMDB builder:

```bash
python examples/conformation_generation/04_csv_to_lmdb_rdkit.py \
  --csv screening.csv \
  --out-dir screening_lmdb \
  --split test \
  --smiles-col smiles \
  --solvent-col solvent_smi
```

Pretrained assets:

- Uni-Mol+ small checkpoint from official Uni-Mol+
- Chemprop solvent model at `models/chemprop/fold_0/model_1/model.pt`
- Zenodo UniProp checkpoints:
  - `uniprop_rdkit_to_dft_implicit.pt`
  - `uniprop_rdkit_to_xtb.pt`
  - `uniprop_xtb_to_dft_implicit.pt`
  - `uniprop_xtb_to_dft_vacuum.pt`

## Stage 1 Data-Model Decision

Design contract:

- One cached geometry per canonical unique chromophore.
- One supervised record per chromophore-solvent observation.
- Upstream code remains separated from FluorCast adapters.
- Geometry cache must not include solvent-specific target values.
- Supervised records reference geometry by canonical chromophore key.
- Solvent embedding and target labels are observation-level data.

Leakage prevention:

- Geometry generation runs before or independently from target splits and does
  not inspect target values.
- Train-only imputers, scalers, calibration, and target normalization are fit
  inside each split.
- Molecule and scaffold splits group before expansion to chromophore-solvent
  observations.
- Pretrained UniProp is treated as an external prior; no FluorCast test labels
  enter head pretraining, finetuning, early stopping, or model selection.

Proposed module structure:

```text
src/chemfluor/uniprop/
  __init__.py
  schemas.py
  assets.py
  geometry.py
  lmdb_io.py
  datasets.py
  splits.py
  commands.py
  results.py
  validation.py

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

## Stage 1 ConforFormer Read-Only Audit

Old experiment location:

- `C:\Users\CL\OneDrive\Desktop\python\fluorcast-conforformer`

Reviewed read-only:

- `src/chemfluor/conforformer/schemas.py`
- `src/chemfluor/conforformer/config.py`
- `src/chemfluor/conforformer/preprocess.py`
- `src/chemfluor/conforformer/cache.py`
- `src/chemfluor/conforformer/dictionary.py`
- `src/chemfluor/conforformer/conformers.py`
- `src/chemfluor/conforformer/adapter.py`
- `scripts/build_conformer_cache.py`
- ConforFormer docs and tests

Reusable unchanged:

- Stable JSON payload hashing.
- Atomic cache write and hash-verified cache read.
- Explicit asset path validation and checksum recording.
- Error taxonomy with reason codes.

Reusable after refactoring:

- Conformer-generation request/config schema.
- Cache-key payloads.
- CLI dry-run, resumability, and cache-hit behavior.
- Test style for deterministic cache keys, corruption detection, dry-run, and
  validation errors.
- Environment/asset inspection pattern.

Not reusable:

- ConforFormer dictionary loading.
- Uni-Mol contrast preprocessing, edge types, hydrogen policy, CLS embedding.
- ConforFormer checkpoint architecture inspection and upstream model-building.
- Vendored `third_party/ConforFormer` source and ConforFormer-specific docs or
  scripts.

Decision:

- Reuse design patterns only.
- Do not depend on ConforFormer-specific preprocessing, encoder loading,
  dictionary handling, or vendored source.

## Stage 1 Tests Run

- `python -m pytest tests\test_uniprop_design_docs.py`
  - `4 passed in 0.03s`
- `python -m pytest`
  - `215 passed, 4 warnings in 62.29s`
- Final doc guard:
  - `python -m pytest tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py`
  - `7 passed in 0.05s`

## Stage 2 Prompt

Prompt: Reproducible dependency bootstrap.

Acceptance criteria:

- Target Python 3.10.
- Add pinned upstream revision file such as
  `third_party/nablacolors.REVISION`.
- Add `scripts/bootstrap_uniprop.sh`.
- Clone exact pinned nablaColors commit into ignored
  `third_party/nablacolors`.
- Refuse to silently use a different commit.
- Install Uni-Core and Uni-Mol+ in editable mode.
- Support CPU validation and CUDA/Nibi mode.
- Do not install globally.
- Safe to rerun.
- Add checkpoint manifest with filename, expected size, source, checksum.
- Do not commit checkpoints.
- Add `scripts/audit_uniprop_environment.py`.
- Report Python, PyTorch, CUDA availability, CUDA runtime, GPU name, RDKit,
  LMDB, Uni-Core, Uni-Mol+, Chemprop, upstream Git revision, checkpoint
  presence and hashes.
- Add dependency documentation for local WSL and Nibi.
- Do not reuse Python 3.14 ConforFormer environment.
- Add `--dry-run` and `--json-output` support where appropriate.
- Add tests for shell syntax, dry-run behavior, revision mismatch detection,
  missing-checkpoint reporting, JSON schema, Python 3.10 import smoke, and
  existing FluorCast tests.
- Environment report clearly says whether machine is ready for preprocessing,
  CPU smoke testing, and GPU training.
- Rerunning bootstrap does not corrupt existing valid setup.
- No downloaded dependency or checkpoint is tracked by Git.

## Stage 2 Work Completed

Files added or updated:

- `.gitignore`
- `third_party/nablacolors.REVISION`
- `configs/uniprop/checkpoint_manifest.json`
- `scripts/bootstrap_uniprop.sh`
- `scripts/audit_uniprop_environment.py`
- `docs/UNIPROP_DEPENDENCIES.md`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
- `tests/test_uniprop_bootstrap_environment.py`

Pinned upstream:

- Repo: `https://github.com/AI4DD/nablaColors.git`
- Ref: `v1.0.0`
- Commit: `39095389c0a4ecb47872ef74d00b8d13597939c8`
- The commit was confirmed with:

```bash
git ls-remote https://github.com/AI4DD/nablaColors.git refs/tags/v1.0.0 refs/heads/main
```

Output:

```text
39095389c0a4ecb47872ef74d00b8d13597939c8 refs/heads/main
39095389c0a4ecb47872ef74d00b8d13597939c8 refs/tags/v1.0.0
```

`.gitignore` was updated to keep local generated/dependency assets out of Git:

- `.venv-uniprop/`
- `*.pt`
- `*.ckpt`
- `*.lmdb`
- `third_party/nablacolors/`
- `assets/uniprop/`
- `assets/nablacolors/`
- `outputs/uniprop*/`
- `models/uniprop*/`

## Stage 2 Bootstrap Script

File:

- `scripts/bootstrap_uniprop.sh`

Main behavior:

- Supports:
  - `--mode cpu|cuda`
  - `--python PATH`
  - `--venv PATH`
  - `--upstream-dir PATH`
  - `--revision-file PATH`
  - `--repo-url URL`
  - `--dry-run`
  - `--json-output PATH`
- Reads `third_party/nablacolors.REVISION`.
- Clones `third_party/nablacolors/` only when absent.
- If checkout exists, verifies it is a Git checkout and exactly matches pinned
  commit.
- Refuses revision mismatch before installs.
- Creates/reuses `.venv-uniprop/`.
- Verifies Python 3.10.
- Verifies the interpreter is inside a virtualenv before installs.
- Updates pip/setuptools/wheel inside the isolated venv.
- Runs upstream `install_unicore.sh`.
- Installs Uni-Core editable if `Uni-Core/` is present.
- Installs `unimol_plus` editable.
- In CPU mode exports `CUDA_VISIBLE_DEVICES=-1` unless already set.
- Writes JSON status if requested.
- Dry-run prints planned actions and does not create clone or venv.

Important non-goal:

- Non-dry-run bootstrap was not executed in this chat. No dependencies were
  installed.

## Stage 2 Audit Script

File:

- `scripts/audit_uniprop_environment.py`

Main behavior:

- Safe to run from default FluorCast environment.
- Does not install anything.
- Supports:
  - `--revision-file`
  - `--upstream-dir`
  - `--manifest`
  - `--checkpoint-dir`
  - `--json-output`
  - `--dry-run`
- Reports:
  - Python executable and version
  - PyTorch version
  - CUDA availability
  - CUDA runtime
  - GPU name
  - RDKit
  - LMDB
  - Uni-Core
  - Uni-Mol+
  - Uni-Mol
  - Chemprop
  - pinned revision
  - actual upstream Git revision
  - checkpoint presence
  - checkpoint sizes
  - checkpoint hashes
- Emits readiness booleans:
  - `preprocessing_ready`
  - `cpu_smoke_ready`
  - `gpu_training_ready`
- Emits reason flags explaining why each readiness value is true or false.

Current local expectation:

- In the current Windows session, Python is 3.14. Therefore UniProp readiness is
  expected to be false until run from `.venv-uniprop` with Python 3.10.

## Stage 2 Checkpoint Manifest

File:

- `configs/uniprop/checkpoint_manifest.json`

Contains four Zenodo checkpoint records:

- `uniprop_rdkit_to_dft_implicit.pt`
  - Source:
    `https://zenodo.org/records/18061300/files/uniprop_rdkit_to_dft_implicit.pt`
  - MD5: `c87305171142e1c0898a0e2b67a7236a`
- `uniprop_rdkit_to_xtb.pt`
  - Source:
    `https://zenodo.org/records/18061300/files/uniprop_rdkit_to_xtb.pt`
  - MD5: `7be9b8858e70a85718429cd17dd0670b`
- `uniprop_xtb_to_dft_implicit.pt`
  - Source:
    `https://zenodo.org/records/18061300/files/uniprop_xtb_to_dft_implicit.pt`
  - MD5: `b9768e7b4f69b4d54b5d436b7403e883`
- `uniprop_xtb_to_dft_vacuum.pt`
  - Source:
    `https://zenodo.org/records/18061300/files/uniprop_xtb_to_dft_vacuum.pt`
  - MD5: `369b98e9bc9915396822c8274bf89d2f`

Default checkpoint directory:

- `assets/uniprop/checkpoints`

Configurable environment variable:

- `FLUORCAST_UNIPROP_CHECKPOINT_DIR`

Limitation:

- Expected sizes are currently recorded from Zenodo's published MB display
  (`459500000` bytes placeholders). Tighten to exact byte counts when real
  checkpoint files are staged.

## Stage 2 Dependency Documentation

File:

- `docs/UNIPROP_DEPENDENCIES.md`

Documents:

- Tracked inputs.
- Ignored generated/downloaded files.
- Local WSL CPU dry-run and bootstrap.
- Nibi/CUDA setup.
- Environment audit commands.
- Rerun behavior.
- Checkpoint staging policy.

Local WSL dry run:

```bash
bash scripts/bootstrap_uniprop.sh --mode cpu --dry-run
```

Local WSL bootstrap:

```bash
bash scripts/bootstrap_uniprop.sh --mode cpu --python python3.10
```

Nibi/CUDA bootstrap:

```bash
module purge
module load python/3.10
module load gcc

export FLUORCAST_UNIPROP_CHECKPOINT_DIR="$SCRATCH/fluorcast_uniprop_checkpoints"
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10
```

Audit:

```bash
.venv-uniprop/bin/python scripts/audit_uniprop_environment.py \
  --checkpoint-dir "$FLUORCAST_UNIPROP_CHECKPOINT_DIR" \
  --json-output outputs/uniprop_environment_report_nibi.json
```

## Stage 2 Tests Added

File:

- `tests/test_uniprop_bootstrap_environment.py`

Tests:

- Bootstrap shell syntax with `bash -n`.
- Bootstrap dry-run does not create clone or venv.
- Revision mismatch detection.
- Missing-checkpoint reporting.
- Environment report JSON schema.
- Checkpoint manifest schema.
- Pinned revision file schema.
- Python 3.10 import smoke when Python 3.10 is available.

Windows/WSL note:

- Tests include WSL path conversion for this Windows workspace when `bash.exe`
  is exposed through the WindowsApps WSL shim.

## Stage 2 Test Runs

Focused test initial run:

- `python -m pytest tests\test_uniprop_bootstrap_environment.py`
- Initial result exposed:
  - Windows/WSL bash path handling issue.
  - Missing explicit `importlib.util` import.
- Both were fixed.

Focused final:

- `python -m pytest tests\test_uniprop_bootstrap_environment.py`
- Result:
  - `6 passed, 1 skipped in 3.03s`
- Skipped test:
  - Python 3.10 smoke because no `python3.10` executable was available in the
    current Windows environment.

Combined doc/bootstrap guard:

- `python -m pytest tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py`
- Result:
  - `13 passed, 1 skipped in 2.80s`

Full suite:

- `python -m pytest`
- Result:
  - `221 passed, 1 skipped, 4 warnings in 43.78s`

Final doc/bootstrap guard after implementation log update:

- `python -m pytest tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py`
- Result:
  - `13 passed, 1 skipped in 2.76s`

The 4 warnings in the full suite are existing warning-path tests around invalid
scaffold/SMILES rows in hybrid and paired workflows.

## Current Git Status At Handoff

Status after Stage 2 before this handoff file was created:

```text
 M .gitignore
?? configs/
?? docs/UNIPROP_3D_DESIGN.md
?? docs/UNIPROP_ASSET_MAP.md
?? docs/UNIPROP_DEPENDENCIES.md
?? docs/UNIPROP_IMPLEMENTATION_LOG.md
?? scripts/audit_uniprop_environment.py
?? scripts/bootstrap_uniprop.sh
?? tests/test_uniprop_bootstrap_environment.py
?? tests/test_uniprop_design_docs.py
?? third_party/
```

Expected new tracked candidates:

- `.gitignore`
- `configs/uniprop/checkpoint_manifest.json`
- `docs/UNIPROP_3D_DESIGN.md`
- `docs/UNIPROP_ASSET_MAP.md`
- `docs/UNIPROP_DEPENDENCIES.md`
- `docs/UNIPROP_IMPLEMENTATION_LOG.md`
- `scripts/audit_uniprop_environment.py`
- `scripts/bootstrap_uniprop.sh`
- `tests/test_uniprop_bootstrap_environment.py`
- `tests/test_uniprop_design_docs.py`
- `third_party/nablacolors.REVISION`
- `development md/UniProp Handoff/README.md`

Expected ignored local/generated paths:

- `.venv-uniprop/`
- `third_party/nablacolors/`
- `assets/uniprop/`
- `assets/nablacolors/`
- `*.pt`
- `*.ckpt`
- `*.lmdb`
- `outputs/uniprop*/`
- `models/uniprop*/`

No downloaded nablaColors clone or checkpoint was created during the chat.

## Commands Run Across The Chat

Repository/audit:

```text
git branch --show-current
git status --short
rg --files
Get-ChildItem -Force
Get-Content -Path requirements.txt
Get-Content -Path src\config.py
Get-Content -Path src\data.py
Get-Content -Path src\splitting.py
Get-Content -Path src\models.py
Get-Content -Path src\train.py
Get-Content -Path src\chemfluor\data_standardization.py
Get-Content -Path scripts\train_combined_predictors.py
Get-Content -Path scripts\manuscript\manuscript_splits.py
Get-Content -Path scripts\manuscript\run_paper_comparison_experiments.py
Get-Content -Path scripts\run_graph_model_experiments.py
Get-Content -Path scripts\predict_all_models.py
Get-Content -Path slurm\run_paper_comparison_experiments.sbatch
```

Upstream:

```text
git ls-remote https://github.com/AI4DD/nablaColors.git refs/tags/v1.0.0 refs/heads/main
```

ConforFormer read-only:

```text
rg --files C:\Users\CL\OneDrive\Desktop\python\fluorcast-conforformer
Get-ChildItem -Force C:\Users\CL\OneDrive\Desktop\python\fluorcast-conforformer
Get-Content for relevant ConforFormer schemas/config/cache/preprocess/adapter/docs/tests
```

Tests:

```text
python -m pytest tests\test_uniprop_design_docs.py
python -m pytest
python -m pytest tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py
python -m pytest tests\test_uniprop_bootstrap_environment.py
python -m pytest tests\test_uniprop_design_docs.py tests\test_uniprop_bootstrap_environment.py tests\test_docs_slurm_references.py
```

## Important Sources Used

- AI4DD/nablaColors GitHub:
  `https://github.com/AI4DD/nablaColors`
- Zenodo record:
  `https://zenodo.org/records/18061300`
- Uni-Mol+ upstream:
  `https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus`
- Chemprop v1.3.0 reference:
  `https://github.com/chemprop/chemprop/tree/v1.3.0`

## Remaining Limitations

- No UniProp dependency was installed.
- No nablaColors source was cloned into the repository.
- No checkpoint was downloaded.
- No LMDB was created or read locally.
- No UniProp model was trained, validated, or used for inference.
- Local test environment is Python 3.14, not Python 3.10.
- Python 3.10 smoke test is skipped until a `python3.10` executable is
  available.
- Checkpoint exact byte sizes need tightening after real staging.

## Recommended Next Stage

Recommended Stage 3:

1. Run the bootstrap on WSL or Nibi with Python 3.10:

```bash
bash scripts/bootstrap_uniprop.sh --mode cpu --python python3.10
```

or on Nibi/CUDA:

```bash
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10
```

2. Stage checkpoints outside Git under either:

```text
assets/uniprop/checkpoints/
```

or:

```text
$FLUORCAST_UNIPROP_CHECKPOINT_DIR
```

3. Run:

```bash
.venv-uniprop/bin/python scripts/audit_uniprop_environment.py \
  --json-output outputs/uniprop_environment_report.json
```

4. Implement adapter foundation:

```text
src/chemfluor/uniprop/__init__.py
src/chemfluor/uniprop/schemas.py
src/chemfluor/uniprop/assets.py
src/chemfluor/uniprop/lmdb_io.py
```

5. Add tests:

- schema validation
- asset manifest validation
- checkpoint hash behavior with tiny temp files
- synthetic LMDB write/read round trip
- missing dependency and missing checkpoint errors

6. Do not train yet.

## Practical Notes For The Next Agent

- Do not revert any untracked Stage 1 or Stage 2 files.
- `development md/UniProp Handoff/` existed before the handoff write and was
  empty.
- This file is intentionally placed at
  `development md/UniProp Handoff/README.md`.
- Continue to use `apply_patch` for edits.
- Use `rg`/`rg --files` for searches.
- Run new tests first, then the relevant existing suite, then the full suite if
  practical.
- Keep generated data, third-party source trees, checkpoints, LMDBs, and logs
  out of Git.
