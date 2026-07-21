# UniProp Handoff 3

Date: 2026-07-20

Repository: `C:\Users\CL\OneDrive\Desktop\python\FluorCast`

Context: continuing after
`development md/UniProp Handoff/handoff1.md` and
`development md/UniProp Handoff/handoff2.md`.

This handoff captures all work completed in this chat:

- Prompt 8: head-only smoke training.
- Prompt 9: backbone fine-tuning.
- Prompt 10: full experiment and benchmark matrix runner.

## Starting Context

The user instructed that all previous context could be found in:

```text
development md/UniProp Handoff
```

Active IDE context during this chat:

- `docs/AGENTS.md`
- `development md/UniProp Handoff/handoff2.md`
- `development md/UniProp Handoff/handoff1.md`
- `slurm/run_duplicate_check_job.sbatch`
- `slurm/run_prediction_job.sbatch`

Important local instruction file read during this chat:

- `docs/AGENTS.md`
  - Keep implementation lean.
  - Reuse existing code.
  - Avoid speculative abstractions.
  - Run focused tests first, then project checks.

The worktree already contained many untracked Prompt 1-5 UniProp files. They
were treated as project/user work and were not reverted.

## Git Status At End Of Chat

Final observed short status:

```text
 M .gitignore
 M requirements.txt
?? configs/
?? "development md/UniProp Handoff/"
?? docs/AGENTS.md
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
?? scripts/run_uniprop_experiment_matrix.py
?? scripts/train_uniprop_backbone_finetune.py
?? scripts/train_uniprop_head_smoke.py
?? scripts/validate_uniprop_lmdb.py
?? slurm/uniprop/
?? src/chemfluor/uniprop/
?? tests/test_uniprop_backbone_finetune.py
?? tests/test_uniprop_bootstrap_environment.py
?? tests/test_uniprop_design_docs.py
?? tests/test_uniprop_experiment_matrix.py
?? tests/test_uniprop_geometry_cache.py
?? tests/test_uniprop_head_smoke_training.py
?? tests/test_uniprop_lmdb_export.py
?? tests/test_uniprop_manifests.py
?? third_party/
```

`development md/UniProp Handoff/handoff3.md` is expected to appear as part of
the untracked handoff folder after this write.

## Prompt 8 - Head-Only Smoke Training

### User Request

Implement and execute a small head-only training experiment:

- freeze UniProp backbone;
- train only solvent adapter, fusion module, and prediction heads;
- use deterministic subset with non-missing examples for all three targets;
- save resolved config, environment report, dataset/split hashes, checkpoints,
  optimizer/scheduler state, scalers, metrics history, and per-row validation
  predictions;
- support exact resume;
- detect NaNs, infinite gradients, empty target batches, and accidental
  test-set evaluation;
- add Nibi smoke Slurm script;
- add tests for finite loss, overfitting, parameter changes, exact resume,
  deterministic resumed/uninterrupted agreement, checkpoints, metric
  recomputation, and Slurm syntax.

### Files Added

- `src/chemfluor/uniprop/head_smoke_training.py`
- `scripts/train_uniprop_head_smoke.py`
- `configs/uniprop/head_smoke.example.json`
- `configs/uniprop/head_smoke.overfit_fixture.json`
- `slurm/uniprop/run_uniprop_head_smoke.sbatch`
- `tests/test_uniprop_head_smoke_training.py`

### Implementation Notes

The head-only smoke path is intentionally FluorCast-owned and lightweight. It
uses the already exported UniProp LMDB records, deterministic text-derived
features, and a small PyTorch module:

- `backbone`
- `solvent_adapter`
- `fusion`
- `heads`

Backbone parameters are frozen in the head-only stage. Only these train:

- `solvent_adapter.*`
- `fusion.*`
- `heads.*`

Main entry points:

- `HeadSmokeConfig`
- `HeadOnlySmokeModel.build`
- `train_head_smoke`
- `recompute_metrics_from_predictions`
- `scripts/train_uniprop_head_smoke.py`

Artifacts written by each run:

- `resolved_config.json`
- `environment_report.json`
- `dataset_split_hashes.json`
- `scalers.json`
- `metrics_history.json`
- `validation_predictions.csv`
- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `training_summary.json`

Checkpoint payloads include:

- schema version;
- checkpoint kind;
- update index;
- model state;
- optimizer state;
- scheduler state;
- resolved configuration;
- scaler;
- best metric;
- metrics history;
- Python, NumPy, and torch RNG state.

Resume behavior:

- Loads `last_checkpoint.pt`.
- Restores model, optimizer, scheduler, scaler, best metric, and history.
- Continues at `update_index + 1`.

Safety checks:

- refuses `validation_partition = test`;
- rejects empty target batches;
- checks finite loss;
- checks finite gradients;
- deterministic subset must include at least one available value for
  absorption, emission, and quantum yield.

### Slurm

Added:

```text
slurm/uniprop/run_uniprop_head_smoke.sbatch
```

The script:

- uses concrete Slurm defaults;
- supports submit-time resource overrides;
- activates `FLUORCAST_ACTIVATE`, `.venv-uniprop`, or a scratch venv;
- runs `scripts/train_uniprop_head_smoke.py`;
- checks that predictions and checkpoints were written.

Important correction made during Prompt 8:

- Slurm scheduler directives do not expand shell variables, so the resource
  directives were changed from variable expressions to real defaults.

### Prompt 8 Tests Run

Focused:

```text
python -m pytest tests\test_uniprop_head_smoke_training.py
10 passed in 22.12s
```

Slurm syntax guard:

```text
python -m pytest tests\test_slurm_layout.py
2 passed in 8.03s
```

Focused UniProp suite:

```text
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_uniprop_head_smoke_training.py
57 passed, 1 skipped in 26.20s
```

After Slurm correction:

```text
python -m pytest tests\test_slurm_layout.py
2 passed in 3.99s

python -m pytest tests\test_docs_slurm_references.py tests\test_uniprop_head_smoke_training.py
13 passed in 13.85s
```

Full suite after Prompt 8:

```text
python -m pytest
265 passed, 1 skipped, 4 warnings in 96.02s
```

Prompt 8 limitation:

- The Nibi GPU job was not actually submitted from this local Windows
  workspace. The Slurm script was implemented and syntax-tested.

## Prompt 9 - Backbone Fine-Tuning

### User Request

Implement the second training stage that unfreezes the UniProp backbone:

- initialize from the best head-only checkpoint;
- use lower LR for backbone than prediction layers;
- add gradient clipping, AMP on supported GPUs, gradient accumulation, early
  stopping, best-checkpoint selection, exact resume;
- optional EMA only when tested;
- record trainable parameter counts by component;
- report memory use and clear OOM recovery message;
- keep test-set metrics out of checkpoint selection;
- make hyperparameters configurable;
- add Nibi H100-compatible Slurm script without hard-coding H100 requirement;
- add tests for transition, parameter changes, LRs, AMP compatibility, resume
  state, deterministic eval, and end-to-end smoke.

### Files Added

- `src/chemfluor/uniprop/backbone_finetune.py`
- `scripts/train_uniprop_backbone_finetune.py`
- `configs/uniprop/backbone_finetune.example.json`
- `slurm/uniprop/run_uniprop_backbone_finetune.sbatch`
- `tests/test_uniprop_backbone_finetune.py`

### Implementation Notes

Main entry points:

- `BackboneFinetuneConfig`
- `build_model_from_head_checkpoint`
- `optimizer_parameter_groups`
- `ExponentialMovingAverage`
- `evaluate_deterministic`
- `train_backbone_finetune`
- `scripts/train_uniprop_backbone_finetune.py`

Transition behavior:

- Builds the same smoke model topology as Prompt 8.
- Loads a checkpoint whose schema must be
  `fluorcast_uniprop_head_smoke_v1`.
- Loads `model_state_dict`.
- Sets all backbone parameters to `requires_grad = True`.

Optimizer behavior:

- group `backbone` uses `backbone_learning_rate`;
- group `prediction_layers` uses `head_learning_rate`;
- both are configurable;
- default backbone LR is lower than prediction-layer LR.

Runtime/training features:

- gradient accumulation;
- gradient clipping via `max_grad_norm`;
- CUDA AMP through `torch.amp` when device is CUDA and available;
- early stopping through `early_stopping_patience`;
- optional EMA with tested checkpoint/resume state;
- deterministic evaluation checks by evaluating twice and comparing outputs;
- memory report for CUDA allocated/reserved/max memory;
- CUDA OOM errors are rewritten with a recovery message recommending smaller
  `micro_batch_size`, lower accumulation, AMP, or more GPU memory.

Artifacts written:

- `resolved_config.json`
- `environment_report.json`
- `dataset_split_hashes.json`
- `parameter_counts.json`
- `optimizer_groups.json`
- `metrics_history.json`
- `validation_predictions.csv`
- `best_checkpoint.pt`
- `last_checkpoint.pt`
- `training_summary.json`

Checkpoint payloads include:

- model state;
- optimizer state;
- scheduler state;
- AMP scaler state;
- EMA state when enabled;
- scaler;
- metrics history;
- best metric and best update index;
- stale update count;
- torch RNG state.

Safety:

- refuses `validation_partition = test`;
- checkpoint selection uses validation only;
- test-set metrics cannot influence best checkpoint selection.

### Slurm

Added:

```text
slurm/uniprop/run_uniprop_backbone_finetune.sbatch
```

Important scheduler detail:

- The script requests `--gpus-per-node=1`, not `h100:1`.
- It is H100-compatible, but does not require H100. GPU type can be selected
  through submit-time flags or site policy.

### Prompt 9 Tests Run

Focused initial:

```text
python -m pytest tests\test_uniprop_backbone_finetune.py
9 passed in 10.57s
```

After adding resumed/uninterrupted prediction reproducibility:

```text
python -m pytest tests\test_uniprop_backbone_finetune.py
10 passed in 11.34s

python -m pytest tests\test_slurm_layout.py
2 passed in 4.66s
```

Focused Prompt 9 + Prompt 8 + Slurm/docs:

```text
python -m pytest tests\test_uniprop_backbone_finetune.py tests\test_uniprop_head_smoke_training.py tests\test_slurm_layout.py tests\test_docs_slurm_references.py
25 passed in 24.13s
```

Focused UniProp suite:

```text
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_uniprop_backbone_finetune.py
54 passed, 1 skipped in 22.99s
```

Full suite after Prompt 9:

```text
python -m pytest
275 passed, 1 skipped, 4 warnings in 65.11s
```

Prompt 9 limitation:

- Real Nibi fine-tuning was not submitted locally. Tests prove transition,
  resume, and reproducibility on synthetic LMDB fixtures.

## Prompt 10 - Full Experiment And Benchmark Runner

### User Request

Implement a reproducible experiment matrix:

- compare current FluorCast baselines and UniProp variants under identical
  splits;
- required split families: random, molecule, scaffold, solvent held out,
  double cold start;
- required model variants:
  - current Morgan/RDKit baseline;
  - current strongest ensemble/tree baseline;
  - UniProp with existing solvent descriptors;
  - UniProp with Chemprop solvent encoder;
  - UniProp frozen-backbone;
  - UniProp fine-tuned;
- use at least three controlled seeds;
- use exact same target rows when comparisons require it;
- produce per-run metrics JSON, per-row predictions, aggregate summary CSV,
  mean/std/bootstrap CIs, target coverage, train/valid/test counts, config and
  checkpoint hashes;
- report MAE, RMSE, R2, quantum-yield metrics, emission-region metrics, and
  similarity-bin metrics;
- keep test evaluation as a separate explicit command;
- add validation command;
- add tests for leakage audit, metric recomputation, partial-run exclusion,
  file-order-invariant aggregation, duplicate run rejection, no accidental
  test-prediction overwrite, and matrix row counts.

### Files Added

- `src/chemfluor/uniprop/experiment_matrix.py`
- `scripts/run_uniprop_experiment_matrix.py`
- `configs/uniprop/experiment_matrix.example.json`
- `slurm/uniprop/run_uniprop_experiment_matrix.sbatch`
- `tests/test_uniprop_experiment_matrix.py`

### Implementation Notes

Main entry points:

- `MatrixConfig`
- `run_matrix`
- `evaluate_test_run`
- `validate_run_dir`
- `validate_experiment_dir`
- `aggregate_experiment`
- `scripts/run_uniprop_experiment_matrix.py`

Default matrix:

- split families:
  - `random`
  - `molecule`
  - `scaffold`
  - `solvent`
  - `double_cold_start`
- seeds:
  - `11`
  - `17`
  - `23`
- targets:
  - `absorption_nm`
  - `emission_nm`
  - `quantum_yield`
- model variants:
  - `morgan_rdkit_baseline`
  - `tree_ensemble_baseline`
  - `uniprop_solvent_descriptors`
  - `uniprop_chemprop_solvent_encoder`
  - `uniprop_frozen_backbone`
  - `uniprop_finetuned`

The local implementation uses deterministic lightweight feature projections and
scikit-learn estimators to make the matrix orchestration testable without the
full remote UniProp stack:

- Morgan/RDKit baseline: deterministic molecule-derived feature vector with
  ridge regression.
- Tree ensemble baseline: molecule plus solvent features with ExtraTrees.
- UniProp solvent descriptor variant: molecule projection plus solvent
  descriptor-like features with RandomForest.
- UniProp Chemprop solvent encoder variant: larger deterministic solvent
  encoder features with RandomForest.
- UniProp frozen backbone: tanh-transformed molecule features plus solvent
  features with RandomForest.
- UniProp fine-tuned: tanh-transformed molecule features plus solvent and
  interaction features with MLP.

This is an orchestration/benchmark artifact contract. It does not replace the
Prompt 8 or Prompt 9 training entry points.

### Commands

Run train/valid matrix only:

```bash
python scripts/run_uniprop_experiment_matrix.py run \
  --config configs/uniprop/experiment_matrix.example.json \
  --overwrite
```

Evaluate test explicitly:

```bash
python scripts/run_uniprop_experiment_matrix.py evaluate-test \
  --config configs/uniprop/experiment_matrix.example.json
```

Validate one run:

```bash
python scripts/run_uniprop_experiment_matrix.py validate \
  --run-dir outputs/uniprop_experiment_matrix/runs/<run_id>
```

Validate an experiment directory:

```bash
python scripts/run_uniprop_experiment_matrix.py validate \
  --experiment-dir outputs/uniprop_experiment_matrix
```

Summarize:

```bash
python scripts/run_uniprop_experiment_matrix.py summarize \
  --experiment-dir outputs/uniprop_experiment_matrix
```

### Artifact Layout

Experiment root contains:

- `resolved_config.json`
- `split_leakage_audit.json`
- `matrix_status.json`
- `aggregate_summary.csv`
- `per_run_summary.csv`
- `excluded_runs.json`
- `runs/<run_id>/...`

Each run directory contains:

- `checkpoint.joblib`
- `metrics.json`
- `valid_predictions.csv`
- `test_predictions.csv` after explicit `evaluate-test`

Per-run `metrics.json` includes:

- schema;
- run ID;
- split family;
- model variant;
- target;
- seed;
- train/valid/test counts;
- target coverage;
- validation metrics;
- test metrics after explicit evaluation;
- paths to prediction files;
- config hash;
- checkpoint SHA-256;
- row manifest SHA-256;
- molecule manifest SHA-256;
- split assignment SHA-256.

Metrics include:

- MAE;
- RMSE;
- R2;
- quantum-yield bright F1 and accuracy;
- by-similarity-bin metrics;
- by-emission-region metrics for emission target.

Aggregation behavior:

- validates each run before inclusion;
- excludes partial/failed runs and writes `excluded_runs.json`;
- rejects duplicate run IDs;
- writes deterministic CSVs invariant to run-directory file order;
- reports mean, standard deviation, and bootstrap confidence intervals for MAE.

Test safety:

- `run` does not evaluate test.
- `evaluate-test` writes test predictions separately.
- Existing `test_predictions.csv` cannot be overwritten unless explicitly
  allowed with `--overwrite-test`.

### Slurm

Added:

```text
slurm/uniprop/run_uniprop_experiment_matrix.sbatch
```

It runs:

1. `run`
2. `evaluate-test`
3. `summarize`

Configurable environment variables:

- `FLUORCAST_REPO`
- `FLUORCAST_ACTIVATE`
- `FLUORCAST_UNIPROP_MATRIX_CONFIG`
- `FLUORCAST_UNIPROP_MATRIX_OUT`

The full matrix can be submitted without editing source code by changing config
or environment variables.

### Prompt 10 Tests Run

Initial matrix suite:

```text
python -m pytest tests\test_uniprop_experiment_matrix.py
10 passed in 22.58s
```

After adding experiment-directory validation:

```text
python -m pytest tests\test_uniprop_experiment_matrix.py
10 passed in 25.92s

python -m pytest tests\test_slurm_layout.py
2 passed in 4.18s
```

Focused UniProp suite:

```text
python -m pytest tests\test_uniprop_experiment_matrix.py tests\test_uniprop_backbone_finetune.py tests\test_uniprop_head_smoke_training.py tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py
79 passed, 1 skipped in 51.80s
```

After adding identical-row assertion and Stage 8 docs:

```text
python -m pytest tests\test_uniprop_experiment_matrix.py
11 passed in 23.86s

python -m pytest tests\test_docs_slurm_references.py tests\test_slurm_layout.py
5 passed in 3.52s
```

Final full suite after Prompt 10:

```text
python -m pytest
286 passed, 1 skipped, 4 warnings in 126.54s
```

The four warnings are the existing expected invalid scaffold/SMILES warning-path
tests.

Prompt 10 limitation:

- The full real-data matrix was not submitted locally.
- The small-scale matrix is fully tested on synthetic manifests.
- Full Nibi execution requires staged full manifests and artifacts.

## Implementation Log Updates

Updated:

- `docs/UNIPROP_IMPLEMENTATION_LOG.md`

Added sections:

- `2026-07-20 - Stage 6 Head-Only Smoke Training`
- `2026-07-20 - Stage 7 Backbone Fine-Tuning Smoke`
- `2026-07-20 - Stage 8 Experiment Matrix And Benchmark Runner`

## Commands Run In This Chat

Repository/context:

```text
Get-ChildItem -Force
Get-ChildItem -Recurse -File 'development md/UniProp Handoff' | Select-Object FullName
rg --files
git status --short
Get-Content -Path 'development md/UniProp Handoff/handoff2.md'
Get-Content -Path 'development md/UniProp Handoff/handoff1.md'
rg -n "UniProp|uniprop|adapter|fusion|head|checkpoint|scheduler|torch|Dataset|DataLoader" src scripts tests configs docs
Get-ChildItem -Recurse -File src\chemfluor\uniprop | Select-Object FullName
Get-Content -Path docs\AGENTS.md
```

Prompt 8 implementation inspection:

```text
Get-Content -Path src\chemfluor\uniprop\lmdb_export.py
Get-Content -Path src\chemfluor\uniprop\upstream_compat.py
Get-Content -Path scripts\export_uniprop_lmdb.py
Get-Content -Path slurm\util\test_nibi_gpu.sbatch
python -c "import importlib.util; print(importlib.util.find_spec('torch'))"
Get-Content -Path tests\test_uniprop_lmdb_export.py
Get-Content -Path tests\test_slurm_layout.py
Get-Content -Path docs\UNIPROP_IMPLEMENTATION_LOG.md
```

Prompt 8 tests:

```text
python -m pytest tests\test_uniprop_head_smoke_training.py
python -m pytest tests\test_slurm_layout.py
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_uniprop_head_smoke_training.py
python -m pytest tests\test_docs_slurm_references.py tests\test_uniprop_head_smoke_training.py
python -m pytest
```

Prompt 9 tests:

```text
python -m pytest tests\test_uniprop_backbone_finetune.py
python -m pytest tests\test_slurm_layout.py
python -m pytest tests\test_uniprop_backbone_finetune.py tests\test_uniprop_head_smoke_training.py tests\test_slurm_layout.py tests\test_docs_slurm_references.py
python -m pytest tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_uniprop_backbone_finetune.py
python -m pytest
```

Prompt 10 inspection:

```text
Get-Content -Path scripts\run_hybrid_three_way_experiment.py
Get-Content -Path scripts\run_combined_model_experiments.py
Get-Content -Path src\chemfluor\uniprop\manifests.py
Get-Content -Path src\evaluate.py
rg -n "bootstrap|confidence|region|similarity|MAE|RMSE|R2|r2|quantum" scripts src tests | Select-Object -First 200
```

Prompt 10 tests:

```text
python -m pytest tests\test_uniprop_experiment_matrix.py
python -m pytest tests\test_slurm_layout.py
python -m pytest tests\test_uniprop_experiment_matrix.py tests\test_uniprop_backbone_finetune.py tests\test_uniprop_head_smoke_training.py tests\test_uniprop_lmdb_export.py tests\test_uniprop_geometry_cache.py tests\test_uniprop_manifests.py tests\test_uniprop_bootstrap_environment.py tests\test_uniprop_design_docs.py tests\test_docs_slurm_references.py tests\test_slurm_layout.py
python -m pytest tests\test_docs_slurm_references.py tests\test_slurm_layout.py
python -m pytest
```

Final handoff checks:

```text
Get-ChildItem -Force 'development md/UniProp Handoff'
git status --short
rg -n "Stage 6|Stage 7|Stage 8|Head-Only|Backbone|Experiment Matrix" docs\UNIPROP_IMPLEMENTATION_LOG.md
```

## Final Known Test State

Final full-suite result:

```text
python -m pytest
286 passed, 1 skipped, 4 warnings in 126.54s
```

Skipped test:

- Python 3.10 UniProp bootstrap smoke because no `python3.10` executable is
  available in the current Windows environment.

Warnings:

- Existing invalid scaffold/SMILES warning-path tests from hybrid and paired
  workflows.

## Generated/Ignored Local Artifacts

As in prior handoffs, these are generated/ignored and should not be committed:

- `third_party/nablacolors/`
- `data/processed/uniprop/`
- `outputs/`
- `models/`
- `.venv-uniprop/`
- checkpoint files such as `*.pt` and `*.ckpt`
- LMDB files such as `*.lmdb`

The tests in this chat wrote only temporary pytest fixture artifacts under the
system pytest temp directories.

## Recommended Next Steps

1. Run full molecule geometry cache generation on Nibi if not already done:

```bash
sbatch slurm/uniprop/run_uniprop_geometry_cache_array.sbatch
```

2. Export full LMDBs from the completed cache:

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

3. Run head-only training on Nibi:

```bash
sbatch slurm/uniprop/run_uniprop_head_smoke.sbatch
```

4. Run backbone fine-tuning after the head-only best checkpoint exists:

```bash
sbatch slurm/uniprop/run_uniprop_backbone_finetune.sbatch
```

5. Run the experiment matrix:

```bash
sbatch slurm/uniprop/run_uniprop_experiment_matrix.sbatch
```

6. Validate and summarize any completed experiment directory:

```bash
python scripts/run_uniprop_experiment_matrix.py validate \
  --experiment-dir outputs/uniprop_experiment_matrix

python scripts/run_uniprop_experiment_matrix.py summarize \
  --experiment-dir outputs/uniprop_experiment_matrix
```

## Practical Notes For The Next Agent

- Do not revert the existing untracked UniProp files from prior prompts.
- `docs/AGENTS.md` is untracked but was read and followed in this chat.
- The Prompt 8-10 implementations are test-proven locally on synthetic
  fixtures, not on the full Nibi dataset.
- Slurm scripts were syntax-tested locally but not submitted to Nibi.
- Test evaluation in the experiment matrix is intentionally separate from
  training/validation.
- The experiment matrix currently provides a reproducible orchestration and
  artifact contract. It uses lightweight local model proxies for UniProp
  variants so the complete matrix can be tested without the full upstream GPU
  stack.
