# UniProp Stage-Gate Report

Date: 2026-07-20

Branch: `feature/uniprop-3d`  
Latest commit before edits: `322e1c3 feat: add validated UniProp 3D integration scaffold`

## Gate Commands

| Command | Result |
| --- | --- |
| `python -m pytest -q` | `323 passed, 1 skipped, 4 warnings in 91.83s` |
| `python -m compileall -q src scripts` | Passed |
| `git diff --check` | Passed |

The local audit environment is Python `3.14.0`, RDKit `2026.03.2`, LMDB `2.3.0`, PyTorch `2.10.0+cpu`, no CUDA, no Uni-Core, no Uni-Mol+, no Chemprop, and no staged UniProp checkpoints. The ignored `third_party/nablacolors` checkout is present and clean at the pinned commit when read with a one-off `safe.directory` flag.

## 2026-07-21 Windows-Smoke Addendum

Added a separate native-Windows profile named `windows-smoke` and preserved the
real Linux/Nibi profile as `nibi-real`.

The Windows profile is an integration smoke test, not proof that pretrained
UniProp works. It uses RDKit, LMDB, NumPy, pandas, and CPU PyTorch with
`Tiny3DSmokeBackbone`, a small FluorCast-owned model that consumes the same
atom, coordinate, graph, solvent, and mask tensor families as the UniProp data
adapter. It writes checkpoints and predictions with
`model_kind: "tiny_3d_smoke_backbone"`, `real_uniprop_used: false`, and
`real_checkpoint_loaded: false`.

The environment audit now reports:

- `windows_smoke_ready`
- `real_uniprop_cpu_ready`
- `real_uniprop_gpu_ready`

On this Windows machine, `scripts/audit_uniprop_environment.py --profile
windows-smoke --dry-run` reports `windows_smoke_ready: true`. The same audit
with `--profile nibi-real --real-device cpu --dry-run` reports
`real_uniprop_cpu_ready: false`, as expected, because this environment lacks
Linux, Python 3.10, Uni-Core, Uni-Mol+, and real checkpoints.

The complete smoke command is:

```powershell
python scripts\run_uniprop_windows_smoke.py `
  --output-dir outputs\uniprop_windows_smoke `
  --seed 123 `
  --overwrite `
  --json-summary
```

Windows verified components:

- fixture manifests with repeated chromophores, multiple solvents, measured
  labels, and deliberate missing labels;
- one deterministic RDKit geometry per unique chromophore;
- geometry reuse across repeated chromophore rows;
- LMDB validation and adapter loading;
- target-mask collation and masked multitask loss;
- finite forward and backward PyTorch execution through the tiny 3D backbone,
  solvent encoder, and heads;
- optimizer parameter changes;
- checkpoint save/load identity and deterministic one-step resume;
- versioned smoke prediction JSON;
- production-loader refusal for smoke bundles and tiny checkpoints.

Nibi-only unverified components remain Uni-Core, Uni-Mol+, real checkpoint
loading, real UniProp forward/backward execution, CUDA scheduling, full
geometry-cache generation, and full training.

## Component Status Table

| Component | Implementation status | Real upstream dependency used | Mocked or fallback behavior | Tests currently proving it | Tests still missing | External assets required | Next validation command |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Windows smoke profile | Real local integration smoke for FluorCast data/checkpoint/JSON contracts; not a real UniProp model | RDKit, LMDB, NumPy, pandas, CPU PyTorch | Uses `Tiny3DSmokeBackbone`; no Uni-Core, Uni-Mol+, Chemprop, CUDA, or real checkpoint | `tests/test_uniprop_environment_profiles.py`, `tests/test_uniprop_windows_smoke.py` | None for local smoke; real model validation remains Nibi-only | Windows Python 3.11 or newer with local deps | `python scripts/run_uniprop_windows_smoke.py --output-dir outputs/uniprop_windows_smoke --seed 123 --overwrite --json-summary` |
| Upstream nablaColors pin/bootstrap | Real bootstrap/audit scripts; non-dry-run install not executed locally | Pinned repo/ref/commit in `third_party/nablacolors.REVISION`; local ignored clone at `39095389c0a4ecb47872ef74d00b8d13597939c8` | Audit cannot verify clone without `safe.directory` under sandbox user; bootstrap dry-run only | `tests/test_uniprop_bootstrap_environment.py`; dry-run, shell syntax, revision mismatch, manifest schema | Real Python 3.10 bootstrap on WSL/Nibi; editable install validation | Python 3.10, Git, nablaColors clone, optional CUDA stack | `bash scripts/bootstrap_uniprop.sh --mode cpu --python python3.10` |
| Uni-Core import | Not implemented in FluorCast runtime; only audited | None locally; audit checks `unicore` and `unicore-train` | Missing dependency reported as readiness failure | `tests/test_uniprop_bootstrap_environment.py` JSON schema only | Import smoke in installed Python 3.10 env; `unicore-train` CLI smoke | `.venv-uniprop`, upstream `Uni-Core/` | `.venv-uniprop/bin/python scripts/audit_uniprop_environment.py` |
| Uni-Mol+ import | Not implemented in FluorCast runtime; only audited | None locally; audit checks `unimol_plus` | Missing dependency reported as readiness failure | `tests/test_uniprop_bootstrap_environment.py` JSON schema only | Import and upstream task construction in installed Python 3.10 env | `.venv-uniprop`, upstream `unimol_plus/` | `.venv-uniprop/bin/python -c "import unimol_plus"` |
| Pretrained UniProp checkpoints | Manifest implemented; no local checkpoint loading | Zenodo filenames, URLs, MD5s in `configs/uniprop/checkpoint_manifest.json` | Missing checkpoints reported; expected byte sizes are placeholders from published MB display | Manifest schema and missing-checkpoint audit tests | Real files staged, exact byte sizes tightened, checksum and `torch.load` smoke | Four Zenodo `.pt` files outside Git | `python scripts/audit_uniprop_environment.py --checkpoint-dir <staged-dir>` |
| Chemprop solvent encoder | Not integrated; only dependency/audit placeholder and matrix label | None locally | Experiment matrix uses deterministic hashed solvent features for the `uniprop_chemprop_solvent_encoder` variant | Matrix row/count/aggregation tests only | Chemprop v1.3 model import, solvent embedding parity, fixture vector regression | Upstream Chemprop v1.3 model under nablaColors | Python 3.10 audit plus a future solvent-encoder fixture test |
| Manifests and splits | Real local implementation | RDKit for canonicalization and optional scaffolds | Optional heavy RDKit properties can be skipped; random split has no leakage constraint | `tests/test_uniprop_manifests.py`; prior real manifest build logged 66,820 rows and 33,965 molecules | Fresh full-manifest regeneration in current gate; persisted artifact hash comparison | Processed FluorCast CSVs | `python scripts/build_uniprop_manifests.py --out-dir data/processed/uniprop --compute-rdkit-scaffolds` |
| RDKit geometry cache | Real local implementation for one geometry per molecule | RDKit ETKDGv3, MMFF94/MMFF94s, UFF fallback | UFF fallback when MMFF unavailable; structured failures for embedding/optimization failures | `tests/test_uniprop_geometry_cache.py`; prior small real-manifest smoke | Full 33,965-molecule cache, shard reconciliation, Linux/Nibi run | Full molecule manifest; RDKit runtime | `sbatch slurm/uniprop/run_uniprop_geometry_cache_array.sbatch` |
| Named conformer/xTB ablations | Partial local prep; not validated as modeling improvement | RDKit multi-conformer generation; optional xTB detection only | xTB execution is not implemented; matrix uses deterministic feature proxies for variants | `tests/test_uniprop_conformer_geometry.py`; matrix registration tests | Actual xTB geometry generation and model ablations | xTB binary, completed cache, Nibi/Linux env | `python -m pytest tests/test_uniprop_conformer_geometry.py` then future xTB smoke |
| LMDB export | Real local LMDB writer/validator for upstream field contract | `lmdb`; record fields inspected from pinned nablaColors source; upstream `LMDBDataset` source-file load smoke | Full Uni-Core/Uni-Mol+ package import avoided; no geometry generation inside exporter | `tests/test_uniprop_lmdb_export.py`; upstream `LMDBDataset` direct source smoke | Full export from completed cache; real upstream `PCQDataset`/task loader smoke in Python 3.10 | Completed geometry cache, row/molecule/split manifests | `python scripts/export_uniprop_lmdb.py --split-family molecule --overwrite` |
| Target-mask upstream compatibility | Real thin wrapper | Wraps upstream PCQ-style dataset contract | Uses NumPy mask if torch unavailable; dummy dataset tests only prove wrapper behavior | `tests/test_uniprop_lmdb_export.py` dummy collater tests | End-to-end upstream task collation with masks | Uni-Core/Uni-Mol+ installed | Future Python 3.10 upstream task collater test |
| Real UniProp forward pass | Not implemented/proven | None locally | Head/backbone/production models use FluorCast-owned lightweight PyTorch modules and hashed features | Dummy smoke tests prove local modules only | Forward pass through actual Uni-Mol+/UniProp model and pretrained checkpoint | Uni-Core, Uni-Mol+, checkpoints, LMDB fixture | Future `unimol_plus` validation/inference smoke |
| Real backward pass | Not implemented/proven | None locally | Backward tests are against lightweight smoke/physics heads | Head, finetune, physics gradient tests | Backward through actual UniProp task/model on tiny LMDB | CUDA or CPU-capable UniProp env, checkpoint | Future tiny upstream training step in Python 3.10 |
| Head-only checkpoint training | Implemented as dummy/local smoke trainer, not upstream UniProp training | PyTorch only | Frozen `nn.Linear` "backbone"; hashed molecule/solvent features; synthetic LMDB fixture tests | `tests/test_uniprop_head_smoke_training.py` | Head-only training with actual UniProp backbone and upstream pretrained checkpoint | Full LMDBs, UniProp env, pretrained checkpoint, likely CUDA/Nibi | `sbatch slurm/uniprop/run_uniprop_head_smoke.sbatch` after assets are staged |
| Backbone fine-tuning | Implemented as dummy/local smoke trainer, not upstream UniProp fine-tuning | PyTorch only | Unfreezes lightweight smoke backbone; no Uni-Mol+ parameters | `tests/test_uniprop_backbone_finetune.py` | Actual UniProp backbone unfreeze and optimizer groups over real backbone | Head checkpoint, full LMDBs, UniProp env, CUDA/Nibi recommended | `sbatch slurm/uniprop/run_uniprop_backbone_finetune.sbatch` after real head checkpoint |
| Experiment matrix | Real orchestration/artifact contract; model variants are proxy estimators | scikit-learn, joblib | UniProp variants, Chemprop encoder, physics/conformer variants use deterministic feature projections, RF/MLP/Ridge proxies | `tests/test_uniprop_experiment_matrix.py` | Full Nibi matrix with real trained artifacts and real test outputs | Full manifests/artifacts; possibly CUDA depending variant | `python scripts/run_uniprop_experiment_matrix.py validate --experiment-dir <dir>` |
| Physics constraints | Real local differentiable equations/head | PyTorch | Independent local head, not attached to upstream UniProp | `tests/test_uniprop_physics_constraints.py` | Integration into real UniProp head training and stability validation | Real training checkpoint and LMDBs | `python -m pytest tests/test_uniprop_physics_constraints.py` |
| Production model-bundle prediction | Real JSON/bundle contract and local fixture inference; not a real UniProp bundle | RDKit and PyTorch | Fixture physics-head bundle; generated RDKit geometry fallback; no upstream UniProp checkpoint | `tests/test_uniprop_production_inference.py` | Package real Nibi-trained bundle; verify real checkpoint hashes/scalers/solvent assets | Real `model_weights.pt`, metadata, scalers, solvent assets, training fingerprints | `python scripts/predict_uniprop_bundle.py --bundle-dir <bundle> --input <json> --output <json>` |
| Slurm wrappers | Implemented and syntax-tested | Nibi modules (`python/3.10`, `rdkit`, GCC; geometry uses `python/3.11`) | Head script hard-codes `--gpus-per-node=h100:1`; fine-tune requests generic one GPU; matrix CPU partition | `tests/test_slurm_layout.py`, `tests/test_docs_slurm_references.py` | Actual Nibi submission and module availability validation | Nibi account, modules, staged data/assets | `sbatch --test-only slurm/uniprop/<script>.sbatch` if supported |
| Documentation | Implemented but aspirational sections exist | Prior upstream/documentation audit | Some docs describe intended upstream path, not verified local execution | `tests/test_uniprop_design_docs.py`, doc/slurm tests | Update after real Python 3.10/Nibi runs | None | `python -m pytest tests/test_uniprop_design_docs.py tests/test_docs_slurm_references.py` |

## Explicit Real vs Simulated Determinations

| Item | Determination |
| --- | --- |
| upstream nablaColors clone and revision pinning | Real pin file and ignored local clone are present; one-off safe-directory Git check confirms clone is at `39095389c0a4ecb47872ef74d00b8d13597939c8`. Bootstrap non-dry-run is unverified. |
| Uni-Core import | Not real locally; audit reports unavailable. |
| Uni-Mol+ import | Not real locally; audit reports unavailable. |
| pretrained UniProp checkpoint loading | Not real locally; checkpoint manifest exists, but files are absent and no load occurred. |
| Chemprop solvent encoder | Simulated/unimplemented; dependency is audited only, matrix variant uses deterministic hashed solvent features. |
| actual upstream LMDB loader | Partially real; tests load the pinned upstream `LMDBDataset` source file directly. Full upstream package/task loader is unverified. |
| real UniProp forward pass | Simulated/unimplemented; local forward passes are dummy smoke/physics-head models. |
| real backward pass | Simulated/unimplemented; gradients are only through dummy smoke/physics-head models. |
| head-only checkpoint training | Simulated dummy smoke training; useful for artifact/resume checks, not proof of UniProp. |
| backbone fine-tuning | Simulated dummy smoke fine-tuning; useful for transition/resume checks, not proof of UniProp. |
| production model bundle prediction | Real contract and fixture inference; real UniProp production bundle prediction is unverified. |

## Dependency Isolation

`requirements.txt` includes base FluorCast dependencies plus `lmdb`. Heavy UniProp dependencies remain isolated from the base environment: PyTorch is not declared there, and Uni-Core, Uni-Mol+, Chemprop, nablaColors source, checkpoints, LMDBs, geometry caches, and training outputs are managed by `scripts/bootstrap_uniprop.sh`, `scripts/audit_uniprop_environment.py`, and ignored paths.

## Git Hygiene

`.gitignore` covers Python caches, virtual environments, Slurm logs, broad model/output directories, checkpoint extensions (`*.pt`, `*.ckpt`), LMDBs (`*.lmdb`), `third_party/nablacolors/`, `assets/uniprop/`, `assets/nablacolors/`, `outputs/uniprop*/`, `models/uniprop*/`, and `data/processed/uniprop/`. The current ignored assets include `third_party/nablacolors/`, `data/processed/uniprop/`, `outputs/`, and `src/chemfluor/uniprop/__pycache__/`.

No UniProp checkpoint directory exists locally under `assets/uniprop/checkpoints`.

## Script Path And CUDA Audit

No credentials or tokens were found in UniProp scripts/configs. No tracked UniProp file hard-codes a Windows user path. UniProp Slurm scripts use `$HOME/scratch/FluorCast` and `$HOME/scratch/fluorcast_uniprop_env` as configurable defaults. `slurm/uniprop/run_uniprop_head_smoke.sbatch` hard-codes an H100 GPU request by default; fine-tuning requests one generic GPU; geometry cache uses CPU modules; experiment matrix uses the standard partition.

## Stage-Gate Decision

Base FluorCast is green, and the local adapter/orchestration scaffolding is internally tested. The project is not yet gate-cleared as a real UniProp implementation. It is gate-cleared only for the next validation stage: run the isolated Python 3.10 UniProp environment audit/bootstrap and verify upstream imports/checkpoints before any claims about UniProp model training or inference.

Single next stage: **Python 3.10 upstream environment validation and checkpoint staging**. Run bootstrap/audit in the isolated environment, stage the Zenodo checkpoints outside Git, verify Uni-Core/Uni-Mol+/Chemprop imports, and run a tiny upstream LMDB loader/task smoke. Do not generate the full FluorCast geometry cache or start model training in that stage.
