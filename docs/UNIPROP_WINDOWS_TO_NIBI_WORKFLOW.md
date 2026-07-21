# UniProp Windows To Nibi Workflow

## Profiles

`windows-smoke` is a native-Windows integration smoke test. It uses RDKit,
LMDB, NumPy, pandas, and CPU PyTorch with a tiny FluorCast-owned model kind:
`tiny_3d_smoke_backbone`.

`nibi-real` is the real UniProp profile. It must run on Linux/Nibi with the
pinned nablaColors revision, Uni-Core, Uni-Mol+, and staged real UniProp
checkpoint files. CUDA is required only for GPU mode.

## Windows Command

```powershell
python scripts\audit_uniprop_environment.py --profile windows-smoke `
  --json-output outputs\uniprop_windows_smoke_environment_report.json

python scripts\run_uniprop_windows_smoke.py `
  --output-dir outputs\uniprop_windows_smoke `
  --seed 123 `
  --overwrite `
  --json-summary
```

Run the ordinary Windows suite by excluding real-only markers explicitly:

```powershell
python -m pytest -q -m "not real_uniprop and not cuda"
python -m pytest -q -m windows_smoke
python -m compileall -q src scripts
```

The marker exclusion is intentional. Real UniProp and CUDA tests should not be
silently skipped or treated as locally verified on Windows.

## What Windows Proves

- Fixture manifests preserve repeated chromophore rows across solvents and
  deliberate missing labels.
- RDKit generates one deterministic ETKDGv3/MMFF geometry for each unique
  chromophore.
- Repeated chromophore rows reuse the same cached geometry.
- LMDB records round-trip through the FluorCast adapter.
- Target masks prevent missing labels from becoming zero-valued labels.
- `Tiny3DSmokeBackbone` consumes atom, coordinate, graph, solvent, and mask
  tensors, then performs a real CPU PyTorch forward pass.
- The solvent encoder and multitask heads run with masked losses.
- Backward propagation produces finite gradients and an optimizer step changes
  trainable parameters.
- Checkpoints save, reload, and resume deterministically.
- The smoke prediction JSON validates against a versioned schema.
- Smoke artifacts carry `profile: "windows-smoke"`,
  `model_kind: "tiny_3d_smoke_backbone"`, `real_uniprop_used: false`, and
  `real_checkpoint_loaded: false`.
- Production loading refuses smoke bundles and tiny smoke checkpoints.

## What Remains Nibi-Only

- Uni-Core import and `unicore-train` execution.
- Uni-Mol+ import and real task/model loading.
- Real UniProp checkpoint staging, checksum validation, and `torch.load`.
- Real UniProp forward and backward passes.
- Scheduled CUDA execution when GPU mode is requested.
- Full FluorCast geometry-cache generation.
- Full model training, validation, and production bundle packaging.

## Nibi Command

```bash
module purge
module load python/3.10
module load gcc

export FLUORCAST_UNIPROP_CHECKPOINT_DIR="$SCRATCH/fluorcast_uniprop_checkpoints"
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10

.venv-uniprop/bin/python scripts/audit_uniprop_environment.py \
  --profile nibi-real \
  --real-device gpu \
  --checkpoint-dir "$FLUORCAST_UNIPROP_CHECKPOINT_DIR" \
  --json-output outputs/uniprop_environment_report_nibi.json
```

Do not start full training until `real_uniprop_cpu_ready` or
`real_uniprop_gpu_ready` is true for the intended execution mode.
