# UniProp Dependency Bootstrap

This bootstrap is intentionally separate from the default FluorCast environment.
Do not install UniProp, Uni-Core, Uni-Mol+, PyTorch, or checkpoints into the
existing Python 3.14/FluorCast environment.

## Tracked Inputs

- Pinned upstream revision: `third_party/nablacolors.REVISION`
- Checkpoint manifest: `configs/uniprop/checkpoint_manifest.json`
- Bootstrap script: `scripts/bootstrap_uniprop.sh`
- Environment audit: `scripts/audit_uniprop_environment.py`

Generated or downloaded files are ignored by Git:

- `.venv-uniprop/`
- `third_party/nablacolors/`
- `assets/uniprop/`
- `*.pt`
- `*.lmdb`

## Local WSL

Use Python 3.10 inside WSL. First inspect the planned actions:

```bash
bash scripts/bootstrap_uniprop.sh --mode cpu --dry-run
```

Then bootstrap the isolated environment:

```bash
bash scripts/bootstrap_uniprop.sh --mode cpu --python python3.10
```

Audit readiness:

```bash
.venv-uniprop/bin/python scripts/audit_uniprop_environment.py \
  --json-output outputs/uniprop_environment_report.json
```

The report has three top-level readiness booleans:

- `preprocessing_ready`
- `cpu_smoke_ready`
- `gpu_training_ready`

CPU mode is expected to report `gpu_training_ready: false`.

## Nibi / CUDA

Create the environment from a compute job or interactive allocation, not by
training on a login node. Keep all paths configurable:

```bash
module purge
module load python/3.10
module load gcc

export FLUORCAST_UNIPROP_CHECKPOINT_DIR="$SCRATCH/fluorcast_uniprop_checkpoints"
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10
```

After checkpoint files are staged outside Git, audit:

```bash
.venv-uniprop/bin/python scripts/audit_uniprop_environment.py \
  --checkpoint-dir "$FLUORCAST_UNIPROP_CHECKPOINT_DIR" \
  --json-output outputs/uniprop_environment_report_nibi.json
```

The audit verifies Python, PyTorch, CUDA availability, CUDA runtime, GPU name,
RDKit, LMDB, Uni-Core, Uni-Mol+, Chemprop, upstream Git revision, checkpoint
presence, checkpoint size, and checkpoint hashes.

## Rerun Behavior

The bootstrap is safe to rerun:

- If `third_party/nablacolors/` already exists, its Git revision must exactly
  match `third_party/nablacolors.REVISION`.
- A revision mismatch stops the script before installs.
- The virtual environment is reused only after verifying it is Python 3.10 and
  isolated from global site installs.
- Editable installs are rerun inside `.venv-uniprop/`.

## Checkpoints

Do not commit checkpoints. Download or stage the files listed in
`configs/uniprop/checkpoint_manifest.json` under
`assets/uniprop/checkpoints/` or the directory named by
`FLUORCAST_UNIPROP_CHECKPOINT_DIR`.
