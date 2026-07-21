# UniProp Dependency Bootstrap

FluorCast now uses two explicit UniProp execution profiles.

- `windows-smoke` is the native-Windows integration profile. It uses Python
  3.11 or newer, RDKit, LMDB, NumPy, pandas, and CPU PyTorch. It deliberately
  avoids Uni-Core, Uni-Mol+, Chemprop, CUDA, and real UniProp checkpoints.
- `nibi-real` is the Linux/Nibi profile for the real UniProp dependency stack:
  Python 3.10, PyTorch, LMDB, Uni-Core, Uni-Mol+, the pinned nablaColors
  revision, and staged real checkpoint files. CUDA is required only when GPU
  mode is requested.

Do not install Uni-Core, Uni-Mol+, Chemprop, CUDA tooling, or real checkpoints
into the native-Windows smoke environment.

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
  --profile nibi-real \
  --real-device cpu \
  --json-output outputs/uniprop_environment_report.json
```

The report includes profile-specific readiness booleans:

- `windows_smoke_ready`
- `real_uniprop_cpu_ready`
- `real_uniprop_gpu_ready`

The older `preprocessing_ready`, `cpu_smoke_ready`, and `gpu_training_ready`
fields remain for compatibility with earlier scripts.

## Native Windows Smoke

Run the native-Windows audit from the local Windows Python environment:

```powershell
python scripts\audit_uniprop_environment.py --profile windows-smoke `
  --json-output outputs\uniprop_windows_smoke_environment_report.json
```

Then run the full integration smoke:

```powershell
python scripts\run_uniprop_windows_smoke.py `
  --output-dir outputs\uniprop_windows_smoke `
  --seed 123 `
  --overwrite `
  --json-summary
```

This profile proves the local FluorCast data path only. Every checkpoint and
prediction output carries `profile: "windows-smoke"`,
`model_kind: "tiny_3d_smoke_backbone"`, `real_uniprop_used: false`, and
`real_checkpoint_loaded: false`.

The Windows smoke run validates:

- fixture row and molecule manifest creation;
- one deterministic RDKit ETKDGv3/MMFF geometry per unique chromophore;
- geometry reuse for repeated chromophore rows in different solvents;
- LMDB write/read/validation;
- the FluorCast LMDB dataset adapter and missing-label target masks;
- a real CPU PyTorch forward/backward/optimizer step through
  `Tiny3DSmokeBackbone`, the solvent encoder, and multitask heads;
- checkpoint save/load identity and one-step resume;
- a production-style smoke JSON output schema.

It does not validate real UniProp inference, Uni-Core, Uni-Mol+, Chemprop,
real checkpoint loading, CUDA execution, the full geometry cache, or full model
training.

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
  --profile nibi-real \
  --real-device gpu \
  --checkpoint-dir "$FLUORCAST_UNIPROP_CHECKPOINT_DIR" \
  --json-output outputs/uniprop_environment_report_nibi.json
```

The audit verifies Python, PyTorch, optional CUDA availability, CUDA runtime,
GPU name, RDKit, LMDB, Uni-Core, Uni-Mol+, upstream Git revision, checkpoint
presence, checkpoint size, and checkpoint hashes. Chemprop is still reported
but does not gate the real UniProp readiness booleans until the solvent encoder
is wired to a real Chemprop asset.

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
