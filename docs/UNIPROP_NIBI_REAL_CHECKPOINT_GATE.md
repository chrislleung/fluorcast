# Nibi Real UniProp Checkpoint Gate

This gate proves compatibility only. It does not report scientific model
performance, start full training, or build the full FluorCast geometry cache.

## Inputs

- Branch: `feature/uniprop-3d`
- Starting commit for this stage: `e4c49bc`
- Recovery tag: `uniprop-windows-gate-passed`
- Upstream nablaColors repo: `https://github.com/AI4DD/nablaColors.git`
- Upstream commit: `39095389c0a4ecb47872ef74d00b8d13597939c8`
- Expected Nibi Python: `3.10`
- Verified local Windows audit environment: `C:\Users\CL\.venvs\fluorcast-uniprop-win`
- Uni-Core: pinned inside `third_party/nablacolors/Uni-Core`
- Uni-Mol+: pinned inside `third_party/nablacolors/unimol_plus`
- First checkpoint: `uniprop_rdkit_to_dft_implicit.pt`
- First checkpoint MD5: `c87305171142e1c0898a0e2b67a7236a`
- Feature schema: `configs/uniprop/feature_schema.json`
- Feature schema kind: categorical RDKit atom/bond feature channels, not a token dictionary
- Feature schema SHA-256:
  `93e2a5aaf19617b7420a0020cea3c4d5a8550680fe4d2fd410b16d17081577f8`
- The checkpoint manifest is `configs/uniprop/checkpoint_manifest.json`; the
  published `459500000` byte sizes are approximate until real staged byte
  counts are audited. MD5 remains strict.

## Nibi Commands

```bash
ssh <nibi-login-host>
cd "$HOME/scratch/FluorCast"
git fetch --all --tags
git checkout feature/uniprop-3d
git rev-parse --short HEAD
```

The starting point for this stage should be `e4c49bc`, or a later commit that
contains this gate.

```bash
module purge
module load python/3.10
module load gcc
module load cuda
```

Create the isolated FluorCast-owned environment. This bootstrap creates only
`.venv-uniprop`, installs PyTorch from the pip configuration exposed on Nibi,
installs Alliance-compatible NumPy from the same configuration,
installs pinned Uni-Core directly from `third_party/nablacolors/Uni-Core`, and
installs pinned Uni-Mol+ from `third_party/nablacolors/unimol_plus`. It does not
install Conda, create `unimol_env`, download checkpoints, or run training.

```bash
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10 --clean
source .venv-uniprop/bin/activate
```

The explicit `--mode cuda` flag is required for the Nibi GPU bootstrap. Running
`bash scripts/bootstrap_uniprop.sh --clean` alone uses the script's CPU default
and is only suitable for CPU-mode environment checks.

The default NumPy requirement is the public version pin `numpy==2.2.2`. Alliance
may satisfy that with a local distributor wheel such as
`2.2.2+computecanada`; the bootstrap validates the public/base version while
allowing that local suffix. The requirement is intentionally not written with
`+computecanada`, so standard pip version matching can select the Alliance wheel
without making the script unusable outside that wheelhouse.

Uni-Core's `setup.py` imports `torch` while pip is preparing the build, before
the package itself is installed. PyTorch is already installed in `.venv-uniprop`,
but pip's temporary PEP 517 build-isolation environment cannot see it, which
causes `ModuleNotFoundError: No module named 'torch'` during metadata/build
requirements discovery. The direct Uni-Core install therefore runs from the
pinned checkout with pip build isolation disabled:

```bash
.venv-uniprop/bin/python -m pip install --no-build-isolation third_party/nablacolors/Uni-Core
```

The bootstrap first installs build prerequisites into `.venv-uniprop`, including
`setuptools`, `wheel`, `packaging`, Alliance-compatible `numpy==2.2.2`, and the
requested PyTorch wheel. This removes the PyTorch NumPy initialization warning
and ensures Uni-Core's build process imports the same environment-installed
PyTorch package that the final runtime gate will use.

Expected post-bootstrap import diagnostic:

```bash
python - <<'PY'
import hashlib
import importlib
import subprocess
import sys
from pathlib import Path

import torch
import unicore
import unimol_plus
from unimol_plus.models.uniprop import UniPropModel

upstream_dir = Path("third_party/nablacolors")
model_module = importlib.import_module(UniPropModel.__module__)
schema_hash = hashlib.sha256(Path("configs/uniprop/feature_schema.json").read_bytes()).hexdigest()
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Environment path: {sys.prefix}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torch compiled CUDA version: {getattr(torch.version, 'cuda', None)}")
print(f"CUDA available at runtime: {torch.cuda.is_available()}")
print(f"Uni-Core import path: {unicore.__file__}")
print(f"Uni-Mol+ import path: {unimol_plus.__file__}")
print(f"Real UniProp model class: {UniPropModel.__module__}.{UniPropModel.__name__}")
print(f"Real UniProp model source path: {model_module.__file__}")
print(f"Pinned upstream Git commit: {subprocess.check_output(['git', '-C', str(upstream_dir), 'rev-parse', 'HEAD'], text=True).strip()}")
print(f"Feature-schema SHA-256: {schema_hash}")
PY
```

CUDA may report unavailable on a login node. The tiny GPU gate performs the
actual CUDA runtime check inside the Slurm allocation. In CUDA bootstrap mode,
however, `torch.version.cuda` must not be `None`; a `None` compiled CUDA version
means pip selected a CPU-only PyTorch wheel and the bootstrap must fail.

The loaded CUDA toolkit module and PyTorch's compiled CUDA runtime are separate
facts. For the default bootstrap, optional Uni-Core fused CUDA extensions are not
compiled, so a loaded toolkit such as `cuda/12.6` may differ from a PyTorch wheel
compiled for CUDA `12.2` without failing the bootstrap. When
`--enable-cuda-ext` is supplied, the bootstrap checks `nvcc --version` against
`torch.version.cuda` and fails before compilation if they differ; load a CUDA
module matching the PyTorch build before compiling those optional extensions.
The default bootstrap still requires all UniProp imports used by this gate to
succeed.

Stage one checkpoint only:

```bash
export FLUORCAST_UNIPROP_CHECKPOINT_DIR="$SCRATCH/fluorcast_uniprop_checkpoints"
mkdir -p "$FLUORCAST_UNIPROP_CHECKPOINT_DIR"
cd "$FLUORCAST_UNIPROP_CHECKPOINT_DIR"
wget https://zenodo.org/records/18061300/files/uniprop_rdkit_to_dft_implicit.pt
cd "$HOME/scratch/FluorCast"
```

Verify hashes through the audit:

```bash
python scripts/run_uniprop_real_checkpoint_gate.py \
  --audit-only \
  --device cpu \
  --checkpoint-dir "$FLUORCAST_UNIPROP_CHECKPOINT_DIR" \
  --checkpoint-id uniprop_rdkit_to_dft_implicit.pt \
  --feature-schema configs/uniprop/feature_schema.json \
  --output-dir outputs/uniprop_real_checkpoint_gate_cpu_audit
```

Submit the tiny GPU gate:

```bash
sbatch slurm/uniprop/run_uniprop_real_checkpoint_gate.sbatch
```

Read the summary after Slurm completes:

```bash
cat outputs/uniprop_real_checkpoint_gate/<job-id>/summary.json
```

Success requires `real_uniprop_used=true`, `real_checkpoint_loaded=true`,
`finite_forward_outputs=true`, `finite_loss=true`,
`nonzero_gradient_count > 0`, at least one `changed_parameter_names` entry,
`reload_agreement.passed=true`, and `all_stages_passed=true`.

## Failure Categories

- `missing_dependency`: activate the `.venv-uniprop` environment and rerun
  `scripts/bootstrap_uniprop.sh`.
- `wrong_upstream_commit`: restore `third_party/nablacolors` to
  `39095389c0a4ecb47872ef74d00b8d13597939c8`.
- `missing_checkpoint`: stage only the requested checkpoint in
  `$FLUORCAST_UNIPROP_CHECKPOINT_DIR`.
- `checkpoint_hash_mismatch`: remove and restage the checkpoint; do not use it.
- `missing_feature_schema`: restore `configs/uniprop/feature_schema.json`.
- `feature_schema_hash_mismatch`: confirm the expected hash argument or restore
  the tracked schema file.
- `upstream_source_hash_mismatch`: restore the pinned nablaColors checkout.
- `feature_schema_fixture`: fixture schemas are not allowed in real mode.
- `unsupported_feature_schema`: generated categorical node or edge feature
  indices are outside the schema bounds.
- `preprocessing_incompatibility`: inspect the selected molecule geometry and
  upstream `pcq` preprocessing imports.
- `checkpoint_key_incompatibility`: the checkpoint is not compatible with the
  real `uniprop_small` backbone.
- `cuda_unavailable`: request one GPU or run the CPU audit only.
- `out_of_memory`: reduce the gate to CPU audit or request more GPU memory.
- `nonfinite_forward_output`, `nonfinite_loss`, `missing_gradients`,
  `optimizer_no_op`, `reload_mismatch`: keep the JSON summary and Slurm logs;
  these indicate real model/runtime incompatibility that should be fixed before
  cache generation or training.

## Feature Channels

Atom channels, in upstream order:
`atomic_number`, `chirality`, `total_degree`, `formal_charge`, `total_num_h`,
`radical_electrons`, `hybridization`, `is_aromatic`, `is_in_ring`.

Edge channels, in upstream order:
`bond_type`, `bond_stereo`, `is_conjugated`.

The runtime gate validates generated `node_attr` and `edge_attr` channel
indices before upstream preprocessing and model execution.
