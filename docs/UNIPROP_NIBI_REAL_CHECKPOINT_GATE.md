# Nibi Real UniProp Checkpoint Gate

This gate proves compatibility only. It does not report scientific model
performance, start full training, or build the full FluorCast geometry cache.

## Inputs

- Branch: `feature/uniprop-3d`
- Starting commit for this stage: `bcdb1fc`
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

The starting point for this stage should be `bcdb1fc`, or a later commit that
contains this gate.

```bash
module purge
module load python/3.10
module load gcc
module load cuda
```

Create the isolated FluorCast-owned environment. This bootstrap creates only
`.venv-uniprop`, installs PyTorch from the pip configuration exposed on Nibi,
installs Alliance-compatible NumPy from the same configuration, installs
Uni-Core runtime dependencies from
`configs/uniprop/unicore_runtime_requirements.txt`, installs pinned Uni-Core
directly from `third_party/nablacolors/Uni-Core`, and installs pinned Uni-Mol+
from `third_party/nablacolors/unimol_plus`. It does not install Conda, create
`unimol_env`, download checkpoints, run training, or compile arbitrary Rust
source packages on the login node.

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

The exact CUDA bootstrap command for this Nibi repair is:

```bash
module purge
module load python/3.10
module load gcc
module load cuda
bash scripts/bootstrap_uniprop.sh --mode cuda --python python3.10 --clean
```

Uni-Core's `setup.py` imports `torch` while pip is preparing the build, before
the package itself is installed. PyTorch is already installed in `.venv-uniprop`,
but pip's temporary PEP 517 build-isolation environment cannot see it, which
causes `ModuleNotFoundError: No module named 'torch'` during metadata/build
requirements discovery. The direct Uni-Core install therefore runs from the
pinned checkout with pip build isolation disabled.

Uni-Core has a second Python 3.10-specific build compatibility issue on Nibi:
the pinned `third_party/nablacolors/Uni-Core/setup.py` imports in this order:

```python
import torch
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import find_packages, setup
```

PyTorch can leave Python 3.10's standard-library `distutils` loaded before
modern `setuptools` starts its default vendored-distutils override. In that
state, `_distutils_hack.ensure_local_distutils()` can assert because
`distutils.core.__file__` resolves under
`.../python/3.10.13/lib/python3.10/distutils/core.py` instead of
`.../site-packages/setuptools/_distutils/core.py`.

The bootstrap does not patch the pinned upstream Uni-Core source or reorder its
imports, and it does not downgrade `setuptools`. Because Python 3.10 still has
stdlib `distutils` available, the Uni-Core compatibility probe and Uni-Core pip
build subprocess are run with the official scoped setting
`SETUPTOOLS_USE_DISTUTILS=stdlib`. This fix is distinct from disabling pip build
isolation: `--no-build-isolation` makes the already-installed PyTorch visible to
the local Uni-Core build, while `SETUPTOOLS_USE_DISTUTILS=stdlib` prevents the
setuptools vendored-distutils collision after PyTorch has been imported. Do not
substitute Python 3.12 for this gate; Python 3.12 removed stdlib `distutils`, so
this compatibility mode is not available there.

That exception is intentionally scoped only to the local Uni-Core source
package. Uni-Core declares normal runtime dependencies including `lmdb`, `tqdm`,
`ml_collections`, `scipy`, `tensorboardX`, `tokenizers`, and `wandb`. Those
third-party packages are resolved and installed in a separate phase before
Uni-Core, using pip's normal build-isolation behavior and binary-only selection:

```bash
.venv-uniprop/bin/python -m pip install \
  --only-binary=:all: \
  -r configs/uniprop/unicore_runtime_requirements.txt
```

Before running that install, the bootstrap runs a structured pip resolver
preflight:

```bash
.venv-uniprop/bin/python -m pip install \
  --dry-run \
  --report <temporary-runtime-report.json> \
  --only-binary=:all: \
  -r configs/uniprop/unicore_runtime_requirements.txt
```

The bootstrap parses the JSON report, not ordinary pip console text. It fails
before installation if any selected artifact is not a wheel, if `wandb` is not
exactly `0.17.9`, if a selected package or normal dependency introduces
`pydantic`, `pydantic-core`, or `maturin`, or if dependency resolution would
replace the already validated NumPy or PyTorch installations. Every selected
package is printed with its report name and version, original artifact URL,
decoded artifact filename, parsed wheel name and version, Alliance-wheelhouse
status, and whether the wheel version has a local label.

Pip installation reports store selected artifact locations under
`download_info.url`. URL path characters may be percent-encoded there. Alliance
local-version wheel labels such as `+computecanada` can therefore appear in the
raw URL as `%2Bcomputecanada`, for example
`requests-2.34.2%2Bcomputecanada-py3-none-any.whl`. The bootstrap splits the
artifact URL structurally, percent-decodes only the URL path with
`urllib.parse.unquote`, extracts the final path component, and then validates
the decoded filename with `packaging.utils.parse_wheel_filename()`. That parser
remains the authoritative wheel check; the bootstrap does not rely on raw
`.whl` suffix checks. Parsed wheel names are compared with pip report metadata
using `packaging.utils.canonicalize_name`, and parsed wheel versions are
compared with report metadata using `packaging.version.Version`, so valid local
versions such as `2.34.2+computecanada` are accepted.

Valid Alliance wheels and PyPI wheels are both accepted. The runtime dependency
phase still requires every selected artifact to be a valid wheel, and actual
source distributions such as `.tar.gz`, `.zip`, and `.tar.bz2` remain
prohibited.

LMDB is the one runtime dependency that must be more than merely wheel-shaped.
The verified Nibi survey showed that pip previously selected
`lmdb-1.7.5-py3-none-any.whl`; that wheel installed, but it did not contain the
native CPython module and `import lmdb.cpython` failed. LMDB then fell back to
CFFI and attempted to compile against host headers, ending with
`fatal error: lmdb.h: No such file or directory` and `cffi.VerificationError`.
That host-header compilation path is prohibited for this bootstrap.

The Nibi-available `1.7.1`, `1.7.2`, `1.7.3`, and `1.7.5` LMDB wheels are all
`py3-none-any` universal wheels and are rejected. Nibi also cannot resolve
`lmdb==2.3.0`, so this environment must not request it. The usable surveyed
candidate is:

```text
lmdb-1.4.1+computecanada-cp310-cp310-linux_x86_64.whl
```

It contains:

```text
lmdb/cpython.cpython-310-x86_64-linux-gnu.so
```

`configs/uniprop/unicore_runtime_requirements.txt` therefore pins the portable
public requirement `lmdb==1.4.1`. On Alliance, pip may satisfy that with the
local-version distribution `1.4.1+computecanada`; outside Alliance, the policy
does not require a local suffix. The resolver preflight validates that the
selected LMDB artifact has public/base version `1.4.1`, canonical package name
`lmdb`, valid wheel metadata, tags intersecting `packaging.tags.sys_tags()`, a
CPython 3.10 implementation tag, a CPython 3.10 ABI tag, and a Linux platform
tag. Source archives, PyPy wheels, universal wheels, and `py3-none-any` LMDB
wheels are rejected before runtime dependency installation.

On Nibi, the expected LMDB resolver diagnostic is equivalent to:

```text
name=lmdb
version=1.4.1+computecanada
filename=lmdb-1.4.1+computecanada-cp310-cp310-linux_x86_64.whl
native_candidate=True
```

After runtime dependencies install, the bootstrap validates LMDB before
Uni-Core. It starts a clean subprocess with `LMDB_FORCE_CFFI`,
`LMDB_FORCE_SYSTEM`, `LMDB_INCLUDEDIR`, and `LMDB_LIBDIR` removed, imports both
`lmdb` and `lmdb.cpython`, prints the LMDB version, package path, native
extension path, virtual-environment path, active implementation, and those
environment-variable values, and requires the native extension to live inside
`.venv-uniprop` with a valid CPython extension suffix. CFFI selection, generated
CFFI products, and compiler launches during import are failures. The next stage
runs a temporary LMDB database round-trip and prints `LMDB_NATIVE_SMOKE_OK`.
The temporary database is created under `tempfile.TemporaryDirectory()` and
leaves no persistent database.

The `wandb==0.17.9` pin is a compatibility policy, not a scientific model
dependency. Normal `wandb 0.17.9` supports Python 3.10 and does not require
Pydantic. The bootstrap does not install the optional `wandb[launch]` extra,
because that path can introduce Pydantic. Pydantic, `pydantic-core`, Maturin,
Rust, and Cargo are not required for the UniProp checkpoint gate. If a newer
unconstrained `wandb` release causes pip to select `pydantic-core` as a source
distribution, pip may try to import Maturin or build Rust code. That is
prohibited on the Nibi login-node bootstrap path; the repair is to keep the
runtime dependency phase wheel-only and pinned, not to install Rust build tools.

The effective Uni-Core install command is:

```bash
env SETUPTOOLS_USE_DISTUTILS=stdlib \
  .venv-uniprop/bin/python -m pip install \
  --no-build-isolation \
  --no-deps \
  third_party/nablacolors/Uni-Core
```

`--no-deps` is used only here, after the complete Uni-Core runtime dependency
set has already passed the wheel-only dry-run report validation and install.
`--only-binary=:all:` is not applied to this local source path.

After Uni-Core installs, the bootstrap runs:

```bash
.venv-uniprop/bin/python -m pip check
```

It then imports and prints versions and import paths for `lmdb`, `tqdm`,
`ml_collections`, `scipy`, `tensorboardX`, `tokenizers`, `wandb`, `torch`,
`numpy`, and `unicore`; asserts `wandb.__version__ == "0.17.9"`; and verifies
from package metadata that `pydantic`, `pydantic-core`, and `maturin` are not
installed. Uni-Mol+ installation and the final `UniPropModel` import happen
only after these checks pass.

The bootstrap first installs build prerequisites into `.venv-uniprop`, including
`setuptools`, `wheel`, `packaging`, Alliance-compatible `numpy==2.2.2`, and the
requested PyTorch wheel. This removes the PyTorch NumPy initialization warning
and ensures Uni-Core's build process imports the same environment-installed
PyTorch package that the final runtime gate will use.

Uni-Mol+ is installed from `third_party/nablacolors/unimol_plus`. Its setup file
imports `setuptools` before declaring runtime dependencies and does not match
the Uni-Core `torch`-then-`setuptools` failure path, so the
`SETUPTOOLS_USE_DISTUTILS=stdlib` setting remains Uni-Core-specific unless a
future Nibi source-install failure proves Uni-Mol+ needs the same scoped
compatibility mode.

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
