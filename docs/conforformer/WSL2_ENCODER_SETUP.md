# WSL2 ConforFormer Encoder Setup

These instructions are for a separate local Linux environment for the real
ConforFormer encoder. They are based on the pinned Uni-Mol fork README and
requirements, but they have not been run end-to-end in this workspace. Treat
commands as a reproducible setup recipe to verify, not as a completed
installation report.

Do not install these dependencies into the normal Windows FluorCast `.venv`.

## Prerequisites

- Windows with WSL2 enabled.
- Ubuntu under WSL2.
- Git available in WSL2.
- Enough disk space for checkpoints and optional data artifacts.
- Optional CUDA support only if the local machine has a compatible NVIDIA stack.
  The first FluorCast smoke should use CPU.

## Repository Location

For best filesystem performance, copy or clone the repository into the WSL
filesystem, for example under `~/src/fluorcast-conforformer`, instead of
running heavy Python imports from `/mnt/c/...`.

If you keep the Windows checkout as the source of truth, copy only the local
asset files and generated diagnostic JSON back and forth. Do not duplicate large
embedding caches in this stage.

## Recommended Python

Use Python 3.9 or 3.10 for the ConforFormer environment. The pinned Uni-Mol
`setup.py` advertises Python 3.7-3.10, the notebooks reference Python 3.9
Uni-Core wheels, and the upstream README targets older PyTorch/RDKit versions.

The current Windows FluorCast environment uses Python 3.14.0, which is not a
good target for Uni-Core.

## Create Environment

Unverified WSL2 commands:

```bash
sudo apt update
sudo apt install -y git build-essential python3.10 python3.10-venv python3.10-dev

cd ~/src
git clone <your FluorCast repo URL> fluorcast-conforformer
cd fluorcast-conforformer
git checkout feature/conforformer

python3.10 -m venv .venv-conforformer
source .venv-conforformer/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Install PyTorch

CPU-first smoke, unverified:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For CUDA, choose the command from the official PyTorch selector for the local
driver/CUDA stack. Do not enable fp16 for the first compatibility smoke.

## Install Uni-Core

The pinned `third_party/ConforFormer/unimol/requirements.txt` uses a `git://`
Uni-Core URL. Prefer HTTPS if `git://` is blocked:

```bash
python -m pip install "git+https://github.com/dptech-corp/Uni-Core.git@stable#egg=Uni-Core"
```

If source builds fail, use a Uni-Core wheel compatible with the selected Python
and PyTorch versions. Record the wheel URL and version in the environment report.

## Install Uni-Mol Fork and Dependencies

Unverified:

```bash
python -m pip install rdkit-pypi==2022.9.3
python -m pip install lmdb pandas scikit-learn-extra scipy numpy
python -m pip install -e third_party/ConforFormer/unimol
```

`lmdb` is needed for the upstream LMDB task pathway. The direct Stage 4 adapter
does not require LMDB for importing, but the pinned upstream task imports may
expect it when loading datasets.

## Assets

Place official assets under ignored local paths:

```text
assets/conforformer/checkpoints/ConforFormer.pt
assets/conforformer/dictionaries/dict_omol_full.txt
```

The dictionary path is illustrative until the exact official dictionary is
obtained. Do not create a replacement dictionary from the example Uni-Mol
`dict.txt`.

## Verification Commands

From the repository root with `.venv-conforformer` active:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import unicore; print('unicore ok')"
python -c "import lmdb; print('lmdb ok')"
python -c "import sys; sys.path.insert(0, 'third_party/ConforFormer/unimol'); import unimol.tasks.unimol_contrast; import unimol.models.unimol_contrast; print('upstream ok')"
python scripts/smoke_conforformer_encoder.py --env-report
```

After official checkpoint and dictionary assets are present:

```bash
python scripts/smoke_conforformer_encoder.py \
  --inspect-only \
  --checkpoint assets/conforformer/checkpoints/ConforFormer.pt \
  --dictionary assets/conforformer/dictionaries/dict_omol_full.txt \
  --output conforformer_encoder_diagnostics/inspect_conforformer.json
```

Only if inspection succeeds:

```bash
python scripts/smoke_conforformer_encoder.py \
  --cache-dir data/processed/conforformer_cache/conformers \
  --smiles "CCO" \
  --checkpoint assets/conforformer/checkpoints/ConforFormer.pt \
  --dictionary assets/conforformer/dictionaries/dict_omol_full.txt \
  --device cpu \
  --max-conformers 1 \
  --repeat-check \
  --output conforformer_encoder_diagnostics/cco_cpu_embedding.json
```

## Environment Report

Record:

```bash
python scripts/smoke_conforformer_encoder.py --env-report \
  --checkpoint assets/conforformer/checkpoints/ConforFormer.pt \
  --dictionary assets/conforformer/dictionaries/dict_omol_full.txt \
  --output conforformer_encoder_diagnostics/wsl2_env_report.json
```

This reports Python version, OS, PyTorch version, CUDA availability, Uni-Core
import status, LMDB import status, pinned upstream import status, upstream
commit, and asset availability.

## Remove and Recreate

Unverified:

```bash
deactivate 2>/dev/null || true
rm -rf .venv-conforformer
python3.10 -m venv .venv-conforformer
source .venv-conforformer/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Then repeat the installation steps above.

## Current Status

This WSL2 environment has not been created or verified in this stage. The
current native Windows environment can import PyTorch CPU, but Uni-Core is not
installed and the pinned upstream modules fail to import because `unicore` is
missing. Real model inference remains blocked until the exact dictionary and a
working separate ConforFormer environment are available.
