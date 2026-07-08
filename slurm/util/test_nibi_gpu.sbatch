#!/usr/bin/env bash
# Utility job for validating GPU and PyTorch availability on Nibi.
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/test_gpu_%j.out
#SBATCH --error=outputs/slurm/test_gpu_%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Submit from the repo root or set FLUORCAST_REPO." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p outputs/slurm

module purge || true
module load python/3.11
module load gcc
module load rdkit

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  source "$FLUORCAST_ACTIVATE"
elif [[ -f "$HOME/scratch/chemfluor_env/bin/activate" ]]; then
  source "$HOME/scratch/chemfluor_env/bin/activate"
elif [[ -f "$HOME/scratch/fluorcast_env/bin/activate" ]]; then
  source "$HOME/scratch/fluorcast_env/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  echo "WARNING: No virtual environment found; using loaded Python." >&2
fi

echo "GPU test started on $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
python --version

echo "Checking PyTorch CUDA..."
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(2000, 2000, device="cuda")
    y = x @ x
    print("CUDA matmul OK:", y.shape)
else:
    raise SystemExit("CUDA is not available inside this GPU job.")
PY

echo "GPU test finished successfully at $(date)"
