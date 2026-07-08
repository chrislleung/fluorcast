#!/bin/bash
#SBATCH --job-name=test_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/test_gpu_%j.out
#SBATCH --error=outputs/slurm/test_gpu_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project
mkdir -p outputs/slurm

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

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
