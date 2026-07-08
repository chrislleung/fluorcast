#!/bin/bash
#SBATCH --job-name=graph_gin_mpnn
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/graph_gin_mpnn_emission_gpu_%j.out
#SBATCH --error=outputs/slurm/graph_gin_mpnn_emission_gpu_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/graph_gin_mpnn_emission_gpu
mkdir -p outputs/graph_gin_mpnn_emission_gpu

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "GPU graph GIN/MPNN emission job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking RDKit..."
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

echo "Checking PyTorch/CUDA..."
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. Stop before training.")
PY

echo "Running graph_gin and graph_mpnn emission_nm on GPU..."
python -u scripts/run_graph_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --neural-compare-dir outputs/neural_model_experiments_fluodb \
  --out-root models/graph_gin_mpnn_emission_gpu \
  --compare-out outputs/graph_gin_mpnn_emission_gpu \
  --models graph_gin,graph_mpnn \
  --targets emission_nm \
  --epochs 100 \
  --patience 15 \
  --batch-size 512 \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "GPU graph GIN/MPNN emission job completed."

echo "Outputs:"
ls -lh outputs/graph_gin_mpnn_emission_gpu || true
