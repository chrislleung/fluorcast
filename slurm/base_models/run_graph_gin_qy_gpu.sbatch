#!/bin/bash
#SBATCH --job-name=gin_qy_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/graph_gin_qy_gpu_%j.out
#SBATCH --error=outputs/slurm/graph_gin_qy_gpu_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/graph_gin_qy_gpu
mkdir -p outputs/graph_gin_qy_gpu

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Graph GIN quantum_yield GPU job started on $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
python --version

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available. Stop before training.")
PY

python -u scripts/run_graph_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --neural-compare-dir outputs/neural_model_experiments_fluodb \
  --out-root models/graph_gin_qy_gpu \
  --compare-out outputs/graph_gin_qy_gpu \
  --models graph_gin \
  --targets quantum_yield \
  --epochs 100 \
  --patience 15 \
  --batch-size 512 \
  --hidden-dim 256 \
  --num-layers 4 \
  --dropout 0.2 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --seed 0 \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Graph GIN quantum_yield GPU job completed."
ls -lh outputs/graph_gin_qy_gpu || true
