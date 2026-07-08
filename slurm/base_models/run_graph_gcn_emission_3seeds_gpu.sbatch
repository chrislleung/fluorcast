#!/bin/bash
#SBATCH --job-name=gcn_em_3seed
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/graph_gcn_emission_3seeds_gpu_%j.out
#SBATCH --error=outputs/slurm/graph_gcn_emission_3seeds_gpu_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/graph_gcn_emission_3seeds_gpu
mkdir -p outputs/graph_gcn_emission_3seeds_gpu

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Graph GCN emission 3-seed GPU job started on $(hostname)"
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

for SEED in 0 1 2; do
  echo "============================================================"
  echo "Running graph_gcn emission_nm seed ${SEED}"
  echo "Seed start time: $(date)"
  echo "============================================================"

  mkdir -p models/graph_gcn_emission_3seeds_gpu/seed_${SEED}
  mkdir -p outputs/graph_gcn_emission_3seeds_gpu/seed_${SEED}

  python -u scripts/run_graph_model_experiments.py \
    --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
    --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
    --tree-compare-dir outputs/model_experiments_fluodb \
    --neural-compare-dir outputs/neural_model_experiments_fluodb \
    --out-root models/graph_gcn_emission_3seeds_gpu/seed_${SEED} \
    --compare-out outputs/graph_gcn_emission_3seeds_gpu/seed_${SEED} \
    --models graph_gcn \
    --targets emission_nm \
    --epochs 100 \
    --patience 15 \
    --batch-size 512 \
    --hidden-dim 256 \
    --num-layers 4 \
    --dropout 0.2 \
    --learning-rate 1e-3 \
    --weight-decay 1e-4 \
    --seed ${SEED} \
    --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
    --benchmark-solvent-smiles "CS(=O)C" \
    --known-emission-nm 539 \
    --known-quantum-yield 0.196

  echo "Finished graph_gcn emission_nm seed ${SEED} at $(date)"
done

echo "End time: $(date)"
echo "Graph GCN emission 3-seed GPU job completed."

echo "Seed outputs:"
find outputs/graph_gcn_emission_3seeds_gpu -maxdepth 2 -type f | sort
