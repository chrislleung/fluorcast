#!/bin/bash
#SBATCH --job-name=fc_scaffold_full
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /home/chrisl/scratch/ChemFluor_Project

module purge
module load python/3.11.5
module load gcc rdkit

source .venv/bin/activate

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job started on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Started at: $(date)"

echo ""
echo "=== Running paired absorption/emission/Stokes scaffold experiment ==="

python scripts/run_paired_spectral_three_way_experiment.py \
  --split-type scaffold \
  --seed 0 \
  --models rf extratrees histgb gbdt mlp \
  --invalid-smiles-policy drop \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --out-dir outputs/paired_stokes_three_way/scaffold \
  --model-out-dir models/paired_stokes_three_way/scaffold

echo ""
echo "=== Running QY scaffold experiment ==="

python scripts/run_hybrid_three_way_experiment.py \
  --target-name quantum_yield \
  --split-type scaffold \
  --seed 0 \
  --models rf extratrees histgb gbdt mlp \
  --invalid-smiles-policy drop \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --out-dir outputs/hybrid_three_way/scaffold/quantum_yield \
  --model-out-dir models/hybrid_three_way/scaffold/quantum_yield

echo ""
echo "Scaffold full experiment finished at: $(date)"
