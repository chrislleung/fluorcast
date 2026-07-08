#!/bin/bash
# Archived workflow; retained for reproducibility, not routine use.
#SBATCH --job-name=hyb_qy_mol
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

cd /scratch/chrisl/FluorCast

PYTHON=/scratch/chrisl/FluorCast/.venv/bin/python

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "Python: $PYTHON"

$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

mkdir -p outputs/hybrid_three_way/molecule/quantum_yield
mkdir -p models/hybrid_three_way/molecule/quantum_yield
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON scripts/run_hybrid_three_way_experiment.py \
  --target-name quantum_yield \
  --split-type molecule \
  --models rf extratrees histgb gbdt mlp \
  --out-dir outputs/hybrid_three_way/molecule/quantum_yield \
  --model-out-dir models/hybrid_three_way/molecule/quantum_yield \
  --n-jobs $SLURM_CPUS_PER_TASK

echo "DONE: $(date)"
find outputs/hybrid_three_way/molecule/quantum_yield -maxdepth 2 -type f -print
