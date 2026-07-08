#!/bin/bash
#SBATCH --job-name=hyb_abs_scaf
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/chrisl/ChemFluor_Project

PYTHON=/scratch/chrisl/ChemFluor_Project/.venv/bin/python

echo "PWD: $(pwd)"
echo "Python: $PYTHON"
$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

mkdir -p outputs/hybrid_evaluation/scaffold/absorption_nm
mkdir -p models/hybrid_evaluation/scaffold/absorption_nm
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON scripts/run_hybrid_three_way_experiment.py \
  --target-name absorption_nm \
  --split-type scaffold \
  --models rf extratrees histgb gbdt mlp \
  --out-dir outputs/hybrid_evaluation/scaffold/absorption_nm \
  --model-out-dir models/hybrid_evaluation/scaffold/absorption_nm \
  --n-jobs $SLURM_CPUS_PER_TASK

echo "DONE: $(date)"
find outputs/hybrid_evaluation/scaffold/absorption_nm -maxdepth 2 -type f -print
