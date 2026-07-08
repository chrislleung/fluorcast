#!/bin/bash
#SBATCH --job-name=stokes_cmp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/chrisl/ChemFluor_Project

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack
module load rdkit

module list

PYTHON=/scratch/chrisl/ChemFluor_Project/.venv/bin/python

echo "Python: $PYTHON"
$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Creating derived Stokes-shift target..."
$PYTHON scripts/manuscript/add_stokes_shift_target.py

echo "Running Stokes-shift paper comparison..."
$PYTHON scripts/manuscript/run_paper_comparison_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated_with_stokes.csv \
  --targets stokes_shift_nm \
  --splits random,molecule,scaffold \
  --models rf,extratrees,histgb,gbdt,mlp \
  --seeds 0,1,2 \
  --out-dir outputs/paper_comparison_stokes \
  --n-jobs $SLURM_CPUS_PER_TASK

echo "DONE: $(date)"
find outputs/paper_comparison_stokes -type f | sort
