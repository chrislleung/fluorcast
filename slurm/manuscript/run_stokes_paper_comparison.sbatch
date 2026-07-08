#!/usr/bin/env bash
# Historical manuscript comparison: directly models a derived Stokes target.
#SBATCH --job-name=stokes_cmp
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Submit from the repo root or set FLUORCAST_REPO." >&2
  exit 2
fi

cd "$PROJECT_ROOT"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"

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

mkdir -p outputs/slurm
PYTHON="${FLUORCAST_PYTHON:-python}"

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
