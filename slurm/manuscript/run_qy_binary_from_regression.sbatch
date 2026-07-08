#!/usr/bin/env bash
# Historical manuscript verification derived from QY regression outputs.
#SBATCH --job-name=qy_bin_reg
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

echo "Checking for QY regression prediction files..."
find outputs/paper_comparison/predictions -name "quantum_yield__*__*__seed*.csv" | sort | head -n 20

echo "Running thresholded-regression PLQY binary classification verification..."
$PYTHON scripts/manuscript/verify_qy_binary_from_regression.py

echo "DONE: $(date)"
find outputs/paper_comparison/qy_binary_from_regression -type f | sort
