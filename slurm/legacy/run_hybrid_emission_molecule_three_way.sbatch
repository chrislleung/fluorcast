#!/bin/bash
# Archived workflow; retained for reproducibility, not routine use.
#SBATCH --job-name=hyb_emi_mol
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "FLUORCAST_REPO must point to the FluorCast repository." >&2
  exit 2
fi
cd "$PROJECT_ROOT"

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$FLUORCAST_ACTIVATE"
fi
PYTHON="${FLUORCAST_PYTHON:-python}"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"
echo "Python: $PYTHON"

$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

mkdir -p outputs/hybrid_three_way/molecule/emission_nm
mkdir -p models/hybrid_three_way/molecule/emission_nm
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON scripts/run_hybrid_three_way_experiment.py \
  --target-name emission_nm \
  --split-type molecule \
  --models rf extratrees histgb gbdt mlp \
  --out-dir outputs/hybrid_three_way/molecule/emission_nm \
  --model-out-dir models/hybrid_three_way/molecule/emission_nm \
  --n-jobs $SLURM_CPUS_PER_TASK

echo "DONE: $(date)"
find outputs/hybrid_three_way/molecule/emission_nm -maxdepth 2 -type f -print
