#!/bin/bash
# Archived workflow; retained for reproducibility, not routine use.
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "FLUORCAST_REPO must point to the FluorCast repository." >&2
  exit 2
fi
cd "$PROJECT_ROOT"

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack
module load rdkit

module list

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  # shellcheck disable=SC1090
  source "$FLUORCAST_ACTIVATE"
fi
PYTHON="${FLUORCAST_PYTHON:-python}"

echo "Python: $PYTHON"
$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

: "${TARGET_NAME:?ERROR: TARGET_NAME is required}"
: "${SPLIT_TYPE:?ERROR: SPLIT_TYPE is required}"

OUT_DIR="outputs/hybrid_three_way/${SPLIT_TYPE}/${TARGET_NAME}"
MODEL_OUT_DIR="models/hybrid_three_way/${SPLIT_TYPE}/${TARGET_NAME}"

mkdir -p "$OUT_DIR"
mkdir -p "$MODEL_OUT_DIR"
mkdir -p logs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Target: $TARGET_NAME"
echo "Split: $SPLIT_TYPE"
echo "Output: $OUT_DIR"
echo "Model output: $MODEL_OUT_DIR"
echo "CPUs: $SLURM_CPUS_PER_TASK"

$PYTHON scripts/run_hybrid_three_way_experiment.py \
  --target-name "$TARGET_NAME" \
  --split-type "$SPLIT_TYPE" \
  --models rf extratrees histgb gbdt mlp \
  --out-dir "$OUT_DIR" \
  --model-out-dir "$MODEL_OUT_DIR" \
  --n-jobs "$SLURM_CPUS_PER_TASK"

echo "DONE: $(date)"
find "$OUT_DIR" -maxdepth 2 -type f -print
