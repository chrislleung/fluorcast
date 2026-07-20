#!/usr/bin/env bash
# Submit the full FluorCast baseline and UniProp experiment matrix.
#SBATCH --job-name=uniprop_matrix
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm/uniprop_matrix_%j.out
#SBATCH --error=outputs/slurm/uniprop_matrix_%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p outputs/slurm

module purge || true
module load python/3.10
module load gcc
module load rdkit

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  source "$FLUORCAST_ACTIVATE"
elif [[ -f ".venv-uniprop/bin/activate" ]]; then
  source ".venv-uniprop/bin/activate"
elif [[ -f "$HOME/scratch/fluorcast_uniprop_env/bin/activate" ]]; then
  source "$HOME/scratch/fluorcast_uniprop_env/bin/activate"
fi

CONFIG="${FLUORCAST_UNIPROP_MATRIX_CONFIG:-configs/uniprop/experiment_matrix.example.json}"
OUT_DIR="${FLUORCAST_UNIPROP_MATRIX_OUT:-outputs/uniprop_experiment_matrix/${SLURM_JOB_ID:-local}}"

python scripts/run_uniprop_experiment_matrix.py run \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --overwrite

python scripts/run_uniprop_experiment_matrix.py evaluate-test \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR"

python scripts/run_uniprop_experiment_matrix.py summarize \
  --experiment-dir "$OUT_DIR"

test -s "$OUT_DIR/aggregate_summary.csv"
test -s "$OUT_DIR/per_run_summary.csv"
