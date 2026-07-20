#!/usr/bin/env bash
# Head-only UniProp smoke training on Nibi.
#SBATCH --job-name=uniprop_head_smoke
# Override these resources at submit time when needed, for example:
# sbatch --time=00:10:00 --mem=8G --cpus-per-task=2 --gpus-per-node=h100:1 slurm/uniprop/run_uniprop_head_smoke.sbatch
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/uniprop_head_smoke_%j.out
#SBATCH --error=outputs/slurm/uniprop_head_smoke_%j.err

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
else
  echo "WARNING: No UniProp virtual environment found; using loaded Python." >&2
fi

CONFIG="${FLUORCAST_UNIPROP_HEAD_SMOKE_CONFIG:-configs/uniprop/head_smoke.example.json}"
DATA_DIR="${FLUORCAST_UNIPROP_LMDB_DIR:-data/processed/uniprop/lmdb}"
OUT_DIR="${FLUORCAST_UNIPROP_HEAD_SMOKE_OUT:-outputs/uniprop_head_smoke_nibi/${SLURM_JOB_ID:-local}}"
MAX_UPDATES="${FLUORCAST_UNIPROP_HEAD_SMOKE_UPDATES:-20}"
DEVICE="${FLUORCAST_UNIPROP_HEAD_SMOKE_DEVICE:-cuda}"

echo "UniProp head-only smoke training started on $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
python --version

python scripts/train_uniprop_head_smoke.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --output-dir "$OUT_DIR" \
  --max-updates "$MAX_UPDATES" \
  --device "$DEVICE" \
  --print-summary

test -s "$OUT_DIR/validation_predictions.csv"
test -s "$OUT_DIR/best_checkpoint.pt"
test -s "$OUT_DIR/last_checkpoint.pt"

echo "UniProp head-only smoke training finished at $(date)"
