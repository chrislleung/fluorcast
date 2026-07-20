#!/usr/bin/env bash
# Second-stage UniProp backbone fine-tuning on Nibi.
# Override resources with sbatch flags when needed; any CUDA GPU with enough
# memory can run the smoke job. Request a specific GPU type with sbatch flags
# only when a queue policy or benchmark run needs that.
#SBATCH --job-name=uniprop_finetune
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --output=outputs/slurm/uniprop_finetune_%j.out
#SBATCH --error=outputs/slurm/uniprop_finetune_%j.err

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

CONFIG="${FLUORCAST_UNIPROP_FINETUNE_CONFIG:-configs/uniprop/backbone_finetune.example.json}"
DATA_DIR="${FLUORCAST_UNIPROP_LMDB_DIR:-data/processed/uniprop/lmdb}"
HEAD_CHECKPOINT="${FLUORCAST_UNIPROP_HEAD_CHECKPOINT:-outputs/uniprop_head_smoke/best_checkpoint.pt}"
OUT_DIR="${FLUORCAST_UNIPROP_FINETUNE_OUT:-outputs/uniprop_backbone_finetune_nibi/${SLURM_JOB_ID:-local}}"
MAX_UPDATES="${FLUORCAST_UNIPROP_FINETUNE_UPDATES:-20}"
DEVICE="${FLUORCAST_UNIPROP_FINETUNE_DEVICE:-cuda}"

echo "UniProp backbone fine-tuning started on $(hostname)"
echo "Start time: $(date)"
echo "Python: $(which python)"
python --version

python scripts/train_uniprop_backbone_finetune.py \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --head-checkpoint "$HEAD_CHECKPOINT" \
  --output-dir "$OUT_DIR" \
  --max-updates "$MAX_UPDATES" \
  --device "$DEVICE" \
  --print-summary

test -s "$OUT_DIR/validation_predictions.csv"
test -s "$OUT_DIR/best_checkpoint.pt"
test -s "$OUT_DIR/last_checkpoint.pt"
test -s "$OUT_DIR/parameter_counts.json"

echo "UniProp backbone fine-tuning finished at $(date)"
