#!/bin/bash
#SBATCH --job-name=cf_finalize
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
cd "$PROJECT_ROOT"
mkdir -p outputs/slurm

module purge || true
module load python/3.11
module load gcc
module load rdkit

PYTHON="${FLUORCAST_PYTHON:-$HOME/scratch/venvs/fluorcast-conforformer/bin/python}"
: "${FLUORCAST_RUN_ROOT:?FLUORCAST_RUN_ROOT is required}"
: "${FLUORCAST_CHECKPOINT:?FLUORCAST_CHECKPOINT is required}"
: "${FLUORCAST_DICTIONARY:?FLUORCAST_DICTIONARY is required}"

"$PYTHON" scripts/finalize_conforformer_embeddings.py \
  --run-root "$FLUORCAST_RUN_ROOT" \
  --checkpoint "$FLUORCAST_CHECKPOINT" \
  --dictionary "$FLUORCAST_DICTIONARY"

