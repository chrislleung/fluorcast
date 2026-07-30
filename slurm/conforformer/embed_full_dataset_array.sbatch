#!/bin/bash
#SBATCH --job-name=cf_embed
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
cd "$PROJECT_ROOT"
mkdir -p outputs/slurm

module purge || true
module load python/3.11
module load gcc
module load rdkit

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PYTHON="${FLUORCAST_PYTHON:-$HOME/scratch/venvs/fluorcast-conforformer/bin/python}"
: "${FLUORCAST_RUN_ROOT:?FLUORCAST_RUN_ROOT is required}"
: "${FLUORCAST_CHECKPOINT:?FLUORCAST_CHECKPOINT is required}"
: "${FLUORCAST_DICTIONARY:?FLUORCAST_DICTIONARY is required}"

"$PYTHON" scripts/embed_conforformer_shard.py \
  --run-root "$FLUORCAST_RUN_ROOT" \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --checkpoint "$FLUORCAST_CHECKPOINT" \
  --dictionary "$FLUORCAST_DICTIONARY" \
  --device cuda

