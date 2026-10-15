#!/bin/bash
#SBATCH --job-name=cf_downstream
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

PYTHON="${FLUORCAST_PYTHON:-$HOME/scratch/venvs/fluorcast-conforformer/bin/python}"
: "${FLUORCAST_RUN_ROOT:?FLUORCAST_RUN_ROOT is required}"
RUN_ID="$(basename "$FLUORCAST_RUN_ROOT")"
DOWNSTREAM_ROOT="${FLUORCAST_DOWNSTREAM_ROOT:-outputs/conforformer/downstream/$RUN_ID}"
MODEL_ROOT="${FLUORCAST_MODEL_ROOT:-models/conforformer_downstream/$RUN_ID}"
DATASET="${FLUORCAST_DATASET:-data/processed/fluodb_lite/combined_deduplicated.csv}"
SOLVENT_DESCRIPTORS="${FLUORCAST_SOLVENT_DESCRIPTORS:-data/solvent_descriptors_expanded_deep4chem.csv}"

"$PYTHON" scripts/train_conforformer_downstream.py \
  --dataset "$DATASET" \
  --embedding-run-root "$FLUORCAST_RUN_ROOT" \
  --solvent-descriptors "$SOLVENT_DESCRIPTORS" \
  --out-dir "$DOWNSTREAM_ROOT" \
  --model-out-dir "$MODEL_ROOT" \
  --split-type molecule \
  --n-jobs "${SLURM_CPUS_PER_TASK}"

