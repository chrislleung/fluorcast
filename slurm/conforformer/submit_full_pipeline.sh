#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-$(pwd)}"
cd "$PROJECT_ROOT"

PYTHON="${FLUORCAST_PYTHON:-$HOME/scratch/venvs/fluorcast-conforformer/bin/python}"
DATASET="${FLUORCAST_DATASET:-data/processed/fluodb_lite/combined_deduplicated.csv}"
CHECKPOINT="${FLUORCAST_CHECKPOINT:-models/conforformer/ConforFormer.pt}"
DICTIONARY="${FLUORCAST_DICTIONARY:-configs/conforformer/OMOL_full_dict.txt}"
SHARD_SIZE="${FLUORCAST_SHARD_SIZE:-128}"
RUN_ID="${FLUORCAST_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${FLUORCAST_RUN_ROOT:-outputs/conforformer/full_dataset/$RUN_ID}"
CPU_LIMIT="${FLUORCAST_CPU_ARRAY_LIMIT:-16}"
GPU_LIMIT="${FLUORCAST_GPU_ARRAY_LIMIT:-4}"

for path in "$PYTHON" "$DATASET" "$CHECKPOINT" "$DICTIONARY"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path does not exist: $path" >&2
    exit 2
  fi
done

export FLUORCAST_PYTHON="$PYTHON"
export FLUORCAST_DATASET="$DATASET"
export FLUORCAST_CHECKPOINT="$CHECKPOINT"
export FLUORCAST_DICTIONARY="$DICTIONARY"
export FLUORCAST_SHARD_SIZE="$SHARD_SIZE"
export FLUORCAST_RUN_ROOT="$RUN_ROOT"

mkdir -p "$RUN_ROOT" outputs/slurm

"$PYTHON" scripts/build_conforformer_inventory.py \
  --dataset "$DATASET" \
  --run-root "$RUN_ROOT" \
  --shard-size "$SHARD_SIZE"

SHARD_COUNT="$("$PYTHON" -c "import json; print(json.load(open('$RUN_ROOT/inventory/inventory_manifest.json'))['shard_count'])")"
if [[ "$SHARD_COUNT" -le 0 ]]; then
  echo "ERROR: inventory has no shards" >&2
  exit 2
fi
LAST_INDEX=$((SHARD_COUNT - 1))

CONF_JOB="$(sbatch --parsable --array=0-${LAST_INDEX}%${CPU_LIMIT} slurm/conforformer/build_conformer_cache_array.sbatch)"
EMBED_JOB="$(sbatch --parsable --dependency=afterok:${CONF_JOB} --array=0-${LAST_INDEX}%${GPU_LIMIT} slurm/conforformer/embed_full_dataset_array.sbatch)"
FINAL_JOB="$(sbatch --parsable --dependency=afterok:${EMBED_JOB} slurm/conforformer/finalize_embeddings.sbatch)"

echo "run_root=$RUN_ROOT"
echo "shard_count=$SHARD_COUNT"
echo "conformer_cache_job=$CONF_JOB"
echo "embedding_job=$EMBED_JOB"
echo "finalize_job=$FINAL_JOB"

if [[ "${FLUORCAST_SKIP_TRAINING:-0}" == "1" ]]; then
  echo "training_job=skipped"
else
  TRAIN_JOB="$(sbatch --parsable --dependency=afterok:${FINAL_JOB} slurm/conforformer/train_downstream.sbatch)"
  echo "training_job=$TRAIN_JOB"
fi

