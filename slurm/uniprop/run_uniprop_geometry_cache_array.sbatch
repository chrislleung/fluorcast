#!/usr/bin/env bash
#SBATCH --job-name=uniprop-geom
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-99

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
    echo "Could not locate the FluorCast repository." >&2
    echo "Submit from the project root or set FLUORCAST_REPO." >&2
    exit 2
fi

cd "$PROJECT_ROOT"

module purge || true
module load python/3.11
module load gcc
module load rdkit

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
    if [[ ! -f "$FLUORCAST_ACTIVATE" ]]; then
        echo "Environment activation script does not exist: $FLUORCAST_ACTIVATE" >&2
        exit 2
    fi
    # shellcheck disable=SC1090
    source "$FLUORCAST_ACTIVATE"
elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "No virtual environment found; using python from the loaded modules." >&2
fi

mkdir -p outputs/slurm outputs/uniprop_geometry

MANIFEST="${FLUORCAST_UNIPROP_MOLECULE_MANIFEST:-data/processed/uniprop/molecule_manifest.csv}"
CACHE_DIR="${FLUORCAST_UNIPROP_GEOMETRY_CACHE:-data/processed/uniprop/geometry_cache}"
SHARD_COUNT="${FLUORCAST_UNIPROP_GEOMETRY_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-100}}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
WORKERS="${SLURM_CPUS_PER_TASK:-1}"

python -u scripts/build_uniprop_geometry_cache.py \
    --molecule-manifest "$MANIFEST" \
    --cache-dir "$CACHE_DIR" \
    --workers "$WORKERS" \
    --resume \
    --overwrite-invalid \
    --shard-index "$SHARD_INDEX" \
    --shard-count "$SHARD_COUNT" \
    --failure-json "outputs/uniprop_geometry/failures_${SHARD_INDEX}.json" \
    --failure-csv "outputs/uniprop_geometry/failures_${SHARD_INDEX}.csv" \
    --status-json "outputs/uniprop_geometry/status_${SHARD_INDEX}.json"
