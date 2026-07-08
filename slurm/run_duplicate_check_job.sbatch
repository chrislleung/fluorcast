#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-duplicate
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euo pipefail

: "${FLUORCAST_INPUT_JSON:?FLUORCAST_INPUT_JSON must point to the input JSON}"
: "${FLUORCAST_OUTPUT_JSON:?FLUORCAST_OUTPUT_JSON must point to the output JSON}"

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
elif [[ -f "$HOME/scratch/chemfluor_env/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/chemfluor_env/bin/activate"
elif [[ -f "$HOME/scratch/fluorcast_env/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/scratch/fluorcast_env/bin/activate"
elif [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "No virtual environment found; using python from the loaded modules." >&2
fi

mkdir -p outputs/slurm

runner_args=(
    --input "$FLUORCAST_INPUT_JSON"
    --output "$FLUORCAST_OUTPUT_JSON"
)
if [[ -n "${FLUORCAST_DATASET:-}" ]]; then
    runner_args+=(--dataset "$FLUORCAST_DATASET")
fi

python scripts/run_duplicate_check_job.py "${runner_args[@]}"
