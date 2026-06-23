#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-predict
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

: "${FLUORCAST_REPO:?FLUORCAST_REPO must point to the FluorCast repository}"
: "${FLUORCAST_INPUT_JSON:?FLUORCAST_INPUT_JSON must point to the input JSON}"
: "${FLUORCAST_OUTPUT_JSON:?FLUORCAST_OUTPUT_JSON must point to the output JSON}"

if [[ ! -d "$FLUORCAST_REPO" ]]; then
    echo "FluorCast repository directory does not exist: $FLUORCAST_REPO" >&2
    exit 2
fi

cd "$FLUORCAST_REPO"
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
    echo "No .venv found; using python from PATH. Set FLUORCAST_ACTIVATE to customize activation." >&2
fi

python scripts/run_prediction_job.py \
    --input "$FLUORCAST_INPUT_JSON" \
    --output "$FLUORCAST_OUTPUT_JSON"
