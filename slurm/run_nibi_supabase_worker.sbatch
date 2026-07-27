#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-work
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

: "${FLUORCAST_WORKER_ENV:?FLUORCAST_WORKER_ENV must point to the worker environment file}"
if [[ ! -f "$FLUORCAST_WORKER_ENV" ]]; then
    echo "Worker environment file does not exist: $FLUORCAST_WORKER_ENV" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$FLUORCAST_WORKER_ENV"
set +a

: "${FLUORCAST_JOBS_DIR:?FLUORCAST_JOBS_DIR must point to the jobs directory}"
: "${SUPABASE_URL:?SUPABASE_URL must be set}"
: "${SUPABASE_SERVICE_ROLE_KEY:?SUPABASE_SERVICE_ROLE_KEY must be set}"

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

mkdir -p outputs/slurm

python scripts/nibi_supabase_worker.py \
    --loop \
    --interval-seconds 30 \
    --exit-after-idle-seconds 900
