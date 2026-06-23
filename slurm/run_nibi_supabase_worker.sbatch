#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-worker
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -euo pipefail

: "${FLUORCAST_REPO:?FLUORCAST_REPO must point to the FluorCast repository}"
: "${FLUORCAST_JOBS_DIR:?FLUORCAST_JOBS_DIR must point to the jobs directory}"
: "${SUPABASE_URL:?SUPABASE_URL must be set}"
: "${SUPABASE_SERVICE_ROLE_KEY:?SUPABASE_SERVICE_ROLE_KEY must be set}"

if [[ ! -d "$FLUORCAST_REPO" ]]; then
    echo "FluorCast repository directory does not exist: $FLUORCAST_REPO" >&2
    exit 2
fi

cd "$FLUORCAST_REPO"

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

python scripts/nibi_supabase_worker.py --once
