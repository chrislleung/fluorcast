#!/usr/bin/env bash
set -euo pipefail

WORKER_JOB_NAME="fluorcast-work"
DEFAULT_REPO="${SCRATCH:-$HOME/scratch}/FluorCast"
WORKER_ENV="${FLUORCAST_WORKER_ENV:-${SCRATCH:-$HOME/scratch}/fluorcast_worker.env}"

FLUORCAST_REPO="${FLUORCAST_REPO:-$DEFAULT_REPO}"

if [[ ! -d "$FLUORCAST_REPO" ]]; then
    echo "FluorCast repository directory does not exist: $FLUORCAST_REPO" >&2
    exit 2
fi

cd "$FLUORCAST_REPO"

if [[ ! -f "$WORKER_ENV" ]]; then
    echo "Worker environment file does not exist: $WORKER_ENV" >&2
    exit 2
fi

set -a
# shellcheck disable=SC1090
source "$WORKER_ENV"
set +a

: "${SUPABASE_URL:?SUPABASE_URL must be set in $WORKER_ENV}"
: "${SUPABASE_SERVICE_ROLE_KEY:?SUPABASE_SERVICE_ROLE_KEY must be set in $WORKER_ENV}"
: "${FLUORCAST_REPO:?FLUORCAST_REPO must be set in $WORKER_ENV or the launcher environment}"
: "${FLUORCAST_JOBS_DIR:?FLUORCAST_JOBS_DIR must be set in $WORKER_ENV}"

if [[ ! -d "$FLUORCAST_REPO" ]]; then
    echo "Configured FLUORCAST_REPO directory does not exist: $FLUORCAST_REPO" >&2
    exit 2
fi

cd "$FLUORCAST_REPO"

# This duplicate-worker guard keeps cron from stacking multiple polling workers
# when an earlier worker is still queued or running.
existing_worker="$(
    squeue \
        --noheader \
        --user "$USER" \
        --name "$WORKER_JOB_NAME" \
        --states PENDING,RUNNING,CONFIGURING,COMPLETING \
        --format "%i %j %T" || true
)"

if [[ -n "$existing_worker" ]]; then
    echo "A $WORKER_JOB_NAME Slurm job is already queued or running:"
    echo "$existing_worker"
    exit 0
fi

echo "No $WORKER_JOB_NAME Slurm worker is queued or running; submitting one."
sbatch slurm/run_nibi_supabase_worker.sbatch
