#!/usr/bin/env bash
#SBATCH --job-name=hybrid-3way
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G

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

TARGET_NAME="${FLUORCAST_TARGET_NAME:-emission_nm}"
SPLIT_TYPE="${FLUORCAST_SPLIT_TYPE:-molecule}"
SEED="${FLUORCAST_SEED:-0}"
OUT_DIR="${FLUORCAST_OUT_DIR:-outputs/hybrid_three_way/${SPLIT_TYPE}/${TARGET_NAME}/seed_${SEED}}"
MODEL_OUT_DIR="${FLUORCAST_MODEL_OUT_DIR:-models/hybrid_three_way/${SPLIT_TYPE}/${TARGET_NAME}/seed_${SEED}}"
N_JOBS="${SLURM_CPUS_PER_TASK:-1}"

mkdir -p "$OUT_DIR" "$MODEL_OUT_DIR" outputs/slurm

args=(
    --target-name "$TARGET_NAME"
    --split-type "$SPLIT_TYPE"
    --seed "$SEED"
    --models ${FLUORCAST_MODELS:-rf extratrees histgb gbdt mlp}
    --out-dir "$OUT_DIR"
    --model-out-dir "$MODEL_OUT_DIR"
    --n-jobs "$N_JOBS"
)

if [[ -n "${FLUORCAST_MAX_ROWS:-}" ]]; then
    args+=(--max-rows "$FLUORCAST_MAX_ROWS")
fi

python -u scripts/run_hybrid_three_way_experiment.py "${args[@]}"
