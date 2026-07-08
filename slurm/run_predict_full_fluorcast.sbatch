#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-full
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

: "${FLUORCAST_SMILES:?FLUORCAST_SMILES must contain the molecule SMILES}"
: "${FLUORCAST_SOLVENT_SMILES:?FLUORCAST_SOLVENT_SMILES must contain the solvent SMILES}"
: "${FLUORCAST_OUT_DIR:?FLUORCAST_OUT_DIR must point to the output directory}"

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

mkdir -p "$FLUORCAST_OUT_DIR" outputs/slurm

args=(
    --smiles "$FLUORCAST_SMILES"
    --solvent-smiles "$FLUORCAST_SOLVENT_SMILES"
    --out-dir "$FLUORCAST_OUT_DIR"
    --tree-model-dir "${FLUORCAST_TREE_MODEL_DIR:-models/experiments_fluodb}"
    --neural-model-dir "${FLUORCAST_NEURAL_MODEL_DIR:-models/neural_experiments_fluodb}"
    --absorption-hybrid-model-dir "${FLUORCAST_ABS_HYBRID_DIR:-models/production_hybrid/absorption_nm}"
    --emission-hybrid-model-dir "${FLUORCAST_EM_HYBRID_DIR:-models/production_hybrid/emission_nm}"
    --quantum-yield-hybrid-model-dir "${FLUORCAST_QY_HYBRID_DIR:-models/production_hybrid/quantum_yield}"
)

if [[ "${FLUORCAST_SKIP_HYBRID:-0}" == "1" ]]; then
    args+=(--skip-hybrid)
fi

if [[ -n "${FLUORCAST_KNOWN_ABSORPTION_NM:-}" ]]; then
    args+=(--known-absorption-nm "$FLUORCAST_KNOWN_ABSORPTION_NM")
fi
if [[ -n "${FLUORCAST_KNOWN_EMISSION_NM:-}" ]]; then
    args+=(--known-emission-nm "$FLUORCAST_KNOWN_EMISSION_NM")
fi
if [[ -n "${FLUORCAST_KNOWN_QUANTUM_YIELD:-}" ]]; then
    args+=(--known-quantum-yield "$FLUORCAST_KNOWN_QUANTUM_YIELD")
fi

python -u scripts/predict_full_fluorcast.py "${args[@]}"
