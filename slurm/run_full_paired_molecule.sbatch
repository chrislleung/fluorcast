#!/bin/bash
#SBATCH --job-name=fc_molecule_full
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

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

mkdir -p outputs/slurm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "Job started on: $(hostname)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Working directory: $(pwd)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Started at: $(date)"

echo ""
echo "=== Running paired absorption/emission/Stokes molecule experiment ==="

python scripts/run_paired_spectral_three_way_experiment.py \
  --split-type molecule \
  --seed 0 \
  --models rf extratrees histgb gbdt mlp \
  --invalid-smiles-policy drop \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --out-dir outputs/paired_stokes_three_way/molecule \
  --model-out-dir models/paired_stokes_three_way/molecule

echo ""
echo "=== Running QY molecule experiment ==="

python scripts/run_hybrid_three_way_experiment.py \
  --target-name quantum_yield \
  --split-type molecule \
  --seed 0 \
  --models rf extratrees histgb gbdt mlp \
  --invalid-smiles-policy drop \
  --n-jobs "${SLURM_CPUS_PER_TASK}" \
  --out-dir outputs/hybrid_three_way/molecule/quantum_yield \
  --model-out-dir models/hybrid_three_way/molecule/quantum_yield

echo ""
echo "Molecule full experiment finished at: $(date)"
