#!/usr/bin/env bash
#SBATCH --job-name=paper-comparison
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=48:00:00
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

mkdir -p outputs/slurm outputs/paper_comparison

N_JOBS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting manuscript comparison experiments at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(command -v python)"
echo "Allocated CPUs: $N_JOBS"
python --version
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

python -u scripts/manuscript/run_paper_comparison_experiments.py \
    --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
    --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
    --out-dir outputs/paper_comparison \
    --models rf,extratrees,histgb,gbdt,mlp \
    --targets absorption_nm,emission_nm,quantum_yield \
    --splits random,molecule,scaffold \
    --seeds 0,1,2 \
    --n-jobs "$N_JOBS"

echo "Manuscript comparison experiments finished at $(date)"
echo "Summary outputs:"
find outputs/paper_comparison -maxdepth 1 -type f -printf "%f\n" | sort
