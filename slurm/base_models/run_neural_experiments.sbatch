#!/usr/bin/env bash
#SBATCH --job-name=fluorcast_neural
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/chemfluor_neural_%j.out
#SBATCH --error=outputs/slurm/chemfluor_neural_%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Submit from the repo root or set FLUORCAST_REPO." >&2
  exit 2
fi

cd "$PROJECT_ROOT"

mkdir -p outputs/slurm
NEURAL_OUT_ROOT="${FLUORCAST_NEURAL_OUT_ROOT:-models/neural_experiments_fluodb}"
NEURAL_COMPARE_OUT="${FLUORCAST_NEURAL_COMPARE_OUT:-outputs/neural_model_experiments_fluodb}"
TREE_COMPARE_DIR="${FLUORCAST_TREE_COMPARE_OUT:-outputs/model_experiments_fluodb}"
mkdir -p "$NEURAL_OUT_ROOT"
mkdir -p "$NEURAL_COMPARE_OUT"

module purge || true
module load python/3.11
module load gcc
module load rdkit

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  source "$FLUORCAST_ACTIVATE"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  echo "WARNING: No virtual environment found; using loaded Python." >&2
fi

echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python scripts/run_neural_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir "$TREE_COMPARE_DIR" \
  --out-root "$NEURAL_OUT_ROOT" \
  --compare-out "$NEURAL_COMPARE_OUT" \
  --models mlp_small,mlp_medium,mlp_large,pytorch_mlp \
  --targets absorption_nm,emission_nm,quantum_yield \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Job completed successfully."
