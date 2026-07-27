#!/bin/bash
# Archived workflow; retained for reproducibility, not routine use.
#SBATCH --job-name=chemfluor
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/slurm/chemfluor_%j.out
#SBATCH --error=outputs/slurm/chemfluor_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.11
module load gcc
module load rdkit

if [[ -z "${FLUORCAST_ACTIVATE:-}" || ! -f "$FLUORCAST_ACTIVATE" ]]; then
  echo "FLUORCAST_ACTIVATE must point to the environment activation script." >&2
  exit 2
fi
# shellcheck disable=SC1090
source "$FLUORCAST_ACTIVATE"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p outputs outputs/models outputs/metrics outputs/plots

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python -c "from rdkit import Chem; print('RDKit test:', Chem.MolFromSmiles('CCO'))"

python -m src.train
