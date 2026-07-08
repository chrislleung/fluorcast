#!/bin/bash
#SBATCH --job-name=chemfluor
#SBATCH --account=def-yzhao
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/chemfluor_%j.out
#SBATCH --error=outputs/chemfluor_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

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
