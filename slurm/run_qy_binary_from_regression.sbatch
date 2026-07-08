#!/bin/bash
#SBATCH --job-name=qy_bin_reg
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd /scratch/chrisl/ChemFluor_Project

echo "PWD: $(pwd)"
echo "HOST: $(hostname)"
echo "DATE: $(date)"

module purge
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack
module load rdkit

module list

PYTHON=/scratch/chrisl/ChemFluor_Project/.venv/bin/python

echo "Python: $PYTHON"
$PYTHON --version
$PYTHON -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"

echo "Checking for QY regression prediction files..."
find outputs/paper_comparison/predictions -name "quantum_yield__*__*__seed*.csv" | sort | head -n 20

echo "Running thresholded-regression PLQY binary classification verification..."
$PYTHON scripts/manuscript/verify_qy_binary_from_regression.py

echo "DONE: $(date)"
find outputs/paper_comparison/qy_binary_from_regression -type f | sort
