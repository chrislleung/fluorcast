#!/bin/bash
#SBATCH --job-name=chemfluor_fluodb
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/fluodb_train_%j.out
#SBATCH --error=outputs/fluodb_train_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project_git

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs
mkdir -p models/chemfluor_combined_fluodb

echo "Starting FluoDB combined training at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking RDKit..."
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

echo "Checking input files..."
ls -lh data/processed/fluodb_lite/combined_deduplicated.csv
ls -lh data/solvent_descriptors_expanded_deep4chem.csv

echo "Running training..."
python -u scripts/train_combined_predictors.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-dir models/chemfluor_combined_fluodb \
  --model rf

echo "Training finished at $(date)"
echo "Model output files:"
find models/chemfluor_combined_fluodb -maxdepth 1 -type f -printf "%f\n" | sort
