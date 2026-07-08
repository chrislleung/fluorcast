#!/bin/bash
#SBATCH --job-name=fluodb_models
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=outputs/fluodb_models_%j.out
#SBATCH --error=outputs/fluodb_models_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs
mkdir -p models/experiments_fluodb

echo "Starting model experiments at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

python -u scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb \
  --models rf,extratrees,histgb,gbdt \
  --targets emission_nm,quantum_yield \
  --compare-out outputs/model_experiments_fluodb \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "Finished model experiments at $(date)"
