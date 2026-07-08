#!/bin/bash
#SBATCH --job-name=debug_models
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=outputs/debug_models_%j.out
#SBATCH --error=outputs/debug_models_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs

python -u scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb_debug \
  --models rf,histgb \
  --targets emission_nm \
  --max-train-rows 5000 \
  --compare-out outputs/model_experiments_fluodb_debug \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196
