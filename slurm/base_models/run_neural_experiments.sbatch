#!/bin/bash
#SBATCH --job-name=chemfluor_neural
#SBATCH --account=def-yzhao
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/chemfluor_neural_%j.out
#SBATCH --error=outputs/slurm/chemfluor_neural_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/neural_experiments_fluodb
mkdir -p outputs/neural_model_experiments_fluodb

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python scripts/run_neural_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --out-root models/neural_experiments_fluodb \
  --compare-out outputs/neural_model_experiments_fluodb \
  --models mlp_small,mlp_medium,mlp_large,pytorch_mlp \
  --targets emission_nm,quantum_yield \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Job completed successfully."
