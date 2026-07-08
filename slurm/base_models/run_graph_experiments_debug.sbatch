#!/bin/bash
#SBATCH --job-name=chemfluor_graph_dbg
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=outputs/slurm/chemfluor_graph_debug_%j.out
#SBATCH --error=outputs/slurm/chemfluor_graph_debug_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/graph_experiments_debug
mkdir -p outputs/graph_model_experiments_debug

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

echo "Debug job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

python -u scripts/run_graph_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --neural-compare-dir outputs/neural_model_experiments_fluodb \
  --out-root models/graph_experiments_debug \
  --compare-out outputs/graph_model_experiments_debug \
  --models graph_gcn \
  --targets emission_nm \
  --epochs 2 \
  --max-train-rows 200 \
  --hidden-dim 64 \
  --num-layers 2 \
  --batch-size 64 \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Debug job completed successfully."
ls -lh outputs/graph_model_experiments_debug || true
