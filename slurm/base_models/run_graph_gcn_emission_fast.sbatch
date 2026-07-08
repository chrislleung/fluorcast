#!/bin/bash
#SBATCH --job-name=graph_gcn_em
#SBATCH --partition=cpubase_bycore_b2
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/graph_gcn_emission_fast_%j.out
#SBATCH --error=outputs/slurm/graph_gcn_emission_fast_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/graph_gcn_emission_fast
mkdir -p outputs/graph_gcn_emission_fast

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Fast graph GCN emission job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking RDKit..."
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

echo "Checking PyTorch..."
python -c "import torch; print('Torch OK:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo "Running graph_gcn emission_nm only..."
python -u scripts/run_graph_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --neural-compare-dir outputs/neural_model_experiments_fluodb \
  --out-root models/graph_gcn_emission_fast \
  --compare-out outputs/graph_gcn_emission_fast \
  --models graph_gcn \
  --targets emission_nm \
  --epochs 30 \
  --patience 6 \
  --batch-size 256 \
  --hidden-dim 128 \
  --num-layers 3 \
  --dropout 0.2 \
  --learning-rate 1e-3 \
  --weight-decay 1e-4 \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Fast graph GCN emission job completed."

echo "Outputs:"
ls -lh outputs/graph_gcn_emission_fast || true
