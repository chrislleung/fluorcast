#!/bin/bash
#SBATCH --job-name=chemfluor_combined
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/combined_train_%j.out
#SBATCH --error=outputs/combined_train_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project_git

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs
mkdir -p models/chemfluor_combined

echo "Starting ChemFluor + Deep4Chem training at $(date)"
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

python -u scripts/train_combined_predictors.py \
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" \
  --chemfluor data/chemfluor_data.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-dir models/chemfluor_combined \
  --model rf

echo "ChemFluor + Deep4Chem training finished at $(date)"
find models/chemfluor_combined -maxdepth 1 -type f -printf "%f\n" | sort
