#!/bin/bash
#SBATCH --job-name=chemfluor_predict
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/slurm/predict_all_models_%j.out
#SBATCH --error=outputs/slurm/predict_all_models_%j.err

cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs/predictions outputs/slurm

python scripts/predict_all_models.py \
  --smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196 \
  --graph-model-dirs \
    models/graph_gin_emission_3seeds_gpu/seed_0/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_1/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_2/graph_gin \
    models/graph_gcn_emission_3seeds_gpu/seed_0/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_1/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_2/graph_gcn \
    models/graph_gin_qy_gpu/graph_gin \
    models/graph_gcn_qy_gpu/graph_gcn \
  --out outputs/predictions/difficult_benchmark_all_models_with_graphs_and_qy.csv
