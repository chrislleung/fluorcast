#!/usr/bin/env bash
#SBATCH --job-name=fluorcast_predict
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=outputs/slurm/predict_all_models_%j.out
#SBATCH --error=outputs/slurm/predict_all_models_%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  echo "Submit from the repo root or set FLUORCAST_REPO." >&2
  exit 2
fi

cd "$PROJECT_ROOT"

module purge || true
module load python/3.11
module load gcc
module load rdkit

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  source "$FLUORCAST_ACTIVATE"
elif [[ -f "$HOME/scratch/chemfluor_env/bin/activate" ]]; then
  source "$HOME/scratch/chemfluor_env/bin/activate"
elif [[ -f "$HOME/scratch/fluorcast_env/bin/activate" ]]; then
  source "$HOME/scratch/fluorcast_env/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
else
  echo "WARNING: No virtual environment found; using loaded Python." >&2
fi

mkdir -p outputs/predictions outputs/slurm

python scripts/predict_all_models.py \
  --smiles "${FLUORCAST_SMILES:-O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2}" \
  --solvent-smiles "${FLUORCAST_SOLVENT_SMILES:-CS(=O)C}" \
  --known-emission-nm "${FLUORCAST_KNOWN_EMISSION_NM:-539}" \
  --known-quantum-yield "${FLUORCAST_KNOWN_QY:-0.196}" \
  --tree-model-dir "${FLUORCAST_TREE_MODEL_DIR:-models/experiments_fluodb}" \
  --neural-model-dir "${FLUORCAST_NEURAL_MODEL_DIR:-models/neural_experiments_fluodb}" \
  --graph-model-dirs \
    models/graph_gin_emission_3seeds_gpu/seed_0/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_1/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_2/graph_gin \
    models/graph_gcn_emission_3seeds_gpu/seed_0/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_1/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_2/graph_gcn \
    models/graph_gin_qy_gpu/graph_gin \
    models/graph_gcn_qy_gpu/graph_gcn \
  --out "${FLUORCAST_OUT:-outputs/predictions/difficult_benchmark_all_models_with_graphs_and_qy.csv}"
