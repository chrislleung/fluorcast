#!/bin/bash
#SBATCH --job-name=zhao_morgan
#SBATCH --output=outputs/slurm/zhao_morgan-%j.out
#SBATCH --error=outputs/slurm/zhao_morgan-%j.err
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

set -euo pipefail

cd "$HOME/scratch/FluorCast-conforformer"

module purge
module load StdEnv/2023
module load python/3.12
module load scipy-stack
module load rdkit

PY="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"

MODEL_ROOT="$HOME/scratch/FluorCast-conforformer/models/conforformer_downstream/full_20260806T004836Z/molecule_seed0"

mkdir -p outputs/case_studies
mkdir -p outputs/slurm

echo "============================================================"
echo "Zhao Case Studies - Morgan Only"
echo "============================================================"
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Model root: $MODEL_ROOT"
echo

"$PY" scripts/predict_conforformer_case.py \
    --input-csv inputs/case_studies/zhao_fluorophores.csv \
    --output-csv outputs/case_studies/zhao_morgan_predictions.csv \
    --model-root "$MODEL_ROOT" \
    --split-type molecule \
    --pooling not_applicable \
    --feature-set morgan_solvent \
    --device cpu

echo
echo "===== MORGAN RESULTS ====="

"$PY" - <<'PY'
import pandas as pd

df = pd.read_csv(
    "outputs/case_studies/zhao_morgan_predictions.csv"
)

cols = [
    "name",
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "stokes_shift_nm",
    "derived_stokes_shift_nm",
    "status",
    "error",
]

cols = [c for c in cols if c in df.columns]

print(df[cols].to_string(index=False))
PY
