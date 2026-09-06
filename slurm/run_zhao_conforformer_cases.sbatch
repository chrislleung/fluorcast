#!/bin/bash
#SBATCH --job-name=zhao_cf
#SBATCH --output=outputs/slurm/zhao_cf-%j.out
#SBATCH --error=outputs/slurm/zhao_cf-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=h100:1

set -euo pipefail

cd "$HOME/scratch/FluorCast-conforformer"

module purge
module load StdEnv/2023
module load python/3.12
module load scipy-stack
module load rdkit

PY="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"

MODEL_ROOT="$HOME/scratch/FluorCast-conforformer/models/conforformer_downstream/full_20260806T004836Z/molecule_seed0_hybrid_mean"

mkdir -p outputs/case_studies
mkdir -p outputs/slurm

echo "============================================================"
echo "FluorCast-ConforFormer - Zhao Case Studies"
echo "============================================================"
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Python:     $("$PY" --version)"
echo
echo "Model root:"
echo "$MODEL_ROOT"
echo

echo "Checking trained model artifacts..."
find "$MODEL_ROOT/molecule/mean/conforformer_morgan_solvent" \
    -type f \
    \( -name 'model.joblib' -o -name 'feature_metadata.json' \) \
    -print | sort

echo

"$PY" - <<'PY'
import torch
import rdkit

print("PyTorch:", torch.__version__)
print("RDKit:", rdkit.__version__)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside GPU allocation")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "Starting inference..."
echo

"$PY" scripts/predict_conforformer_case.py \
    --input-csv inputs/case_studies/zhao_fluorophores.csv \
    --output-csv outputs/case_studies/zhao_conforformer_predictions.csv \
    --model-root "$MODEL_ROOT" \
    --split-type molecule \
    --pooling mean \
    --feature-set conforformer_morgan_solvent \
    --device cuda

echo
echo "Inference finished."
echo

"$PY" - <<'PY'
import pandas as pd

p = "outputs/case_studies/zhao_conforformer_predictions.csv"
df = pd.read_csv(p)

wanted = [
    "name",
    "absorption_nm",
    "emission_nm",
    "quantum_yield",
    "stokes_shift_nm",
    "derived_stokes_shift_nm",
    "stokes_source",
    "status",
    "error",
]

wanted = [c for c in wanted if c in df.columns]

print(df[wanted].to_string(index=False))
PY
