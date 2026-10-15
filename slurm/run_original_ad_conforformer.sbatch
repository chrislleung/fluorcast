#!/bin/bash
#SBATCH --job-name=ad_cf
#SBATCH --output=outputs/slurm/ad_cf-%j.out
#SBATCH --error=outputs/slurm/ad_cf-%j.err
#SBATCH --time=00:30:00
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
echo "Original Applicability-Domain Case - ConforFormer"
echo "============================================================"
echo "Date:       $(date)"
echo "Host:       $(hostname)"
echo "Git commit: $(git rev-parse HEAD)"
echo "Model root: $MODEL_ROOT"
echo

"$PY" - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
PY

"$PY" scripts/predict_conforformer_case.py \
    --input-csv inputs/case_studies/original_ad_case.csv \
    --output-csv outputs/case_studies/original_ad_conforformer.csv \
    --model-root "$MODEL_ROOT" \
    --split-type molecule \
    --pooling mean \
    --feature-set conforformer_morgan_solvent \
    --device cuda

echo
echo "===== RESULT ====="

"$PY" - <<'PY'
import pandas as pd

df = pd.read_csv(
    "outputs/case_studies/original_ad_conforformer.csv"
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

print(df[[c for c in cols if c in df.columns]].to_string(index=False))
PY
