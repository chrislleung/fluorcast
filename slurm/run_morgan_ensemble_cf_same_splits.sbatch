#!/bin/bash
#SBATCH --job-name=mg_cf_ensemble
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=outputs/slurm/mg_cf_ensemble-%A_%a.out
#SBATCH --error=outputs/slurm/mg_cf_ensemble-%A_%a.err

set -euo pipefail

module purge
module load StdEnv/2023
module load python/3.11
module load scipy-stack
module load rdkit

source "$HOME/scratch/venvs/fluorcast-cf-benchmark/bin/activate"

cd "$HOME/scratch/FluorCast-conforformer"

echo "Python: $(which python)"
python --version

python - <<'PY'
import numpy
import pandas
import sklearn
import joblib
from rdkit import Chem

print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("joblib:", joblib.__version__)
print("SLURM ENVIRONMENT CHECK: PASS")
PY

RUNS=(
    "random_seed0"
    "molecule_seed0"
    "scaffold_seed0"
)

RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"

ROOT="outputs/conforformer/downstream/full_20260806T004836Z"

echo
echo "============================================================"
echo "Morgan ensemble on exact ConforFormer split"
echo "Run: $RUN"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Host: $(hostname)"
echo "Started: $(date)"
echo "============================================================"

python scripts/run_morgan_ensemble_on_cf_splits.py \
    --run-dir "$ROOT/$RUN" \
    --embedding-run-root \
        "outputs/conforformer/full_dataset/full_20260806T004836Z" \
    --seed 0 \
    --n-jobs "$SLURM_CPUS_PER_TASK"

echo
echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
