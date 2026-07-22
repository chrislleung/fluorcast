#!/usr/bin/env bash
# Minimal real UniProp checkpoint compatibility gate on Nibi.
# Submit with:
# sbatch --time=00:10:00 --mem=16G --cpus-per-task=2 --gpus-per-node=1 slurm/uniprop/run_uniprop_real_checkpoint_gate.sbatch
#SBATCH --job-name=uniprop_real_gate
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --output=uniprop_real_gate_%j.out
#SBATCH --error=uniprop_real_gate_%j.err

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-$HOME/scratch/FluorCast}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "ERROR: Could not locate FluorCast repository at PROJECT_ROOT=$PROJECT_ROOT" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p outputs/slurm

module purge || true
module load python/3.10
module load gcc

if [[ -n "${FLUORCAST_ACTIVATE:-}" ]]; then
  source "$FLUORCAST_ACTIVATE"
elif [[ -f ".venv-uniprop/bin/activate" ]]; then
  source ".venv-uniprop/bin/activate"
elif [[ -f "$HOME/scratch/fluorcast_uniprop_env/bin/activate" ]]; then
  source "$HOME/scratch/fluorcast_uniprop_env/bin/activate"
else
  echo "ERROR: No UniProp virtual environment found. Set FLUORCAST_ACTIVATE." >&2
  exit 2
fi

CHECKPOINT_DIR="${FLUORCAST_UNIPROP_CHECKPOINT_DIR:-$SCRATCH/fluorcast_uniprop_checkpoints}"
CHECKPOINT_ID="${FLUORCAST_UNIPROP_REAL_GATE_CHECKPOINT:-uniprop_rdkit_to_dft_implicit.pt}"
OUT_DIR="${FLUORCAST_UNIPROP_REAL_GATE_OUT:-outputs/uniprop_real_checkpoint_gate/${SLURM_JOB_ID:-local}}"
FEATURE_SCHEMA_PATH="${FLUORCAST_UNIPROP_FEATURE_SCHEMA:-configs/uniprop/feature_schema.json}"
DEVICE="${FLUORCAST_UNIPROP_REAL_GATE_DEVICE:-cuda}"
EXPECTED_MD5="c87305171142e1c0898a0e2b67a7236a"

echo "UniProp real checkpoint gate on $(hostname)"
echo "Start time: $(date)"
echo "Project root: $(pwd)"
echo "Python executable: $(which python)"
python --version
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Checkpoint id: $CHECKPOINT_ID"
echo "Feature schema: $FEATURE_SCHEMA_PATH"
echo "Output dir: $OUT_DIR"

python -c "import hashlib, pathlib, sys; p=pathlib.Path(sys.argv[1]); expected=sys.argv[2]; actual=hashlib.md5(p.read_bytes()).hexdigest(); print(f'Checkpoint MD5: {actual}'); raise SystemExit(0 if actual == expected else 1)" "$CHECKPOINT_DIR/$CHECKPOINT_ID" "$EXPECTED_MD5"

python scripts/run_uniprop_real_checkpoint_gate.py \
  --audit-only \
  --device "$DEVICE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-id "$CHECKPOINT_ID" \
  --feature-schema "$FEATURE_SCHEMA_PATH" \
  --output-dir "$OUT_DIR"

python -c "import json, sys; p=json.load(open(sys.argv[1], encoding='utf-8')); ready=bool(p.get('environment_ready')); print(f'Environment ready: {ready}'); raise SystemExit(0 if ready else 1)" "$OUT_DIR/environment_audit.json"

python scripts/run_uniprop_real_checkpoint_gate.py \
  --device "$DEVICE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --checkpoint-id "$CHECKPOINT_ID" \
  --feature-schema "$FEATURE_SCHEMA_PATH" \
  --output-dir "$OUT_DIR" \
  --overwrite

test -s "$OUT_DIR/environment_audit.json"
test -s "$OUT_DIR/summary.json"

echo "Summary JSON: $OUT_DIR/summary.json"
echo "UniProp real checkpoint gate finished at $(date)"
