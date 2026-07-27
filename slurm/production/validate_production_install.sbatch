#!/usr/bin/env bash
#SBATCH --job-name=fluorcast-validate
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_ROOT="${FLUORCAST_REPO:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$PROJECT_ROOT" || ! -d "$PROJECT_ROOT" ]]; then
  echo "Could not locate the FluorCast repository." >&2
  exit 2
fi
if [[ -z "${FLUORCAST_ACTIVATE:-}" || ! -f "$FLUORCAST_ACTIVATE" ]]; then
  echo "FLUORCAST_ACTIVATE must point to the environment activation script." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
module purge
module load python/3.11
module load gcc
module load rdkit
# shellcheck disable=SC1090
source "$FLUORCAST_ACTIVATE"

python scripts/remote/validate_production_install.py \
  --repo-dir "$PROJECT_ROOT" \
  --artifact-dir "${FLUORCAST_ARTIFACT_DIR:-$PROJECT_ROOT/models}" \
  --state-file "${FLUORCAST_INSTALL_STATE:-$PROJECT_ROOT/install-state.json}"
