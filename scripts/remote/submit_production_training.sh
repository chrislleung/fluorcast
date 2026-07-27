#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s --account ACCOUNT --repo-dir PATH --env-activate PATH [--split-type molecule|scaffold] [--state-file PATH]\n' "$0" >&2
}

json_escape() {
  printf '%s' "$1" | python -c 'import json,sys; print(json.dumps(sys.stdin.read())[1:-1])'
}

emit() {
  local status="$1"
  local code="$2"
  local message="$3"
  printf '{"schema_version":1,"status":"%s","code":"%s","message":"%s"}\n' "$status" "$code" "$(json_escape "$message")"
}

emit_state_file() {
  python - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.dumps(json.load(handle), sort_keys=True, separators=(",", ":")))
PY
}

account=""
repo_dir=""
env_activate=""
state_file=""
split_type="molecule"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --account)
      account="${2:-}"
      shift 2
      ;;
    --repo-dir)
      repo_dir="${2:-}"
      shift 2
      ;;
    --env-activate)
      env_activate="${2:-}"
      shift 2
      ;;
    --state-file)
      state_file="${2:-}"
      shift 2
      ;;
    --split-type)
      split_type="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      emit "failed" "INVALID_ARGUMENT" "Unknown argument."
      exit 2
      ;;
  esac
done

if [[ -z "$account" || ! "$account" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{1,63}$ ]]; then
  emit "failed" "INVALID_SLURM_ACCOUNT" "A non-empty Slurm account containing only letters, numbers, dot, underscore, or dash is required."
  exit 2
fi
if [[ -z "$repo_dir" || ! -d "$repo_dir" ]]; then
  emit "failed" "REPO_MISSING" "Repository directory is required."
  exit 2
fi
if [[ -z "$env_activate" || ! -f "$env_activate" ]]; then
  emit "failed" "ENV_ACTIVATE_MISSING" "Environment activation script is required."
  exit 2
fi
if [[ "$split_type" != "molecule" && "$split_type" != "scaffold" ]]; then
  emit "failed" "INVALID_SPLIT_TYPE" "Split type must be molecule or scaffold."
  exit 2
fi

repo_dir="$(cd "$repo_dir" && pwd -P)"
state_root="${XDG_STATE_HOME:-$HOME/.local/state}/fluorcast"
if [[ -z "$state_file" ]]; then
  mkdir -p -m 700 "$state_root"
  chmod 700 "$state_root" 2>/dev/null || true
  state_file="$state_root/provisioning-state.json"
else
  mkdir -p -m 700 "$(dirname "$state_file")"
  chmod 700 "$(dirname "$state_file")" 2>/dev/null || true
fi
state_file="$(python - "$state_file" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
install_state="${state_root}/install-state.json"
mkdir -p -m 700 "$(dirname "$install_state")"
chmod 700 "$(dirname "$install_state")" 2>/dev/null || true
install_state="$(python - "$install_state" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
lock_dir="$state_file.lock"

if ! mkdir "$lock_dir" 2>/dev/null; then
  emit "failed" "PROVISIONING_LOCKED" "Another production training submission is already active."
  exit 1
fi
cleanup() {
  rmdir "$lock_dir" 2>/dev/null || true
}
trap cleanup EXIT

if [[ -f "$state_file" ]]; then
  emit "success" "TRAINING_ALREADY_SUBMITTED" "Using recorded Slurm jobs from provisioning state."
  emit_state_file "$state_file"
  exit 0
fi

cd "$repo_dir"
mkdir -p outputs/slurm models/production

submit_job() {
  local job_id
  job_id="$(sbatch --parsable --account "$account" "$@")"
  if [[ ! "$job_id" =~ ^[0-9]+([_;.][A-Za-z0-9_.-]+)?$ ]]; then
    emit "failed" "SBATCH_INVALID_JOB_ID" "sbatch did not return a parsable job id."
    exit 1
  fi
  printf '%s' "$job_id"
}

common_exports="FLUORCAST_REPO=$repo_dir,FLUORCAST_ACTIVATE=$env_activate"
tree_exports="$common_exports,FLUORCAST_TREE_OUT_ROOT=models/production/tree,FLUORCAST_TREE_COMPARE_OUT=outputs/production/tree"
neural_exports="$common_exports,FLUORCAST_NEURAL_OUT_ROOT=models/production/neural,FLUORCAST_NEURAL_COMPARE_OUT=outputs/production/neural,FLUORCAST_TREE_COMPARE_OUT=outputs/production/tree"
tree_job="$(submit_job --export="$tree_exports" slurm/base_models/run_model_experiments_fluodb.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted tree production training as $tree_job."
neural_job="$(submit_job --dependency="afterok:$tree_job" --export="$neural_exports" slurm/base_models/run_neural_experiments.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted neural production training as $neural_job."

hybrid_abs_exports="$common_exports,FLUORCAST_TARGET_NAME=absorption_nm,FLUORCAST_SPLIT_TYPE=$split_type,FLUORCAST_MODEL_OUT_DIR=models/production/hybrid/absorption_nm,FLUORCAST_OUT_DIR=outputs/production/hybrid/absorption_nm"
hybrid_em_exports="$common_exports,FLUORCAST_TARGET_NAME=emission_nm,FLUORCAST_SPLIT_TYPE=$split_type,FLUORCAST_MODEL_OUT_DIR=models/production/hybrid/emission_nm,FLUORCAST_OUT_DIR=outputs/production/hybrid/emission_nm"
hybrid_qy_exports="$common_exports,FLUORCAST_TARGET_NAME=quantum_yield,FLUORCAST_SPLIT_TYPE=$split_type,FLUORCAST_MODEL_OUT_DIR=models/production/hybrid/quantum_yield,FLUORCAST_OUT_DIR=outputs/production/hybrid/quantum_yield"
hybrid_dependency="afterok:$tree_job:$neural_job"
hybrid_abs_job="$(submit_job --dependency="$hybrid_dependency" --export="$hybrid_abs_exports" slurm/run_hybrid_three_way_experiment.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted hybrid absorption production training as $hybrid_abs_job."
hybrid_em_job="$(submit_job --dependency="$hybrid_dependency" --export="$hybrid_em_exports" slurm/run_hybrid_three_way_experiment.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted hybrid emission production training as $hybrid_em_job."
hybrid_qy_job="$(submit_job --dependency="$hybrid_dependency" --export="$hybrid_qy_exports" slurm/run_hybrid_three_way_experiment.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted hybrid quantum yield production training as $hybrid_qy_job."
validation_exports="$common_exports,FLUORCAST_ARTIFACT_DIR=$repo_dir/models/production,FLUORCAST_INSTALL_STATE=$install_state"
validation_job="$(submit_job --dependency="afterok:$hybrid_abs_job:$hybrid_em_job:$hybrid_qy_job" --export="$validation_exports" slurm/production/validate_production_install.sbatch)"
emit "running" "SBATCH_SUBMITTED" "Submitted production validation as $validation_job."

tmp_state="$state_file.$$.tmp"
python - "$state_file" "$install_state" "$split_type" "$tree_job" "$neural_job" "$hybrid_abs_job" "$hybrid_em_job" "$hybrid_qy_job" "$validation_job" > "$tmp_state" <<'PY'
import json
import sys

_, state_file, install_state, split_type, tree, neural, hybrid_abs, hybrid_em, hybrid_qy, validation = sys.argv
payload = {
    "schema_version": 1,
    "status": "submitted",
    "state_file": state_file,
    "install_state_file": install_state,
    "split_type": split_type,
    "jobs": {
        "tree": tree,
        "neural": neural,
        "hybrid_absorption_nm": hybrid_abs,
        "hybrid_emission_nm": hybrid_em,
        "hybrid_quantum_yield": hybrid_qy,
        "validation": validation,
    },
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
mv "$tmp_state" "$state_file"
emit "success" "TRAINING_SUBMITTED" "Production training and validation jobs were submitted."
emit_state_file "$state_file"
