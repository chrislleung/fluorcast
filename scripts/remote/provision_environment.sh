#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s --repo-dir PATH --env-dir PATH [--recreate]\n' "$0" >&2
}

json_event() {
  local status="$1"
  local code="$2"
  local message="$3"
  printf '{"schema_version":1,"status":"%s","code":"%s","message":"%s"}\n' "$status" "$code" "$message"
}

repo_dir=""
env_dir=""
recreate="0"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      repo_dir="${2:-}"
      shift 2
      ;;
    --env-dir)
      env_dir="${2:-}"
      shift 2
      ;;
    --recreate)
      recreate="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      usage
      json_event "failed" "INVALID_ARGUMENT" "Unknown argument."
      exit 2
      ;;
  esac
done

if [[ -z "$repo_dir" || -z "$env_dir" ]]; then
  usage
  json_event "failed" "INVALID_ARGUMENT" "Repository and environment directories are required."
  exit 2
fi
if [[ ! -d "$repo_dir" ]]; then
  json_event "failed" "REPO_MISSING" "Repository directory does not exist."
  exit 2
fi

repo_dir="$(cd "$repo_dir" && pwd -P)"
constraints="$repo_dir/scripts/remote/nibi-python311-constraints.txt"
if [[ ! -f "$constraints" ]]; then
  json_event "failed" "CONSTRAINTS_MISSING" "Pinned constraints file is missing."
  exit 2
fi
if ! command -v module >/dev/null 2>&1; then
  json_event "failed" "MODULE_COMMAND_MISSING" "Environment modules are not available."
  exit 1
fi

tmp_requirements="$(mktemp "${TMPDIR:-/tmp}/fluorcast-requirements.XXXXXX")"
cleanup() {
  rm -f "$tmp_requirements"
}
trap cleanup EXIT

json_event "running" "MODULES_LOADING" "Loading Alliance runtime modules."
module purge
module load python/3.11
module load gcc
module load rdkit

if [[ -d "$env_dir" && "$recreate" != "1" ]]; then
  if [[ ! -f "$env_dir/bin/activate" ]]; then
    json_event "failed" "ENV_EXISTS_INVALID" "Environment directory exists but has no activation script."
    exit 1
  fi
  json_event "running" "ENV_EXISTS" "Reusing existing Python environment."
else
  if [[ -e "$env_dir" && "$recreate" == "1" ]]; then
    json_event "running" "ENV_RECREATE" "Recreating existing Python environment."
    rm -rf "$env_dir"
  fi
  json_event "running" "ENV_CREATE" "Creating Python virtual environment."
  python -m venv --system-site-packages "$env_dir"
fi

# shellcheck disable=SC1090
source "$env_dir/bin/activate"

grep -v -E '^rdkit($|[=<> ])' "$repo_dir/requirements.txt" > "$tmp_requirements"
json_event "running" "PIP_BOOTSTRAP" "Upgrading Python packaging tools."
python -m pip install --upgrade pip setuptools wheel
json_event "running" "PIP_INSTALL" "Installing pinned FluorCast runtime dependencies."
python -m pip install --constraint "$constraints" -r "$tmp_requirements"
json_event "running" "PIP_CHECK" "Checking installed package compatibility."
python -m pip check
json_event "running" "IMPORT_CHECK" "Checking runtime imports."
python - <<'PY'
import importlib
for name in ("numpy", "pandas", "sklearn", "scipy", "xgboost", "lightgbm", "catboost", "rdkit"):
    importlib.import_module(name)
PY
json_event "success" "ENV_READY" "FluorCast Python environment is ready."
