#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd -P)"
env_dir="${FLUORCAST_ENV_DIR:-$repo_dir/.venv}"

exec "$repo_dir/scripts/remote/provision_environment.sh" \
  --repo-dir "$repo_dir" \
  --env-dir "$env_dir" \
  "$@"
