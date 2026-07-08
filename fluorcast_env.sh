#!/usr/bin/env bash
set -euo pipefail

# Helper for finding a working FluorCast Python environment on Nibi.
# This file can be executed:
#   bash fluorcast_env.sh
# or sourced:
#   source fluorcast_env.sh

FLUORCAST_REPO="${FLUORCAST_REPO:-$HOME/scratch/FluorCast}"

CANDIDATES=(
  "${FLUORCAST_PYTHON:-}"
  "${FLUORCAST_REPO}/.venv/bin/python"
  "$HOME/scratch/chemfluor_env/bin/python"
  "$HOME/scratch/fluorcast_env/bin/python"
  "/scratch/${USER}/FluorCast/.venv/bin/python"
  "/scratch/${USER}/chemfluor_env/bin/python"
  "/scratch/${USER}/fluorcast_env/bin/python"
  "/home/${USER}/scratch/FluorCast/.venv/bin/python"
  "/home/${USER}/scratch/chemfluor_env/bin/python"
  "/home/${USER}/scratch/fluorcast_env/bin/python"
)

PYTHON=""

for candidate in "${CANDIDATES[@]}"; do
  if [[ -z "$candidate" ]]; then
    continue
  fi

  if [[ -x "$candidate" ]]; then
    echo "Testing Python candidate: $candidate"
    if "$candidate" -c "import pandas, numpy, sklearn, rdkit; print('imports ok')" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    else
      echo "Candidate failed imports: $candidate"
    fi
  fi
done

if [[ -z "$PYTHON" ]]; then
  echo "ERROR: No working FluorCast Python environment found." >&2
  echo "Checked:" >&2
  printf '  %s\n' "${CANDIDATES[@]}" >&2
  echo >&2
  echo "Fix one environment manually, for example:" >&2
  echo "  cd ~/scratch/FluorCast" >&2
  echo "  module purge" >&2
  echo "  module load python/3.11" >&2
  echo "  module load gcc" >&2
  echo "  module load rdkit" >&2
  echo "  python -m venv --system-site-packages ~/scratch/chemfluor_env" >&2
  echo "  source ~/scratch/chemfluor_env/bin/activate" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  echo "  python -c \"import pandas, numpy, sklearn, rdkit; print('imports ok')\"" >&2

  if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    return 1
  else
    exit 1
  fi
fi

export PYTHON
export FLUORCAST_REPO

echo "Using FluorCast repo: $FLUORCAST_REPO"
echo "Using Python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"
