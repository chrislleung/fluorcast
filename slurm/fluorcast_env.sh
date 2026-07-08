#!/bin/bash
set -euo pipefail

CANDIDATES=(
  "/scratch/chrisl/ChemFluor_Project/chemfluor_env/bin/python"
  "/scratch/chrisl/ChemFluor_Project/.venv/bin/python"
  "/home/chrisl/scratch/ChemFluor_Project/chemfluor_env/bin/python"
  "/home/chrisl/scratch/ChemFluor_Project/.venv/bin/python"
)

PYTHON=""

for candidate in "${CANDIDATES[@]}"; do
  if [ -x "$candidate" ]; then
    echo "Testing Python candidate: $candidate"
    if "$candidate" -c "import pandas, numpy, sklearn, rdkit; print('imports ok')" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    else
      echo "Candidate failed imports: $candidate"
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: No working FluorCast Python environment found."
  echo "Checked:"
  printf '%s\n' "${CANDIDATES[@]}"
  echo
  echo "Fix one environment manually, for example:"
  echo "  cd /scratch/chrisl/ChemFluor_Project"
  echo "  source .venv/bin/activate"
  echo "  python -m pip install -r requirements.txt"
  echo "  python -c \"import pandas, numpy, sklearn, rdkit; print('imports ok')\""
  exit 1
fi

export PYTHON

echo "Using Python: $PYTHON"
"$PYTHON" --version
"$PYTHON" -c "import pandas, numpy, sklearn, rdkit; print('imports ok')"
