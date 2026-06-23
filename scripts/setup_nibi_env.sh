#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

module load gcc
module load rdkit

rm -rf .venv
python -m venv --system-site-packages .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
grep -v -E '^rdkit($|[=<> ])' requirements.txt > /tmp/fluorcast_requirements_nibi.txt
pip install -r /tmp/fluorcast_requirements_nibi.txt

python -c "import pandas; print('pandas OK')"
python -c "import sklearn; print('sklearn OK')"
python -c "from rdkit import Chem; print('RDKit OK')"

echo "FluorCast NIBI environment ready."
