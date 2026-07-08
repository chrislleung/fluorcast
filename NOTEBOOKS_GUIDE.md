# ChemFluor Notebooks Folder Guide

Put these six files inside `FluorCast/notebooks/`.

## Required notebook files

```text
notebooks/
  00_project_overview.ipynb
  01_data_and_features.ipynb
  02_train_models.ipynb
  03_view_results.ipynb
  04_predict_new_molecule.ipynb
  05_batch_prediction.ipynb
```

## 00_project_overview.ipynb

Purpose: non-technical explanation for chemists.

Contents:

- Project goal
- Inputs: SMILES and solvent
- Outputs: emission wavelength, PLQY, bright/dim class
- Feature engineering summary
- Random split vs scaffold split
- Solvent descriptors
- Applicability-domain warnings
- Recommended notebook order

## 01_data_and_features.ipynb

Purpose: show how raw data becomes machine learning features.

Contents:

- Load `chemfluor_data.csv`
- Run `load_raw_data()`
- Run `clean_data()`
- Show cleaned dataset stats
- Run `build_feature_matrix()`
- Show feature matrix shape and feature group counts

## 02_train_models.ipynb

Purpose: submit and monitor Slurm training jobs.

Contents:

- `!sbatch ../slurm/base_models/run_model_experiments_fluodb.sbatch`
- `!squeue -u $USER`
- List recent output logs
- Tail a selected job log

Do not run full training directly inside the notebook.

## 03_view_results.ipynb

Purpose: view model results after training.

Contents:

- Load `outputs/metrics/metrics.csv`
- Display predicted-vs-actual plots
- Display residual plots
- Display PLQY confusion matrix
- Display worst 20 wavelength predictions
- Display worst 20 PLQY predictions
- Preview uncertainty CSVs

## 04_predict_new_molecule.ipynb

Purpose: user-facing notebook for one molecule.

Contents:

- Editable variables: `SMILES`, `SOLVENT`, `NAME`
- Run `python -m src.predict`
- Save prediction report as JSON
- Explain confidence levels

## 05_batch_prediction.ipynb

Purpose: user-facing notebook for candidate lists.

Contents:

- Create or load a CSV with `name`, `SMILES`, and `solvent`
- Run batch prediction
- Display the saved results table

## Common setup cell

Every notebook starts with this cell:

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))
print("Project root:", PROJECT_ROOT)
```

