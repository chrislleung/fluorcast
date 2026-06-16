# ChemFluor Molecular Property Pipeline

ChemFluor is a machine-learning workflow for predicting photophysical properties of organic fluorescent molecules from molecular structure and solvent information. The project began as a ChemFluor-style reproduction pipeline and has now been extended into a larger solvent-aware fluorescence modeling workflow using **ChemFluor**, **Deep4Chem**, and **FluoDB-Lite**.

The current workflow supports two related tasks:

```text
molecule + solvent → predicted optical properties
```

and first-pass candidate screening:

```text
target emission + solvent + candidate molecules → ranked candidate fluorophores
```

This is not full neural molecular generation. The current inverse-design step uses rule-based scaffold enumeration, then trained machine-learning models rank generated candidates by predicted emission, predicted quantum yield, estimated brightness, and applicability-domain similarity.

---

## Project Overview

The model predicts:

* **Absorption wavelength** in nanometers
* **Emission wavelength** in nanometers, which corresponds to fluorescence color
* **Quantum yield / PLQY**, which corresponds to brightness
* **Lifetime**, when available
* **Log extinction coefficient**, when available
* **Bright/dim class** in the original ChemFluor workflow, using `PLQY > 0.25`

Inputs:

```text
chromophore SMILES
solvent or solvent SMILES
```

Core feature types:

* Morgan fingerprints
* MACCS keys
* RDKit molecular descriptors
* solvent identity features
* solvent physical descriptors
* RDKit-derived solvent descriptors
* applicability-domain similarity checks

The project supports:

* original ChemFluor training
* random-split evaluation
* scaffold-split evaluation
* grouped chromophore split for combined datasets
* ChemFluor + Deep4Chem training
* ChemFluor + Deep4Chem + FluoDB-Lite training
* candidate generation
* candidate screening for target wavelengths
* candidate-screening summaries
* Compute Canada / Nibi Slurm training

---

## Current Development Summary

### Original ChemFluor Workflow

The original ChemFluor workflow evaluates both random splits and scaffold splits.

Approximate best results after feature expansion and solvent descriptors:

| Workflow | Split type | Main emission/wavelength MAE | Interpretation |
|---|---:|---:|---|
| ChemFluor-only baseline | Random split | ~22 nm | First working fluorescence predictor |
| Improved ChemFluor workflow | Random split | ~16.7 nm | Strong interpolation performance |
| Improved ChemFluor workflow | Scaffold split | ~31–32 nm | More realistic unseen-scaffold test |

Random split performance is optimistic because similar molecules can appear in both train and test. Scaffold split is harder because molecules with the same Bemis-Murcko scaffold are kept together.

### Combined Dataset Workflow

The combined workflow uses a grouped split by canonical chromophore SMILES. This prevents the exact same chromophore from appearing in both training and test sets, even if it appears in multiple solvents.

| Workflow | Split type | Emission MAE | Interpretation |
|---|---:|---:|---|
| ChemFluor + Deep4Chem | Grouped by chromophore | ~31.4 nm | Larger chemical space made prediction harder |
| ChemFluor + Deep4Chem + FluoDB-Lite | Grouped by chromophore | **23.4 nm** | FluoDB-Lite improved broad chemical-space prediction |
| FluoDB-expanded red/NIR region | Wavelength-region analysis | **39.2 nm** | Red/NIR remains the hardest region |

Current FluoDB-expanded Random Forest model metrics:

```text
absorption_nm MAE:     20.19 nm
emission_nm MAE:       23.38 nm
quantum_yield MAE:     0.151
lifetime_ns MAE:       4.32 ns
log_extinction MAE:    0.397
```

The current 600 nm candidate-screening run still ranks the best candidates around **560–564 nm**, which suggests that the next bottleneck is not only training data, but also **candidate-library coverage**. The next development step is to expand the candidate generator with more red-shifted fluorophore families such as BODIPY-like, cyanine-like, rhodamine-like, fluorescein-like, and larger donor-acceptor scaffolds.

---

## Data Sources

### ChemFluor

The original ChemFluor dataset is publicly available on Figshare:

* Dataset: **ChemFluor**
* DOI: `10.6084/m9.figshare.12110619`
* Figshare page: https://figshare.com/articles/dataset/ChemFluor/12110619
* License: **CC BY 4.0**
* Authors listed on Figshare: Cheng-Wei Ju, Rizhang Liu, Bo Li, Hanzhi Bai

Expected path:

```text
data/chemfluor_data.csv
```

The original workflow also supports older root-level paths:

```text
chemfluor_data.csv
solvent_descriptors.csv
```

### Deep4Chem

Deep4Chem adds a larger chromophore dataset with absorption, emission, lifetime, quantum yield, and extinction-coefficient information.

Expected path:

```text
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
```

### FluoDB-Lite

FluoDB-Lite is an aggregated public fluorophore dataset from the FLAME/FluoDB work. It contains fluorophore SMILES, solvent information, photophysical properties, scaffold/category labels, source labels, and references.

Expected path:

```text
data/raw/fluodb/FluoDB-Lite.csv
```

Important: FluoDB-Lite includes rows sourced from datasets such as Deep4Chem and ChemFluor. It should not be blindly appended to the existing training set. The workflow first analyzes FluoDB-Lite, standardizes all datasets into a shared schema, reports overlap, removes exact duplicate measurements, and then trains from a deduplicated combined file.

---

## Repository Structure

```text
ChemFluor_Project/
  README.md
  NOTEBOOKS_GUIDE.md
  requirements.txt
  run_chemfluor.sh
  run_fluodb_training.sh
  example_candidates.csv
  data/
    chemfluor_data.csv
    solvent_descriptors.csv
    solvent_descriptors_expanded_deep4chem.csv
    raw/
      deep4chem/
        DB for chromophore_Sci_Data_rev03.csv
      fluodb/
        FluoDB-Lite.csv
    processed/
      fluodb_lite/
        combined_deduplicated.csv
    generated_candidates/
      scaffold_candidates.csv
  scripts/
    analyze_deep4chem_dataset.py
    make_deep4chem_solvent_descriptors.py
    analyze_fluodb_lite.py
    prepare_fluodb_lite.py
    train_combined_predictors.py
    report_combined_model_results.py
    compare_model_results.py
    analyze_prediction_errors.py
    generate_scaffold_candidates.py
    screen_candidate_molecules.py
    summarize_candidate_screening.py
  src/
    config.py
    data.py
    features.py
    splitting.py
    models.py
    evaluate.py
    plots.py
    train.py
    predict.py
    applicability.py
    chemfluor/
      data_standardization.py
  notebooks/
  outputs/
  models/
  tests/
```

Generated model files, reports, plots, Slurm logs, processed FluoDB files, and raw FluoDB files should usually remain untracked by Git.

Recommended `.gitignore` entries:

```gitignore
models/
outputs/
*.joblib
*.pkl
*.pickle
*.out
*.err
__pycache__/
.pytest_cache/
.venv/
```

---

## Installation

### Local Installation

Install Python 3.11

```bash
winget install Python.Python.3.11
py -3.11 --version
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest matplotlib scipy
```

If RDKit is difficult to install with pip, use conda:

```bash
conda install -c conda-forge rdkit
```

Recommended Python version: **Python 3.11**.
This project uses scientific Python packages such as RDKit, LightGBM, XGBoost, CatBoost, scikit-learn, NumPy, and pandas. These are more reliable on stable Python versions such as 3.11.

---

## Required Files *No Action Needed

For the original ChemFluor workflow:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
```

For the expanded ChemFluor + Deep4Chem + FluoDB-Lite workflow:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
data/raw/fluodb/FluoDB-Lite.csv
```

After FluoDB-Lite preprocessing, full FluoDB-expanded training uses:

```text
data/processed/fluodb_lite/combined_deduplicated.csv
```

---

## Feature Engineering

### Molecular Features

The molecular structure is represented using:

* Morgan fingerprints, radius 2, 2048 bits
* MACCS keys
* RDKit molecular descriptors

RDKit descriptors include molecular weight, logP, hydrogen-bond donors/acceptors, topological polar surface area, ring count, rotatable bonds, fraction sp3 carbons, heavy atom count, aromatic rings, molecular refractivity, BalabanJ, and Bertz complexity.

### Solvent Features

Solvent information is represented using:

* one-hot encoded solvent identity
* optional physical solvent descriptors from `solvent_descriptors.csv`
* RDKit-derived solvent descriptors in the expanded workflow

---

## Train-Test Splits

### Random Split

Random split is an 80/20 train-test split. It is useful for baseline comparison, but it can overestimate performance when similar molecules appear in both the training and test sets.

### Scaffold Split

The scaffold split uses Bemis-Murcko scaffolds. Molecules with the same scaffold are kept together so that the same scaffold does not appear in both training and test sets.

### Grouped Chromophore Split

The combined ChemFluor + Deep4Chem + FluoDB workflow uses a grouped split by canonical chromophore SMILES. This prevents the exact same chromophore from appearing in both train and test. It is stricter than a random row split but not as strict as a Bemis-Murcko scaffold split.

---

# Original ChemFluor Workflow

## Models

The pipeline compares several machine learning models.

### Regression Models

Regression models are used for emission wavelength and PLQY value prediction:

- LightGBM
- Random Forest
- Extra Trees
- Gradient Boosting
- Support Vector Regression
- XGBoost, if installed
- CatBoost, if installed

The three best regressors by validation MAE are also averaged as a simple ensemble.

### Classification Models

Classification models are used for PLQY bright/dim prediction:

- LightGBM
- Random Forest
- Extra Trees
- XGBoost, if installed
- CatBoost, if installed

---

## Metrics

### Regression Metrics

Regression models are evaluated using:

- MAE, mean absolute error
- RMSE, root mean squared error
- R², coefficient of determination
- Spearman rank correlation

### Classification Metrics

Classification models are evaluated using:

- accuracy
- precision
- recall
- F1 score
- confusion matrix

---

## Training

### Local Training

From the project root:

```bash
python -m src.train
```

or:

```bash
python src/train.py
```

The training pipeline will:

1. Load the dataset.
2. Clean the data.
3. Canonicalize SMILES.
4. Merge duplicate molecule-solvent pairs.
5. Build molecular and solvent features.
6. Train models.
7. Evaluate random and scaffold splits.
8. Save metrics, plots, models, and metadata.

---

## Running Original ChemFluor Training on Compute Canada / Nibi

This project was developed and tested on the Digital Research Alliance of Canada / Compute Canada Nibi cluster.

This section is for training the **original ChemFluor-only workflow**, not the expanded ChemFluor + Deep4Chem + FluoDB workflow.

### 1. Log in to Nibi

From your local terminal:

```bash
ssh <your_username>@nibi.alliancecan.ca
```

Example:

```bash
ssh johndoe@nibi.alliancecan.ca
```

### 2. Go to the Project Folder

If the project already exists on Nibi:

```bash
cd ~/scratch/ChemFluor_Project
```

If setting it up for the first time from GitHub:

```bash
cd ~/scratch
git clone https://github.com/chrislleung/ChemFluor.git ChemFluor_Project
cd ChemFluor_Project
```

### 3. Add Required Data Files

Place the following files in `data/` for the current project layout, or in the project root for the older layout:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
```

After setup, the project should look like:

```text
ChemFluor_Project/
  data/
    chemfluor_data.csv
    solvent_descriptors.csv
  run_chemfluor.sh
  src/
  notebooks/
```

### 4. Load Required Nibi Modules

RDKit should be loaded through the Nibi module system rather than installed with pip:

```bash
module purge
module load python/3.11
module load gcc
module load rdkit
```

Check the Python version:

```bash
python --version
```

### 5. Create and Activate the Python Environment

If the environment does not exist yet, create it with system site packages so the virtual environment can see module-provided RDKit:

```bash
python -m venv --system-site-packages ~/scratch/chemfluor_env
source ~/scratch/chemfluor_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If the environment already exists:

```bash
source ~/scratch/chemfluor_env/bin/activate
```

Check that RDKit and the machine learning packages work:

```bash
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
python -c "import lightgbm, xgboost, catboost; print('ML packages ready')"
```

If optional packages such as XGBoost or CatBoost are not installed, the pipeline can still run with the installed models.

### 6. Submit the Training Job

Training should be run through Slurm, not directly on the login node.

Submit the job:

```bash
sbatch run_chemfluor.sh
```

Check job status:

```bash
squeue -u $USER
```

When the job starts, watch the output using the job ID:

```bash
tail -f outputs/chemfluor_<JOBID>.out
```

Example:

```bash
tail -f outputs/chemfluor_15656967.out
```

To exit `tail` without cancelling the job:

```text
Ctrl + C
```

### 7. View Saved Outputs

After training finishes, outputs are saved in:

```text
outputs/metrics/
outputs/plots/
outputs/models/
outputs/predictions/
```

Important files include:

```text
outputs/metrics/metrics.csv
outputs/metrics/metrics.json
outputs/models/best_wavelength_lightgbm.pkl
outputs/models/best_plqy_lightgbm.pkl
outputs/models/best_plqy_classifier.pkl
outputs/models/feature_artifacts.pkl
outputs/models/inference_metadata.pkl
```

### 8. Update the Nibi Copy from GitHub

If the GitHub repository has been updated, pull the newest code on Nibi:

```bash
cd ~/scratch/ChemFluor_Project
git pull origin main
```

Then rerun training if the code, features, or model logic changed:

```bash
sbatch run_chemfluor.sh
```

### Important Nibi Note

Do not run the full training pipeline directly on the login node with:

```bash
python -m src.train
```

That command is acceptable only for quick debugging. Full training should be submitted with:

```bash
sbatch run_chemfluor.sh
```

The Slurm script requests compute resources, loads the correct modules, activates the Python environment, and runs:

```bash
python -u -m src.train
```

---

## Outputs

The pipeline saves outputs to:

```text
outputs/
```

Important output files include:

```text
outputs/metrics/metrics.json
outputs/metrics/metrics.csv
outputs/metrics/wavelength_uncertainty.csv
outputs/metrics/plqy_uncertainty.csv
outputs/metrics/wavelength_feature_importance.csv
outputs/metrics/plqy_feature_importance.csv
outputs/plots/*.png
outputs/plots/worst_20_wavelength_predictions.csv
outputs/plots/worst_20_plqy_predictions.csv
outputs/models/*.pkl
outputs/models/feature_artifacts.pkl
outputs/models/inference_metadata.pkl
outputs/predictions/*.json
outputs/predictions/*.csv
```

Plots include:

- predicted vs actual wavelength
- predicted vs actual PLQY
- residual plots
- error by solvent
- PLQY classifier confusion matrix

---

## Predicting New Molecules

After training, run prediction from the project root:

```bash
python -m src.predict --smiles "CCO" --solvent "MeOH"
```

You can include a candidate name and save the full report as JSON:

```bash
python -m src.predict \
  --smiles "c1ccccc1" \
  --solvent "MeCN" \
  --name "candidate_1" \
  --output outputs/predictions/candidate_1_prediction.json
```

For batch prediction, make a CSV with required columns:

```text
SMILES
solvent
```

A `name` column is optional.

Example batch file:

```csv
name,SMILES,solvent
benzene_example,c1ccccc1,MeCN
ethanol_example,CCO,MeOH
triethylamine_example,CCN(CC)CC,MeCN
```

Run batch prediction:

```bash
python -m src.predict \
  --csv example_candidates.csv \
  --save-csv outputs/predictions/batch_predictions.csv
```

# Expanded ChemFluor + Deep4Chem + FluoDB-Lite Workflow

## Workflow Overview

The full expanded workflow is:

```text
1. Analyze Deep4Chem
2. Build expanded solvent descriptors
3. Analyze FluoDB-Lite
4. Prepare and deduplicate ChemFluor + Deep4Chem + FluoDB-Lite
5. Train combined models
6. Generate model reports
7. Analyze prediction errors
8. Generate scaffold-based candidate molecules
9. Screen candidates for target emission wavelengths
10. Summarize candidate-screening results
```

---

## 1. Analyze Deep4Chem

```powershell
python scripts/analyze_deep4chem_dataset.py `
  --input "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
```

Default output:

```text
outputs/deep4chem_analysis/
```

Generated files include:

```text
deep4chem_summary.txt
top_solvents.csv
missing_values.csv
numeric_summary.csv
invalid_solvents.csv
```

---

## 2. Build Expanded Solvent Descriptors

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv
```

Optional report path:

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv `
  --report outputs/deep4chem_analysis/solvent_descriptor_report.txt
```

---

## 3. Analyze FluoDB-Lite

Analyze the raw FluoDB-Lite file before adding it to the combined training data:

```powershell
python scripts/analyze_fluodb_lite.py `
  --input data/raw/fluodb/FluoDB-Lite.csv `
  --out-dir outputs/fluodb_lite_analysis
```

Important outputs:

```text
outputs/fluodb_lite_analysis/fluodb_lite_summary.txt
outputs/fluodb_lite_analysis/source_counts.csv
outputs/fluodb_lite_analysis/scaffold_counts.csv
outputs/fluodb_lite_analysis/top_red_scaffolds_ge600.csv
outputs/fluodb_lite_analysis/top_red_sources_ge600.csv
```

Use these files to inspect target coverage, source overlap, scaffold distribution, and red/orange/NIR representation.

---

## 4. Prepare and Deduplicate FluoDB-Lite

```powershell
python scripts/prepare_fluodb_lite.py `
  --fluodb data/raw/fluodb/FluoDB-Lite.csv `
  --chemfluor data/chemfluor_data.csv `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --out-dir data/processed/fluodb_lite
```

Generated outputs:

```text
data/processed/fluodb_lite/chemfluor_standardized.csv
data/processed/fluodb_lite/deep4chem_standardized.csv
data/processed/fluodb_lite/fluodb_lite_standardized.csv
data/processed/fluodb_lite/combined_before_dedup.csv
data/processed/fluodb_lite/combined_deduplicated.csv
data/processed/fluodb_lite/overlap_report.md
data/processed/fluodb_lite/overlap_summary.json
data/processed/fluodb_lite/red_region_summary.csv
data/processed/fluodb_lite/molecule_solvent_replicates.csv
```

Exact duplicate measurements are defined as rows with the same canonical chromophore SMILES, canonical solvent SMILES, absorption, emission, quantum yield, and log extinction. Source priority is:

```text
ChemFluor first
Deep4Chem second
FluoDB-Lite third
```

Molecule-solvent pairs with different reported values are reported as replicates rather than collapsed automatically.

---

## 5. Train Combined Models Locally

### ChemFluor + Deep4Chem

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined `
  --model rf
```

### ChemFluor + Deep4Chem + FluoDB-Lite

```powershell
python scripts/train_combined_predictors.py `
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined_fluodb `
  --model rf
```

The FluoDB-expanded dataset is large. Local training can slow down a laptop, so full training is recommended on Compute Canada / Nibi.

Useful options:

```text
--model rf
--model histgb
--n-bits 2048
--radius 2
--standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv
```

Model outputs include:

```text
metrics.json
feature_metadata.json
combined_standardized_training_rows.csv
combined_modeling_rows_after_feature_merge.csv
predictions_absorption_nm.csv
predictions_emission_nm.csv
predictions_lifetime_ns.csv
predictions_quantum_yield.csv
predictions_log_extinction.csv
*_rf.joblib or *_histgb.joblib
```

These outputs are generated artifacts and should not be committed to GitHub.

---

# Transferring Prepared Data to Compute Canada / Nibi

After analyzing and preparing the datasets locally, move the large training job to Compute Canada / Nibi.

Recommended workflow:

```text
1. Analyze and prepare datasets locally.
2. Commit/push only source code and documentation to GitHub.
3. Clone or pull the GitHub repo on Nibi.
4. Transfer only required data files from the local machine to Nibi.
5. Train the model on Nibi with Slurm.
6. Copy selected trained results back to the local machine if needed.
```

Do not commit generated model folders, reports, raw FluoDB, or processed FluoDB artifacts unless intentionally archiving a release.

---

## Files Needed on Nibi

Different training modes require different files.

### Original ChemFluor-only training

For the original ChemFluor workflow on Nibi, only these files are required:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
```

This mode runs the original `src.train` pipeline, evaluates random and scaffold splits, and writes outputs under `outputs/`.

### ChemFluor + Deep4Chem training

For combined ChemFluor + Deep4Chem training on Nibi, these files are required:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
```

### ChemFluor + Deep4Chem + FluoDB-Lite training

For FluoDB-expanded training on Nibi, these files should exist:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
data/raw/fluodb/FluoDB-Lite.csv
data/processed/fluodb_lite/combined_deduplicated.csv
```

The most important training input for the FluoDB-expanded model is:

```text
data/processed/fluodb_lite/combined_deduplicated.csv
```

If this file already exists locally after analysis/preparation, it can be transferred directly to Nibi rather than regenerated there.

---

## Transfer from Windows to Nibi

Run these commands from **Windows PowerShell**, not from the Nibi terminal.

Go to the local project folder:

```powershell
cd "C:\Users\CL\OneDrive\Desktop\python\ChemFluor_Project"
```

Transfer the full local `data/` folder:

```powershell
scp -r data chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/
```

Or transfer only specific files/folders:

```powershell
scp data/chemfluor_data.csv chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/
scp data/solvent_descriptors.csv chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/
scp data/solvent_descriptors_expanded_deep4chem.csv chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/
scp -r data/raw/deep4chem chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/raw/
scp -r data/raw/fluodb chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/raw/
scp -r data/processed/fluodb_lite chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/data/processed/
```

If the destination directories do not exist yet, create them on Nibi first:

```bash
cd ~/scratch/ChemFluor_Project
mkdir -p data/raw/deep4chem
mkdir -p data/raw/fluodb
mkdir -p data/processed/fluodb_lite
```

---

## Verify Files on Nibi

Log in:

```bash
ssh chrisl@nibi.alliancecan.ca
```

Go to the project:

```bash
cd ~/scratch/ChemFluor_Project
```

Verify original ChemFluor-only files:

```bash
ls -lh data/chemfluor_data.csv
ls -lh data/solvent_descriptors.csv
```

Verify expanded workflow files, if using Deep4Chem or FluoDB-Lite:

```bash
ls -lh data/solvent_descriptors_expanded_deep4chem.csv
ls -lh "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
ls -lh data/raw/fluodb/FluoDB-Lite.csv
ls -lh data/processed/fluodb_lite/combined_deduplicated.csv
```

---

## Pull Latest Code on Nibi

If the repo already exists on Nibi:

```bash
cd ~/scratch/ChemFluor_Project
git pull origin main
```

If setting up for the first time:

```bash
cd ~/scratch
git clone https://github.com/chrislleung/ChemFluor.git ChemFluor_Project
cd ChemFluor_Project
```

---

# Compute Canada / Nibi Training

## Environment Setup

Load Nibi modules and activate the virtual environment:

```bash
cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate
```

Confirm RDKit works:

```bash
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"
```

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
```

If packages are missing:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest typing_extensions matplotlib scipy
```

If RDKit works outside the virtual environment but not inside it, recreate the environment with system site packages:

```bash
cd ~/scratch
mv chemfluor_env chemfluor_env_old

module purge
module load python/3.11
module load gcc
module load rdkit

python -m venv --system-site-packages ~/scratch/chemfluor_env
source ~/scratch/chemfluor_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ~/scratch/ChemFluor_Project/requirements.txt
python -m pip install pytest typing_extensions matplotlib scipy
```

---


  ## Slurm Training Script for Combined ChemFluor + Deep4Chem Model

Use this section when you want to train the combined ChemFluor + Deep4Chem model on Nibi, but not the FluoDB-expanded model. This requires:

```text
data/chemfluor_data.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
```

Create a script such as `run_combined_training.sh` (skip if already in directory):

```bash
#!/bin/bash
#SBATCH --job-name=chemfluor_combined
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=outputs/combined_train_%j.out
#SBATCH --error=outputs/combined_train_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs
mkdir -p models/chemfluor_combined

echo "Starting ChemFluor + Deep4Chem training at $(date)"
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

python -u scripts/train_combined_predictors.py \
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" \
  --chemfluor data/chemfluor_data.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-dir models/chemfluor_combined \
  --model rf

echo "ChemFluor + Deep4Chem training finished at $(date)"
find models/chemfluor_combined -maxdepth 1 -type f -printf "%f\n" | sort
```

Submit and monitor:

```bash
chmod +x run_combined_training.sh
sbatch run_combined_training.sh
squeue -u $USER
tail -f outputs/combined_train_<JOBID>.out
```

---

## Slurm Training Script for FluoDB-Expanded Model

Create a script (skip if already in directory):

```bash
nano run_fluodb_training.sh
```

Paste:

```bash
#!/bin/bash
#SBATCH --job-name=chemfluor_fluodb
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/fluodb_train_%j.out
#SBATCH --error=outputs/fluodb_train_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

mkdir -p outputs
mkdir -p models/chemfluor_combined_fluodb

echo "Starting FluoDB combined training at $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
python --version

echo "Checking RDKit..."
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"

echo "Checking input files..."
ls -lh data/processed/fluodb_lite/combined_deduplicated.csv
ls -lh data/solvent_descriptors_expanded_deep4chem.csv

echo "Running training..."
python -u scripts/train_combined_predictors.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-dir models/chemfluor_combined_fluodb \
  --model rf

echo "Training finished at $(date)"
find models/chemfluor_combined_fluodb -maxdepth 1 -type f -printf "%f\n" | sort
```

If your allocation requires an account line, copy it from your existing script:

```bash
grep account run_chemfluor.sh
```

Then add the same `#SBATCH --account=...` line near the top of `run_fluodb_training.sh`.

Submit:

```bash
chmod +x run_fluodb_training.sh
sbatch run_fluodb_training.sh
```

Monitor:

```bash
squeue -u $USER
tail -f outputs/fluodb_train_<JOBID>.out
tail -f outputs/fluodb_train_<JOBID>.err
```

To exit `tail` without stopping the job:

```text
Ctrl + C
```

To cancel a job:

```bash
scancel <JOBID>
```

---

## Copy Trained Results Back to Local Machine

After training finishes, copy selected results back from Nibi to the local machine.

Run these from Windows PowerShell:

```powershell
cd "C:\Users\CL\OneDrive\Desktop\python\ChemFluor_Project_synced"
```

Copy the original ChemFluor workflow outputs, if you trained the original dataset:

```powershell
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/metrics outputs/
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/plots outputs/
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/models outputs/
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/predictions outputs/
```

Copy the combined or FluoDB-expanded trained model folder, if needed:

```powershell
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/models/chemfluor_combined models/
scp -r chrisl@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/models/chemfluor_combined_fluodb models/
```

Copy model reports:

```powershell
scp -r johndoe@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/combined_model_report_fluodb outputs/
scp -r johndoe@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/error_analysis_fluodb outputs/
```

Copy candidate-screening outputs if needed:

```powershell
scp -r johndoe@nibi.alliancecan.ca:~/scratch/ChemFluor_Project/outputs/candidate_screening outputs/
```

These copied files are useful for local inspection and reporting, but they should usually remain untracked by Git.

---

# Reporting and Error Analysis

Generate a model report:

```bash
python scripts/report_combined_model_results.py \
  --model-dir models/chemfluor_combined_fluodb \
  --out-dir outputs/combined_model_report_fluodb
```

Generated outputs:

```text
outputs/combined_model_report_fluodb/model_summary.md
outputs/combined_model_report_fluodb/metrics_table.csv
outputs/combined_model_report_fluodb/figures/
```

Analyze prediction errors:

```bash
python scripts/analyze_prediction_errors.py \
  --model-dir models/chemfluor_combined_fluodb \
  --out-dir outputs/error_analysis_fluodb
```

Check wavelength-region error:

```bash
cat outputs/error_analysis_fluodb/error_by_wavelength_region_emission_nm.csv
```

The current FluoDB-expanded run showed that red/NIR remains the hardest emission region.

---

# Candidate Generation and Screening

## Generate Scaffold-Based Candidates

```bash
python scripts/generate_scaffold_candidates.py
```

Default output:

```text
data/generated_candidates/scaffold_candidates.csv
```

Current default generator:

```text
Scaffold templates used: 5
Substituents used: 12
Raw combinations attempted: 60
Unique valid molecules saved: 59
```

The current scaffold set is mainly coumarin-like and naphthalimide-like. This is useful for testing but does not yet cover enough red-shifted fluorophore families.

---

## Screen Candidate Molecules

Example: screen for 600 nm emission in ethanol using the FluoDB-trained model.

```bash
python scripts/screen_candidate_molecules.py \
  --candidates data/generated_candidates/scaffold_candidates.csv \
  --solvent-smiles CCO \
  --target-emission 600 \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --applicability-reference-csv models/chemfluor_combined_fluodb/combined_modeling_rows_after_feature_merge.csv \
  --applicability-threshold 0.30 \
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600_fluodb.csv
```

Ranked output columns:

```text
name
scaffold
substituent
smiles
canonical_smiles
solvent_smiles
predicted_absorption_nm
predicted_emission_nm
predicted_quantum_yield
predicted_log_extinction
nearest_training_similarity
nearest_training_smiles
outside_applicability_domain
emission_error_from_target
score
estimated_brightness_score
```

Applicability-domain columns use Morgan fingerprint Tanimoto similarity to the closest reference chromophore.

---

## Screen 450, 520, and 600 nm Targets

```bash
python scripts/screen_candidate_molecules.py \
  --candidates data/generated_candidates/scaffold_candidates.csv \
  --solvent-smiles CCO \
  --target-emission 450 \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --applicability-reference-csv models/chemfluor_combined_fluodb/combined_modeling_rows_after_feature_merge.csv \
  --applicability-threshold 0.30 \
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450_fluodb.csv

python scripts/screen_candidate_molecules.py \
  --candidates data/generated_candidates/scaffold_candidates.csv \
  --solvent-smiles CCO \
  --target-emission 520 \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --applicability-reference-csv models/chemfluor_combined_fluodb/combined_modeling_rows_after_feature_merge.csv \
  --applicability-threshold 0.30 \
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520_fluodb.csv

python scripts/screen_candidate_molecules.py \
  --candidates data/generated_candidates/scaffold_candidates.csv \
  --solvent-smiles CCO \
  --target-emission 600 \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --applicability-reference-csv models/chemfluor_combined_fluodb/combined_modeling_rows_after_feature_merge.csv \
  --applicability-threshold 0.30 \
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600_fluodb.csv
```

---

## Summarize Candidate Screening

```bash
python scripts/summarize_candidate_screening.py \
  --inputs outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450_fluodb.csv outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520_fluodb.csv outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600_fluodb.csv \
  --targets 450 520 600 \
  --out outputs/candidate_screening/screening_summary_fluodb.csv \
  --markdown outputs/candidate_screening/screening_summary_fluodb.md \
  --top-n 10
```

The Markdown report includes:

* best candidate by target emission wavelength
* blue/green/orange-red interpretation notes
* scaffold counts among top candidates
* substituent counts
* warning if the best candidate is more than 30 nm from the target
* warning if one scaffold family dominates the top candidates
* caution that model-ranked candidates are not experimentally validated fluorophores

---

## Predicting a New Molecule with the Combined Model

Use this script for models trained by `scripts/train_combined_predictors.py`. It builds the same Morgan fingerprint plus solvent-descriptor feature representation used by the combined ChemFluor + Deep4Chem workflow.

**run these on the Compute Canada Servers

ChemFluor + Deep4Chem model:

```bash
python scripts/predict_combined_molecule.py \
  --smiles "c1ccccc1" \
  --solvent-smiles "CCO" \
  --model-dir models/chemfluor_combined \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --name candidate_1 \
  --out outputs/predictions/candidate_1_combined_prediction.json
```

FluoDB-expanded model:

```bash
python scripts/predict_combined_molecule.py \
  --smiles "c1ccccc1" \
  --solvent-smiles "CCO" \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --name candidate_1 \
  --out outputs/predictions/candidate_1_fluodb_prediction.json \
  --out-csv outputs/predictions/candidate_1_fluodb_prediction.csv
```

The script predicts whichever target model files are present in the model directory and warns about missing targets. When `combined_modeling_rows_after_feature_merge.csv` is available, it also reports the nearest training-set Tanimoto similarity and flags molecules below the applicability-domain threshold.

### Confidence and Known-Value Evaluation

For single-molecule prediction, the default applicability-domain threshold is 0.50. Candidate screening may use a lower threshold such as 0.30, but individual literature-style predictions should use the stricter threshold.

Example:

```bash
python scripts/predict_combined_molecule.py \
  --smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --solvent-smiles "CS(=O)C" \
  --model-dir models/chemfluor_combined_fluodb \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --name literature_candidate \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196 \
  --check-exact-reference-match \
  --out outputs/predictions/literature_candidate.json \
  --out-csv outputs/predictions/literature_candidate.csv
```

The output includes predicted values, known values, absolute errors, residuals, nearest-training similarity, a confidence label, and an applicability-domain warning.

## Comparing Alternative Models

Random Forest is a strong baseline for the combined ChemFluor + Deep4Chem + FluoDB-Lite workflow, but it may underfit difficult red-shifted substituent effects. Gradient boosting models, Extra Trees, and neural networks can capture different structure-property patterns, so compare overall emission MAE and red/NIR-region error separately before choosing a production model.

Run a local comparison:

```bash
python scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb \
  --models rf,extratrees,histgb,gbdt,mlp \
  --targets emission_nm,quantum_yield \
  --compare-out outputs/model_experiments_fluodb
```

Add a known-molecule benchmark:

```bash
python scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb \
  --models rf,extratrees,histgb,gbdt,mlp \
  --targets emission_nm,quantum_yield \
  --compare-out outputs/model_experiments_fluodb \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196
```

The comparison writes:

```text
outputs/model_experiments_fluodb/model_comparison.csv
outputs/model_experiments_fluodb/model_comparison.md
outputs/model_experiments_fluodb/error_by_region_comparison.csv
outputs/model_experiments_fluodb/benchmark_prediction_comparison.csv
```

For Nibi, create a Slurm script such as `run_model_experiments.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=chemfluor_models
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=outputs/model_experiments_%j.out
#SBATCH --error=outputs/model_experiments_%j.err

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

python scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb \
  --models rf,extratrees,histgb,gbdt,mlp \
  --targets emission_nm,quantum_yield \
  --compare-out outputs/model_experiments_fluodb \
  --n-jobs 16
```

Submit it with:

```bash
sbatch run_model_experiments.sh
```

---

## Neural Model Experiments

Use `scripts/run_neural_model_experiments.py` to test stronger neural baselines against the existing RF, ExtraTrees, HistGB, and GBDT comparison outputs. The script trains separate models for each requested target, saves every sklearn alpha variant, skips PyTorch gracefully when `torch` is unavailable, and writes neural-only plus all-model comparison tables.

Example full run:

```bash
python scripts/run_neural_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --out-root models/neural_experiments_fluodb \
  --compare-out outputs/neural_model_experiments_fluodb \
  --models mlp_small,mlp_medium,mlp_large,pytorch_mlp \
  --targets emission_nm,quantum_yield \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196
```

Neural model outputs are written to:

```text
outputs/neural_model_experiments_fluodb/neural_model_comparison.csv
outputs/neural_model_experiments_fluodb/neural_model_comparison.md
outputs/neural_model_experiments_fluodb/neural_error_by_region_comparison.csv
outputs/neural_model_experiments_fluodb/neural_benchmark_prediction_comparison.csv
```

Combined tree-plus-neural outputs are written to:

```text
outputs/neural_model_experiments_fluodb/all_model_comparison.csv
outputs/neural_model_experiments_fluodb/all_model_comparison.md
outputs/neural_model_experiments_fluodb/all_error_by_region_comparison.csv
outputs/neural_model_experiments_fluodb/all_benchmark_prediction_comparison.csv
```

Interpret the neural results by region and use case, not only by global MAE. If RF remains best overall, it should stay the default production model. A neural model is still scientifically useful if it improves low-similarity benchmark molecules, red/NIR emission errors, quantum-yield MAE, or provides a useful model-family disagreement signal for extrapolative molecules.

## Graph Neural Network Experiments

Use `scripts/run_graph_model_experiments.py` to test whether molecular graphs improve the cases where fingerprint/tree models are weakest: unfamiliar molecules, red/NIR emitters, difficult benchmarks, and model-family disagreement. The first implementation is pure PyTorch and does not require PyTorch Geometric, DGL, Chemprop, or CUDA.

Example full run:

```bash
python scripts/run_graph_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --neural-compare-dir outputs/neural_model_experiments_fluodb \
  --out-root models/graph_experiments_fluodb \
  --compare-out outputs/graph_model_experiments_fluodb \
  --models graph_gcn,graph_mpnn,graph_gin \
  --targets emission_nm,quantum_yield \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196
```

Graph-only outputs:

```text
outputs/graph_model_experiments_fluodb/graph_model_comparison.csv
outputs/graph_model_experiments_fluodb/graph_model_comparison.md
outputs/graph_model_experiments_fluodb/graph_error_by_region_comparison.csv
outputs/graph_model_experiments_fluodb/graph_benchmark_prediction_comparison.csv
outputs/graph_model_experiments_fluodb/performance_by_similarity_bin.csv
outputs/graph_model_experiments_fluodb/performance_by_similarity_bin.md
```

Merged tree, MLP, and graph outputs:

```text
outputs/graph_model_experiments_fluodb/all_model_comparison.csv
outputs/graph_model_experiments_fluodb/all_model_comparison.md
outputs/graph_model_experiments_fluodb/all_error_by_region_comparison.csv
outputs/graph_model_experiments_fluodb/all_benchmark_prediction_comparison.csv
outputs/graph_model_experiments_fluodb/model_disagreement_summary.csv
```

Interpret graph results cautiously. A graph neural network may not beat RF globally, especially on modest tabular chemistry datasets. The key test is whether it improves low-similarity bins, red/NIR error, benchmark molecules, QY behavior, or disagreement-based uncertainty detection. If improvement appears only in high-similarity bins, it is mostly interpolation; improvement in the `0.00-0.30` or `0.30-0.50` similarity bins is stronger evidence for extrapolative value.

Submit the full graph workflow on Nibi:

```bash
sbatch run_graph_experiments.sh
```

Submit the short debug run:

```bash
sbatch run_graph_experiments_debug.sh
```

For Nibi, create a Slurm script such as `run_neural_experiments.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=chemfluor_neural
#SBATCH --account=def-yzhao
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm/chemfluor_neural_%j.out
#SBATCH --error=outputs/slurm/chemfluor_neural_%j.err

set -euo pipefail

cd ~/scratch/ChemFluor_Project

mkdir -p outputs/slurm
mkdir -p models/neural_experiments_fluodb
mkdir -p outputs/neural_model_experiments_fluodb

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate

echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
Aecho "Python: $(which python)"
python --version

python scripts/run_neural_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --tree-compare-dir outputs/model_experiments_fluodb \
  --out-root models/neural_experiments_fluodb \
  --compare-out outputs/neural_model_experiments_fluodb \
  --models mlp_small,mlp_medium,mlp_large,pytorch_mlp \
  --targets emission_nm,quantum_yield \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196

echo "End time: $(date)"
echo "Job completed successfully."
```

Submit it with:

```bash
sbatch run_model_experiments.sh
```

---

# Applicability Domain and Confidence Warnings

The prediction and screening workflows report whether a molecule is similar to the model reference data.

Applicability checks include:

* maximum Morgan fingerprint Tanimoto similarity to reference molecules
* nearest training/reference SMILES
* whether the candidate falls below the selected similarity threshold
* scaffold seen/unseen checks in the original prediction workflow
* solvent seen/unseen checks in the original prediction workflow

For candidate screening, important columns are:

```text
nearest_training_similarity
nearest_training_smiles
outside_applicability_domain
```

Low-confidence predictions should be treated as rough screening estimates, not definitive experimental predictions.

---

# Testing

Run all tests locally:

```powershell
python -m pytest tests
```

On Nibi, use:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
```

Run specific test groups:

```powershell
python -m pytest tests/test_data_standardization.py
python -m pytest tests/test_applicability_domain.py
python -m pytest tests/test_fluodb_lite_standardization.py
python -m pytest tests/test_standardized_deduplication.py
python -m pytest tests/test_summarize_candidate_screening.py
```

---

# Saving Work to GitHub

Before committing, check status:

```bash
git status
```

Do not commit generated artifacts:

```text
models/
outputs/
*.joblib
*.out
*.err
```

Safe add command:

```bash
git add scripts src tests README.md requirements.txt .gitignore
```

If development notes are included:

```bash
git add "development md/Development2.md"
```

Check staged files:

```bash
git diff --cached --stat
```

If generated files were accidentally staged:

```bash
git restore --staged models outputs data/raw/fluodb data/processed
```

Commit:

```bash
git commit -m "Add FluoDB-Lite integration and training documentation"
```

Push:

```bash
git push origin main
```

If Git shows modified tracked data files that should not be committed, restore them only after confirming they are not needed:

```bash
git restore data/chemfluor_data.csv
git restore data/solvent_descriptors.csv
git restore data/solvent_descriptors_expanded_deep4chem.csv
git restore data/test1_candidate_molecules.csv
```

---

# Jupyter Notebooks

The `notebooks/` folder provides department-facing notebooks.

Recommended order:

1. `00_project_overview.ipynb`
2. `03_view_results.ipynb`
3. `04_predict_new_molecule.ipynb`
4. `05_batch_prediction.ipynb`

Technical users may also use:

1. `01_data_and_features.ipynb`
2. `02_train_models.ipynb`

The training notebook should submit a Slurm job rather than running full training directly inside the notebook.

---

# Known Limitations

* PLQY is experimentally noisy and difficult to predict.
* Solvent descriptor values should be verified before publication-quality use.
* Random split can overestimate performance when similar molecules appear in train and test.
* Scaffold split is a better estimate of performance on new molecular cores.
* The grouped chromophore split prevents exact molecule leakage but is not equivalent to a scaffold split.
* Red/NIR emission remains the hardest wavelength region.
* The current candidate generator is too small and does not yet cover enough red-shifted fluorophore families.
* Predictions for low-similarity molecules should be treated as rough estimates.

---

# Next Development Step

The next major development step is to expand `scripts/generate_scaffold_candidates.py` with red-shifted fluorophore families, such as:

```text
BODIPY-like scaffolds
cyanine-like scaffolds
rhodamine-like scaffolds
fluorescein-like scaffolds
larger donor-acceptor systems
extended aromatic systems
```

The next measurable milestone is:

```text
old 59-candidate library vs expanded red-shifted library
→ screen both with the FluoDB-trained model
→ compare best 600 nm emission error
```

---

# Citation

If using this project, cite the original ChemFluor dataset and paper.

Dataset:

```text
ChemFluor. Figshare. DOI: 10.6084/m9.figshare.12110619
```

Original paper:

```text
Ju, C.-W.; Bai, H.; Li, B.; Liu, R. Machine Learning Enables Highly Accurate Predictions of Photophysical Properties of Organic Fluorescent Materials: Emission Wavelengths and Quantum Yields. Journal of Chemical Information and Modeling, 2021.
```
