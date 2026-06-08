# ChemFluor Molecular Property Pipeline

ChemFluor is a machine learning pipeline for predicting fluorescent molecule properties from molecular structure and solvent information.

The project predicts:

- **Emission wavelength** in nanometers, which corresponds to fluorescence color.
- **PLQY** as a continuous brightness/quantum-yield value.
- **Bright/dim class**, where molecules with `PLQY > 0.25` are labeled bright.

The code is organized as a Python package in `src/`, while the notebooks in `notebooks/` provide a department-friendly interface for viewing data, training models, inspecting results, and predicting new molecules.

---

## Repository Structure

```text
ChemFluor_Project/
  README.md
  requirements.txt
  run_chemfluor.sh
  example_candidates.csv
  src/
    __init__.py
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
    utils.py
    test_prediction.py
  notebooks/
    00_project_overview.ipynb
    01_data_and_features.ipynb
    02_train_models.ipynb
    03_view_results.ipynb
    04_predict_new_molecule.ipynb
    05_batch_prediction.ipynb
  outputs/
    models/
    metrics/
    plots/
    predictions/
```

The `src/` folder contains the actual reproducible code. The `notebooks/` folder is the user-facing interface for chemists and collaborators.

---

## Files Not Included in GitHub

For privacy, size, and reproducibility reasons, the following files should usually **not** be committed to GitHub:

```text
chemfluor_data.csv
solvent_descriptors.csv
outputs/
*.pkl
*.joblib
```

The project requires `chemfluor_data.csv` for training. The `solvent_descriptors.csv` file is optional but recommended.

---

## Data

The ChemFluor dataset is publicly available on Figshare under the CC BY 4.0 license:

- Dataset: ChemFluor
- DOI: https://doi.org/10.6084/m9.figshare.12110619
- Figshare page: https://figshare.com/articles/dataset/ChemFluor/12110619
- Authors: Cheng-Wei Ju, Rizhang Liu, Bo Li, Hanzhi Bai

To run this project, download the dataset from Figshare and place the processed CSV in the project root as:

```text
chemfluor_data.csv
```

---

## Solvent Descriptors

This repository includes `solvent_descriptors.csv`, an auxiliary solvent-property table used by the model. It contains descriptors such as dielectric constant, refractive index, dipole moment, hydrogen-bond donor/acceptor parameters, and ET(30) polarity.

These descriptors are included to make the feature generation process reproducible. Values should be checked against primary or reference sources before publication-quality use, since solvent properties can depend on temperature, source, and definition.

---

## Required Dataset Columns

The training dataset should be named:

```text
chemfluor_data.csv
```

It must contain these columns:

```text
SMILES
solvent
Emission/nm
PLQY
```

The dataset may contain additional columns, but these four are required.

---

## Feature Engineering

The model uses both molecular and solvent features.

### Molecular Features

- Morgan fingerprints, radius 2, 2048 bits
- MACCS keys
- RDKit molecular descriptors, including molecular weight, logP, TPSA, ring counts, aromatic rings, rotatable bonds, fraction sp3, MolMR, BalabanJ, and BertzCT

### Solvent Features

- One-hot solvent identity
- Optional physical solvent descriptors from `solvent_descriptors.csv`

The solvent descriptor file should contain:

```text
solvent,dielectric_constant,refractive_index,dipole_moment,hbond_donor,hbond_acceptor,polarity_ET30
```

If `solvent_descriptors.csv` is missing, the pipeline creates a blank template. If some descriptor values are missing, the pipeline reports them and median-imputes numeric values.

---

## Targets

### Wavelength Regression

Wavelength is modeled in two ways:

1. Direct prediction of `Emission/nm`
2. Energy-space prediction using:

\[
E = \frac{1240}{\lambda}
\]

where `E` is emission energy in electronvolts and `lambda` is wavelength in nanometers. Energy predictions are converted back to nanometers with:

\[
\lambda = \frac{1240}{E}
\]

### PLQY Regression

PLQY is modeled in two ways:

1. Raw PLQY regression
2. Logit-transformed PLQY regression, which accounts for PLQY being bounded between 0 and 1

### PLQY Classification

PLQY classification uses:

```text
PLQY > 0.25 = bright
PLQY <= 0.25 = dim
```

The threshold can be changed in `src/config.py`.

---

## Train/Test Splits

The pipeline reports two validation modes.

### Random Split

The random split measures how well the model predicts molecules similar to examples it has already seen. This is useful for baseline comparison, but it can overestimate real-world performance.

### Bemis-Murcko Scaffold Split

The scaffold split keeps molecular scaffolds separated between train and test sets. This gives a more honest estimate of how well the model generalizes to new molecular families.

For prospective chemistry and general lab use, the scaffold split should be treated as the more important score.

---

## Models

Regression models:

- LightGBM
- Random Forest
- Extra Trees
- Gradient Boosting
- Support Vector Regression
- XGBoost, if installed
- CatBoost, if installed

Classification models:

- LightGBM
- Random Forest
- Extra Trees
- XGBoost, if installed
- CatBoost, if installed

The pipeline also tests a simple ensemble that averages the top three regression models by validation MAE.

---

## Installation on Compute Canada / Nibi

On Compute Canada, RDKit should be loaded through the module system rather than installed with pip.

```bash
cd ~/scratch/ChemFluor_Project

module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate
```

Install Python packages inside the virtual environment:

```bash
pip install -r requirements.txt
```

If `catboost` is installed, keep `numpy<2.0` to avoid dependency conflicts.

---

## Installation on a Local Computer

A local installation is useful for viewing notebooks and running predictions after trained models are available.

```bash
pip install -r requirements.txt
```

If RDKit is difficult to install with pip, use conda:

```bash
conda install -c conda-forge rdkit
```

---

## Training on Compute Canada

Full training should be run through Slurm, not directly on the login node.

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_chemfluor.sh
```

Check job status:

```bash
squeue -u $USER
```

Watch output after the job starts:

```bash
tail -f outputs/chemfluor_<JOBID>.out
```

Replace `<JOBID>` with the Slurm job number.

The Slurm script runs:

```bash
python -u -m src.train
```

---

## Running Training Directly

For small tests only:

```bash
python -m src.train
```

Do not run the full training pipeline directly on a Compute Canada login node.

---

## Training Outputs

Training saves outputs to:

```text
outputs/metrics/
outputs/plots/
outputs/models/
```

Important files include:

```text
outputs/metrics/metrics.json
outputs/metrics/metrics.csv
outputs/metrics/wavelength_uncertainty.csv
outputs/metrics/plqy_uncertainty.csv
outputs/metrics/wavelength_feature_importance.csv
outputs/metrics/plqy_feature_importance.csv
outputs/plots/predicted_vs_actual_wavelength.png
outputs/plots/predicted_vs_actual_plqy.png
outputs/plots/confusion_matrix_plqy_classifier.png
outputs/plots/worst_20_wavelength_predictions.csv
outputs/plots/worst_20_plqy_predictions.csv
outputs/models/best_wavelength_lightgbm.pkl
outputs/models/best_plqy_lightgbm.pkl
outputs/models/best_plqy_classifier.pkl
outputs/models/feature_artifacts.pkl
outputs/models/inference_metadata.pkl
```

---

## Predicting a New Molecule

After training, run:

```bash
python -m src.predict --smiles "c1ccccc1" --solvent "MeCN"
```

With a name and saved JSON report:

```bash
python -m src.predict \
  --smiles "c1ccccc1" \
  --solvent "MeCN" \
  --name "candidate_1" \
  --output outputs/predictions/candidate_1_prediction.json
```

For batch prediction, create a CSV with columns:

```text
name,SMILES,solvent
```

Then run:

```bash
python -m src.predict \
  --csv example_candidates.csv \
  --save-csv outputs/predictions/batch_predictions.csv
```

---

## Applicability-Domain Warning

The prediction command reports whether the model is likely to be reliable for the input molecule.

The warning uses:

- Morgan fingerprint Tanimoto similarity to training molecules
- Top-5 nearest-neighbor similarity
- Whether the Bemis-Murcko scaffold appeared in training
- Whether the solvent appeared in training
- Optional seed-ensemble uncertainty, if available

Confidence levels:

- **High**: scaffold seen, solvent seen, and nearest-neighbor similarity is high
- **Medium**: molecule is moderately similar to training molecules, but scaffold may be new
- **Low**: molecule is structurally unfamiliar, solvent is unknown, descriptors are missing, or uncertainty is high

Low-confidence predictions should be treated as rough screening estimates, not final experimental conclusions.

---

## Department Notebook Guide

Most users should start with:

```text
notebooks/00_project_overview.ipynb
notebooks/03_view_results.ipynb
notebooks/04_predict_new_molecule.ipynb
notebooks/05_batch_prediction.ipynb
```

Technical users can also use:

```text
notebooks/01_data_and_features.ipynb
notebooks/02_train_models.ipynb
```

---

## Notebook Descriptions

### `00_project_overview.ipynb`

Explains the project goal, inputs, outputs, model strategy, random split, scaffold split, solvent descriptors, and applicability-domain warnings.

### `01_data_and_features.ipynb`

Loads the dataset, cleans it, canonicalizes SMILES, builds the feature matrix, and shows how molecular and solvent features are created.

### `02_train_models.ipynb`

Shows how to submit the Slurm training job, monitor it, and inspect logs. Full model training should happen through Slurm.

### `03_view_results.ipynb`

Loads metrics and plots from the `outputs/` folder. Displays model performance, predicted-vs-actual plots, residual plots, confusion matrix, and worst-prediction tables.

### `04_predict_new_molecule.ipynb`

Allows a user to enter one SMILES string and solvent, then returns predicted emission wavelength, PLQY, bright/dim classification, confidence level, and nearest training molecules.

### `05_batch_prediction.ipynb`

Allows a user to predict many candidate molecules from a CSV file.

---

## Known Limitations

- PLQY is experimentally noisy and depends on conditions such as concentration, oxygen, aggregation, purity, and instrument settings.
- Random split can overestimate performance.
- Scaffold split is a more realistic estimate for new chemistry.
- Solvent descriptors must be complete and consistently named.
- Low-confidence predictions should be validated experimentally.
- The dataset is modest in size, so strong classical ML baselines are currently more practical than large deep learning models.

---

## Recommended Workflow

For developers:

```text
Edit code locally → commit to GitHub → sync to Nibi → run Slurm training → inspect outputs → update notebooks/results
```

For department users:

```text
Open notebooks → view results → enter candidate molecule → inspect prediction and confidence warning
```
