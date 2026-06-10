# ChemFluor Molecular Property Pipeline

This project recreates and extends a ChemFluor-style machine learning pipeline for predicting photophysical properties of organic fluorescent molecules from molecular structure and solvent information.

The model predicts:

* **Emission wavelength** in nanometers, which corresponds to fluorescence color
* **PLQY**, photoluminescence quantum yield, which corresponds to brightness
* **Bright/dim class**, using the threshold `PLQY > 0.25`

The pipeline supports model training, random-split evaluation, scaffold-split evaluation, uncertainty analysis, and prediction on new user-provided molecules with applicability-domain warnings.

---

## Project Overview

The goal of this project is to use molecular and solvent features to predict fluorescence behavior.

Inputs:

```text
SMILES
solvent
```

Main targets:

```text
Emission/nm
PLQY
```

The model uses a combination of:

* molecular fingerprints
* molecular descriptors
* solvent identity features
* solvent physical descriptors
* random and scaffold-based validation
* multiple machine learning models
* applicability-domain checks for new molecules

This project is designed for both research use and department-facing Jupyter notebook demonstrations.

---

## Data Source

The ChemFluor dataset is publicly available on Figshare:

* Dataset: **ChemFluor**
* DOI: `10.6084/m9.figshare.12110619`
* Figshare page: https://figshare.com/articles/dataset/ChemFluor/12110619
* License: **CC BY 4.0**
* Authors listed on Figshare: Cheng-Wei Ju, Rizhang Liu, Bo Li, Hanzhi Bai

The pipeline expects at least these columns:

```text
SMILES
solvent
Emission/nm
PLQY
```

The dataset file is not committed directly to this repository by default so that users download the official source version and cite it properly.

---

## Repository Structure

```text
ChemFluor_Project/
  README.md
  NOTEBOOKS_GUIDE.md
  requirements.txt
  run_chemfluor.sh
  example_candidates.csv
  solvent_descriptors.csv
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
    metrics/
    plots/
    models/
    predictions/
```

---

## Installation

### Local Installation

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
pip install -r requirements.txt
```

If RDKit is difficult to install with pip, use conda:

```bash
conda install -c conda-forge rdkit
```

### Recommended Python Version

Python 3.11 is recommended.

This project uses scientific Python packages such as RDKit, LightGBM, XGBoost, CatBoost, scikit-learn, NumPy, and pandas. These are more reliable on stable Python versions such as 3.11.

---

## Required Files

For full training, place these files in the project root:

```text
chemfluor_data.csv
solvent_descriptors.csv
```

`chemfluor_data.csv` contains the main ChemFluor dataset.

`solvent_descriptors.csv` contains auxiliary solvent-property features used by the model.

Expected solvent descriptor columns:

```text
solvent
dielectric_constant
refractive_index
dipole_moment
hbond_donor
hbond_acceptor
polarity_ET30
```

If `solvent_descriptors.csv` is missing, the training pipeline can create a blank template from the solvents found in the dataset. If descriptor values are partially missing, the pipeline reports the missing solvents and median-imputes the missing numeric values.

---

## Solvent Descriptors

This repository includes `solvent_descriptors.csv`, an auxiliary table of solvent properties used for feature generation.

The table includes descriptors such as:

* dielectric constant
* refractive index
* dipole moment
* hydrogen-bond donor parameter
* hydrogen-bond acceptor parameter
* ET(30) polarity

These descriptors are included to make feature generation reproducible. Values should be checked against primary or reference sources before publication-quality use, because solvent properties can depend on temperature, source, and definition.

---

## Feature Engineering

The feature matrix is built from molecular and solvent information.

### Molecular Features

The molecular structure is represented using:

* Morgan fingerprints, radius 2, 2048 bits
* MACCS keys
* RDKit molecular descriptors

RDKit descriptors include features related to:

* molecular weight
* logP
* hydrogen bond donors and acceptors
* topological polar surface area
* ring count
* rotatable bonds
* fraction sp3 carbons
* heavy atom count
* aromatic rings
* aliphatic rings
* molecular refractivity
* BalabanJ
* Bertz complexity

### Solvent Features

Solvent information is represented using:

* one-hot encoded solvent identity
* optional physical solvent descriptors from `solvent_descriptors.csv`

The combination of molecular and solvent features allows the model to learn both structure-property relationships and solvent effects.

---

## Target Engineering

Emission wavelength is modeled in two ways.

### Direct Wavelength Prediction

The first approach predicts emission wavelength directly in nanometers:

```text
Emission/nm
```

### Energy-Space Prediction

The second approach converts wavelength to photon energy before training:

$$
E = \frac{1240}{\lambda}
$$

where:

* (E) is photon energy in electronvolts, eV
* (\lambda) is emission wavelength in nanometers, nm

After the model predicts energy, the prediction is converted back to wavelength:

$$
\lambda = \frac{1240}{E}
$$

The final error is still reported in nanometers. This formulation is useful because molecular emission is often more physically related to photon energy than to wavelength directly.

### PLQY Regression

PLQY is modeled two ways:

* raw PLQY regression
* logit-transformed PLQY regression

The logit transform is used because PLQY is bounded between 0 and 1.

### PLQY Classification

PLQY is also converted into a binary bright/dim label:

```text
PLQY > 0.25  -> bright
PLQY <= 0.25 -> dim
```

The threshold can be changed in:

```text
src/config.py
```

---

## Train-Test Splits

The pipeline evaluates models using two splitting methods.

### Random Split

Random split is an 80/20 train-test split. It is useful for baseline comparison, but it can overestimate performance when similar molecules appear in both the training and test sets.

### Scaffold Split

The scaffold split uses Bemis-Murcko scaffolds. Molecules with the same scaffold are kept together so that the same scaffold does not appear in both training and test sets.

The scaffold split is the more honest generalization test for new chemistry because it evaluates performance on molecular cores that the model did not see during training.

---

## Models

The pipeline compares several machine learning models.

### Regression Models

Regression models are used for emission wavelength and PLQY value prediction:

* LightGBM
* Random Forest
* Extra Trees
* Gradient Boosting
* Support Vector Regression
* XGBoost, if installed
* CatBoost, if installed

The three best regressors by validation MAE are also averaged as a simple ensemble.

### Classification Models

Classification models are used for PLQY bright/dim prediction:

* LightGBM
* Random Forest
* Extra Trees
* XGBoost, if installed
* CatBoost, if installed

---

## Metrics

### Regression Metrics

Regression models are evaluated using:

* MAE, mean absolute error
* RMSE, root mean squared error
* R², coefficient of determination
* Spearman rank correlation

### Classification Metrics

Classification models are evaluated using:

* accuracy
* precision
* recall
* F1 score
* confusion matrix

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

1. load the dataset
2. clean the data
3. canonicalize SMILES
4. merge duplicate molecule-solvent pairs
5. build molecular and solvent features
6. train models
7. evaluate random and scaffold splits
8. save metrics, plots, models, and metadata

---

## Running on Compute Canada / Nibi

This project was developed and tested on the Digital Research Alliance of Canada / Compute Canada Nibi cluster.

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

Place the following files in the project root:

```text
chemfluor_data.csv
solvent_descriptors.csv
```

After setup, the project should look like:

```text
ChemFluor_Project/
  chemfluor_data.csv
  solvent_descriptors.csv
  run_chemfluor.sh
  src/
  notebooks/
```

### 4. Load Required Nibi Modules

RDKit should be loaded through the Nibi module system rather than installed with pip:

```bash
module load python/3.11
module load gcc
module load rdkit
```

### 5. Create and Activate the Python Environment

If the environment does not exist yet:

```bash
python -m venv ~/scratch/chemfluor_env
source ~/scratch/chemfluor_env/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
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

* predicted vs actual wavelength
* predicted vs actual PLQY
* residual plots
* error by solvent
* PLQY classifier confusion matrix

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

---

## Applicability Domain and Confidence Warnings

The prediction command reports whether a new molecule is similar to the training data.

It checks:

* maximum Morgan fingerprint Tanimoto similarity to training molecules
* average similarity to the five nearest training molecules
* whether the Bemis-Murcko scaffold was seen during training
* whether the solvent was seen during training
* optional prediction uncertainty from seed ensemble models

Confidence levels:

### High Confidence

The molecule is close to the training domain.

Typical conditions:

* scaffold was seen during training
* maximum Tanimoto similarity is at least 0.60
* solvent was seen during training

### Medium Confidence

The molecule is moderately similar to training examples.

Typical conditions:

* maximum Tanimoto similarity is at least 0.40
* solvent was seen during training
* scaffold may be new

### Low Confidence

The molecule is outside or weakly supported by the training domain.

Possible reasons:

* low similarity to training molecules
* unseen scaffold
* unknown solvent
* missing or imputed solvent descriptors
* high model uncertainty

Low-confidence predictions should be treated as rough screening estimates, not definitive experimental predictions.

---

## Jupyter Notebooks

The `notebooks/` folder provides department-facing notebooks.

Recommended order:

1. `00_project_overview.ipynb`
2. `03_view_results.ipynb`
3. `04_predict_new_molecule.ipynb`
4. `05_batch_prediction.ipynb`

Technical users may also use:

1. `01_data_and_features.ipynb`
2. `02_train_models.ipynb`

### Notebook Descriptions

#### `00_project_overview.ipynb`

Explains:

* project goal
* inputs and outputs
* dataset source
* feature engineering
* random split vs scaffold split
* prediction and confidence warnings

#### `01_data_and_features.ipynb`

Demonstrates:

* loading the raw dataset
* cleaning the data
* canonicalizing SMILES
* building the feature matrix
* showing feature matrix shape

#### `02_train_models.ipynb`

Shows how to:

* submit the Slurm training job
* check job status
* inspect output logs

This notebook should submit the Slurm job rather than train directly inside the notebook.

#### `03_view_results.ipynb`

Displays:

* metrics CSV
* random split results
* scaffold split results
* predicted-vs-actual plots
* residual plots
* confusion matrix
* worst prediction tables

#### `04_predict_new_molecule.ipynb`

Allows users to enter:

* SMILES
* solvent
* candidate name

Then it runs a prediction and reports:

* predicted emission wavelength
* predicted PLQY
* bright/dim prediction
* confidence level
* nearest training molecules
* Tanimoto similarity
* scaffold seen or unseen
* solvent seen or unseen

#### `05_batch_prediction.ipynb`

Allows users to predict multiple molecules from a CSV file.

---

## Example Candidate File

This repository includes:

```text
example_candidates.csv
```

Example contents:

```csv
name,SMILES,solvent
benzene_example,c1ccccc1,MeCN
ethanol_example,CCO,MeOH
triethylamine_example,CCN(CC)CC,MeCN
```

Use it with:

```bash
python -m src.predict \
  --csv example_candidates.csv \
  --save-csv outputs/predictions/batch_predictions.csv
```

---

## Current Best Results

After adding solvent descriptors and testing XGBoost/CatBoost, the strongest clean results were approximately:

### Random Split

```text
Best wavelength MAE: 16.67 nm
Best PLQY MAE: 0.1233
Best PLQY classifier: 83.99% accuracy / 0.853 F1
```

### Scaffold Split

```text
Best wavelength MAE: 31.62 nm
Best PLQY MAE: 0.2116
Best PLQY classifier: 76.95% accuracy / 0.812 F1
```

The random split results show strong interpolation performance. The scaffold split results are more realistic for new molecular families and show that scaffold generalization remains the main challenge.

---

## Known Limitations

* PLQY is experimentally noisy and difficult to predict.
* Solvent descriptor values should be verified before publication-quality use.
* Random split can overestimate performance when similar molecules appear in train and test.
* Scaffold split is a better estimate of performance on new molecular cores.
* The dataset is modest for deep learning, so classical ML baselines remain important.
* Predictions for low-similarity molecules should be treated as rough estimates.

---

## Citation

If using this project, cite the original ChemFluor dataset and paper.

Dataset:

```text
ChemFluor. Figshare. DOI: 10.6084/m9.figshare.12110619
```

Original paper:

```text
Ju, C.-W.; Bai, H.; Li, B.; Liu, R. Machine Learning Enables Highly Accurate Predictions of Photophysical Properties of Organic Fluorescent Materials: Emission Wavelengths and Quantum Yields. Journal of Chemical Information and Modeling, 2021.
```
