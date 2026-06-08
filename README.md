## Target Engineering

Emission wavelength is modeled in two ways.

### Direct Wavelength Prediction

The first model predicts emission wavelength directly in nanometers:

```text
Emission/nm
```

### Energy-Space Prediction

The second model converts wavelength to photon energy before training:

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

The final error is still reported in nanometers. This is useful because molecular emission is often more physically related to photon energy than to wavelength directly.

PLQY is also modeled two ways:

* raw PLQY regression
* logit-transformed PLQY regression

The logit transform is used because PLQY is bounded between 0 and 1.

---

## Running on Compute Canada / Nibi

This project was developed to run on the Digital Research Alliance of Canada / Compute Canada Nibi cluster.

### 1. Log in to Nibi

From your local terminal:

```bash
ssh <your_username>@nibi.alliancecan.ca
```

For example:

```bash
ssh chrisl@nibi.alliancecan.ca
```

### 2. Go to the project folder

If the project already exists on Nibi:

```bash
cd ~/scratch/ChemFluor_Project
```

If you are setting it up for the first time from GitHub:

```bash
cd ~/scratch
git clone https://github.com/chrislleung/ChemFluor.git ChemFluor_Project
cd ChemFluor_Project
```

### 3. Add the required data files

The full dataset is not included directly in this repository. Download the ChemFluor dataset from Figshare and place it in the project root as:

```text
chemfluor_data.csv
```

The dataset source is:

```text
https://figshare.com/articles/dataset/ChemFluor/12110619
```

The project also uses:

```text
solvent_descriptors.csv
```

This file contains auxiliary solvent-property descriptors used during feature generation.

After setup, the project root should contain:

```text
ChemFluor_Project/
  chemfluor_data.csv
  solvent_descriptors.csv
  run_chemfluor.sh
  src/
  notebooks/
  outputs/
```

### 4. Load the required Nibi modules

RDKit should be loaded through the Nibi module system rather than installed with pip:

```bash
module load python/3.11
module load gcc
module load rdkit
```

### 5. Create and activate the Python environment

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

Check that the environment works:

```bash
python -c "from rdkit import Chem; print(Chem.MolFromSmiles('CCO'))"
python -c "import lightgbm, xgboost, catboost; print('ML packages ready')"
```

### 6. Submit the training job

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

### 7. View saved outputs

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

### 8. Run prediction on a new molecule

After training has created the model files and metadata, run:

```bash
python -m src.predict --smiles "c1ccccc1" --solvent "MeCN"
```

With a custom name and saved JSON output:

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

### 9. Updating the Nibi copy from GitHub

If the GitHub repository has been updated, pull the newest code on Nibi:

```bash
cd ~/scratch/ChemFluor_Project
git pull origin main
```

Then rerun training if the code, features, or model logic changed:

```bash
sbatch run_chemfluor.sh
```

### 10. Important Nibi notes

Do not run the full training pipeline directly on the login node with:

```bash
python -m src.train
```

That command is okay only for quick debugging. Full training should be submitted with:

```bash
sbatch run_chemfluor.sh
```

The Slurm script requests compute resources, loads the correct modules, activates the Python environment, and runs:

```bash
python -u -m src.train
```
