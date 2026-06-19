# FluorCast

FluorCast is a solvent-aware machine learning workflow for predicting fluorescent molecule properties from:

```text
chromophore SMILES + solvent SMILES/name → emission wavelength + quantum yield
```

This README covers the current combined workflow with datasets:

```text
ChemFluor + Deep4Chem + FluoDB-Lite
```

---

## Current Results

### Overall Model Performance by MAE

| Model | Family | Emission MAE ↓ | Quantum Yield MAE ↓ | Main Use |
|---|---|---:|---:|---|
| **RF** | Tree | **23.85 nm** | 0.1505 | Best global emission model |
| **ExtraTrees** | Tree | 28.20 nm | **0.1464** | Best global QY MAE |
| **HistGB** | Tree boosting | 29.31 nm | 0.1749 | Strong non-graph baseline |
| **graph_gin, 3-seed mean** | Graph NN | 29.49 ± 0.90 nm | 0.1681 | Best stable graph emission model |
| **graph_gcn, 3-seed mean** | Graph NN | 29.73 ± 2.57 nm | 0.1610 | Best single graph seed, less stable |
| **Best MLP** | Fingerprint NN | 30.78 nm | 0.1519 | Competitive QY model |
| **GBDT** | Tree boosting | 40.10 nm | 0.2087 | Weaker tree baseline |
| **graph_mpnn** | Graph NN | 62.09 nm | — | Current implementation underperformed |

Main conclusion:

```text
RF remains the best global emission model.
ExtraTrees is best for global quantum-yield MAE.
Graph GIN is the strongest and most stable graph emission model.
```

---

## Required Data

The current workflow expects these files. They are included in the GitHub repository:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
data/raw/fluodb/FluoDB-Lite.csv
data/processed/fluodb_lite/combined_deduplicated.csv
```

Most important training file:

```text
data/processed/fluodb_lite/combined_deduplicated.csv
```

Most important solvent descriptor file:

```text
data/solvent_descriptors_expanded_deep4chem.csv
```

---

## Pull Project to Compute Canada / Nibi

Log in:

```bash
ssh [username]@nibi.alliancecan.ca
```

Clone the repo if it is not already on Nibi:

```bash
cd ~/scratch
git clone https://github.com/chrislleung/fluorcast.git ChemFluor_Project
cd ChemFluor_Project
```

If it already exists:

```bash
cd ~/scratch/ChemFluor_Project
git pull origin main
```

Load the environment:

```bash
module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate
```

If the environment does not exist yet:

```bash
cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

python -m venv --system-site-packages ~/scratch/chemfluor_env
source ~/scratch/chemfluor_env/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install pytest typing_extensions matplotlib scipy
```

Check RDKit:

```bash
python -c "from rdkit import Chem; print('RDKit OK:', Chem.MolFromSmiles('CCO'))"
```

---

## Data Files

The required CSV files are included in the GitHub repository, so no separate `scp` transfer is needed after cloning or pulling the repo.

Verify the files on Nibi:

```bash
cd ~/scratch/ChemFluor_Project

ls -lh data/chemfluor_data.csv
ls -lh data/solvent_descriptors.csv
ls -lh data/solvent_descriptors_expanded_deep4chem.csv
ls -lh "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
ls -lh data/raw/fluodb/FluoDB-Lite.csv
ls -lh data/processed/fluodb_lite/combined_deduplicated.csv
```

If any file is missing, pull the latest repository version:

```bash
git pull origin main
```

## Run Training

Training should be run with Slurm, not directly on the login node.

The Slurm scripts are already included in the GitHub repository, so you do not need to create or paste them manually.

### 1. Tree Model Experiments

This trains RF, ExtraTrees, HistGB, and GBDT on emission and quantum yield.

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_model_experiments_fluodb.sh
```

Monitor:

```bash
squeue -u $USER
ls -lh outputs/slurm | tail -20
```

Outputs:

```text
models/experiments_fluodb/
outputs/model_experiments_fluodb/
```

---

### 2. Neural MLP Experiments

This trains MLP baselines and compares them with the tree-model results.

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_neural_experiments.sh
```

Monitor:

```bash
squeue -u $USER
ls -lh outputs/slurm | tail -20
```

Outputs:

```text
models/neural_experiments_fluodb/
outputs/neural_model_experiments_fluodb/
```

---

### 3. GPU Graph Neural Network Experiments

Graph models should be run on GPU.

Main graph experiment scripts already included in the repo:

```text
run_graph_gin_emission_3seeds_gpu.sh
run_graph_gcn_emission_3seeds_gpu.sh
run_graph_gin_qy_gpu.sh
run_graph_gcn_qy_gpu.sh
run_graph_gin_mpnn_emission_gpu.sh
```

Recommended emission stability runs:

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_graph_gin_emission_3seeds_gpu.sh
sbatch run_graph_gcn_emission_3seeds_gpu.sh
```

Optional graph QY runs:

```bash
sbatch run_graph_gin_qy_gpu.sh
sbatch run_graph_gcn_qy_gpu.sh
```

Monitor:

```bash
squeue -u $USER
ls -lh outputs/slurm | tail -20
```

Outputs:

```text
models/graph_gin_emission_3seeds_gpu/
models/graph_gcn_emission_3seeds_gpu/
models/graph_gin_qy_gpu/
models/graph_gcn_qy_gpu/
outputs/graph_gin_emission_3seeds_gpu/
outputs/graph_gcn_emission_3seeds_gpu/
outputs/graph_gin_qy_gpu/
outputs/graph_gcn_qy_gpu/
```

---

### 4. All-Model Prediction Job

The prediction Slurm script is also included.

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_predict_all_models.sh
```

Outputs:

```text
outputs/predictions/
outputs/slurm/
```

## Run All-Model Prediction

Use `scripts/predict_all_models.py` after trained model artifacts exist.

For the prepared benchmark/presentation prediction, use the included Slurm script:

```bash
cd ~/scratch/ChemFluor_Project
sbatch run_predict_all_models.sh
```

For a custom molecule, replace the "python scripts/predict_all_models.py..." portion of the script with:

```bash
python scripts/predict_all_models.py \
  --smiles "YOUR_CHROMOPHORE_SMILES" \
  --solvent-smiles "YOUR_SOLVENT_SMILES" \
  --graph-model-dirs \
    models/graph_gin_emission_3seeds_gpu/seed_0/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_1/graph_gin \
    models/graph_gin_emission_3seeds_gpu/seed_2/graph_gin \
    models/graph_gcn_emission_3seeds_gpu/seed_0/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_1/graph_gcn \
    models/graph_gcn_emission_3seeds_gpu/seed_2/graph_gcn \
  --out outputs/predictions/new_molecule_prediction.csv
```

The prediction table includes:

```text
model
model_family
seed
predicted_emission_nm
predicted_quantum_yield
emission_abs_error_nm
quantum_yield_abs_error
nearest_training_similarity
nearest_training_smiles
confidence_label
outside_applicability_domain
```

## Check Results

Main output folders:

```text
outputs/model_experiments_fluodb/
outputs/neural_model_experiments_fluodb/
outputs/graph_gin_emission_3seeds_gpu/
outputs/graph_gcn_emission_3seeds_gpu/
outputs/predictions/
models/
```

Useful commands:

```bash
cat outputs/model_experiments_fluodb/model_comparison.md
cat outputs/neural_model_experiments_fluodb/all_model_comparison.md
cat outputs/graph_seed_summary_grouped.csv
cat outputs/predictions/difficult_benchmark_all_models_with_graphs_and_qy.csv
```

View Queue
```bash
squeue -u $USER
```

Cancel Job
```bash
scancel JOBID
```

Find recent Slurm logs:

```bash
ls -lh outputs/slurm | tail -20
```

View a Slurm log:

```bash
cat outputs/slurm/<LOG_FILE>.out
cat outputs/slurm/<LOG_FILE>.err
```

---

## What the Models Are Doing

### Tree and Fingerprint Models

RF, ExtraTrees, HistGB, GBDT, and MLP models use fixed molecular features:

```text
chromophore SMILES → Morgan fingerprint + molecular descriptors
solvent SMILES/name → solvent descriptor vector
combined vector → prediction
```

These models are strong baselines. RF is currently the best global emission model.

### Graph Neural Networks

Graph models use the molecule as a graph:

```text
atoms = nodes
bonds = edges
atom/bond features = graph features
```

Workflow:

```text
SMILES → RDKit molecule → molecular graph → GCN/GIN/MPNN → learned molecular embedding
learned molecular embedding + solvent descriptors → emission/QY prediction
```

Current graph results:

```text
graph_gin = best stable graph emission model
graph_gcn = can perform well but is seed-sensitive
graph_mpnn = weak in current implementation
```

### Applicability Domain

The predictor reports nearest training-set similarity using Morgan fingerprint Tanimoto similarity.

Important columns:

```text
nearest_training_similarity
nearest_training_smiles
confidence_label
outside_applicability_domain
```

Low-similarity predictions should be treated as rough estimates, not confirmed experimental values.

### Model Disagreement

The all-model predictor compares outputs from tree, MLP, and graph models.

High disagreement means:

```text
prediction uncertainty is high
```

This is especially important for outside-domain molecules.

---

## Git Notes

Do not commit generated artifacts:

```text
models/
outputs/
*.joblib
*.pt
*.out
*.err
```

Safe commit command:

```bash
git add scripts src tests README.md requirements.txt .gitignore
git commit -m "Update README for FluorCast workflow"
git push origin main
```

## Check for an existing molecule-solvent pair

Canonicalize a proposed molecule with RDKit and check it against one or more CSV
datasets. Solvent names are compared case-insensitively with normalized whitespace.

```bash
python scripts/check_molecule_in_dataset.py \
  --smiles "CC1=CC=C(C=C1)N" \
  --solvent "ethanol" \
  --dataset data/chemfluor_data.csv \
  --dataset "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" \
  --out outputs/molecule_matches.csv
```

Use `--smiles-column` and `--solvent-column` for datasets with nonstandard column
names. Invalid dataset SMILES are skipped and counted in the terminal summary.
