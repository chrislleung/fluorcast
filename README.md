# FluorCast

FluorCast is a solvent-aware machine learning workflow for predicting fluorescent molecule properties from:

```text
chromophore SMILES + solvent SMILES/name -> absorption wavelength, emission wavelength, calculated Stokes shift, quantum yield, and brightness class
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

## External Known-Fluorophore Benchmark Diagnostics

Use this diagnostic pipeline after generating known-fluorophore prediction CSVs with
`scripts/predict_all_models.py`. It consolidates per-molecule prediction files,
compares predictions with literature benchmark values, checks training-set overlap,
classifies likely failure modes, scores confidence, and writes manuscript-ready
CSV/Markdown summaries.

```bash
python scripts/diagnose_external_benchmark.py \
  --prediction-dir outputs/predictions-6-18 \
  --training-csv models/experiments_fluodb/rf/combined_standardized_training_rows.csv \
  --out-dir outputs/predictions-6-18/diagnostics
```

Main outputs:

```text
external_benchmark_all_predictions.csv
external_benchmark_model_summary.csv
external_benchmark_family_summary.csv
external_benchmark_molecule_summary.csv
external_benchmark_training_overlap.csv
external_benchmark_failure_modes.csv
external_benchmark_report.md
```

Key interpretation points:

- `external_benchmark_training_overlap.csv` separates molecule overlap, solvent
  overlap, and exact molecule-solvent-pair overlap. This explains why
  `nearest_training_similarity = 1.0` can still fail.
- `external_benchmark_failure_modes.csv` flags likely issues such as
  benchmark/training label mismatch, solvent or condition mismatch, high model
  disagreement, QY condition sensitivity, and structural extrapolation.
- Confidence scores combine molecule/solvent/pair overlap, nearest-training
  similarity, training-label consistency, and model agreement.
- Optional PNG plots are written when matplotlib is installed; plotting is skipped
  gracefully otherwise.

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
git clone https://github.com/chrislleung/fluorcast.git FluorCast
cd FluorCast
```

If it already exists:

```bash
cd ~/scratch/FluorCast
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
cd ~/scratch/FluorCast

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
cd ~/scratch/FluorCast

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

## Slurm Script Organization

Most prediction users should run the full hybrid prediction workflow:

```bash
mkdir -p outputs/slurm
sbatch slurm/run_predict_full_fluorcast.sbatch
```

Most experiment users should use the hybrid and paired spectral workflows under
`slurm/`.

The Slurm folders are organized by purpose:

```text
slurm/              Current recommended workflows and app-facing jobs
slurm/base_models/  Supported original RF, ExtraTrees, MLP, and graph workflows
slurm/manuscript/   Manuscript/history workflows
slurm/util/         Utility/test jobs
slurm/legacy/       Archived old workflows not recommended for routine use
```

`models/production_hybrid/` is reserved for explicit production artifacts and
should not be committed. Generic experiment wrappers default to split- and
seed-specific paths outside that directory.

## Run Full Experiments on Nibi

Training and full experiments should be run with Slurm, not directly on the
login node. The current recommended path is the hybrid workflow in `slurm/`.
The original base-model workflows remain supported in `slurm/base_models/` for
reproducibility, comparison, and retraining.

The Slurm scripts are already included in the GitHub repository, so you do not
need to create or paste them manually.

### 1. Base Tree Model Experiments

This trains RF, ExtraTrees, HistGB, and GBDT on absorption, emission, and
quantum yield. A production full report requires absorption-capable base
artifacts; `models/experiments_fluodb/` should therefore contain artifacts for
`absorption_nm`, `emission_nm`, and `quantum_yield`.

```bash
cd ~/scratch/FluorCast
mkdir -p outputs/slurm
sbatch slurm/base_models/run_model_experiments_fluodb.sbatch
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
cd ~/scratch/FluorCast
mkdir -p outputs/slurm
sbatch slurm/base_models/run_neural_experiments.sbatch
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

### 3. GPU Graph-Only Experiments

Graph models should be run on GPU. These remain emission/QY comparison
workflows unless absorption graph training is implemented and tested later.

Main graph experiment scripts already included in the repo:

```text
slurm/base_models/run_graph_gin_emission_3seeds_gpu.sbatch
slurm/base_models/run_graph_gcn_emission_3seeds_gpu.sbatch
slurm/base_models/run_graph_gin_qy_gpu.sbatch
slurm/base_models/run_graph_gcn_qy_gpu.sbatch
slurm/base_models/run_graph_gin_mpnn_emission_gpu.sbatch
```

Recommended emission stability runs:

```bash
cd ~/scratch/FluorCast
mkdir -p outputs/slurm
sbatch slurm/base_models/run_graph_gin_emission_3seeds_gpu.sbatch
sbatch slurm/base_models/run_graph_gcn_emission_3seeds_gpu.sbatch
```

Optional graph QY runs:

```bash
sbatch slurm/base_models/run_graph_gin_qy_gpu.sbatch
sbatch slurm/base_models/run_graph_gcn_qy_gpu.sbatch
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

### 4. Hybrid Three-Way Experiments

Hybrid experiments train base models on one split, train the hybrid ensemble on a second split, and evaluate only on the final held-out split. Run one target and split at a time.

For production molecule-split hybrid artifacts:

```bash
cd ~/scratch/FluorCast
mkdir -p outputs/slurm

export FLUORCAST_TARGET_NAME="absorption_nm"
export FLUORCAST_SPLIT_TYPE="molecule"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/molecule/absorption_nm"
export FLUORCAST_MODEL_OUT_DIR="models/production_hybrid/absorption_nm"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

Repeat for emission:

```bash
export FLUORCAST_TARGET_NAME="emission_nm"
export FLUORCAST_SPLIT_TYPE="molecule"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/molecule/emission_nm"
export FLUORCAST_MODEL_OUT_DIR="models/production_hybrid/emission_nm"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

Repeat for quantum yield:

```bash
export FLUORCAST_TARGET_NAME="quantum_yield"
export FLUORCAST_SPLIT_TYPE="molecule"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/molecule/quantum_yield"
export FLUORCAST_MODEL_OUT_DIR="models/production_hybrid/quantum_yield"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

For scaffold or comparison experiments, write to split-specific folders so production models are not overwritten:

```bash
export FLUORCAST_TARGET_NAME="absorption_nm"
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/scaffold/absorption_nm"
export FLUORCAST_MODEL_OUT_DIR="models/hybrid_three_way/scaffold/absorption_nm"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

For scaffold emission:

```bash
export FLUORCAST_TARGET_NAME="emission_nm"
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/scaffold/emission_nm"
export FLUORCAST_MODEL_OUT_DIR="models/hybrid_three_way/scaffold/emission_nm"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

For scaffold quantum yield:

```bash
export FLUORCAST_TARGET_NAME="quantum_yield"
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/scaffold/quantum_yield"
export FLUORCAST_MODEL_OUT_DIR="models/hybrid_three_way/scaffold/quantum_yield"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

Production hybrid artifacts should be staged on Nibi as:

```text
models/production_hybrid/absorption_nm/
models/production_hybrid/emission_nm/
models/production_hybrid/quantum_yield/
```

Comparison hybrid artifacts should be kept separate, for example:

```text
models/hybrid_three_way/scaffold/absorption_nm/
models/hybrid_three_way/scaffold/emission_nm/
models/hybrid_three_way/scaffold/quantum_yield/
```

Do not commit trained model artifacts.


---

### 5. Paired Absorption/Emission Stokes Experiments

Paired spectral experiments use the same molecule-solvent rows for absorption
and emission, then calculate Stokes shift from those paired predictions. Stokes
shift is not directly modeled in this workflow.

```bash
cd ~/scratch/FluorCast

export FLUORCAST_SPLIT_TYPE="molecule"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/paired_stokes_three_way/molecule"
export FLUORCAST_MODEL_OUT_DIR="models/paired_stokes_three_way/molecule"
sbatch slurm/run_paired_spectral_three_way_experiment.sbatch
```

For scaffold splits:

```bash
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_OUT_DIR="outputs/paired_stokes_three_way/scaffold"
export FLUORCAST_MODEL_OUT_DIR="models/paired_stokes_three_way/scaffold"
sbatch slurm/run_paired_spectral_three_way_experiment.sbatch
```

Convenience wrappers run paired absorption/emission/Stokes plus QY for the
molecule and scaffold splits:

```bash
sbatch slurm/run_full_paired_molecule.sbatch
sbatch slurm/run_full_paired_scaffold.sbatch
```

## Predicting New Molecules

### Predicting custom molecules with ConforFormer

Use this path when you want the trained ConforFormer downstream models rather
than the hybrid production workflow. It generates a fresh ConforFormer embedding
for a new chromophore SMILES, combines it with the same Morgan fingerprint and
solvent descriptor representation used in downstream training, validates the
saved feature schema, and loads saved models only.

Single case:

```bash
cd ~/scratch/FluorCast-conforformer
PY="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"

"$PY" scripts/predict_conforformer_case.py \
  --smiles "YOUR_CHROMOPHORE_SMILES" \
  --solvent-smiles "O" \
  --model-root "models/conforformer_downstream/<run_id>" \
  --device cuda
```

Batch CSV:

```bash
"$PY" scripts/predict_conforformer_case.py \
  --input-csv case_studies.csv \
  --output-csv case_study_predictions.csv \
  --model-root "models/conforformer_downstream/<run_id>" \
  --device auto
```

The batch input requires `smiles` and `solvent_smiles`; `name` is optional. The
output includes canonicalized SMILES, `absorption_nm`, `emission_nm`,
`quantum_yield`, `stokes_shift_nm`, `derived_stokes_shift_nm`, `stokes_source`,
`status`, and `error`.

By default the script expects the molecule split, `mean` pooled ConforFormer
embedding, and `conforformer_morgan_solvent` representation. The corresponding
model directories are:

```text
models/conforformer_downstream/<run_id>/molecule/mean/conforformer_morgan_solvent/<target>/
```

`stokes_shift_nm` is reported from the direct trained Stokes model when that
target artifact exists. The script always also reports `derived_stokes_shift_nm`
as predicted emission minus predicted absorption. If the direct Stokes target is
not present, `stokes_shift_nm` is derived and `stokes_source` is `derived`.

Use `--model-root` to point at a specific downstream run. If omitted, the script
looks for a single discoverable run under `models/conforformer_downstream/`, or
uses `training_manifest.json` metadata to choose among runs with saved models.
Use `--device auto`, `--device cpu`, or `--device cuda` for encoder inference.
New pooled embeddings are cached by default under
`.cache/conforformer_inference/`; cache keys include the canonical SMILES,
checkpoint hash, dictionary hash, conformer configuration, preprocessing
configuration, and pooling configuration.

JSON output for one case:

```bash
"$PY" scripts/predict_conforformer_case.py \
  --smiles "YOUR_CHROMOPHORE_SMILES" \
  --solvent-smiles "O" \
  --model-root "models/conforformer_downstream/<run_id>" \
  --json
```

### Full hybrid prediction, recommended

The preferred end-to-end workflow predicts absorption, emission, quantum yield,
calculates Stokes shift from the paired absorption/emission predictions, and
writes full JSON and Markdown reports.

```bash
cd ~/scratch/FluorCast

export FLUORCAST_SMILES="YOUR_CHROMOPHORE_SMILES"
export FLUORCAST_SOLVENT_SMILES="YOUR_SOLVENT_SMILES"
export FLUORCAST_OUT_DIR="outputs/predictions/example_full"
export FLUORCAST_TREE_MODEL_DIR="models/experiments_fluodb"
export FLUORCAST_NEURAL_MODEL_DIR="models/neural_experiments_fluodb"
export FLUORCAST_ABS_HYBRID_DIR="models/production_hybrid/absorption_nm"
export FLUORCAST_EM_HYBRID_DIR="models/production_hybrid/emission_nm"
export FLUORCAST_QY_HYBRID_DIR="models/production_hybrid/quantum_yield"

mkdir -p outputs/slurm

sbatch slurm/run_predict_full_fluorcast.sbatch
```

The wrapper uses `models/experiments_fluodb` for tree models and
`models/neural_experiments_fluodb` for neural models. Override them with
`FLUORCAST_TREE_MODEL_DIR` and `FLUORCAST_NEURAL_MODEL_DIR`. Set
`FLUORCAST_SKIP_HYBRID=1` to produce a base-model-only report. Production full
reports require the selected base-model directories to include absorption
artifacts.

For a base-only smoke test against an older absorption-capable tree directory:

```bash
export FLUORCAST_TREE_MODEL_DIR="models/chemfluor_combined_fluodb"
export FLUORCAST_NEURAL_MODEL_DIR="models/does_not_exist"
export FLUORCAST_SKIP_HYBRID="1"
mkdir -p outputs/slurm
sbatch slurm/run_predict_full_fluorcast.sbatch
```

Only explicitly trained production hybrid artifacts belong under:

```text
models/production_hybrid/absorption_nm/
models/production_hybrid/emission_nm/
models/production_hybrid/quantum_yield/
```

Override those locations with `FLUORCAST_ABS_HYBRID_DIR`,
`FLUORCAST_EM_HYBRID_DIR`, and `FLUORCAST_QY_HYBRID_DIR`. These directories
contain trained model artifacts and should not be committed.

### Base-model comparison prediction

Use `scripts/predict_all_models.py` after trained base-model artifacts exist.
For the prepared benchmark/presentation prediction, use the included Slurm script:

```bash
cd ~/scratch/FluorCast
mkdir -p outputs/slurm
sbatch slurm/base_models/run_predict_all_models.sbatch
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

### Desktop app JSON prediction

The desktop-app runner accepts `model_choice: "hybrid"` and returns one full
FluorCast prediction record.

```json
{
  "job_id": "job-example-001",
  "user_id": "user-example-001",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "model_choice": "hybrid",
  "requested_at": "2026-07-03T14:30:00Z"
}
```

Submit with:

```bash
export FLUORCAST_INPUT_JSON="jobs/job-example-001/input.json"
export FLUORCAST_OUTPUT_JSON="jobs/job-example-001/output.json"
sbatch slurm/run_prediction_job.sbatch
```

## Check Results

Main output folders:

```text
outputs/model_experiments_fluodb/
outputs/neural_model_experiments_fluodb/
outputs/graph_gin_emission_3seeds_gpu/
outputs/graph_gcn_emission_3seeds_gpu/
outputs/hybrid_three_way/
outputs/paired_stokes_three_way/
outputs/paired_spectral_three_way/
outputs/predictions/
models/production_hybrid/
models/paired_stokes_three_way/
models/hybrid_three_way/
models/experiments_fluodb/
models/neural_experiments_fluodb/
models/graph_gin_emission_3seeds_gpu/
models/graph_gcn_emission_3seeds_gpu/
```

Useful commands:

```bash
cat outputs/model_experiments_fluodb/model_comparison.md
cat outputs/neural_model_experiments_fluodb/all_model_comparison.md
cat outputs/graph_seed_summary_grouped.csv
cat outputs/hybrid_three_way/*/*/metrics_summary.md
cat outputs/paired_stokes_three_way/*/paired_spectral_metrics_summary.md
cat outputs/paired_spectral_three_way/*/*/paired_spectral_metrics_summary.md
cat outputs/predictions/example_full/full_fluorcast_report.md
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

 
## ConforFormer: 3D Molecular Representations

FluorCast integrates **ConforFormer**, a pretrained molecular representation model that uses 3D molecular conformations to generate learned molecular embeddings.

Unlike conventional Morgan fingerprints, which encode local molecular connectivity, ConforFormer incorporates conformer-dependent geometric information into its learned representations.

The ConforFormer workflow supports prediction of:

- Absorption wavelength (`absorption_nm`)
- Emission wavelength (`emission_nm`)
- Quantum yield (`quantum_yield`)
- Stokes shift (`stokes_shift_nm`)

### Original Implementation and Attribution

The ConforFormer molecular encoder was originally developed by the authors of the [ConforFormer project](https://github.com/EPiCs-group/ConforFormer).

FluorCast builds upon the pretrained ConforFormer architecture by integrating its 3D molecular embeddings into a solvent-aware machine learning pipeline for predicting absorption wavelengths, emission wavelengths, quantum yields, and Stokes shifts.

The original ConforFormer implementation is included as a Git submodule under `third_party/ConforFormer`. Pretrained model weights are available through the [official Hugging Face repository](https://huggingface.co/ConforFormer/ConforFormer).

FluorCast's downstream training, molecular feature fusion, and fluorescence prediction workflows are implemented separately from the original ConforFormer project.

### 1. Model Architecture

The ConforFormer pipeline follows these steps:

```text
Chromophore SMILES
    |
    v
RDKit canonicalization
    |
    v
3D conformer generation
    |
    v
Pretrained ConforFormer encoder
    |
    v
512-dimensional conformer embeddings
    |
    v
Conformer pooling
    |
    +---- Mean pooling
    +---- Lowest-energy pooling
    +---- Boltzmann-weighted pooling (298 K)
    |
    v
Molecular embedding + solvent descriptors
    |
    v
Downstream supervised regression
    |
    v
Absorption / Emission / Quantum Yield / Stokes Shift
```

Three molecular feature configurations are supported:

| Feature set | Description |
|---|---|
| `conforformer_solvent` | ConforFormer embedding + solvent descriptors |
| `morgan_solvent` | Morgan fingerprint + solvent descriptors |
| `conforformer_morgan_solvent` | ConforFormer embedding + Morgan fingerprint + solvent descriptors |

The Morgan-only configuration provides a baseline for evaluating the contribution of 3D molecular representations.

### 2. Environment Setup

The ConforFormer encoder requires a separate compatible Python environment with PyTorch, RDKit, and the upstream Uni-Core/Uni-Mol dependencies.

The Nibi workflow uses the following environment path:

```bash
cd ~/scratch/FluorCast

module purge
module load python/3.11
module load gcc
module load rdkit

export FLUORCAST_PYTHON="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"
```

The environment must be provisioned with compatible encoder dependencies before running the pipeline. Installing the standard FluorCast requirements alone is not sufficient.

Initialize the upstream ConforFormer submodule:

```bash
git submodule update --init --recursive
```

Verify the environment:

```bash
"$FLUORCAST_PYTHON" -c "import torch, rdkit; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

"$FLUORCAST_PYTHON" scripts/smoke_conforformer_encoder.py --env-report
```

Detailed dependency setup and compatibility instructions are available in:

- `docs/conforformer/WSL2_ENCODER_SETUP.md`
- `docs/conforformer/LOCAL_ENCODER_COMPATIBILITY.md`
- `docs/conforformer/CHECKPOINT_ASSET_MAP.md`

### 3. Required Model Assets

The pretrained ConforFormer checkpoint must be available locally:

```text
models/conforformer/ConforFormer.pt
```

The repository includes the dictionary configuration:

```text
configs/conforformer/OMOL_full_dict.txt
```

**Note:** Pretrained checkpoint weights and trained downstream model artifacts are not tracked in Git. Download or transfer the checkpoint separately and verify its compatibility with the dictionary and encoder.

The upstream checkpoint is available from:

https://huggingface.co/ConforFormer/ConforFormer

### 4. Training ConforFormer Models

The full training workflow uses the standardized FluorCast dataset:

```text
data/processed/fluodb_lite/combined_deduplicated.csv
```

The training process consists of four main stages:

1. Generate an inventory of unique chromophores.
2. Generate and cache 3D molecular conformers.
3. Extract ConforFormer embeddings using GPU inference.
4. Train solvent-aware downstream regression models.

The included Slurm scripts automate the full workflow.

#### Run a Small Validation Experiment

Before submitting the full dataset, run a small canary experiment:

```bash
cd ~/scratch/FluorCast

export FLUORCAST_PYTHON="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"

export FLUORCAST_DATASET="data/processed/fluodb_lite/combined_deduplicated.csv"

export FLUORCAST_CHECKPOINT="models/conforformer/ConforFormer.pt"

export FLUORCAST_DICTIONARY="configs/conforformer/OMOL_full_dict.txt"

export FLUORCAST_MAX_MOLECULES=256
export FLUORCAST_SHARD_SIZE=128

export FLUORCAST_RUN_ID="canary_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p outputs/slurm

bash slurm/conforformer/submit_full_pipeline.sh
```

#### Train on the Full Dataset

Once the validation run is successful:

```bash
cd ~/scratch/FluorCast

export FLUORCAST_PYTHON="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"

export FLUORCAST_DATASET="data/processed/fluodb_lite/combined_deduplicated.csv"

export FLUORCAST_CHECKPOINT="models/conforformer/ConforFormer.pt"

export FLUORCAST_DICTIONARY="configs/conforformer/OMOL_full_dict.txt"

unset FLUORCAST_MAX_MOLECULES

export FLUORCAST_SHARD_SIZE=128

export FLUORCAST_RUN_ID="full_$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p outputs/slurm

bash slurm/conforformer/submit_full_pipeline.sh
```

The submission script automatically schedules CPU conformer generation, GPU embedding extraction, embedding finalization, and downstream model training using Slurm dependencies.

Monitor jobs:

```bash
squeue -u "$USER"
```

Inspect logs:

```bash
ls -lh outputs/slurm/
```

#### Downstream Training Configuration

The default downstream workflow uses a molecule-grouped split to reduce leakage between training and evaluation.

The training, validation, and test proportions are:

```text
Training:   60%
Validation: 20%
Test:       20%
```

A scaffold-grouped split is also supported.

The downstream trainer evaluates ConforFormer representations, Morgan fingerprints, and feature fusion, using available target labels and fixed split assignments.

It supports three ConforFormer pooling methods:

- `mean`
- `lowest_energy`
- `boltzmann_298k`

To train downstream models manually using existing embeddings:

```bash
"$FLUORCAST_PYTHON" scripts/train_conforformer_downstream.py \
  --dataset data/processed/fluodb_lite/combined_deduplicated.csv \
  --embedding-run-root outputs/conforformer/full_dataset/YOUR_RUN_ID \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-dir outputs/conforformer/downstream/YOUR_RUN_ID \
  --model-out-dir models/conforformer_downstream/YOUR_RUN_ID \
  --split-type molecule \
  --seed 0 \
  --n-jobs 16
```

Replace `YOUR_RUN_ID` with the actual training run identifier.

### 5. Training Outputs

The embedding pipeline writes its results under:

```text
outputs/conforformer/full_dataset/<run_id>/
```

This directory contains:

```text
inventory/
conformer_status/
embeddings/
embedding_index.csv
embedding_manifest.json
embedding_summary.json
failed_molecules.csv
```

Downstream training results are written to:

```text
outputs/conforformer/downstream/<run_id>/
```

Important files include:

```text
metrics.csv
metrics.json
selection_results.csv
split_assignments.csv
leakage_check.json
training_manifest.json
predictions/
reports/
```

Trained model artifacts are stored under:

```text
models/conforformer_downstream/<run_id>/<split_type>/<pooling>/<feature_set>/<target>/
```

Each trained target directory contains a saved estimator (`model.joblib`) and its feature metadata (`feature_metadata.json`).

### 6. Predicting New Molecules

The ConforFormer prediction script supports individual chromophores and batch CSV input.

For new molecules, the script:

1. Canonicalizes the chromophore and solvent SMILES.
2. Generates conformers and ConforFormer embeddings.
3. Constructs the selected molecular and solvent features.
4. Loads previously trained downstream estimators.
5. Predicts molecular properties without retraining.

#### Single-Molecule Prediction

For example, predict molecular properties in ethanol:

```bash
cd ~/scratch/FluorCast

"$FLUORCAST_PYTHON" scripts/predict_conforformer_case.py \
  --smiles "c1ccccc1" \
  --solvent-smiles "CCO" \
  --model-root models/conforformer_downstream/YOUR_RUN_ID \
  --checkpoint models/conforformer/ConforFormer.pt \
  --dictionary configs/conforformer/OMOL_full_dict.txt \
  --split-type molecule \
  --pooling mean \
  --feature-set conforformer_morgan_solvent \
  --device cuda
```

Replace `YOUR_RUN_ID` with a trained run containing the requested model artifacts.

This example uses benzene in ethanol to demonstrate the prediction interface; it is not intended as a validated fluorescence benchmark.

For JSON output, add:

```bash
--json
```

For CPU inference, use `--device cpu`. Alternatively, use `--device auto` to select an available device.

#### Batch Prediction

Prepare a CSV with the following columns:

```csv
name,smiles,solvent_smiles
Molecule_1,c1ccccc1,CCO
Molecule_2,c1ccncc1,O
```

Run:

```bash
"$FLUORCAST_PYTHON" scripts/predict_conforformer_case.py \
  --input-csv inputs/my_molecules.csv \
  --output-csv outputs/predictions/conforformer_batch.csv \
  --model-root models/conforformer_downstream/YOUR_RUN_ID \
  --split-type molecule \
  --pooling mean \
  --feature-set conforformer_morgan_solvent \
  --device cuda
```

The output CSV includes:

```text
name
smiles
canonical_smiles
solvent_smiles
canonical_solvent_smiles
absorption_nm
emission_nm
quantum_yield
stokes_shift_nm
derived_stokes_shift_nm
stokes_source
status
error
```

If a direct Stokes-shift model is available, `stokes_shift_nm` contains its prediction. Otherwise, the value is derived from predicted emission minus predicted absorption. `derived_stokes_shift_nm` always reports the latter calculation.

#### Alternative Feature Configurations

To predict using only ConforFormer embeddings and solvent descriptors:

```bash
--feature-set conforformer_solvent --pooling mean
```

To predict using the Morgan baseline:

```bash
--feature-set morgan_solvent --pooling not_applicable
```

The corresponding trained model artifacts must exist.

### 7. Model Validation

Run the ConforFormer tests:

```bash
python -m pytest \
  tests/test_conforformer_adapter.py \
  tests/test_conforformer_cache.py \
  tests/test_conforformer_config.py \
  tests/test_conforformer_dictionary.py \
  tests/test_conformer_generation.py \
  tests/test_conforformer_preprocess.py \
  tests/test_conforformer_full_pipeline.py \
  tests/test_predict_conforformer_case.py
```

The full dataset workflow validates embedding integrity, checks molecule-level leakage, and records reproducibility metadata.

Model comparisons should use the same target, data split, and comparison cohort. Metrics from different cohorts or training protocols should not be treated as directly equivalent.

### 8. Additional Documentation

Detailed implementation and experiment documentation:

- [Full Dataset Workflow](docs/conforformer/FULL_DATASET_WORKFLOW.md)
- [Implementation Plan](docs/conforformer/IMPLEMENTATION_PLAN.md)
- [Checkpoint Asset Map](docs/conforformer/CHECKPOINT_ASSET_MAP.md)
- [Local Encoder Compatibility](docs/conforformer/LOCAL_ENCODER_COMPATIBILITY.md)
- [Upstream Audit](docs/conforformer/UPSTREAM_AUDIT.md)

**Note:** ConforFormer model checkpoints, intermediate embedding caches, trained downstream estimators, and generated predictions should remain outside Git version control. The repository tracks the implementation, configurations, scripts, and documentation needed to reproduce experiments.


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

Before committing, inspect the working tree and stage only the files you intend
to include. Avoid broad `git add` commands when generated outputs or local
artifacts are present.

```bash
git status --short
git diff -- README.md
git add README.md
# Repeat git diff and git add only for other files you intentionally changed.
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
