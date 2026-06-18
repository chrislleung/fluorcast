# ChemFluor Development Notes — Model Comparison + Neural Network Next Steps

## Current Project State

The ChemFluor project now has an expanded combined training workflow using:

* Original ChemFluor dataset
* Deep4Chem dataset
* FluoDB-Lite dataset

The combined standardized dataset is:

```bash
data/processed/fluodb_lite/combined_deduplicated.csv
```

The expanded solvent descriptor file is:

```bash
data/solvent_descriptors_expanded_deep4chem.csv
```

The current Nibi project folder is:

```bash
~/scratch/ChemFluor_Project
```

The Nibi virtual environment is:

```bash
~/scratch/chemfluor_env
```

Standard Nibi environment setup:

```bash
cd ~/scratch/ChemFluor_Project

module purge
module load python/3.11
module load gcc
module load rdkit

source ~/scratch/chemfluor_env/bin/activate
```

The GitHub/local code has been migrated successfully to Nibi. The current Nibi folder is synced with GitHub at commit:

```text
e7b0ca0 Add combined model experiment comparison workflow
```

The new experiment script exists and works:

```bash
scripts/run_combined_model_experiments.py
```

It supports:

```bash
--models rf,extratrees,histgb,gbdt,mlp
--targets emission_nm,quantum_yield
--benchmark-smiles
--benchmark-solvent-smiles
--known-emission-nm
--known-quantum-yield
```

## New Model-Comparison Workflow

The new script trains multiple model families and compares them automatically:

```bash
python scripts/run_combined_model_experiments.py \
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv \
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv \
  --out-root models/experiments_fluodb \
  --models rf,extratrees,histgb,gbdt \
  --targets emission_nm,quantum_yield \
  --compare-out outputs/model_experiments_fluodb \
  --benchmark-smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --benchmark-solvent-smiles "CS(=O)C" \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196
```

This script trains each requested model into:

```text
models/experiments_fluodb/rf/
models/experiments_fluodb/extratrees/
models/experiments_fluodb/histgb/
models/experiments_fluodb/gbdt/
```

and writes comparison outputs to:

```text
outputs/model_experiments_fluodb/model_comparison.csv
outputs/model_experiments_fluodb/model_comparison.md
outputs/model_experiments_fluodb/error_by_region_comparison.csv
outputs/model_experiments_fluodb/benchmark_prediction_comparison.csv
```

## Full Tree-Model Comparison Results

The full Nibi run completed for:

```text
rf
extratrees
histgb
gbdt
```

Targets:

```text
emission_nm
quantum_yield
```

### Overall Emission MAE

| Model      | Emission MAE |    RMSE |     R² |
| ---------- | -----------: | ------: | -----: |
| RF         |   23.8493 nm | 37.8891 | 0.8375 |
| ExtraTrees |   28.2001 nm | 48.2149 | 0.7369 |
| HistGB     |   29.3118 nm | 40.4835 | 0.8145 |
| GBDT       |   40.1038 nm | 51.8589 | 0.6956 |

Conclusion:

```text
Random Forest is still the best general-purpose emission model.
```

### Overall Quantum Yield MAE

| Model      | QY MAE |   RMSE |     R² |
| ---------- | -----: | -----: | -----: |
| ExtraTrees | 0.1464 | 0.2203 | 0.5052 |
| RF         | 0.1505 | 0.2113 | 0.5448 |
| HistGB     | 0.1749 | 0.2254 | 0.4818 |
| GBDT       | 0.2087 | 0.2540 | 0.3420 |

Conclusion:

```text
ExtraTrees is slightly best for overall quantum yield MAE, but RF is very close and has better QY R².
```

### Red/NIR Region MAE

From `error_by_region_comparison.csv`:

| Model      | Red/NIR MAE |
| ---------- | ----------: |
| RF         |  34.5019 nm |
| ExtraTrees |  35.7309 nm |
| HistGB     |  40.5023 nm |
| GBDT       |  59.6276 nm |

Conclusion:

```text
RF is also best overall in the red/NIR region.
```

## Known Difficult Benchmark Molecule

Benchmark molecule:

```text
SMILES:
O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2

Solvent:
DMSO

Solvent SMILES:
CS(=O)C

Known emission:
539 nm

Known quantum yield:
0.196
```

Benchmark prediction comparison:

| Model      | Predicted Emission | Emission Error | Predicted QY | QY Error |
| ---------- | -----------------: | -------------: | -----------: | -------: |
| RF         |        459.5974 nm |     79.4026 nm |       0.1245 |   0.0715 |
| ExtraTrees |        469.8762 nm |     69.1238 nm |       0.0611 |   0.1349 |
| HistGB     |        507.7491 nm |     31.2509 nm |       0.2892 |   0.0932 |
| GBDT       |        472.8330 nm |     66.1670 nm |       0.2704 |   0.0744 |

Important result:

```text
HistGB is worse overall than RF, but it performs much better on this specific difficult benchmark molecule.
```

All models returned:

```text
nearest_training_similarity = 0.4468
confidence_label = low-medium
outside_applicability_domain = True
```

Interpretation:

```text
This molecule is outside the reliable single-molecule prediction domain. The large disagreement between RF and HistGB is useful as an uncertainty signal.
```

## Current Best Interpretation

Use RF as the main production model:

```text
RF emission MAE = 23.8493 nm
RF red/NIR MAE = 34.5019 nm
RF QY MAE = 0.1505
```

Use HistGB as a secondary comparison model for difficult low-similarity molecules:

```text
HistGB may capture some red-shifted donor/acceptor behavior that RF smooths out, but it is worse overall.
```

Use ExtraTrees as a possible QY-specific model:

```text
ExtraTrees QY MAE = 0.1464
RF QY MAE = 0.1505
```

## Current Research Insight

A useful project insight is:

```text
Random Forest remains the strongest general model on the expanded ChemFluor + Deep4Chem + FluoDB-Lite dataset. However, for a difficult low-similarity benchmark molecule, HistGradientBoosting reduced the emission error from 79.4 nm to 31.3 nm. This suggests that model-family disagreement can serve as an uncertainty signal for extrapolative fluorescent molecule predictions.
```

## Next Goal: Try Better Neural Network Models

The first MLP experiment underperformed locally/debug, but it was a basic MLP. The next step should not be “give up on neural networks.” Instead, test better neural-network designs.

The current MLP likely struggled because:

```text
1. Fingerprint vectors are sparse/high-dimensional.
2. MLPs need careful scaling, regularization, and validation.
3. The model may need target transforms or better architecture.
4. The dataset is chemical-structure-heavy, where tree models are strong baselines.
```

## Neural Network Experiments to Try Next

### Experiment 1: Tuned sklearn MLP

Try several MLP architectures:

```text
mlp_small: 256, 128
mlp_medium: 512, 256
mlp_large: 1024, 512, 256
```

Try different regularization:

```text
alpha = 1e-3
alpha = 1e-4
alpha = 1e-5
```

Use early stopping.

Compare on:

```text
overall emission MAE
red/NIR emission MAE
benchmark molecule error
QY MAE
```

### Experiment 2: PyTorch Feedforward Neural Network

Implement a PyTorch MLP using the same feature matrix:

Input:

```text
Morgan FP + MACCS + RDKit descriptors + solvent descriptors
```

Architecture:

```text
Linear(input_dim → 1024)
BatchNorm
ReLU
Dropout(0.2)

Linear(1024 → 512)
BatchNorm
ReLU
Dropout(0.2)

Linear(512 → 256)
ReLU

Linear(256 → output)
```

Use:

```text
MSE loss or SmoothL1Loss
AdamW optimizer
early stopping on validation MAE
target scaling
feature scaling for continuous descriptors
```

Train separate models for:

```text
emission_nm
quantum_yield
```

### Experiment 3: Multi-task Neural Network

Instead of training separate models, train one network with multiple outputs:

```text
absorption_nm
emission_nm
quantum_yield
lifetime_ns
log_extinction
```

Only compute loss for targets that are present. Use a mask for missing labels.

This may help because absorption, emission, and QY are related photophysical properties.

### Experiment 4: Graph Neural Network

A stronger long-term neural model would use molecular graphs instead of only fingerprints.

Potential approach:

```text
SMILES → molecular graph → graph neural network embedding
solvent descriptors → dense vector
concatenate molecule embedding + solvent vector
predict emission/QY
```

Possible frameworks:

```text
PyTorch Geometric
DGL-LifeSci
Chemprop-style message passing neural network
```

This is more work but much more “real neural network for chemistry” than MLP on fingerprints.

## Recommended Next Practical Step

Do not jump straight to GNNs. First implement a better neural-model experiment script:

```text
scripts/run_neural_model_experiments.py
```

Start with:

```text
sklearn MLP hyperparameter sweep
PyTorch MLP on same features
optional multitask PyTorch MLP
```

Then compare them against:

```text
RF
HistGB
ExtraTrees
```

## Useful Next Commands

Check existing full comparison outputs:

```bash
cat outputs/model_experiments_fluodb/model_comparison.md
cat outputs/model_experiments_fluodb/error_by_region_comparison.csv
cat outputs/model_experiments_fluodb/benchmark_prediction_comparison.csv
```

Predict with RF:

```bash
python scripts/predict_combined_molecule.py \
  --smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --solvent-smiles "CS(=O)C" \
  --model-dir models/experiments_fluodb/rf \
  --model-type rf \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196 \
  --name difficult_benchmark_rf \
  --out outputs/predictions/difficult_benchmark_rf.json \
  --out-csv outputs/predictions/difficult_benchmark_rf.csv
```

Predict with HistGB:

```bash
python scripts/predict_combined_molecule.py \
  --smiles "O=C(S/C(SC)=C(SC)/SC)C1=CC2=C(C=C1)NC3=CC=CC=C3S2" \
  --solvent-smiles "CS(=O)C" \
  --model-dir models/experiments_fluodb/histgb \
  --model-type histgb \
  --known-emission-nm 539 \
  --known-quantum-yield 0.196 \
  --name difficult_benchmark_histgb \
  --out outputs/predictions/difficult_benchmark_histgb.json \
  --out-csv outputs/predictions/difficult_benchmark_histgb.csv
```

## What to Tell the Next Chat

The next chat should continue from this point:

```text
We have finished a full tree-model comparison on ChemFluor + Deep4Chem + FluoDB-Lite. RF is best overall for emission and red/NIR. ExtraTrees is slightly best for QY. HistGB is worse overall but much better on one difficult low-similarity benchmark molecule. I now want to implement stronger neural-network experiments, starting with tuned MLPs and PyTorch MLP/multitask models, then eventually graph neural networks.
```
