# ChemFluor Development Notes — GPU Graph Neural Networks + Seed Stability

## Current Project State

The ChemFluor project has now moved beyond tree models and fingerprint-based MLPs into true graph neural network experiments.

The combined dataset is still:

```bash
data/processed/fluodb_lite/combined_deduplicated.csv
```

The solvent descriptor file is still:

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

The graph workflow is handled by:

```bash
scripts/run_graph_model_experiments.py
```

It now supports:

```bash
--models graph_gcn,graph_gin,graph_mpnn
--targets emission_nm,quantum_yield
--seed
--benchmark-smiles
--benchmark-solvent-smiles
--known-emission-nm
--known-quantum-yield
```

Seed support was added in the latest update. Verification on Nibi showed:

```text
[--seed SEED]
--seed SEED           Seed controlling data splitting and graph model
```

This means graph experiments can now be run reproducibly across multiple seeds.

## Graph Representation

For the graph models, SMILES are not being used only as fingerprint vectors.

The graph workflow is:

```text
SMILES string → RDKit molecule → molecular graph → graph neural network
```

The molecular graph representation is:

```text
atoms = nodes
bonds = edges
atom and bond properties = graph features
```

The graph model learns a molecular embedding from atom-bond structure. That learned molecular embedding is then combined with solvent descriptors for prediction.

Current graph model input structure:

```text
chromophore molecular graph + solvent descriptor vector → prediction head
```

The solvent is still represented as numeric descriptors, not as a separate solvent graph.

## GPU Access on Nibi

Nibi GPU training worked successfully using an H100 GPU.

Confirmed GPU run output:

```text
CUDA available: True
CUDA device count: 1
CUDA device name: NVIDIA H100 80GB HBM3
```

The correct Nibi GPU request line used successfully was:

```bash
#SBATCH --gpus-per-node=h100:1
```

Useful GPU partition:

```bash
#SBATCH --partition=gpubase_bygpu_b2
```

The GPU graph GCN run completed 50 epochs in roughly 7 minutes. This was a major improvement over CPU, where graph training was too slow.

## Baseline Non-Graph Model Results

The strongest non-graph models from the expanded ChemFluor + Deep4Chem + FluoDB-Lite comparison remain:

### Emission Wavelength

| Model | Emission MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| RF | 23.8493 nm | 37.8891 | 0.8375 |
| ExtraTrees | 28.2001 nm | 48.2149 | 0.7369 |
| HistGB | 29.3118 nm | 40.4835 | 0.8145 |
| Best MLP | 30.7845 nm | 46.9168 | 0.7509 |
| GBDT | 40.1038 nm | 51.8589 | 0.6956 |

Main conclusion:

```text
RF remains the strongest global emission model.
```

### Quantum Yield

| Model | QY MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| ExtraTrees | 0.1464 | 0.2203 | 0.5052 |
| RF | 0.1505 | 0.2113 | 0.5448 |
| Best MLP | 0.1519 | 0.2131 | 0.5369 |
| HistGB | 0.1749 | 0.2254 | 0.4818 |
| GBDT | 0.2087 | 0.2540 | 0.3420 |

Main conclusion:

```text
ExtraTrees has the lowest QY MAE, while RF has the best QY R² among the tree models.
```

## GPU Graph GCN First Result

A GPU-trained graph GCN on `emission_nm` gave:

| Model | Target | MAE | RMSE | R² | Train Rows | Test Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| graph_gcn | emission_nm | 31.0313 nm | 43.0088 | 0.7906 | 29061 | 8166 |

This improved heavily over the CPU GCN:

| Run | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| CPU GCN | 46.7820 nm | 62.5497 | 0.5572 |
| GPU GCN | 31.0313 nm | 43.0088 | 0.7906 |

Interpretation:

```text
GPU training made graph GCN a real model instead of a failed slow CPU experiment.
```

The GPU GCN did not beat RF globally, but it became competitive with the MLP models.

## GPU Graph GIN and MPNN Results

A follow-up GPU job trained `graph_gin` and `graph_mpnn` on emission.

| Model | Target | MAE | RMSE | R² | Train Rows | Test Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| graph_gin | emission_nm | 28.9787 nm | 39.8780 | 0.8200 | 29061 | 8166 |
| graph_mpnn | emission_nm | 62.0897 nm | 82.8804 | 0.2225 | 29061 | 8166 |

Interpretation:

```text
Graph GIN was the best graph model in the first single-seed architecture comparison.
Graph MPNN failed in the current implementation and should not be prioritized unless the architecture is debugged or redesigned.
```

Graph GIN became competitive with strong non-graph baselines:

| Model | Emission MAE |
| --- | ---: |
| RF | 23.8493 nm |
| ExtraTrees | 28.2001 nm |
| graph_gin | 28.9787 nm |
| HistGB | 29.3118 nm |
| GPU graph_gcn | 31.0313 nm |

Key conclusion:

```text
Graph GIN is a legitimate result. It does not beat RF, but it is competitive with ExtraTrees and HistGB.
```

## Region-Wise Graph GIN Behavior

Graph GIN was especially strong in UV and blue emission regions.

| Region | RF MAE | graph_gin MAE | Better Model |
| --- | ---: | ---: | --- |
| UV | 33.7288 nm | 22.6917 nm | graph_gin |
| blue | 21.3743 nm | 19.5290 nm | graph_gin |
| green | 17.2402 nm | 26.1004 nm | RF |
| yellow/orange | 20.4143 nm | 30.8159 nm | RF |
| red/NIR | 34.5019 nm | 52.1268 nm | RF |

Interpretation:

```text
Graph GIN appears useful for UV/blue emission but still struggles for green through red/NIR emission.
```

The red/NIR weakness remains one of the main unsolved problems.

## Difficult Benchmark Molecule

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

All models mark this molecule as outside the reliable applicability domain:

```text
nearest_training_similarity = 0.4468
confidence_label = low-medium
outside_applicability_domain = True
```

Important benchmark observations:

1. An earlier GPU graph GCN run predicted `536.7621 nm`, giving only `2.2379 nm` emission error.
2. Single-seed graph GIN did not keep this benchmark advantage, predicting `479.5979 nm`, giving `59.4021 nm` error.
3. Multi-seed graph GIN later showed that benchmark predictions are seed-sensitive.

This means:

```text
Graph models can sometimes predict the difficult benchmark very well, but benchmark performance is not yet stable enough to rely on one seed.
```

## Seed Support Added

Seed support was added to `scripts/run_graph_model_experiments.py`.

Files changed by the update:

```text
README.md
scripts/run_graph_model_experiments.py
tests/test_graph_model_experiments.py
```

The seed option now controls graph experiment reproducibility.

Useful verification command:

```bash
python scripts/run_graph_model_experiments.py --help | grep -i seed
```

Successful output showed:

```text
[--seed SEED]
--seed SEED           Seed controlling data splitting and graph model
```

The output CSV/Markdown files now include a `seed` column for graph rows.

## Multi-Seed Emission Stability Results

Three-seed runs were completed for graph GIN and graph GCN emission.

### Graph GIN Emission Seeds

| Seed | MAE | RMSE | R² | Benchmark Prediction | Benchmark Error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30.2299 nm | 41.1515 | 0.8061 | 591.7267 nm | 52.7267 nm |
| 1 | 29.7446 nm | 40.7704 | 0.8150 | 557.0889 nm | 18.0889 nm |
| 2 | 28.4839 nm | 39.9788 | 0.8114 | 541.9888 nm | 2.9888 nm |

Grouped summary:

| Model | Target | Seeds | Mean MAE | MAE Std | Min MAE | Max MAE | Mean R² | Min R² | Max R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_gin | emission_nm | 3 | 29.4861 nm | 0.9013 | 28.4839 | 30.2299 | 0.8108 | 0.8061 | 0.8150 |

Interpretation:

```text
Graph GIN is globally stable across seeds, with emission MAE staying between 28.48 and 30.23 nm.
```

However, the difficult benchmark prediction changes substantially across seeds.

### Graph GCN Emission Seeds

| Seed | MAE | RMSE | R² | Benchmark Prediction | Benchmark Error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32.6955 nm | 44.1589 | 0.7768 | 564.3601 nm | 25.3601 nm |
| 1 | 28.0696 nm | 39.7366 | 0.8242 | 564.3798 nm | 25.3798 nm |
| 2 | 28.4375 nm | 40.7456 | 0.8041 | 637.8196 nm | 98.8196 nm |

Grouped summary:

| Model | Target | Seeds | Mean MAE | MAE Std | Min MAE | Max MAE | Mean R² | Min R² | Max R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_gcn | emission_nm | 3 | 29.7342 nm | 2.5711 | 28.0696 | 32.6955 | 0.8017 | 0.7768 | 0.8242 |

Interpretation:

```text
Graph GCN can reach excellent emission performance, but it is less stable than GIN across seeds.
```

## Graph Quantum Yield Results

Graph GIN and graph GCN were also trained on `quantum_yield` using seed 0.

| Model | Target | Seed | MAE | RMSE | R² | Train Rows | Test Rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| graph_gcn | quantum_yield | 0 | 0.1610 | 0.2271 | 0.4792 | 21783 | 6284 |
| graph_gin | quantum_yield | 0 | 0.1681 | 0.2393 | 0.4219 | 21783 | 6284 |

Compared to non-graph QY baselines:

| Model | QY MAE | R² |
| --- | ---: | ---: |
| ExtraTrees | 0.1464 | 0.5052 |
| RF | 0.1505 | 0.5448 |
| Best MLP | 0.1519 | 0.5369 |
| graph_gcn | 0.1610 | 0.4792 |
| graph_gin | 0.1681 | 0.4219 |

Conclusion:

```text
Graph models are not currently better for quantum yield prediction.
```

GCN is better than GIN for QY, but both are worse than ExtraTrees, RF, and the best MLP.

## Current Best Model Roles

| Role | Best Current Model |
| --- | --- |
| Best global emission model | RF |
| Best stable graph emission model | graph_gin |
| Best single-seed graph emission result | graph_gcn seed 1 |
| Best UV/blue emission model | graph_gin appears promising |
| Best red/NIR emission model | RF |
| Best QY MAE | ExtraTrees |
| Best QY R² | RF |
| Best graph QY | graph_gcn, but not competitive |
| Avoid for now | current graph_mpnn implementation |

## Final Graph Neural Network Interpretation

Use this wording in reports or presentations:

```text
Graph neural networks were tested by converting SMILES into molecular graphs and training GCN, GIN, and MPNN architectures with solvent descriptors appended to the learned molecular embedding. GPU training on Nibi H100 made graph experiments practical and substantially improved performance over CPU training. For emission wavelength, graph GIN was the most stable graph architecture across three seeds, achieving 29.49 ± 0.90 nm MAE and mean R² of 0.811. Graph GCN reached a slightly better best-case MAE of 28.07 nm but had higher seed variability, with 29.73 ± 2.57 nm MAE. RF remained the best global emission model at 23.85 nm MAE, but graph GIN was competitive with HistGB and ExtraTrees and performed especially well in UV/blue regions. For quantum yield, graph models did not outperform existing RF, ExtraTrees, or MLP baselines.
```

Shorter version:

```text
RF remains the best production model, but graph GIN is the strongest and most stable graph neural network for emission prediction. Graph models are promising for emission but not yet useful for quantum yield.
```

## Useful Output Files

Main previous tree/MLP comparison outputs:

```bash
outputs/model_experiments_fluodb/model_comparison.md
outputs/model_experiments_fluodb/error_by_region_comparison.csv
outputs/model_experiments_fluodb/benchmark_prediction_comparison.csv
outputs/neural_model_experiments_fluodb/
```

Graph GCN GPU emission output:

```bash
outputs/graph_gcn_emission_gpu/
```

Graph GIN/MPNN emission output:

```bash
outputs/graph_gin_mpnn_emission_gpu/
```

Graph QY outputs:

```bash
outputs/graph_gin_qy_gpu/
outputs/graph_gcn_qy_gpu/
```

Multi-seed graph outputs:

```bash
outputs/graph_gin_emission_3seeds_gpu/seed_0/
outputs/graph_gin_emission_3seeds_gpu/seed_1/
outputs/graph_gin_emission_3seeds_gpu/seed_2/

outputs/graph_gcn_emission_3seeds_gpu/seed_0/
outputs/graph_gcn_emission_3seeds_gpu/seed_1/
outputs/graph_gcn_emission_3seeds_gpu/seed_2/
```

Grouped seed summary:

```bash
outputs/graph_seed_summary.csv
outputs/graph_seed_summary_grouped.csv
```

Current grouped graph seed summary:

```csv
model,target,seeds,mae_mean,mae_std,mae_min,mae_max,r2_mean,r2_min,r2_max
graph_gcn,emission_nm,3,29.734208556063404,2.571149093320753,28.06959450798909,32.6955065883717,0.8016901659652498,0.7767717601655826,0.8242470929777128
graph_gcn,quantum_yield,1,0.1609601966656532,,0.1609601966656532,0.1609601966656532,0.4791675731348044,0.4791675731348044,0.4791675731348044
graph_gin,emission_nm,3,29.48613133365366,0.9012852907192835,28.4838509521042,30.229945591517858,0.8108273568798205,0.8061419570562337,0.8149829109804628
graph_gin,quantum_yield,1,0.1681400047225688,,0.1681400047225688,0.1681400047225688,0.4218547801693504,0.4218547801693504,0.4218547801693504
```

## What To Do Next

The project now has enough graph-model evidence for a strong presentation. The next steps should focus on making the results easier to communicate and improving only the clearest weak points.

### Priority 1: Make a final combined comparison table

Create one clean table with:

```text
RF
ExtraTrees
HistGB
Best MLP
graph_gin 3-seed mean
graph_gcn 3-seed mean
graph_mpnn
```

Include columns:

```text
model
family
target
MAE
RMSE
R²
seed mean/std if applicable
best use case
```

This will be the main table for the presentation.

### Priority 2: Create one concise graph-model figure

Make a bar plot comparing emission MAE:

```text
RF = 23.8493
ExtraTrees = 28.2001
HistGB = 29.3118
graph_gin mean = 29.4861
graph_gcn mean = 29.7342
Best MLP = 30.7845
GBDT = 40.1038
MPNN = 62.0897
```

This figure should make the key point visually:

```text
RF still wins, but graph GIN/GCN are competitive with strong baselines.
```

### Priority 3: Do not spend much more time on MPNN

The current MPNN result is too weak:

```text
graph_mpnn emission MAE = 62.0897 nm
R² = 0.2225
```

Do not rerun MPNN unless debugging the implementation itself.

### Priority 4: Investigate red/NIR weakness only if time allows

Graph models still struggle in red/NIR. Useful possible next work:

```text
1. Oversample red/NIR examples during graph training.
2. Add wavelength-region weighting to the loss.
3. Train a red/NIR-specialist model.
4. Use ensemble disagreement to flag red/NIR uncertainty.
```

This is a good future-work slide, not necessarily something to finish before a presentation.

### Priority 5: Prepare final presentation story

The clean story is:

```text
1. Expanded the dataset using ChemFluor + Deep4Chem + FluoDB-Lite.
2. Built standardized preprocessing and solvent descriptors.
3. Compared tree models and fingerprint-based neural networks.
4. Added applicability-domain/benchmark analysis.
5. Implemented true graph neural networks from SMILES-derived molecular graphs.
6. Moved graph training to Nibi H100 GPU.
7. Added seed control and ran graph stability experiments.
8. Found that RF remains the best production model, but graph GIN is a strong and stable graph-based emission model.
9. Found that graph models do not yet improve quantum yield.
10. Identified red/NIR prediction and low-similarity extrapolation as remaining weaknesses.
```

## Commands To Recheck Key Results

View graph seed summary:

```bash
cat outputs/graph_seed_summary_grouped.csv
```

View GIN emission seeds:

```bash
for SEED in 0 1 2; do
  echo "--- GIN seed ${SEED} ---"
  cat outputs/graph_gin_emission_3seeds_gpu/seed_${SEED}/graph_model_comparison.md
done
```

View GCN emission seeds:

```bash
for SEED in 0 1 2; do
  echo "--- GCN seed ${SEED} ---"
  cat outputs/graph_gcn_emission_3seeds_gpu/seed_${SEED}/graph_model_comparison.md
done
```

View graph QY results:

```bash
echo "=== GIN QY ==="
cat outputs/graph_gin_qy_gpu/graph_model_comparison.md

echo "=== GCN QY ==="
cat outputs/graph_gcn_qy_gpu/graph_model_comparison.md
```

## What To Tell The Next Chat

Use this context in a new chat:

```text
We have completed GPU graph neural network experiments for ChemFluor. SMILES are now converted into molecular graphs for graph_gcn, graph_gin, and graph_mpnn. GPU training works on Nibi H100. Graph GIN is the best stable graph emission model, with 29.49 ± 0.90 nm MAE across three seeds. Graph GCN reached a best seed of 28.07 nm MAE but is less stable, with 29.73 ± 2.57 nm mean MAE. RF is still best globally for emission at 23.85 nm MAE. Graph models do not beat RF/ExtraTrees/MLP for quantum yield. The next task is to make a final combined comparison table/figure and prepare presentation-ready conclusions.
```

---

# Planned Next Feature — All-Model Prediction Interface

## Motivation

A strong next step is to turn the trained ChemFluor models into a usable prediction system.

Instead of only evaluating models on test sets, the project should allow a user to input:

```text
chromophore SMILES + solvent SMILES or solvent name
```

and receive predictions from all available model families at once.

Desired workflow:

```text
input molecule SMILES + solvent SMILES/name
→ run RF / ExtraTrees / HistGB / MLP / graph_gcn / graph_gin
→ return emission and QY predictions
→ show model agreement/disagreement
→ show applicability-domain similarity
→ label prediction confidence
```

This would make the project feel much more complete because it turns the current model comparison pipeline into a practical fluorescent-molecule prediction tool.

## Why This Is Useful

Different models are strongest in different settings.

| Situation | Best Model / Strategy |
| --- | --- |
| General/global emission prediction | RF |
| UV/blue emission prediction | graph_gin may be useful |
| Unfamiliar or outside-domain molecules | compare RF, graph_gcn, graph_gin, and MLP disagreement |
| Quantum yield prediction | RF / ExtraTrees / best MLP |
| Benchmark-like molecules | graph models may help, but only if agreement is reasonable |

The stronger system is not just one best model. The stronger system is an ensemble-style prediction report:

```text
Use all models together and report uncertainty from disagreement.
```

This is scientifically stronger than blindly trusting a single model.

## Desired Command

Example future command:

```bash
python scripts/predict_all_models.py \
  --smiles "YOUR_CHROMOPHORE_SMILES" \
  --solvent-smiles "YOUR_SOLVENT_SMILES"
```

or:

```bash
python scripts/predict_all_models.py \
  --smiles "YOUR_CHROMOPHORE_SMILES" \
  --solvent "DMSO" \
  --out outputs/predictions/example_prediction.csv
```

## Desired Output Format

For one input molecule, the terminal output should look conceptually like this:

```text
Input molecule: ...
Input solvent: ...

Applicability domain:
Nearest training similarity: 0.447
Confidence: low-medium
Outside applicability domain: True

Emission predictions:
RF: 459.6 nm
ExtraTrees: 469.9 nm
HistGB: 507.7 nm
MLP: 483.1 nm
Graph GCN: 536.8 nm
Graph GIN: 542.0 nm

Emission consensus:
Mean: ... nm
Median: ... nm
Std: ... nm
Range: ... nm

Quantum yield predictions:
RF: 0.125
ExtraTrees: 0.061
MLP: 0.200
Graph GCN: 0.089
Graph GIN: 0.181

QY consensus:
Mean: ...
Median: ...
Std: ...
Range: ...

Recommended interpretation:
High agreement → higher confidence
High disagreement → low confidence / outside-domain warning
```

## Most Important Feature: Model Disagreement

Model disagreement should become one of the best uncertainty signals.

If RF, ExtraTrees, HistGB, MLP, GCN, and GIN all predict similar emission wavelengths, confidence should be higher.

Example:

```text
RF = 522 nm
ExtraTrees = 528 nm
HistGB = 519 nm
MLP = 531 nm
Graph GIN = 526 nm
Graph GCN = 524 nm
```

This is a high-agreement prediction.

However, if models disagree strongly:

```text
RF = 460 nm
HistGB = 508 nm
MLP = 483 nm
Graph GCN = 537 nm
Graph GIN = 592 nm
```

then the output should be flagged as uncertain even if one model may be correct.

This is especially important for molecules outside the applicability domain.

## Implementation Plan

Build the prediction tool in stages.

### Stage 1: Tree + MLP Prediction

Start with the easier model families:

```text
RF
ExtraTrees
HistGB
best MLP
```

These models use fingerprint/descriptor-style feature vectors.

### Stage 2: Add Graph Models

Then add:

```text
graph_gcn
graph_gin
```

These require converting the input SMILES into a molecular graph and loading saved graph model checkpoints.

### Stage 3: Add Consensus and Confidence Report

Add summary statistics:

```text
mean prediction
median prediction
standard deviation
min/max range
nearest training similarity
confidence label
outside applicability domain flag
```

## Proposed New Script

Recommended script name:

```bash
scripts/predict_all_models.py
```

Possible module location if preferred:

```bash
src/chemfluor/predict_all.py
```

The script should reuse existing project utilities as much as possible:

```text
SMILES canonicalization
feature generation
solvent descriptor loading
graph conversion
applicability-domain scoring
model loading
prediction formatting
```

Avoid duplicating large blocks of existing feature-generation or graph-conversion code.

## Codex Prompt For Next Chat

Use this prompt to implement the all-model prediction feature:

```text
Please add a ChemFluor all-model prediction script.

Goal:
Create a script that takes one chromophore SMILES and one solvent SMILES or solvent name, then runs all available trained models and outputs a comparison table of predictions.

Add a new script:
scripts/predict_all_models.py

Inputs:
--smiles
--solvent-smiles optional
--solvent optional
--solvent-descriptors default data/solvent_descriptors_expanded_deep4chem.csv
--standardized-combined default data/processed/fluodb_lite/combined_deduplicated.csv
--tree-model-dir default outputs/model_experiments_fluodb or models/model_experiments_fluodb depending on current saved model location
--neural-model-dir default outputs/neural_model_experiments_fluodb or models/neural_model_experiments_fluodb
--graph-model-dirs optional list of graph model directories, including:
  models/graph_gcn_emission_gpu
  models/graph_gin_mpnn_emission_gpu
  models/graph_gin_emission_3seeds_gpu/seed_0
  models/graph_gin_emission_3seeds_gpu/seed_1
  models/graph_gin_emission_3seeds_gpu/seed_2
  models/graph_gcn_emission_3seeds_gpu/seed_0
  models/graph_gcn_emission_3seeds_gpu/seed_1
  models/graph_gcn_emission_3seeds_gpu/seed_2
--out optional CSV path

Required behavior:
1. Canonicalize the input SMILES with RDKit.
2. Load solvent descriptors from the provided CSV.
3. Build the same features used during training for tree/MLP models.
4. Convert the input SMILES into a molecular graph for graph models.
5. Run every model that can be loaded successfully.
6. Output a table with:
   model
   model_family
   target
   seed if available
   predicted_emission_nm
   predicted_quantum_yield
   nearest_training_similarity
   confidence_label
   outside_applicability_domain
7. Add model disagreement summary:
   emission mean, median, std, min, max, range
   QY mean, median, std, min, max, range
8. If a model cannot be loaded, skip it with a warning instead of crashing.
9. Save results to CSV if --out is given.
10. Print a clear terminal report.
11. Add tests with tiny mock models or minimal fixtures to verify:
   --help works
   invalid SMILES fails clearly
   output CSV is written
   disagreement summary has expected columns

Important:
Reuse existing feature generation, graph conversion, applicability-domain, and prediction utilities as much as possible. Do not duplicate large blocks of code if helpers already exist.
```

## Updated Next-Chat Context

Use this in the next chat:

```text
We have completed GPU graph neural network experiments for ChemFluor. SMILES are now converted into molecular graphs for graph_gcn, graph_gin, and graph_mpnn. GPU training works on Nibi H100. Graph GIN is the best stable graph emission model, with 29.49 ± 0.90 nm MAE across three seeds. Graph GCN reached a best seed of 28.07 nm MAE but is less stable, with 29.73 ± 2.57 nm mean MAE. RF is still best globally for emission at 23.85 nm MAE. Graph models do not beat RF/ExtraTrees/MLP for quantum yield. Seed support has been added to scripts/run_graph_model_experiments.py. The next feature to implement is scripts/predict_all_models.py, where a user inputs chromophore SMILES and solvent SMILES/name, and all trained models predict concurrently with applicability-domain and model-disagreement confidence reporting.
```
