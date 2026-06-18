# ChemFluor Development Notes

This file summarizes the current ChemFluor development state so future work can continue without needing to reconstruct the full chat history.

---

# 1. Current Project Direction

ChemFluor started as a fluorescence-property prediction workflow. It has now been expanded into a combined prediction and first-pass inverse-design workflow.

The current goal is:

```text
molecule + solvent → predicted optical properties
```

and the first inverse-design extension is:

```text
target emission + solvent + candidate molecules → ranked candidate fluorophores
```

The current inverse-design step is not full neural generation yet. It uses rule-based scaffold enumeration to generate chemically reasonable candidates, then uses trained ML models to score and rank those candidates.

The long-term goal is to move toward a Gen-DL-like workflow:

```text
desired optical properties + solvent → generated molecule
```

but the current practical intermediate step is:

```text
known fluorophore scaffolds + substituents → candidate molecules → model-ranked candidates
```

---

# 2. Major Work Completed

## 2.1 Added Deep4Chem Dataset

The project was expanded from the original ChemFluor dataset to include the Deep4Chem chromophore dataset.

Raw Deep4Chem file used:

```text
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
```

Main dataset facts from exploratory analysis:

```text
20,836 total rows
14 columns
6,865 unique chromophores
1,363 unique solvent/environment entries
```

Target-property coverage:

| Target property            | Usable rows |
| -------------------------- | ----------: |
| Absorption max             |      17,703 |
| Emission max               |      18,851 |
| Quantum yield              |      13,978 |
| Lifetime                   |       6,949 |
| Log extinction coefficient |       8,300 |

All chromophore SMILES were valid RDKit molecules. Nearly all solvent entries were valid solvent SMILES, except for gas-phase rows labeled `gas`.

---

## 2.2 Added Data Standardization Layer

New reusable module:

```text
src/chemfluor/data_standardization.py
```

This file standardizes both ChemFluor and Deep4Chem into a shared schema.

Important functions:

```text
canonicalize_smiles
load_deep4chem
load_chemfluor
combine_training_data
```

Standardized output columns:

```text
chromophore_smiles
solvent_original
canonical_chromophore_smiles
canonical_solvent_smiles
absorption_nm
emission_nm
lifetime_ns
quantum_yield
log_extinction
source_dataset
```

Important design choice:

* Invalid chromophore SMILES are dropped.
* Rows with no usable target values are dropped.
* Duplicate standardized rows are removed.
* Solvents like `gas` do not become valid solvent SMILES, but rows can still remain if the molecule and targets are valid.

Tests were added:

```text
tests/test_data_standardization.py
```

Run tests with:

```powershell
python -m pytest tests
```

---

## 2.3 Added Expanded Solvent Descriptor Workflow

Script:

```text
scripts/make_deep4chem_solvent_descriptors.py
```

Purpose:

* Extract unique solvents from Deep4Chem.
* Validate solvent SMILES with RDKit.
* Compute RDKit solvent descriptors.
* Merge existing physical solvent descriptors from the original ChemFluor solvent descriptor file.
* Save an expanded solvent descriptor table.

Current canonical expanded descriptor file:

```text
data/solvent_descriptors_expanded_deep4chem.csv
```

Earlier temporary files included:

```text
data/solvent_descriptors_expanded_deep4chem_chatgpt.csv
data/solvent_descriptors_expanded_deep4chem_local.csv
```

The preferred canonical name is now:

```text
data/solvent_descriptors_expanded_deep4chem.csv
```

Command:

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv
```

---

## 2.4 Added Combined Model Training

Script:

```text
scripts/train_combined_predictors.py
```

This script trains solvent-aware prediction models on the combined ChemFluor + Deep4Chem dataset.

Model inputs:

```text
Morgan fingerprint of canonical chromophore SMILES
+
numeric solvent descriptor vector
```

Targets trained separately:

```text
absorption_nm
emission_nm
lifetime_ns
quantum_yield
log_extinction
```

Supported models:

```text
rf
histgb
```

Important split strategy:

```text
Grouped split by canonical_chromophore_smiles
```

This prevents the same chromophore from appearing in both train and test sets under different solvents.

### Random Forest training command

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined `
  --model rf
```

### HistGradientBoosting training command

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined_histgb `
  --model histgb
```

Model outputs include:

```text
metrics.json
feature_metadata.json
predictions_absorption_nm.csv
predictions_emission_nm.csv
predictions_lifetime_ns.csv
predictions_quantum_yield.csv
predictions_log_extinction.csv
*_rf.joblib or *_histgb.joblib
```

These should not be committed to GitHub.

---

# 3. Model Results

## 3.1 Random Forest Results

Random Forest was the strongest model overall by MAE.

| Target         |     MAE |    RMSE |     R² |
| -------------- | ------: | ------: | -----: |
| absorption_nm  | 23.5941 | 39.9978 | 0.8398 |
| emission_nm    | 31.3509 | 46.0230 | 0.7616 |
| lifetime_ns    |  4.9339 | 17.5009 | 0.5464 |
| quantum_yield  |  0.1755 |  0.2314 | 0.4366 |
| log_extinction |  0.2110 |  0.3323 | 0.7322 |

Interpretation:

* Absorption prediction is strongest.
* Emission prediction is good enough for first-pass color screening.
* Quantum yield is noisy and should be treated as a ranking signal, not an exact value.
* Lifetime has fewer labels and outlier issues.
* Log extinction is promising for future brightness estimation.

## 3.2 HistGradientBoosting Results

| Target         |     MAE |    RMSE |     R² |
| -------------- | ------: | ------: | -----: |
| absorption_nm  | 25.1528 | 37.0026 | 0.8629 |
| emission_nm    | 33.4053 | 45.7103 | 0.7649 |
| lifetime_ns    |  6.2098 | 21.3987 | 0.3219 |
| quantum_yield  |  0.1863 |  0.2370 | 0.4093 |
| log_extinction |  0.2204 |  0.3196 | 0.7522 |

Interpretation:

* HistGB is competitive by R² for some targets.
* Random Forest is still the better main baseline because it has better MAE overall.

---

# 4. Model Reporting and Error Analysis

## 4.1 Model Report Script

Script:

```text
scripts/report_combined_model_results.py
```

Command:

```powershell
python scripts/report_combined_model_results.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/combined_model_report
```

Outputs:

```text
outputs/combined_model_report/model_summary.md
outputs/combined_model_report/metrics_table.csv
outputs/combined_model_report/figures/
```

The report script creates:

```text
predicted vs actual plots
residual histograms
residual vs predicted plots
```

for each target.

## 4.2 Model Comparison Script

Script:

```text
scripts/compare_model_results.py
```

Command:

```powershell
python scripts/compare_model_results.py `
  --rf-dir models/chemfluor_combined `
  --histgb-dir models/chemfluor_combined_histgb `
  --out-dir outputs/model_comparison_report
```

Outputs:

```text
model_comparison.csv
model_comparison.md
mae_comparison.png
rmse_comparison.png
r2_comparison.png
```

## 4.3 Error Analysis Script

Script:

```text
scripts/analyze_prediction_errors.py
```

Command:

```powershell
python scripts/analyze_prediction_errors.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/error_analysis
```

Outputs include:

```text
overall_error_summary.csv
error_analysis_report.md
worst_predictions_<target>.csv
best_predictions_<target>.csv
error_by_source_dataset_<target>.csv
top_error_solvents_<target>.csv
error_by_wavelength_region_absorption_nm.csv
error_by_wavelength_region_emission_nm.csv
```

Purpose:

* Find where model performs best/worst.
* Identify whether errors are concentrated by dataset source, solvent, or wavelength region.
* Useful next debugging target for model improvement.

---

# 5. Candidate Generation

## 5.1 Current Candidate Generator

Script:

```text
scripts/generate_scaffold_candidates.py
```

This is rule-based scaffold enumeration, not neural generation.

It uses hardcoded/default fluorophore scaffold templates and substituent fragments.

Current scaffold families:

```text
coumarin_7_substituted
coumarin_6_substituted
coumarin_4_methyl_7_substituted
naphthalimide_4_substituted
naphthalimide_4_substituted_n_butyl
```

Current built-in substituents:

```text
H
methyl
ethyl
methoxy
ethoxy
dimethylamino
diethylamino
cyano
fluoro
chloro
trifluoromethyl
phenyl
```

Default generation command:

```powershell
python scripts/generate_scaffold_candidates.py
```

Expected output:

```text
Scaffold templates used: 5
Substituents used: 12
Raw combinations attempted: 60
Unique valid molecules saved: 59
Saved candidates to: data/generated_candidates/scaffold_candidates.csv
```

Current generated candidate file:

```text
data/generated_candidates/scaffold_candidates.csv
```

Candidate CSV columns:

```text
name
scaffold
substituent
smiles
canonical_smiles
```

Other useful commands:

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds coumarin `
  --out data/generated_candidates/coumarin_candidates.csv
```

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds naphthalimide `
  --out data/generated_candidates/naphthalimide_candidates.csv
```

```powershell
python scripts/generate_scaffold_candidates.py `
  --scaffolds all `
  --substituents cyano,methoxy,diethylamino,phenyl `
  --out data/generated_candidates/custom_candidates.csv
```

---

# 6. Candidate Screening

## 6.1 Current Screening Script

Script:

```text
scripts/screen_candidate_molecules.py
```

Purpose:

Screen candidate molecules using trained ChemFluor models.

Inputs:

```text
candidate molecule CSV
solvent SMILES
target emission wavelength
trained model directory
solvent descriptor CSV
output path
```

Outputs:

```text
ranked candidate CSV
```

Predicted columns:

```text
predicted_absorption_nm
predicted_emission_nm
predicted_quantum_yield
predicted_log_extinction
```

Ranking score:

```text
score = -abs(predicted_emission_nm - target_emission) + 200 * predicted_quantum_yield
```

Brightness estimate:

```text
estimated_brightness_score = predicted_quantum_yield * 10 ** predicted_log_extinction
```

The ranked output preserves candidate metadata:

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
emission_error_from_target
score
estimated_brightness_score
```

---

# 7. Candidate-Screening Results Completed

The same 59 generated candidates were screened in ethanol for:

```text
450 nm
520 nm
600 nm
```

Solvent:

```text
CCO
```

This means ethanol.

## 7.1 450 nm Screening

Command:

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 450 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450.csv
```

Top candidate:

```text
naphthalimide_4_substituted_n_butyl_phenyl
```

Predicted:

```text
emission = 458.035596 nm
quantum_yield = 0.437496
error_from_target = 8.035596 nm
score = 79.463685
```

Top 10 scaffold counts:

```text
naphthalimide_4_substituted            5
coumarin_4_methyl_7_substituted        2
naphthalimide_4_substituted_n_butyl    1
coumarin_6_substituted                 1
coumarin_7_substituted                 1
```

Interpretation:

* 450 nm target gives a mixed set of naphthalimide and coumarin candidates.
* The closest-to-target molecule was `naphthalimide_4_substituted_phenyl` at about 449.65 nm, but it ranked lower because of lower predicted QY.
* Ranking balances target closeness and predicted QY.

## 7.2 520 nm Screening

Command:

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 520 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520.csv
```

Earlier output file used during testing:

```text
outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520_metadata_test.csv
```

Top candidate:

```text
naphthalimide_4_substituted_n_butyl_cyano
```

Predicted:

```text
emission = 504.521911 nm
quantum_yield = 0.420209
error_from_target = 15.478089 nm
score = 68.563786
```

Top 10 scaffold counts:

```text
naphthalimide_4_substituted_n_butyl    8
naphthalimide_4_substituted            1
coumarin_4_methyl_7_substituted        1
```

Interpretation:

* 520 nm target strongly prefers N-butyl naphthalimide derivatives.
* Top substituents included cyano, ethoxy, trifluoromethyl, chloro, methoxy, methyl, and fluoro.
* Most top candidates predicted around 499–506 nm with QY around 0.39–0.42.

## 7.3 600 nm Screening

Command:

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 600 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600.csv
```

Top candidate:

```text
naphthalimide_4_substituted_n_butyl_dimethylamino
```

Predicted:

```text
emission = 562.335811 nm
quantum_yield = 0.427909
error_from_target = 37.664189 nm
score = 47.917666
```

Second candidate:

```text
naphthalimide_4_substituted_n_butyl_diethylamino
```

Predicted:

```text
emission = 563.416885 nm
quantum_yield = 0.417075
error_from_target = 36.583115 nm
score = 46.831916
```

Top 10 scaffold counts:

```text
naphthalimide_4_substituted_n_butyl    7
naphthalimide_4_substituted            2
coumarin_4_methyl_7_substituted        1
```

Interpretation:

* Current candidate library does not reach close enough to 600 nm.
* Best candidates are amino-substituted N-butyl naphthalimides around 562–563 nm.
* Need more red-shifted scaffolds for true orange/red screening.

## 7.4 Summary Table

| Target emission | Top candidate                                     | Scaffold              | Substituent   | Predicted emission | Predicted QY |   Error |
| --------------: | ------------------------------------------------- | --------------------- | ------------- | -----------------: | -----------: | ------: |
|          450 nm | naphthalimide_4_substituted_n_butyl_phenyl        | N-butyl naphthalimide | phenyl        |           458.0 nm |        0.437 |  8.0 nm |
|          520 nm | naphthalimide_4_substituted_n_butyl_cyano         | N-butyl naphthalimide | cyano         |           504.5 nm |        0.420 | 15.5 nm |
|          600 nm | naphthalimide_4_substituted_n_butyl_dimethylamino | N-butyl naphthalimide | dimethylamino |           562.3 nm |        0.428 | 37.7 nm |

Main interpretation:

* Workflow is functioning.
* Different target wavelengths change the ranking.
* 450 nm gives mixed scaffold preferences.
* 520 nm strongly favors N-butyl naphthalimides.
* 600 nm reveals that the current candidate library is not red-shifted enough.

---

# 8. GitHub / Repository State

Repository:

```text
https://github.com/chrislleung/ChemFluor/tree/main
```

Recent commit made:

```text
Add combined ChemFluor Deep4Chem workflow and candidate screening
```

That commit included:

```text
.gitignore update
README update
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/test1_candidate_molecules.csv
scripts/analyze_deep4chem_dataset.py
scripts/analyze_prediction_errors.py
scripts/compare_model_results.py
scripts/generate_scaffold_candidates.py
scripts/make_deep4chem_solvent_descriptors.py
scripts/report_combined_model_results.py
scripts/screen_candidate_molecules.py
scripts/train_combined_predictors.py
src/chemfluor/__init__.py
src/chemfluor/data_standardization.py
tests/test_data_standardization.py
```

Important `.gitignore` policy:

Do not commit:

```text
models/
outputs/
.venv/
*.joblib
*.pkl
*.pickle
Slurm output logs
cache folders
large raw/private data
```

Current desired README direction:

* Keep the original README sections that explain Nibi/SSH/Compute Canada workflow.
* Append the new combined Deep4Chem workflow section at the end.
* Do not replace the older server instructions.

If README was replaced incorrectly, restore the prior README from Git history and append the combined workflow section instead of replacing the file.

Useful commands:

```powershell
git log --oneline -- README.md
git show HEAD~1:README.md
git restore --source=HEAD~1 -- README.md
```

Then append the new workflow section and commit:

```powershell
git add README.md
git commit -m "Append combined Deep4Chem workflow instructions"
git push
```

---

# 9. Important Local File Paths

Local project root used in development:

```text
C:\Users\CL\OneDrive\Desktop\python\ChemFluor_Project_synced
```

Virtual environment:

```text
.venv
```

Core data paths:

```text
data/chemfluor_data.csv
data/solvent_descriptors.csv
data/solvent_descriptors_expanded_deep4chem.csv
data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv
```

Model directories:

```text
models/chemfluor_combined
models/chemfluor_combined_histgb
```

Generated candidates:

```text
data/generated_candidates/scaffold_candidates.csv
```

Candidate-screening outputs:

```text
outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450.csv
outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520.csv
outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600.csv
```

---

# 10. How to Reproduce the Current Workflow

From project root:

## 10.1 Install

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 10.2 Analyze Deep4Chem

```powershell
python scripts/analyze_deep4chem_dataset.py `
  --input "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv"
```

## 10.3 Generate solvent descriptors

```powershell
python scripts/make_deep4chem_solvent_descriptors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --existing-solvents data/solvent_descriptors.csv `
  --output data/solvent_descriptors_expanded_deep4chem.csv
```

## 10.4 Train Random Forest

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined `
  --model rf
```

## 10.5 Train HistGB

```powershell
python scripts/train_combined_predictors.py `
  --deep4chem "data/raw/deep4chem/DB for chromophore_Sci_Data_rev03.csv" `
  --chemfluor data/chemfluor_data.csv `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out-dir models/chemfluor_combined_histgb `
  --model histgb
```

## 10.6 Generate reports

```powershell
python scripts/report_combined_model_results.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/combined_model_report
```

```powershell
python scripts/compare_model_results.py `
  --rf-dir models/chemfluor_combined `
  --histgb-dir models/chemfluor_combined_histgb `
  --out-dir outputs/model_comparison_report
```

```powershell
python scripts/analyze_prediction_errors.py `
  --model-dir models/chemfluor_combined `
  --out-dir outputs/error_analysis
```

## 10.7 Generate candidates

```powershell
python scripts/generate_scaffold_candidates.py
```

## 10.8 Screen candidates

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 450 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450.csv
```

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 520 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520.csv
```

```powershell
python scripts/screen_candidate_molecules.py `
  --candidates data/generated_candidates/scaffold_candidates.csv `
  --solvent-smiles CCO `
  --target-emission 600 `
  --model-dir models/chemfluor_combined `
  --solvent-descriptors data/solvent_descriptors_expanded_deep4chem.csv `
  --out outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600.csv
```

---

# 11. What Needs To Be Done Next

## 11.1 Immediate Documentation Task

Finish README cleanup:

* Restore old README if needed so Nibi/SSH/Compute Canada instructions remain.
* Append the new combined Deep4Chem workflow section to the end.
* Commit and push.

Suggested commit:

```powershell
git add README.md DEVELOPMENT.md
git commit -m "Add development notes and expanded workflow documentation"
git push
```

## 11.2 Add a Candidate-Screening Summary Script

Create a new script:

```text
scripts/summarize_candidate_screening.py
```

Purpose:

* Read ranked CSVs for 450, 520, and 600 nm.
* Write a concise CSV and Markdown summary.
* Include top candidate for each target.
* Include top scaffold counts among top 10.
* Include interpretation notes.

Suggested CLI:

```powershell
python scripts/summarize_candidate_screening.py `
  --inputs outputs/candidate_screening/ranked_scaffold_candidates_ethanol_450.csv outputs/candidate_screening/ranked_scaffold_candidates_ethanol_520.csv outputs/candidate_screening/ranked_scaffold_candidates_ethanol_600.csv `
  --targets 450 520 600 `
  --out outputs/candidate_screening/screening_summary.csv `
  --markdown outputs/candidate_screening/screening_summary.md `
  --top-n 10
```

Expected columns:

```text
target_emission_nm
rank
name
scaffold
substituent
smiles
predicted_absorption_nm
predicted_emission_nm
predicted_quantum_yield
predicted_log_extinction
emission_error_from_target
score
estimated_brightness_score
```

Markdown report should state:

* 450 nm had mixed scaffold preferences.
* 520 nm was dominated by N-butyl naphthalimides.
* 600 nm did not produce candidates very close to 600 nm, meaning the scaffold library should be expanded.

## 11.3 Expand Candidate Generator

The current scaffold library is too limited, especially for 600 nm.

Add more red-shifted scaffold families.

Candidate families to consider:

```text
BODIPY-like scaffolds
rhodamine-like scaffolds
fluorescein-like scaffolds
cyanine / polymethine-like dyes
extended coumarins
larger donor-acceptor systems
larger aromatic systems
```

Recommended approach:

1. Do not add everything at once.
2. Start with one or two additional scaffold families.
3. Make sure RDKit validates the templates.
4. Generate candidates.
5. Rerun 450/520/600 screening.
6. Compare whether 600 nm improves.

## 11.4 Add Applicability-Domain Warning

Candidate predictions should eventually include a trust/similarity score.

Possible idea:

* Compute Morgan fingerprint similarity between each candidate and nearest training chromophore.
* Add column:

```text
nearest_training_similarity
```

* Add warning when similarity is low:

```text
outside_applicability_domain = True/False
```

This is important because model-ranked generated candidates may be outside the model’s reliable chemical space.

## 11.5 Improve Quantum Yield Modeling

Quantum yield is currently noisy.

Possible next improvements:

* Treat QY as a classification/ranking task instead of exact regression.
* Create classes:

```text
low QY
medium QY
high QY
```

* Or train a binary classifier:

```text
bright vs not bright
```

This may be more useful for screening than exact QY regression.

## 11.6 Improve Lifetime Modeling

Lifetime has fewer labels and outliers.

Possible improvement:

```text
train on log(lifetime_ns)
```

Then convert predictions back if needed.

## 11.7 Improve Red/NIR Performance

Use error-analysis outputs to check whether red/NIR molecules have higher errors.

Files to inspect:

```text
outputs/error_analysis/error_by_wavelength_region_emission_nm.csv
outputs/error_analysis/worst_predictions_emission_nm.csv
```

If red/NIR errors are high, focus future dataset/model improvements there.

---

# 12. Current Professor-Facing Summary

Use this if asked what has been done:

```text
I expanded ChemFluor by adding the Deep4Chem chromophore dataset, standardized both datasets into a shared schema, and trained solvent-aware models using Morgan fingerprints plus solvent descriptors. The Random Forest model performed best overall, with about 31 nm MAE for emission and 24 nm MAE for absorption on a grouped split by chromophore.

I then added a first inverse-design layer. Instead of using neural generation immediately, I implemented rule-based scaffold enumeration using coumarin and naphthalimide templates with different substituents. This generated 59 valid candidate molecules. I screened those candidates in ethanol for target emissions of 450, 520, and 600 nm, and added Morgan fingerprint Tanimoto applicability-domain warnings so lower-similarity candidates can be treated as lower-confidence extrapolations.

For 450 nm, the top candidates were mixed between naphthalimide and coumarin scaffolds. For 520 nm, the model strongly preferred N-butyl naphthalimide derivatives. For 600 nm, the best candidates were around 562–563 nm, which shows the current library is not red-shifted enough and needs expanded scaffold families.
```

Important wording:

```text
These are model-ranked candidates, not experimentally validated hits.
```

---

# 13. Current Technical Status

Completed:

```text
Deep4Chem analysis script
solvent descriptor expansion script
data standardization module
combined RF and HistGB training script
model report script
model comparison script
prediction error analysis script
rule-based scaffold candidate generator
candidate screening script
metadata-preserving ranked outputs
candidate-screening summary script
applicability-domain scoring for screened candidates
path-configurable original ChemFluor training workflow
optional FluoDB-Lite analysis, standardization, overlap, and deduplication workflow
450/520/600 nm screening run in ethanol
README append section drafted
Git ignore updated to exclude models/outputs/joblib files
```

Remaining:

```text
append README section cleanly while preserving Nibi/SSH instructions
add DEVELOPMENT.md to repo
expand scaffold library for red-shifted candidates
improve quantum yield modeling
improve lifetime modeling
inspect red/NIR error behavior
```

---

# 14. Important Caution

The current workflow is useful and coherent, but it should not be oversold.

Correct phrasing:

```text
rule-based scaffold enumeration
model-ranked candidates
first-pass screening
candidate prioritization
```

Avoid saying:

```text
AI generated new molecules from scratch
experimentally validated hits
confirmed fluorescence values
```

The model predictions are hypotheses for prioritization, not experimental truth.
