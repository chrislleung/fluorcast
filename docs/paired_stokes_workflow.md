# Paired absorption/emission Stokes workflow

FluorCast calculates Stokes shift from separately modeled absorption and emission maxima. It does not train a direct Stokes-shift model. This keeps the prediction tied to the two observable spectral endpoints and avoids introducing a third model with an independently learned error profile.

Absorption and emission must be evaluated on the same moleculeâ€“solvent rows. Subtracting predictions from unrelated target-specific test splits would not describe any real paired observation and would make Stokes metrics invalid. The paired workflow therefore builds one dataset, assigns one three-way split, and uses those same row assignments for both targets. Base models train on `base_model_train`, hybrid ensembles train and calibrate on `hybrid_meta_train`, and all reported metrics use only `final_test`.

## Smoke test

```powershell
python scripts/build_paired_stokes_dataset.py `
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv `
  --max-rows 1000 `
  --out-csv outputs/paired_stokes_dataset.csv `
  --summary-json outputs/paired_stokes_summary.json

python scripts/run_paired_spectral_three_way_experiment.py `
  --split-type molecule `
  --seed 0 `
  --models rf extratrees `
  --max-rows 1000 `
  --invalid-smiles-policy drop `
  --out-dir outputs/paired_stokes_three_way_smoke/molecule `
  --model-out-dir models/paired_stokes_three_way_smoke/molecule
```

## Full run

Remove `--max-rows` and select any supported split and models:

```powershell
python scripts/run_paired_spectral_three_way_experiment.py `
  --paired-dataset outputs/paired_stokes_dataset.csv `
  --split-type scaffold `
  --seed 0 `
  --models rf extratrees histgb gbdt mlp `
  --out-dir outputs/paired_stokes_three_way/scaffold `
  --model-out-dir models/paired_stokes_three_way/scaffold
```

On Nibi, submit the reusable Slurm wrapper from the repository root:

```bash
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/paired_stokes_three_way/scaffold"
export FLUORCAST_MODEL_OUT_DIR="models/paired_stokes_three_way/scaffold/seed_0"
sbatch slurm/run_paired_spectral_three_way_experiment.sbatch
```

Direct Stokes-target wrappers are historical. The recommended workflow is
paired absorption/emission prediction followed by calculation of Stokes shift.
The reusable wrapper defaults to split- and seed-specific paths under
`outputs/paired_stokes_three_way/` and `models/paired_stokes_three_way/`.

## Outputs

The dataset builder writes the paired CSV and a JSON filtering/statistics audit. The experiment writes split assignments, invalid-row and leakage audits, four base-prediction tables, target-specific evaluated predictions and metric tables, `final_paired_spectral_predictions.csv`, `stokes_metrics_table.csv`, a Markdown metrics summary, and `experiment_config.json`. Saved models are organized under `base_models/<target>/<model>/` and `hybrid_ensemble/<target>/`. Stokes shift remains calculated from paired absorption/emission predictions, not directly modeled.

To combine user-facing reports:

```powershell
python scripts/render_combined_prediction_report.py `
  --absorption-report-json outputs/absorption_report.json `
  --emission-report-json outputs/emission_report.json `
  --quantum-yield-report-json outputs/qy_report.json `
  --out-json outputs/combined_report.json `
  --out-md outputs/combined_report.md
```

