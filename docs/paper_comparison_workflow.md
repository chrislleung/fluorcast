# Paper comparison workflow

The reproducible manuscript pipeline compares FluorCast models across random,
molecule-grouped, and Bemis-Murcko scaffold-grouped splits. All
manuscript-specific Python scripts live in `scripts/manuscript/`.

## Quick local smoke test

Run this command from the project root:

```bash
python scripts/manuscript/run_paper_comparison_experiments.py --max-rows 200 --models rf --targets emission_nm --splits random --seeds 0 --out-dir outputs/paper_comparison_smoke
```

This small run checks data loading, feature generation, training, metrics,
prediction output, and figure generation without running the complete model
matrix.

## Submit the full job on Nibi

Set up the Nibi Python/RDKit environment first, then submit from the project
root:

```bash
sbatch slurm/run_paper_comparison_experiments.sbatch
```

The job loads the Python, GCC, and RDKit modules. It activates
`.venv/bin/activate` when present, or the existing
`~/scratch/chemfluor_env/bin/activate` environment. A custom environment can be
selected before submission with `FLUORCAST_ACTIVATE=/path/to/bin/activate`.

The repository defaults to Slurm's submission directory. If submitting from
elsewhere, set `FLUORCAST_REPO` to the project root.

Check job status with:

```bash
squeue -u $USER
```

## Inspect results

The full job writes results under `outputs/paper_comparison/`. After completion,
start with:

```bash
cat outputs/paper_comparison/paper_tables.md
cat outputs/paper_comparison/metrics_by_split_model_target.csv
cat outputs/paper_comparison/region_metrics_by_split_model.csv
```

Other important outputs include:

- `dataset_audit.md` and `dataset_audit.csv`
- `split_leakage_report.csv`
- `metrics_with_bootstrap_ci.csv`
- `qy_classifier_metrics.csv`
- per-run CSVs in `predictions/`
- manuscript figures in `figures/`

## Generated files and Git

Do not commit `outputs/paper_comparison/`,
`outputs/paper_comparison_smoke/`, prediction CSVs, trained model files, or
Slurm logs. The repository's `.gitignore` already excludes `outputs/`, model
directories, `*.joblib`, `*.pkl`, `*.out`, and `*.err`.

Source code and tests belong in Git. Manuscript-only Python code must remain
under `scripts/manuscript/`; broadly reusable project logic may instead live
under `src/chemfluor/`.

