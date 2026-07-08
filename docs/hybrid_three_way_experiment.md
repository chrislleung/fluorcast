# Three-way hybrid ensemble experiment

This workflow trains base models on one split, trains the hybrid ensemble from predictions on a second split, and reports metrics only on an untouched final split. Molecule and scaffold modes keep their respective groups disjoint and write a machine-readable leakage check.

Smoke test (PowerShell):

```powershell
python scripts/run_hybrid_three_way_experiment.py `
  --target-name emission_nm --split-type molecule --seed 0 `
  --models rf extratrees --max-rows 1000 `
  --out-dir outputs/hybrid_three_way_smoke/molecule/emission_nm `
  --model-out-dir models/hybrid_three_way_smoke/molecule/emission_nm
```

Full experiment:

```powershell
python scripts/run_hybrid_three_way_experiment.py `
  --target-name absorption_nm --split-type scaffold --seed 0 `
  --models rf extratrees histgb gbdt mlp `
  --out-dir outputs/hybrid_three_way/scaffold/absorption_nm `
  --model-out-dir models/production_hybrid/absorption_nm
```

Use `--standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv` to select an existing standardized dataset explicitly, and `--solvent-descriptors` to override the default descriptor table.

Nibi Slurm submission:

```bash
export FLUORCAST_TARGET_NAME="absorption_nm"
export FLUORCAST_SPLIT_TYPE="scaffold"
export FLUORCAST_SEED="0"
export FLUORCAST_OUT_DIR="outputs/hybrid_three_way/scaffold/absorption_nm"
export FLUORCAST_MODEL_OUT_DIR="models/production_hybrid/absorption_nm"
sbatch slurm/run_hybrid_three_way_experiment.sbatch
```

Repeat with `FLUORCAST_TARGET_NAME=emission_nm` and
`FLUORCAST_TARGET_NAME=quantum_yield` to build the full production layout:

```text
models/production_hybrid/absorption_nm/
models/production_hybrid/emission_nm/
models/production_hybrid/quantum_yield/
```

These trained artifacts live on Nibi and are intentionally excluded from Git.

