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

Full non-production experiment:

```powershell
python scripts/run_hybrid_three_way_experiment.py `
  --target-name absorption_nm --split-type scaffold --seed 0 `
  --models rf extratrees histgb gbdt mlp `
  --out-dir outputs/hybrid_three_way/scaffold/absorption_nm `
  --model-out-dir models/hybrid_three_way/scaffold/absorption_nm/seed_0
```

Use `--standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv` to select an existing standardized dataset explicitly, and `--solvent-descriptors` to override the default descriptor table.

Production molecule-split hybrid artifacts:

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

For scaffold or comparison experiments, write to split-specific folders so
production models are not overwritten:

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

The wrapper defaults to
`outputs/hybrid_three_way/<split>/<target>/seed_<seed>` and
`models/hybrid_three_way/<split>/<target>/seed_<seed>`. Use
`FLUORCAST_TARGET_NAME`, `FLUORCAST_SPLIT_TYPE`, `FLUORCAST_OUT_DIR`, and
`FLUORCAST_MODEL_OUT_DIR` to override those values. Set the model output to a
production path only for an explicit production training run.

Explicit production runs may build this complete layout:

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

These trained artifacts live on Nibi and are intentionally excluded from Git.
Do not commit trained model artifacts.
