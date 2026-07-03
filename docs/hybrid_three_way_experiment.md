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
  --target-name quantum_yield --split-type scaffold --seed 0 `
  --models rf extratrees histgb gbdt mlp `
  --out-dir outputs/hybrid_three_way/scaffold/quantum_yield `
  --model-out-dir models/hybrid_three_way/scaffold/quantum_yield
```

Use `--standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv` to select an existing standardized dataset explicitly, and `--solvent-descriptors` to override the default descriptor table.
