# ConforFormer Full-Dataset Workflow

This workflow builds reusable ConforFormer chromophore embeddings for the
standardized FluorCast dataset, then trains downstream solvent-aware models for
absorption, emission, quantum yield, and Stokes shift.

## Architecture

The pipeline has four resumable stages:

1. Inventory: `scripts/build_conforformer_inventory.py` reads
   `data/processed/fluodb_lite/combined_deduplicated.csv`, keeps unique
   `canonical_chromophore_smiles`, assigns stable SHA-256 molecule IDs, and
   partitions molecules into deterministic shards.
2. Conformer cache: `scripts/build_conformer_cache_shard.py` processes one CPU
   array shard and reuses `chemfluor.conforformer.conformers` plus the hashed
   JSON conformer cache.
3. Embedding shards: `scripts/embed_conforformer_shard.py` loads the
   ConforFormer model once per GPU shard, batches conformers, stores all
   conformer-level 512-dimensional embeddings in compressed NPZ files, and
   writes a separate hash-validated done manifest.
4. Finalization and downstream training:
   `scripts/finalize_conforformer_embeddings.py` validates every expected shard
   and writes the global embedding index. `scripts/train_conforformer_downstream.py`
   joins embeddings to the molecule-solvent dataset and trains the downstream
   model sweep.

## Artifact Schemas

Full-dataset outputs live under:

```text
outputs/conforformer/full_dataset/<run_id>/
  inventory/molecule_inventory.csv
  inventory/inventory_manifest.json
  conformer_status/shard_00000.json
  embeddings/shard_00000.npz
  embeddings/shard_00000.done.json
  embedding_index.csv
  failed_molecules.csv
  embedding_manifest.json
  embedding_summary.json
```

Each NPZ stores molecule IDs, canonical SMILES, molecule offsets, conformer IDs,
per-conformer embeddings, conformer energies, mean/lowest-energy/Boltzmann
pooled molecule embeddings, fallback reasons, and success or terminal-failure
status.

Downstream outputs live under:

```text
outputs/conforformer/downstream/<run_id>/
  split_assignments.csv
  leakage_check.json
  excluded_rows.csv
  selection_results.csv
  metrics.csv
  metrics.json
  predictions/
  reports/
  training_manifest.json
```

Models live under:

```text
models/conforformer_downstream/<run_id>/<split_type>/<pooling>/<feature_set>/<target>/
```

## Resumability And Provenance

Normal reruns skip embedding shards only when both the NPZ and done manifest
validate. Validation checks the NPZ SHA-256, dataset hash, inventory hash,
checkpoint hash, dictionary hash, upstream ConforFormer commit, architecture
identity, preprocessing version, conformer configuration hash, pooling
configuration, expected molecule count, embedding dimension, offsets, and finite
successful embeddings.

Corrupt, truncated, stale, or mismatched shards are recomputed. Terminal
per-molecule chemistry/preprocessing failures are recorded in the shard and do
not fail the whole array task. File-system failures, invalid manifests, and
unexpected model errors fail loudly.

## Local Testing

Run focused tests:

```bash
python -m pytest tests/test_conforformer_full_pipeline.py
python -m pytest tests/test_conforformer_adapter.py tests/test_conforformer_cache.py tests/test_conforformer_dictionary.py tests/test_conformer_generation.py tests/test_conforformer_preprocess.py tests/test_conforformer_full_pipeline.py
```

Run compile and shell checks:

```bash
python -m compileall src scripts
bash -n slurm/conforformer/submit_full_pipeline.sh
bash -n slurm/conforformer/build_conformer_cache_array.sbatch
bash -n slurm/conforformer/embed_full_dataset_array.sbatch
bash -n slurm/conforformer/finalize_embeddings.sbatch
bash -n slurm/conforformer/train_downstream.sbatch
git diff --check
```

## Nibi Canary Submission

Use a small deterministic canary before the full run:

```bash
export FLUORCAST_PYTHON="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"
export FLUORCAST_DATASET="data/processed/fluodb_lite/combined_deduplicated.csv"
export FLUORCAST_CHECKPOINT="models/conforformer/ConforFormer.pt"
export FLUORCAST_DICTIONARY="configs/conforformer/OMOL_full_dict.txt"
export FLUORCAST_MAX_MOLECULES=256
export FLUORCAST_SHARD_SIZE=128
export FLUORCAST_RUN_ID="canary_$(date -u +%Y%m%dT%H%M%SZ)"
bash slurm/conforformer/submit_full_pipeline.sh
```

## Nibi Full Submission

```bash
export FLUORCAST_PYTHON="$HOME/scratch/venvs/fluorcast-conforformer/bin/python"
export FLUORCAST_DATASET="data/processed/fluodb_lite/combined_deduplicated.csv"
export FLUORCAST_CHECKPOINT="models/conforformer/ConforFormer.pt"
export FLUORCAST_DICTIONARY="configs/conforformer/OMOL_full_dict.txt"
unset FLUORCAST_MAX_MOLECULES
bash slurm/conforformer/submit_full_pipeline.sh
```

Optional controls include `FLUORCAST_RUN_ROOT`, `FLUORCAST_SHARD_SIZE`,
`FLUORCAST_CPU_ARRAY_LIMIT`, `FLUORCAST_GPU_ARRAY_LIMIT`, and
`FLUORCAST_SKIP_TRAINING=1`.

## Monitoring And Recovery

Monitor jobs with `squeue -u "$USER"` and inspect `outputs/slurm`. Rerun the
same submit command to resume. Completed shards with matching manifests are
skipped; invalid shards are regenerated.

If finalization fails, inspect the reported shard, remove only the bad shard
NPZ/done pair if needed, and resubmit the embedding array or the full pipeline.

## Downstream Training

The primary feature set is one selected pooled 512-dimensional ConforFormer
embedding plus numeric solvent descriptors and missingness indicators. The
matched comparison feature set is Morgan fingerprint plus the same solvent
features. A combined ConforFormer plus Morgan feature set is also supported by
the training module.

All targets share one fixed 60/20/20 split assignment before target-specific
label filtering. The required production run uses molecule grouping; scaffold
grouping is available as a secondary experiment.

## Acceptance Criteria

- Inventory and embedding manifests hash-validate.
- Every inventory molecule appears exactly once in finalization.
- Successful embeddings are finite and 512-dimensional.
- Leakage checks report zero molecule leakage for the production run.
- Metrics and predictions are written for trained target/pooling/feature
  combinations.
- Quantum-yield metrics report raw and clipped prediction behavior.
- Stokes shift reports direct models and absorption/emission-derived metrics on
  identical final-test rows where available.

## Baseline Comparison

Compare ConforFormer primary metrics with the matched Morgan baseline in
`metrics.csv` by filtering on the same target, split, and model-selection
protocol:

```bash
python - <<'PY'
import pandas as pd
m = pd.read_csv("outputs/conforformer/downstream/<run_id>/metrics.csv")
cols = ["target", "pooling_method", "feature_set", "model", "mae", "rmse", "r2"]
print(m[cols].sort_values(["target", "pooling_method", "feature_set"]).to_string(index=False))
PY
```

