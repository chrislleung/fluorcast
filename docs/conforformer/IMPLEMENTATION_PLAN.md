# ConforFormer Implementation Plan

Goal: add ConforFormer chromophore embeddings as an optional feature provider without changing current FluorCast training or prediction behavior until explicitly enabled.

Immediate work is local-first: Windows, PowerShell, VS Code, Codex, and the existing FluorCast `.venv`. Do not assume Slurm, CUDA, Nibi, Narval, or any cluster is available for the initial implementation. Server execution, Slurm jobs, full cache generation, and comparison experiments belong in a later stage.

## Architecture

Use three explicit cache layers:

1. Conformer cache: canonical SMILES to optimized conformers, conformer energies, optimization status, generation status, and conformer-generation provenance.
2. Per-conformer embedding cache: one 512-dimensional ConforFormer CLS embedding for each successful conformer.
3. Pooled feature table: one chromophore representation per molecule, produced from per-conformer embeddings by mean, Boltzmann, attention, or another pooling method.

Do not store only mean embeddings in the primary embedding cache. Future pooling methods must be able to reuse conformer and per-conformer embedding caches without rerunning RDKit conformer generation or the pretrained encoder.

## Proposed Files

New local implementation files:

- `src/chemfluor/conforformer/__init__.py`
- `src/chemfluor/conforformer/config.py`
- `src/chemfluor/conforformer/schemas.py`
- `src/chemfluor/conforformer/conformers.py`
- `src/chemfluor/conforformer/preprocess.py`
- `src/chemfluor/conforformer/adapter.py`
- `src/chemfluor/conforformer/cache.py`
- `src/chemfluor/conforformer/pooling.py`
- `src/chemfluor/conforformer/features.py`
- `scripts/build_conforformer_embedding_cache.py`
- `scripts/smoke_conforformer_encoder.py`
- `tests/test_conforformer_conformers.py`
- `tests/test_conforformer_preprocess.py`
- `tests/test_conforformer_cache.py`
- `tests/test_conforformer_pooling.py`
- `tests/test_conforformer_feature_provider.py`

Responsibilities:

- `schemas.py`: typed records for conformers, embeddings, pooled features, checkpoint metadata, and cache-key payloads.
- `conformers.py`: canonical SMILES to optimized conformers and energies.
- `preprocess.py`: one conformer to exact ConforFormer tensors.
- `adapter.py`: tensors to one 512-dimensional embedding.
- `cache.py`: deterministic cache keys, cache IO, metadata validation, and failed-record persistence.
- `pooling.py`: per-conformer embeddings to one molecule representation.
- `features.py`: optional FluorCast feature-provider surface for pooled chromophore representations.

Later behavior-compatible integration points, guarded by explicit CLI flags only:

- `scripts/train_combined_predictors.py`: add optional `--chromophore-feature-provider conforformer|morgan` or `--conforformer-pooled-features`.
- `scripts/run_combined_model_experiments.py`: pass optional pooled feature table paths through.
- `scripts/run_hybrid_three_way_experiment.py`: pass optional pooled feature table paths while preserving existing `random`, `molecule`, and `scaffold` split assignments.
- `scripts/predict_all_models.py`: optionally discover ConforFormer-augmented model metadata.

Later server-execution files:

- `slurm/base_models/run_conforformer_cache_smoke.sbatch`
- `slurm/base_models/run_conforformer_cache.sbatch`

## Input Schemas

Conformer request:

```python
{
  "chromophore_id": str,
  "canonical_chromophore_smiles": str,
  "isomeric_canonical_smiles": str,
  "conformer_seed": int,
  "num_conformers": int,
  "etkdg_version": str,
  "prune_rms_thresh": float,
  "optimizer": str,
}
```

Preprocessed encoder tensors for one conformer:

```python
{
  "src_tokens": "int64[1, L]",
  "src_coord": "float32[1, L, 3]",
  "src_distance": "float32[1, L, L]",
  "src_edge_type": "int64[1, L, L]",
  "atom_symbols_no_h": list[str],
  "token_length": int,
}
```

The audited default token sequence is:

```text
[CLS] + heavy-atom tokens + [SEP]
```

Validate token length against the verified checkpoint maximum sequence length before inference.

## Cache Schemas

Conformer cache record:

```python
{
  "conformer_cache_key": str,
  "chromophore_id": str,
  "canonical_smiles": str,
  "isomeric_canonical_smiles": str,
  "generation_provenance": dict,
  "conformers": [
    {
      "conformer_id": str,
      "atom_symbols": list[str],
      "coordinates": "float32[num_atoms, 3]",
      "force_field_energy": float | None,
      "energy_units": str,
      "optimizer": str,
      "convergence_status": str,
      "generation_status": "ok" | "failed",
      "failure_reason": str | None,
    }
  ],
}
```

Per-conformer embedding cache record:

```python
{
  "chromophore_id": str,
  "conformer_id": str,
  "embedding": list[float],  # length 512 for audited defaults
  "associated_conformer_energy": float | None,
  "energy_units": str,
  "checkpoint_sha256": str,
  "dictionary_sha256": str,
  "upstream_commit": str,
  "preprocessing_hash": str,
  "conformer_cache_key": str,
  "embedding_cache_key": str,
  "status": "ok" | "failed",
  "failure_reason": str | None,
}
```

Pooled feature table row:

```python
{
  "chromophore_id": str,
  "canonical_smiles": str,
  "pooling_method": str,
  "num_successful_conformers": int,
  "pooled_embedding": list[float],
  "embedding_std": list[float] | None,
  "pooling_cache_key": str,
}
```

Recommended local storage:

- conformer cache: compressed NPZ or Parquet plus JSONL metadata
- per-conformer embedding cache: compressed NPZ or Parquet plus JSONL metadata
- pooled feature tables: Parquet or CSV for simple FluorCast joins

Do not commit large cache artifacts.

## Cache Keys

Use SHA256 over stable JSON payloads. Define three independent keys:

### `conformer_cache_key`

Depends on:

- canonical SMILES and stereochemistry-preserving canonical SMILES
- RDKit version
- ETKDG settings/version
- requested conformer count
- random seed
- pruning settings
- optimizer and optimization settings
- conformer-generation implementation version

### `embedding_cache_key`

Depends on:

- `conformer_cache_key`
- ConforFormer upstream commit
- checkpoint SHA-256
- dictionary SHA-256
- explicit verified architecture settings
- hydrogen policy
- atom handling policy
- preprocessing implementation version

### `pooling_cache_key`

Depends on:

- `embedding_cache_key`
- pooling method
- pooling temperature where applicable
- pooling implementation version

Changing from mean to Boltzmann pooling must only change `pooling_cache_key`. It must not invalidate or regenerate the conformer cache or per-conformer embedding cache.

## Upstream Preprocessing Contract

Preserve the audited behavior by default:

- remove all hydrogens
- center each conformer independently
- prepend CLS and append SEP
- add zero coordinate rows for special tokens
- calculate Euclidean pairwise distances
- calculate edge type as `token_i * vocab_size + token_j`
- extract the CLS representation

The direct adapter must eventually be validated against an upstream LMDB-path smoke example before FluorCast models are trained using real ConforFormer embeddings.

## Checkpoint Validation

Do not assume every checkpoint uses the audited default dimensions. The adapter must:

- inspect checkpoint metadata when available
- inspect relevant state-dictionary tensor shapes
- verify dictionary size compatibility
- verify embedding dimension
- verify architecture arguments
- verify maximum sequence length
- fail fast on mismatch

Record both:

- audited defaults: `model=contrast`, `arch=contrast`, `layers=15`, `embed_dim=512`, `ffn_dim=2048`, `heads=64`, `max_seq_len=512`
- actual verified checkpoint configuration

## Failure Handling

- Invalid chromophore SMILES: write a failed conformer-cache record with reason.
- Conformer generation failure: retry only according to explicit policy, then persist failed conformers with reason.
- Empty heavy-atom molecule after hydrogen removal: fail.
- Atom absent from checkpoint dictionary: default to `fail`, record every unknown atom symbol, and do not silently produce an `ok` embedding using `[UNK]`.
- Optional future atom policy: `use_unk`, but benchmark experiments must default to `fail`.
- Token length exceeds verified checkpoint limit: fail before inference.
- Checkpoint/dictionary/architecture mismatch: fail during adapter initialization.
- CUDA unavailable: irrelevant for immediate local CPU-first work; checkpoint smoke should support CPU.
- OOM: reduce batch size when building caches; never write a partial record as `ok`.

## Tests

Unit tests:

- conformer cache keys change with stereochemistry, RDKit version, ETKDG settings, conformer count, seed, pruning, optimizer, or conformer-generation version
- conformer records include conformer IDs, coordinates, energies, units, optimizer, convergence status, generation status, and failure reason
- tokenization prepends `[CLS]` and appends `[SEP]`
- all hydrogens are removed when `hydrogen_policy="remove_all"`
- unknown atoms fail by default and record unknown symbols
- sequence length is rejected when `CLS + heavy atoms + SEP` exceeds the verified model limit
- coordinates are centered per conformer and padded with zero rows
- pairwise distance tensor is symmetric with zero diagonal
- edge type equals `token_i * vocab_size + token_j`
- embedding cache keys change with conformer key, checkpoint hash, dictionary hash, architecture, hydrogen policy, atom policy, or preprocessing version
- pooling cache keys change with embedding key, pooling method, temperature, or pooling implementation version
- changing pooling method does not change conformer or embedding keys
- failed records preserve input IDs and failure reasons

Integration smoke tests:

- build conformers for benzene locally without loading a checkpoint
- build tensors for benzene and ethanol without loading a checkpoint
- run adapter initialization with a tiny/fake state dict only where practical
- CLI cache builder with `--max-molecules 2 --dry-run`

No test should require the full pretrained checkpoint by default. Mark checkpoint-dependent tests with an opt-in environment variable such as `FLUORCAST_CONFORMER_CHECKPOINT`.

## Local Smoke-Test Commands

Local preprocessing and cache-key tests:

```powershell
python -m pytest tests/test_conforformer_conformers.py tests/test_conforformer_preprocess.py tests/test_conforformer_cache.py tests/test_conforformer_pooling.py
```

Direct adapter smoke when checkpoint assets are available:

```powershell
python scripts/smoke_conforformer_encoder.py `
  --smiles "c1ccccc1" `
  --checkpoint "$env:CONFORMER_CHECKPOINT" `
  --dictionary "$env:CONFORMER_DICT" `
  --device cpu
```

Small local cache build:

```powershell
python scripts/build_conforformer_embedding_cache.py `
  --standardized-combined data/processed/fluodb_lite/combined_deduplicated.csv `
  --checkpoint "$env:CONFORMER_CHECKPOINT" `
  --dictionary "$env:CONFORMER_DICT" `
  --out data/processed/conforformer_cache/smoke `
  --max-molecules 5 `
  --device cpu
```

## Split Preservation

Do not split on conformers or embeddings. Build conformer, embedding, and pooled feature caches by canonical chromophore, then join pooled features into existing row tables before model feature matrix construction. The existing split code remains authoritative:

- `random`: row-level random split
- `molecule`: group by `canonical_chromophore_smiles`
- `scaffold`: group by Bemis-Murcko scaffold

For three-way experiments, compute or load pooled features before model fitting but after row filtering, and reuse the existing `split_assignments.csv` logic unchanged.

## Server Execution Stage

Slurm and cluster work are later-stage tasks. When local smoke tests pass, add Narval/Nibi-compatible jobs for:

- a checkpoint-gated encoder smoke run
- per-conformer embedding cache generation
- pooled feature table generation
- downstream comparison experiments

Those jobs should use a separate ConforFormer environment rather than modifying the current FluorCast venv. They should write logs under `outputs/slurm` and generated artifacts under non-committed cache/output directories.

## Acceptance Criteria

- Existing tests pass without checkpoint assets.
- Current commands train and predict identical feature shapes unless ConforFormer is explicitly enabled.
- Conformer cache stores per-conformer coordinates, energies, statuses, and provenance.
- Per-conformer embedding cache stores one 512-dimensional embedding per successful conformer.
- Pooling can switch from mean to Boltzmann without regenerating conformers or embeddings.
- Cache metadata records all reproducibility fields.
- Unknown atoms fail by default.
- Sequence-length and checkpoint compatibility are validated before inference.
- Molecule and scaffold leakage checks remain unchanged.
- Documentation names the required checkpoint and dictionary files and their hashes once assets are selected.

## Staged Commit Plan

1. Documentation and corrected implementation plan.
2. Schemas, configuration, deterministic conformer generation, cache-key utilities, and tests.
3. Exact tensor preprocessing and tests, without model loading.
4. Optional direct adapter and checkpoint-gated smoke command.
5. Per-conformer embedding cache builder.
6. Pooling implementation and pooled feature tables.
7. Optional FluorCast feature-provider integration behind explicit flags.
8. Optional prediction support.
9. Narval environment, Slurm jobs, full cache generation, and comparison experiments.

