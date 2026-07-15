# ConforFormer Upstream Audit

Audited repository state:

- FluorCast branch: `feature/conforformer`
- ConforFormer submodule path: `third_party/ConforFormer`
- Pinned upstream commit recorded in `configs/conforformer/upstream_commit.txt`: `f3095c5ea0218b6b4b2780cd1f43122410e80a7a`
- Submodule URL: `https://github.com/EPiCs-group/ConforFormer.git`

No large jobs were run. I attempted to inspect the included LMDB examples directly, but the current FluorCast Python environment does not have `lmdb` installed, so the LMDB schema below is derived from the upstream LMDB readers and writer scripts.

## FluorCast Integration Surface

Current FluorCast model families use:

- `src/features.py`: Morgan fingerprints, MACCS keys, RDKit descriptors, one-hot solvent labels, and numeric solvent descriptors.
- `scripts/train_combined_predictors.py`: combined ChemFluor/Deep4Chem/FluoDB-Lite trainer using chromophore Morgan fingerprint plus solvent descriptor vector.
- `src/chemfluor/combined_prediction.py`: inference-time canonicalization, feature alignment, solvent descriptor lookup, and applicability-domain scoring.
- `src/chemfluor/graph_features.py`: current graph feature provider, independent from ConforFormer.
- `scripts/run_hybrid_three_way_experiment.py`: leakage-safe base/meta/final split workflow for `random`, `molecule`, and `scaffold`.
- `scripts/predict_all_models.py`: discovery and prediction across tree, neural, and graph artifact directories.

Current Slurm layout is organized into `slurm/`, `slurm/base_models/`, `slurm/manuscript/`, `slurm/util/`, and `slurm/legacy/`. Nibi jobs load `python/3.11`, `gcc`, and `rdkit`, then activate a FluorCast venv when available. Existing jobs are CPU-oriented except graph model GPU utilities.

The current test suite covers data standardization, combined-model training/prediction, graph experiments, hybrid three-way leakage behavior, Slurm layout/reference consistency, app JSON contracts, and manuscript split logic. There are no ConforFormer tests yet.

## Upstream Model and Architecture

The ConforFormer inference example is `third_party/ConforFormer/example_scripts/inference/get_embed.py`, launched by `infer_contrast_benchmark.sh`.

The intended encoder is the contrast model:

- task for embedding extraction: `unimol_contrast`
- model registry name: `contrast`
- architecture aliases: `contrast` and `unimol_contrast`
- training example task: `unimol_contrast_head`
- training example architecture: `unimol_contrast`
- inference shell architecture: `contrast`

Architecture defaults in `unimol/unimol/models/unimol_contrast.py`:

- encoder layers: `15`
- embedding dimension: `512`
- FFN dimension: `2048`
- attention heads: `64`
- dropout/embedding dropout/attention dropout: `0.1`
- activation: `gelu`
- max sequence length: `512`
- Gaussian basis channels: `128`

The output chromophore embedding used by the example is the CLS token:

```text
encoder_rep, _, _ = model(..., features_only=True)
embedding = encoder_rep[:, 0, :]
```

Therefore the embedding dimension is `512` for the shipped architecture.

Checkpoint loading in `get_embed.py` and `unimol/infer.py` does:

```text
state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
task = tasks.setup_task(args)
model = task.build_model(args)
model.load_state_dict(state["model"], strict=False)
```

The scripts do not read architecture parameters from `state["args"]`; the command-line args must match the checkpoint. A future adapter should inspect checkpoint metadata when available, but it must also record and pass explicit architecture values.

## Dictionary

The upstream Uni-Mol example dictionary at `unimol/example_data/molecule/dict.txt` contains:

```text
[PAD], [CLS], [SEP], [UNK], C, N, O, S, H, Cl, F, Br, I, Si, P, B, Na, K, Al, Ca, Sn, As, Hg, Fe, Zn, Cr, Se, Gd, Au, Li
```

ConforFormer training uses `--dict-name dict_omol_full.txt`, but that file is not present in this repository. The checkpoint bundle must therefore provide the exact dictionary used for training. The dictionary length affects `embed_tokens`, `gbf`, edge-type embeddings, and checkpoint compatibility.

## Required Encoder Input Schema

The forward signature for `UniMolModel_contrast` is:

```text
src_tokens: LongTensor [B, L]
src_distance: FloatTensor [B_or_BxC, L, L]
src_coord: FloatTensor [B_or_BxC, L, 3]
src_edge_type: LongTensor [B, L, L]
features_only=True
```

For the contrast all-conformer inference task, each LMDB record represents a formula block containing many same-formula molecules and up to 16 conformers per molecule. The task flattens conformers into the batch dimension:

- raw `coordinates_1` and `coordinates_2`: list-like array `[molecule_count, atom_count, 3]`
- `AllCoordsDataset`: takes first 16 entries and transposes to `[atom_count, 3, molecule_count]`
- `FlattenRightPadDatasetCoord`: collates that into `[molecule_count, padded_atom_count + 2, 3]`
- `FlattenDistanceDataset`: computes pairwise distances `[molecule_count, L, L]`
- `src_tokens` and `src_edge_type` are repeated inside the model with `repeat_interleave` to match the flattened conformer count.

For a direct single-molecule adapter, the equivalent valid input is:

- `src_tokens`: one tokenized atom sequence with BOS/CLS prepended and EOS/SEP appended, shape `[1, L]`
- `src_coord`: centered coordinates with zero vectors prepended/appended, shape `[1, L, 3]`
- `src_distance`: Euclidean pairwise distances from `src_coord`, shape `[1, L, L]`
- `src_edge_type`: `src_tokens[:, None] * len(dictionary) + src_tokens[None, :]`, shape `[1, L, L]`

## Hydrogen Handling

The ConforFormer training and inference example set `only_polar=0`. In both `unimol_contrast.py` and `unimol_contrast_head.py`:

```text
only_polar > 0  -> remove_polar_hydrogen=True
only_polar < 0  -> keep all hydrogens
only_polar == 0 -> remove_hydrogen=True
```

`RemoveHydrogenDataset` then removes every atom with symbol `H`. Thus the audited ConforFormer example path removes all hydrogens before tokenization and coordinate tensors are built. Any FluorCast integration should default to no-hydrogen embeddings unless the actual checkpoint metadata proves otherwise.

## Atom Tokens, Coordinates, Distances, Edge Types

The preprocessing sequence in `unimol_contrast.py` is:

1. Load raw LMDB record.
2. Read `atoms`, `coordinates_1`, `coordinates_2`, `smi`, and `formula`.
3. Convert atoms to a NumPy array and trim coordinates to first 16 conformers.
4. Optionally remove hydrogens.
5. Center coordinates by subtracting the per-conformer coordinate mean.
6. Tokenize atom symbols with `TokenizeDataset(dictionary)`.
7. Prepend dictionary BOS (`[CLS]`) and append dictionary EOS (`[SEP]`) to tokens.
8. Prepend and append zero coordinates.
9. Compute pairwise Euclidean distances.
10. Build edge types as `token_i * len(dictionary) + token_j`.
11. Right-pad tokens, coordinates, distances, and edge types.

No bond graph is used by the ConforFormer encoder input. Pairwise distance and edge-type tensors provide the attention bias.

## LMDB Record Schema

The contrast benchmark writer creates records like:

```python
{
  "atoms": list[str],
  "coordinates_1": list[np.ndarray],  # each [num_atoms, 3], float32
  "coordinates_2": list[np.ndarray],  # each [num_atoms, 3], float32
  "formula": str,
  "smi": list[str],
}
```

Keys are ASCII integer strings (`b"0"`, `b"1"`, ...). Values are pickled Python dictionaries.

The older Uni-Mol molecule tasks use:

```python
{
  "atoms": list[str],
  "coordinates": list[np.ndarray],
  "smi": str,
  "target": optional,
}
```

The ConforFormer `unimol_contrast` inference task expects the first schema with `coordinates_1` and `coordinates_2`, not the generic `coordinates` schema.

## Embedding Extraction

`get_embed.py` loads each sample, runs both `net_input_set_1` and `net_input_set_2`, extracts `encoder_rep[:, 0, :]`, and writes each embedding to SQLite as raw `float32` bytes. It also writes an XYZ string for the corresponding coordinates. The example stops after 2500 dataset items.

For FluorCast, the useful output should be a deterministic numeric feature vector per chromophore, likely:

- mean over conformer embeddings, shape `[512]`
- optional standard deviation over conformer embeddings, shape `[512]`
- optional number of successful conformers and failure metadata

## Upstream Pipeline vs Direct Python Adapter

The upstream LMDB task pipeline is the safest way to reproduce ConforFormer behavior exactly, especially for multi-conformer batch flattening. However it requires Uni-Core, LMDB, the upstream package import path, and a temporary LMDB, and it is awkward for one-off FluorCast predictions.

A direct Python inference adapter is better for FluorCast if it faithfully implements the audited preprocessing:

- RDKit conformer generation
- no-hydrogen atom/coordinate filtering
- dictionary tokenization
- coordinate centering
- BOS/EOS coordinate padding
- distance and edge-type tensor construction
- checkpoint/model loading with explicit architecture args

The first implementation should validate direct-adapter outputs against an upstream LMDB smoke sample before training new predictors from embeddings.

## Dependency and Nibi/H100 Risks

Existing FluorCast requirements are lightweight and CPU-friendly: `numpy<2.0`, pandas, RDKit, scikit-learn, scipy, matplotlib, LightGBM, XGBoost, CatBoost, Jupyter, pytest.

ConforFormer/Uni-Mol requires:

- Uni-Core from `git+git://github.com/dptech-corp/Uni-Core.git@stable`
- PyTorch
- LMDB
- SciPy
- `scikit-learn-extra`
- RDKit

Likely conflicts and operational risks:

- The upstream Uni-Mol README targets old RDKit (`rdkit-pypi==2022.9.3`) and Docker with PyTorch 1.11/CUDA 11.3, while FluorCast/Nibi uses Python 3.11 and modern RDKit modules.
- The root ConforFormer baseline environment allows `numpy>=1.26,<3`, while FluorCast pins `numpy<2.0`.
- `git://` URLs are often blocked; use HTTPS or a pinned vendored Uni-Core install.
- Old Uni-Core/fused CUDA kernels may not build or run cleanly on H100/CUDA 12.
- `--fp16` may be brittle with older kernels on H100; smoke tests should support CPU and CUDA fp32 first, then CUDA fp16.
- ConforFormer should probably live in a separate environment from the existing FluorCast training env to avoid breaking production workflows.

## Reproducibility Requirements

Record at minimum:

- ConforFormer submodule commit
- checkpoint path, SHA256, and checkpoint metadata keys
- dictionary path and SHA256
- architecture args
- hydrogen policy
- RDKit version
- conformer generation parameters and random seed
- number of requested and successful conformers
- atom symbols after hydrogen filtering
- canonical chromophore SMILES used for cache keying
- adapter version/source commit
- embedding aggregation method
- failure reason for invalid or failed molecules

