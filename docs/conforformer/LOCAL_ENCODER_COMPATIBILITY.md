# Local ConforFormer Encoder Compatibility

This document records the local ConforFormer encoder import boundary. It does
not claim successful real ConforFormer inference unless a real checkpoint and
its exact matching dictionary have been supplied and validated.

## Local Platform

- Pinned ConforFormer commit: `f3095c5ea0218b6b4b2780cd1f43122410e80a7a`.
- Native Windows check observed locally: Python `3.14.0`, PyTorch
  `2.10.0+cpu`, Uni-Core unavailable, CUDA unavailable.
- WSL bare interpreter on PATH during this update: Python `3.14.4`; it does
  not have NumPy/PyTorch/Uni-Core installed and is not the prepared encode
  environment.
- WSL encoder environment manual verification for this compatibility update:
  PyTorch imports, Uni-Core imports, torchmetrics imports, pandas imports, and
  the pinned ConforFormer package imports successfully after the two temporary
  in-memory dataset shims described below are registered.
- The smoke environment report records exact runtime versions unde
  `python_version`, `pytorch.version`, and `unicore.version` when the prepared
  WSL encoder environment is active. Some Uni-Core installs may not expose a
  `__version__` attribute; in that case the field is reported as `null` while
  import availability remains true.

## Import Compatibility Shims

The pinned upstream source imports `unimol.data.HugeMDB_dataset`, but the file
`third_party/ConforFormer/unimol/unimol/data/HugeMDB_dataset.py` is absent at
commit `f3095c5ea0218b6b4b2780cd1f43122410e80a7a`. FluorCast registers an
in-memory `unimol.data.HugeMDB_dataset` module only for that documented commit,
only when the file is still absent, and only when the real module is not already
importable. The placeholder exposes `HMDBDataset` and raises an actionable
`RuntimeError` if instantiated.

The real file
`third_party/ConforFormer/unimol/unimol/data/OMol_dataset.py` exists, but its
import path can require the optional `fairchem` package. FluorCast first attempts
the real import and registers an in-memory `unimol.data.OMol_dataset` module
only when the failure is specifically `ModuleNotFoundError` for `fairchem`.
Unrelated import failures are allowed to propagate. The placeholder exposes
`OMolDataset` and raises an actionable `RuntimeError` if instantiated.

Neither dataset is used by the direct FluorCast contrast-encoder pathway. The
adapter builds the upstream contrast model and feeds already-preprocessed
`src_tokens`, `src_coord`, `src_distance`, and `src_edge_type` tensors. It does
not use upstream HMDB or OMol dataset loading for checkpoint-gated direct encode
construction.

The smoke report includes:

- `upstream_import_status`: whether `unimol.tasks.unimol_contrast` and
  `unimol.models.unimol_contrast` imported after compatibility registration.
- `applied_compatibility_shims`: HMDB/OMol applied flags, exact reasons, pinned
  upstream commit, and post-shim upstream import status.

Expected CPU-only warnings from Uni-Core/fused extensions, such as unavailable
or unbuilt fused CUDA kernels, are compatibility warnings rather than a reason
to modify third-party source. CUDA unavailable is acceptable for import-only and
CPU smoke checks.

## Asset Requirements

Real inference requires:

- The pretrained ConforFormer checkpoint.
- The exact dictionary used with that checkpoint.
- Matching token embedding vocabulary size.
- Matching edge-type/Gaussian-basis vocabulary size, where inferable.
- Compatible architecture metadata: contrast architecture, embedding dimension,
  layer count, attention-head count, maximum sequence length, and required
  special tokens.

The upstream example dictionary at
`third_party/ConforFormer/unimol/example_data/molecule/dict.txt` is useful fo
preprocessing tests only. It must not be treated as checkpoint-compatible unless
its SHA-256 and vocabulary are verified against the checkpoint bundle.

Stage 4.5 asset mapping is recorded in
`docs/conforformer/CHECKPOINT_ASSET_MAP.md`. Current conclusion: the official
Hugging Face model repository lists checkpoints but not the matching
`dict_omol_full.txt`, so real inference remains blocked until that exact
dictionary is supplied from an official source.

## Smoke Commands

Show CLI arguments without constructing the model:

```powershell
python scripts/smoke_conforformer_encoder.py --help
```

Report environment and upstream import compatibility:

```powershell
python scripts/smoke_conforformer_encoder.py --env-report
```

Inspect assets only:

```powershell
python scripts/smoke_conforformer_encoder.py `
  --inspect-only `
  --dictionary "C:\path\to\matching\dictionary.txt" `
  --checkpoint "C:\path\to\checkpoint.pt"
```

Run one-molecule CPU inference only after inspection succeeds:

```powershell
python scripts/smoke_conforformer_encoder.py `
  --cache-dir data/processed/conforformer_cache/conformers `
  --smiles "CCO" `
  --dictionary "C:\path\to\matching\dictionary.txt" `
  --checkpoint "C:\path\to\checkpoint.pt" `
  --device cpu `
  --max-conformers 1 `
  --repeat-check
```

## Native Windows Result

Native Windows support is not assumed. PyTorch may import, but Uni-Core o
upstream ConforFormer dependencies may fail because of compiled extensions,
older dependency pins, or Unix-oriented build assumptions. A native failure is
recorded as a compatibility result, not as a reason to modify FluorCast defaults
or substitute a fake model.

## WSL2 Fallback

Detailed WSL2 setup instructions are in
`docs/conforformer/WSL2_ENCODER_SETUP.md`. At a high level, create or activate a
separate WSL2 ConforFormer environment, install PyTorch/Uni-Core there, keep the
normal FluorCast environment unchanged, run `--env-report`, then run
`--inspect-only` before one-conformer CPU smoke inference.

During this update, the exact requested WSL command
`python scripts/smoke_conforformer_encoder.py --env-report` could not run from
the bare WSL PATH because `python` was not installed there; `python3` then failed
before adapter import because NumPy was absent. That PATH is separate from the
manually verified WSL encoder environment described above.

## Local vs Narval Tasks

These tasks remain local even if inference later moves to Narval:

- SMILES canonicalization and small conformer-cache smoke generation.
- Dictionary parsing and SHA-256 recording.
- Checkpoint metadata inspection when PyTorch is available.
- Stage 3 preprocessing validation.
- CLI argument and dependency diagnostics.

Narval or another cluster environment may later be used for large embedding
cache generation, but this compatibility layer does not add Slurm scripts or run
the full dataset.

## Known Risks

- PyTorch checkpoint loading may still require trust in the checkpoint file. The
  adapter uses `torch.load(..., weights_only=True)` when the installed PyTorch
  supports it and does not fall back to arbitrary pickle execution during
  inspection.
- Checkpoints may omit architecture metadata, so state-dictionary tensor shapes
  are inspected and audited defaults are used only where unavoidable.
- Uni-Core and the pinned upstream model may not import on native Windows.
- Non-strict model loading is disabled by default and should only be enabled
  with a documented compatibility reason.
- No model training, pooling, full embedding-cache generation, Slurm execution,
  or Narval execution is part of this compatibility layer.