# Local ConforFormer Encoder Compatibility

This document records the Stage 4 local encoder boundary. It does not claim
successful real ConforFormer inference unless a real checkpoint and its exact
matching dictionary have been supplied and validated.

## Local Platform

- Operating system: Windows local development is assumed for this stage; this
  run was performed on native Windows.
- Shell: PowerShell.
- Python version observed locally: `Python 3.14.0`.
- PyTorch version observed locally: `2.10.0+cpu`.
- Uni-Core import result observed locally: `ModuleNotFoundError: No module named 'unicore'`.
- Pinned ConforFormer module import result observed locally:
  `ModuleNotFoundError: No module named 'unicore'`.
- Uni-Core import check: `python -c "import unicore; print('unicore ok')"`.
- Pinned ConforFormer import check:
  `python -c "import sys; sys.path.insert(0, r'third_party/ConforFormer/unimol'); import unimol.tasks.unimol_contrast; import unimol.models.unimol_contrast; print('upstream ok')"`.
- Combined environment report:
  `python scripts/smoke_conforformer_encoder.py --env-report`.

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
`third_party/ConforFormer/unimol/example_data/molecule/dict.txt` is useful for
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

Native Windows support is not assumed. PyTorch may import, but Uni-Core or
upstream ConforFormer dependencies may fail because of compiled extensions,
older dependency pins, or Unix-oriented build assumptions. A native failure is
recorded as a compatibility result, not as a reason to modify FluorCast defaults
or substitute a fake model.

## WSL2 Fallback

Detailed WSL2 setup instructions are in
`docs/conforformer/WSL2_ENCODER_SETUP.md`. At a high level, create a separate
WSL2 ConforFormer environment, install PyTorch/Uni-Core there, keep the normal
FluorCast environment unchanged, run `--env-report`, then run `--inspect-only`
before one-conformer CPU smoke inference.

## Local vs Narval Tasks

These tasks remain local even if inference later moves to Narval:

- SMILES canonicalization and small conformer-cache smoke generation.
- Dictionary parsing and SHA-256 recording.
- Checkpoint metadata inspection when PyTorch is available.
- Stage 3 preprocessing validation.
- CLI argument and dependency diagnostics.

Narval or another cluster environment may later be used for large embedding
cache generation, but Stage 4 does not add Slurm scripts or run the full dataset.

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
  or Narval execution is part of this stage.
