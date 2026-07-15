# ConforFormer Checkpoint Asset Map

Stage 4.5 evidence pass performed on the pinned ConforFormer source at
`third_party/ConforFormer` and the official Hugging Face model repository
`ConforFormer/ConforFormer`.

The Hugging Face metadata query on 2026-07-15 reported repository commit
`7a88f23104a182b65d364d2d9f4ec3ca2259e96c` and these model siblings:

- `ConforFormer.pt`
- `No2D_full.pt`
- `OMOL.pt`
- `OMol_Conf_dataset.pt`
- `Reduced_with_contrast.pt`
- `Replicate.pt`
- `UniMol_fullDataset_contrast.pt`

The same Hugging Face repository did not list a dictionary file. Its README
only states that the weights come from the corresponding GitHub page; it does
not map checkpoint filenames to dictionaries, tasks, or training datasets.

## Strong Local Evidence

Pinned source locations that directly support the encoder path:

- `third_party/ConforFormer/example_scripts/training/Contrast_model_train.sh`
  trains `--task unimol_contrast_head`, `--loss unimol_contrast_head`, and
  `--arch unimol_contrast`.
- The same training script uses `--dict-name dict_omol_full.txt`,
  `--train-db-type lmdb`, `--valid-db-type lmdb`, `--only-polar 0`,
  15 layers, 512 embedding dimension, 2048 FFN dimension, 64 attention heads,
  and contrastive loss.
- `third_party/ConforFormer/example_scripts/inference/infer_contrast_benchmark.sh`
  runs `get_sims.py` or `get_embed.py` with `--arch contrast`,
  `--task unimol_contrast`, `--only-polar 0`, `--path $weight_path`, and a
  caller-supplied `--dict-name`.
- `third_party/ConforFormer/example_scripts/inference/get_embed.py` and
  `get_sims.py` load `state = checkpoint_utils.load_checkpoint_to_cpu(args.path)`,
  build the task/model from CLI args, call `model.load_state_dict(state["model"],
  strict=False)`, run `features_only=True`, and extract `encoder_rep[:, 0, :]`.
- `third_party/ConforFormer/unimol/unimol/tasks/unimol_contrast.py` defaults
  `--dict-name` to `dict.txt`, but the paper training wrapper overrides it to
  `dict_omol_full.txt`.
- `third_party/ConforFormer/unimol/unimol/models/unimol_contrast.py` registers
  model `contrast` and architectures `contrast` and `unimol_contrast`.

## Checkpoint Table

| Checkpoint | Apparent purpose | Training dataset | Contrastive learning | Architecture/task | Dictionary | Confidence | Unresolved questions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ConforFormer.pt` | Likely primary released ConforFormer model because it shares the repository/project name. | Not explicitly mapped by pinned source or HF README. | Not explicitly mapped by filename. Contrast training exists in `Contrast_model_train.sh`. | Likely `contrast`/`unimol_contrast` only after checkpoint inspection proves shapes/metadata. | Likely `dict_omol_full.txt` only if this is the contrast-trained model; not proven. | Low-to-medium. | Need checkpoint metadata and exact dictionary. |
| `No2D_full.pt` | Likely ablation or variant without 2D information, based on filename only. | Unresolved. | Unresolved. | Unresolved until checkpoint inspection. | Unresolved. | Low. | No pinned source line names this file or explains "No2D". |
| `OMOL.pt` | Likely model associated with OpenMolecules/OMol data, based on filename and data-processing pipeline. | OpenMolecules-derived data is described in `data_processing/README.md`; exact checkpoint mapping unresolved. | Unresolved. | Unresolved until checkpoint inspection. | Unresolved. | Low. | Need author documentation or checkpoint metadata. |
| `OMol_Conf_dataset.pt` | Likely OpenMolecules conformer-dataset variant, based on filename. | OMol filtering pipeline exists under `data_processing/filter_omol_to_conformers`; exact checkpoint mapping unresolved. | Unresolved. | Unresolved until checkpoint inspection. | Unresolved. | Low. | Need dictionary and training command for this exact artifact. |
| `Reduced_with_contrast.pt` | Likely reduced Uni-Mol partition with contrastive training, based on filename and reduced-dataset pipeline. | `data_processing/README.md` says partition 1 of Uni-Mol was used for the paper; exact checkpoint mapping unresolved. | Filename suggests contrast, but no source line names the file. | Unresolved until checkpoint inspection. | Unresolved; possibly `dict_omol_full.txt`, not proven. | Low. | Need exact script/run metadata. |
| `Replicate.pt` | Likely replicate/reproducibility run, based on filename and analysis output label. | Unresolved. | Unresolved. | Unresolved until checkpoint inspection. | Unresolved. | Low. | `analysis/analysis.R` references `Replicate_sim.sqlite3`, not this checkpoint file. |
| `UniMol_fullDataset_contrast.pt` | Likely full Uni-Mol dataset with contrastive learning, based on filename. | Full Uni-Mol data; exact source mapping unresolved. | Filename suggests contrast. | Likely `contrast`/`unimol_contrast` only after checkpoint inspection. | Unresolved; possibly `dict_omol_full.txt`, not proven. | Low-to-medium. | Need checkpoint metadata and matching dictionary. |

## Recommended First Smoke Checkpoint

No checkpoint is fully recommended yet, because the official assets available
from Hugging Face do not include the matching dictionary and the pinned source
does not map each checkpoint filename to a dictionary.

The first candidate to inspect is `ConforFormer.pt`, because it is the primary
repository-named checkpoint. This is only an inspection candidate. It should not
be used for real inference until:

1. the checkpoint is downloaded from `ConforFormer/ConforFormer`;
2. its SHA-256 and state-dictionary shapes are recorded;
3. the exact matching dictionary is obtained from an official source;
4. dictionary size and edge-type dimensions match the checkpoint.

## Dictionary Requirement

Dictionary files found in the pinned repository:

- `third_party/ConforFormer/unimol/example_data/molecule/dict.txt`
- `third_party/ConforFormer/unimol/example_data/pocket/dict_coarse.txt`

Dictionary names found in code and notebooks:

- `dict.txt`
- `dict_coarse.txt`
- `dict_fine.txt`
- `dict_mol.txt`
- `dict_pkt.txt`
- `dict_omol_full.txt`

Important distinction:

- The example Uni-Mol molecule dictionary is `dict.txt`. It contains a small
  molecule vocabulary and is used by upstream Uni-Mol demos.
- The ConforFormer contrast training wrapper explicitly uses
  `dict_omol_full.txt`.
- `dict_omol_full.txt` is not present in this repository.
- `dict_omol_full.txt` was not listed as a sibling in the official
  `ConforFormer/ConforFormer` Hugging Face repository metadata.
- No deterministic dictionary-generation script for `dict_omol_full.txt` was
  identified in the pinned source.

Conclusion: the exact checkpoint dictionary is currently absent from local
materials and from the official HF model file listing. It must be supplied as a
separate official artifact or requested from the authors. Do not create a
replacement dictionary unless exact token order is established from
authoritative upstream material.

## Local Asset Paths

Expected local paths:

- Checkpoints: `assets/conforformer/checkpoints/`
- Dictionaries: `assets/conforformer/dictionaries/`
- Download staging: `assets/conforformer/downloads/`
- Encoder diagnostics: `conforformer_encoder_diagnostics/`

These paths are intentionally ignored by Git. The adapter accepts explicit
paths and does not hardcode these locations.

## Inspection Commands

Environment only:

```powershell
python scripts/smoke_conforformer_encoder.py --env-report
```

Checkpoint and dictionary inspection, after official assets are supplied:

```powershell
python scripts/smoke_conforformer_encoder.py `
  --inspect-only `
  --checkpoint assets/conforformer/checkpoints/ConforFormer.pt `
  --dictionary assets/conforformer/dictionaries/dict_omol_full.txt `
  --output conforformer_encoder_diagnostics/inspect_conforformer.json
```

The inspection records checkpoint filename, SHA-256, file size, top-level keys,
state-dictionary tensor shapes, inferred vocabulary size, inferred embedding
dimension, inferred layer count, architecture, and the PyTorch checkpoint trust
warning from the adapter.
