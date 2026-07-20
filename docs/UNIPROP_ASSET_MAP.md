# UniProp Asset Map

Status: Stage 1 audit map. No asset was downloaded or installed in this stage.

## Official Upstream Sources

- Code: https://github.com/AI4DD/nablaColors
- Dataset and checkpoints: https://zenodo.org/records/18061300
- Uni-Mol+ upstream: https://github.com/deepmodeling/Uni-Mol/tree/main/unimol_plus
- Chemprop v1.3.0 reference:
  https://github.com/chemprop/chemprop/tree/v1.3.0

## Required Runtime Assets

| Asset | Source | Required for | Git policy | Validation |
| --- | --- | --- | --- | --- |
| `Uni-Core/` | nablaColors repo or pinned local checkout | `unicore-train`, training, validation | Do not commit as copied source unless a submodule/vendor decision is documented | import check plus `unicore-train` availability |
| `unimol_plus/` | nablaColors repo | UniProp tasks and shell entry points | Do not copy into FluorCast in this stage | package import check after editable install |
| `unimol_plus_pcq_small.pt` | official Uni-Mol+ assets | head pretraining initialization | Do not commit | SHA256, size, readable checkpoint |
| `models/chemprop/fold_0/model_1/model.pt` | nablaColors repo | solvent embedding | Do not commit unless already a tiny upstream pointer; model binary stays out of Git | path, SHA256, Chemprop v1.x compatibility |
| `uniprop_rdkit_to_dft_implicit.pt` | Zenodo | pretrained UniProp checkpoint | Do not commit | MD5 from Zenodo plus local SHA256 |
| `uniprop_rdkit_to_xtb.pt` | Zenodo | pretrained UniProp checkpoint | Do not commit | MD5 from Zenodo plus local SHA256 |
| `uniprop_xtb_to_dft_implicit.pt` | Zenodo | pretrained UniProp checkpoint | Do not commit | MD5 from Zenodo plus local SHA256 |
| `uniprop_xtb_to_dft_vacuum.pt` | Zenodo | pretrained UniProp checkpoint | Do not commit | MD5 from Zenodo plus local SHA256 |
| `absorption_conformations.lmdb` | Zenodo `absorption_conformations.zip` | upstream benchmark reproduction | Do not commit | LMDB opens, gzip/pickle record read, schema inspection |
| split CSVs | Zenodo | upstream benchmark reproduction | Small CSVs may be mirrored only with license/provenance approval | row counts, target columns, split integrity |
| FluorCast standardized combined CSV | local repo data pipeline | FluorCast supervised records | Existing policy applies | schema and leakage checks |
| FluorCast-generated UniProp LMDBs | local build step | FluorCast training/inference | Do not commit | manifest, record count, hash checks |

## Zenodo Checkpoint Names

The nablaColors README lists these checkpoint files:

- `uniprop_rdkit_to_dft_implicit.pt`
- `uniprop_rdkit_to_xtb.pt`
- `uniprop_xtb_to_dft_implicit.pt`
- `uniprop_xtb_to_dft_vacuum.pt`

The Zenodo page also reports MD5 values for the first three files in the README
and record metadata. FluorCast should store both the upstream checksum and a
local SHA256 in an asset manifest, because SHA256 is used consistently by the
existing adapter audit pattern.

## Dataset Files

Upstream nablaColors files to account for:

- `absorption_conformations.zip`
- `absorption_pairs_all.csv`
- `absorption_train.csv`
- `absorption_val.csv`
- `absorption_test.csv`
- `absorption_crossval.zip`
- `multitarget_crossval.zip`
- `smiles_to_replace.csv`
- `smiles_to_remove.csv`

FluorCast should not rely on the upstream train/validation/test split for final
FluorCast claims. Those splits are useful for reproducing nablaColors behavior.
FluorCast evaluation should continue to use FluorCast-controlled molecule and
scaffold splits over chromophore-solvent observations.

## Expected LMDB Interface

Upstream records:

- Database is an LMDB file opened with `subdir=False`, `readonly=True`,
  `lock=False`, `readahead=False`, and `meminit=False` for reading.
- Keys are bytes.
- Values are `gzip.decompress(value)` followed by `pickle.loads(...)`.
- Records are dictionaries. The official README recommends inspecting
  `record.keys()` because the geometry payload is the source of truth.

FluorCast record policy:

- Geometry LMDB records are keyed by canonical chromophore geometry ID.
- Supervised rows live separately and reference the geometry ID.
- Generated LMDBs get a sidecar manifest with record count, schema version,
  source CSV path, source CSV SHA256, geometry config, and creation command.
- Failed geometry records are represented in a resumable cache but excluded from
  trainable LMDB subsets unless explicitly requested for diagnostics.

## Proposed Local Asset Manifest

`configs/uniprop/asset_manifest.example.json` should eventually describe all
paths without embedding machine-specific locations:

```json
{
  "upstream_repo": {
    "path": null,
    "commit": null
  },
  "unicore": {
    "path": null
  },
  "unimol_plus": {
    "path": null
  },
  "chemprop_solvent_model": {
    "path": null,
    "sha256": null
  },
  "unimol_plus_small_checkpoint": {
    "path": null,
    "sha256": null
  },
  "uniprop_checkpoints": {
    "rdkit_to_dft_implicit": {
      "path": null,
      "md5": "c87305171142e1c0898a0e2b67a7236a",
      "sha256": null
    },
    "rdkit_to_xtb": {
      "path": null,
      "md5": "7be9b8858e70a85718429cd17dd0670b",
      "sha256": null
    },
    "xtb_to_dft_implicit": {
      "path": null,
      "md5": "b9768e7b4f69b4d54b5d436b7403e883",
      "sha256": null
    },
    "xtb_to_dft_vacuum": {
      "path": null,
      "md5": null,
      "sha256": null
    }
  }
}
```

## ConforFormer Component Classification

The sibling ConforFormer experiment was reviewed read-only.

Reusable unchanged:

- Stable JSON hashing and deterministic payload serialization.
- Atomic writes with temporary files and final replacement.
- Cache wrapper with payload hash, schema version, and explicit load errors.
- Asset checksum recording and dependency report pattern.

Reusable after refactoring:

- `ConformerGenerationConfig`: rename and adapt to UniProp geometry policies.
- `MoleculeConformerCacheRecord` and related schemas: simplify to one cached
  selected geometry per canonical chromophore.
- `build_conformer_cache.py`: adapt dry-run, cache-hit, overwrite, and CSV
  input behavior for UniProp geometry cache construction.
- Cache-key payload classes: replace ConforFormer checkpoint/dictionary fields
  with UniProp upstream commit, checkpoint, LMDB schema, geometry level, and
  solvent model fields.
- Corruption, cache, and CLI tests: copy test intent, not implementation names.

ConforFormer-specific and not reusable:

- `dictionary.py`
- `preprocess.py`
- ConforFormer `adapter.py` model-building logic
- ConforFormer token, edge-type, hydrogen, and CLS embedding assumptions
- `configs/conforformer/`
- `docs/conforformer/`
- `scripts/smoke_conforformer_preprocess.py`
- `scripts/smoke_conforformer_encoder.py`
- `third_party/ConforFormer/`
- generated ConforFormer results, checkpoints, and diagnostics

## Git Exclusions

New stages must keep these out of Git:

- `assets/uniprop/`
- `assets/nablacolors/`
- `third_party/nablaColors/` unless intentionally added as a submodule in a
  documented migration
- `*.lmdb`
- `*.pt`
- `*.ckpt`
- generated geometry caches
- generated manifests that contain local paths
- Slurm logs and training logs
- `outputs/uniprop*/`
- `models/uniprop*/`

The current repository already ignores the broad `models/`, `outputs/`, and
binary checkpoint patterns used by existing workflows. Later stages should add
specific ignore entries only if a new generated path is not covered.
