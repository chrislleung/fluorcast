# FluorCast Remote Provisioning

This document describes the production provisioning flow for a Compute Canada
or Alliance-hosted FluorCast installation. The flow is idempotent, avoids login
node training, and emits JSON suitable for a Tauri client.

## Fresh Installation

Clone the repository, create a runtime environment, install a signed model
bundle, and validate the installation:

```bash
git clone https://github.com/chrislleung/fluorcast.git FluorCast
cd FluorCast
bash scripts/remote/provision_environment.sh \
  --repo-dir "$PWD" \
  --env-dir "$SCRATCH/fluorcast-env"
python scripts/remote/install_model_bundle.py \
  --archive "$SCRATCH/fluorcast-production.tar.gz" \
  --checksum "$SCRATCH/fluorcast-production.sha256.json" \
  --manifest "$SCRATCH/fluorcast-production.manifest.json" \
  --destination "$SCRATCH/fluorcast-artifacts"
python scripts/remote/check_installation.py \
  --repo-dir "$PWD" \
  --env-dir "$SCRATCH/fluorcast-env" \
  --artifact-dir "$SCRATCH/fluorcast-artifacts" \
  --expected-version "fluorcast-production-2026.07.0"
```

## Existing Installation

`provision_environment.sh` reuses an existing environment when its activation
script exists. `install_model_bundle.py` reports `already_installed` when the
destination manifest matches the incoming bundle. No repository, environment,
or artifact directory is deleted by default.

## Dirty Repository Behavior

`check_installation.py` reports `GIT_DIRTY` and exits nonzero when the checkout
has uncommitted changes. This protects the desktop app from mixing production
artifacts with an unknown source tree. Commit, stash, or clone a clean release
before provisioning production work.

## Model Bundle Installation

Bundles are installed only after the archive SHA-256, archive member paths, and
manifest file checksums pass validation. Extraction happens in a temporary
sibling directory, then the complete bundle is moved into place. A different
existing destination is rejected with `DESTINATION_EXISTS`.

## Retraining

Retraining is an explicit Slurm fallback:

```bash
bash scripts/remote/submit_production_training.sh \
  --account "$FLUORCAST_SLURM_ACCOUNT" \
  --repo-dir "$PWD" \
  --env-activate "$SCRATCH/fluorcast-env/bin/activate"
```

The script submits tree, neural, three hybrid target jobs, and a final
validation job with `afterok` dependencies. It never runs training Python
directly. Re-running the command reads `provisioning-state.json` instead of
submitting duplicate jobs.

## Recovery After Failed Setup

If environment creation fails, rerun the provisioning command. Use `--recreate`
only when you intentionally want to remove and rebuild the environment. If a
bundle install fails, remove only temporary `.installing` or `.validated`
siblings after confirming no install process is active; the working artifact
directory is not partially overwritten.

## Updating To Another Release

Install the new bundle into a new destination directory, validate it with
`check_installation.py`, then update the desktop app configuration to point at
the new artifact path. Keep the old bundle until the new validation passes.

## Uninstalling Without Deleting User Jobs

Remove only the environment directory and artifact bundle directory that were
created for this provisioning flow. Do not remove `outputs/`, Slurm logs, or
job working directories unless the user explicitly asks for that cleanup.

## Machine-Readable Output Contract

Python entry points print exactly one JSON object to stdout. Shell entry points
print JSON-lines progress events. Stable fields:

- `schema_version`: currently `1`
- `status`: `running`, `success`, or `failed`
- `code`: stable event or error code for shell events
- `errors`: list of `{code, message}` objects for Python failures
- `warnings`: nonfatal validation notes

No output contains personal usernames, absolute home paths, credentials, or
allocation accounts.
