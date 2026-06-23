# NIBI Supabase Worker

`scripts/nibi_supabase_worker.py` is the NIBI-side bridge between the FluorCast
portal database and the existing Slurm JSON job runners. It polls Supabase for
queued prediction jobs, writes `input.json`, submits
`slurm/run_prediction_job.sbatch`, collects `output.json`, inserts prediction
results, and updates job status.

The worker currently handles prediction jobs only.

## Required Environment

Set these variables on NIBI before running the worker:

```bash
export SUPABASE_URL="https://example-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="fake-service-role-key-for-docs-only"
export FLUORCAST_REPO="/home/your-user/scratch/ChemFluor_Project"
export FLUORCAST_JOBS_DIR="/home/your-user/scratch/fluorcast-jobs"
export FLUORCAST_POLL_LIMIT="5"
```

`FLUORCAST_POLL_LIMIT` is optional and defaults to `5`.

Example `.env` content, using fake values only:

```bash
SUPABASE_URL=https://example-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=fake-service-role-key-for-docs-only
FLUORCAST_REPO=/home/your-user/scratch/ChemFluor_Project
FLUORCAST_JOBS_DIR=/home/your-user/scratch/fluorcast-jobs
FLUORCAST_POLL_LIMIT=5
```

Do not commit real `.env` files or service role keys.

## Security

The Supabase service role key must live only on trusted NIBI infrastructure. It
must not be shipped to browser code, Vercel client code, public logs, notebooks,
or checked-in files. The portal should create queued rows in Supabase; it should
not SSH into NIBI.

## Supabase Tables

The worker reads queued jobs from `prediction_jobs` with:

- `id`
- `user_id`
- `molecule_smiles`
- `solvent_smiles`
- `model_choice`
- `status`

It selects rows where `status = queued`, ordered by `created_at` ascending, up
to `FLUORCAST_POLL_LIMIT`.

For successful outputs, the worker inserts rows into `prediction_results` with:

- `prediction_job_id`
- `model_name`
- `predicted_emission_nm`
- `predicted_quantum_yield`
- `nearest_training_similarity`
- `nearest_training_smiles`
- `warnings`

The worker tries to store `slurm_job_id` on `prediction_jobs` after submission.
If that optional column is absent, it logs a warning and retries without it. On
failed outputs, it similarly tries to store `error_message` and retries without
that optional column if needed.

## Run Manually

From the FluorCast repository on NIBI:

```bash
python scripts/nibi_supabase_worker.py --once
```

Useful modes:

```bash
python scripts/nibi_supabase_worker.py --submit-only
python scripts/nibi_supabase_worker.py --collect-only
python scripts/nibi_supabase_worker.py --once --dry-run
```

`--dry-run` logs the planned actions without writing job files, submitting
Slurm jobs, inserting results, or updating Supabase rows.

## Run From Slurm

After exporting the required environment variables:

```bash
sbatch slurm/run_nibi_supabase_worker.sbatch
```

The worker wrapper activates `.venv/bin/activate` when it exists. To use a
different environment, set:

```bash
export FLUORCAST_ACTIVATE=/path/to/venv/bin/activate
```

## Manual Test Procedure

1. Insert one `prediction_jobs` row in Supabase with `status = queued` and valid
   `molecule_smiles`, `solvent_smiles`, and `model_choice`.
2. Run:

   ```bash
   python scripts/nibi_supabase_worker.py --submit-only
   ```

3. Confirm a directory like
   `$FLUORCAST_JOBS_DIR/prediction_<job_id>/input.json` exists.
4. Confirm the Supabase job status changed to `running`.
5. Wait for the Slurm prediction job to write `output.json`.
6. Run:

   ```bash
   python scripts/nibi_supabase_worker.py --collect-only
   ```

7. Confirm `prediction_results` has one row per model prediction and
   `prediction_jobs.status` is `completed`.
8. Run the collect command again. It should skip completed jobs and not insert
   duplicate result rows.
