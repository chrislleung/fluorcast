# NIBI Supabase Worker

`scripts/nibi_supabase_worker.py` is the NIBI-side bridge between the FluorCast
portal database and the existing Slurm JSON job runners. It collects completed
`output.json` files, inserts prediction results, updates job status, polls
Supabase for queued prediction jobs, writes `input.json`, and submits
`slurm/run_prediction_job.sbatch`.

The worker currently handles prediction jobs only.

## Required Environment

Keep the real worker environment in this NIBI-only file:

```bash
/home/chrisl/scratch/fluorcast_worker.env
```

Set these variables there or export them before manual runs:

```bash
export SUPABASE_URL="https://example-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="fake-service-role-key-for-docs-only"
export FLUORCAST_REPO="/home/your-user/scratch/FluorCast"
export FLUORCAST_JOBS_DIR="/home/your-user/scratch/fluorcast-jobs"
export FLUORCAST_POLL_LIMIT="5"
```

`FLUORCAST_POLL_LIMIT` is optional and defaults to `5`.

Example `.env` content, using fake values only:

```bash
SUPABASE_URL=https://example-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=fake-service-role-key-for-docs-only
FLUORCAST_REPO=/home/your-user/scratch/FluorCast
FLUORCAST_JOBS_DIR=/home/your-user/scratch/fluorcast-jobs
FLUORCAST_POLL_LIMIT=5
```

Do not commit real `.env` files or service role keys.

## Security

The Supabase service role key must live only on trusted NIBI infrastructure.
For this deployment, keep it in `/home/chrisl/scratch/fluorcast_worker.env`.
It must not be shipped to browser code, Vercel client code, public logs,
notebooks, or checked-in files. The portal should create queued rows in
Supabase; it should not SSH into NIBI.

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

- `job_id`
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

## One-Shot Usage

From the FluorCast repository on NIBI:

```bash
python scripts/nibi_supabase_worker.py --once
```

`--once` runs one automation pass. It collects completed outputs first, then
submits queued jobs.

## Collect-Only Usage

Use this after manual Slurm tests or when you only want to import existing
`output.json` files:

```bash
python scripts/nibi_supabase_worker.py --collect-only
```

The collector is idempotent: completed jobs are skipped, and existing
`prediction_results` rows are not inserted again.

## Loop Usage

Loop mode makes the portal flow automatic:

```bash
python scripts/nibi_supabase_worker.py --loop
```

Each loop:

1. Collects completed prediction outputs.
2. Inserts missing prediction result rows.
3. Marks finished jobs `completed` or `failed`.
4. Submits newly queued jobs.
5. Sleeps before checking again.

The default sleep interval is 30 seconds. Override it for testing or operations:

```bash
python scripts/nibi_supabase_worker.py --loop --interval-seconds 60
python scripts/nibi_supabase_worker.py --loop --interval-seconds 5 --max-loops 3
python scripts/nibi_supabase_worker.py --loop --interval-seconds 30 --exit-after-idle-seconds 900
```

Useful manual modes:

```bash
python scripts/nibi_supabase_worker.py --submit-only
python scripts/nibi_supabase_worker.py --once --dry-run
```

`--dry-run` logs planned work without writing job files, submitting Slurm jobs,
inserting results, or updating Supabase rows.

## Run From Slurm

The Slurm wrapper sources `/home/chrisl/scratch/fluorcast_worker.env` and runs
the worker in loop mode with a 30-second interval and a 15-minute idle exit:

```bash
sbatch slurm/run_nibi_supabase_worker.sbatch
```

To use a different env-file path for testing, set:

```bash
export FLUORCAST_WORKER_ENV=/path/to/test_worker.env
sbatch slurm/run_nibi_supabase_worker.sbatch
```

The worker wrapper activates `.venv/bin/activate` when it exists. To use a
different environment, set:

```bash
export FLUORCAST_ACTIVATE=/path/to/venv/bin/activate
```

## Demo Worker

For demos, it can be useful to run a worker that stays alive for the full Slurm
allocation. Start a 24-hour manual worker with an explicit Slurm wrap command:

```bash
sbatch \
  --job-name=fluorcast-work \
  --time=24:00:00 \
  --cpus-per-task=1 \
  --mem=2G \
  --wrap='cd /home/chrisl/scratch/FluorCast && set -a && source /home/chrisl/scratch/fluorcast_worker.env && set +a && python scripts/nibi_supabase_worker.py --loop --interval-seconds 30'
```

Use this only when an always-on demo is worth the idle allocation. Normal
low-traffic operation should use the scheduled launcher below.

## Scheduled Launcher

For normal low-traffic use, run the launcher from cron. It starts a short worker
only when one is not already queued or running:

```bash
bash /home/chrisl/scratch/FluorCast/scripts/submit_nibi_worker_if_needed.sh
```

The launcher:

- uses `FLUORCAST_REPO` when set, otherwise
  `/home/chrisl/scratch/FluorCast`
- sources `/home/chrisl/scratch/fluorcast_worker.env`
- checks for an existing Slurm job named `fluorcast-work`
- exits without submitting when one is already queued or running
- submits `slurm/run_nibi_supabase_worker.sbatch` when no worker exists

Example crontab entry for every 10 minutes:

```cron
*/10 * * * * /usr/bin/env bash /home/chrisl/scratch/FluorCast/scripts/submit_nibi_worker_if_needed.sh >> /home/chrisl/scratch/fluorcast_worker_launcher.log 2>&1
```

With this schedule, new portal jobs may wait up to the cron interval before a
worker starts. Once started, the worker polls every 30 seconds and exits after
15 idle minutes.

## Check Worker Status

Check whether a worker is queued or running:

```bash
squeue -u "$USER" -n fluorcast-work
```

Review launcher logs:

```bash
tail -n 100 /home/chrisl/scratch/fluorcast_worker_launcher.log
```

## Stop The Worker

Find the Slurm job:

```bash
squeue -u "$USER" -n fluorcast-work
```

Cancel it:

```bash
scancel <slurm_job_id>
```

The worker is safe to restart. On startup, it collects existing completed
outputs before submitting new queued jobs.

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
9. Start loop mode:

   ```bash
   sbatch slurm/run_nibi_supabase_worker.sbatch
   ```

10. Insert another queued job from the portal and confirm the worker logs show
    loop number, collected outputs, submitted jobs, and sleep interval.

