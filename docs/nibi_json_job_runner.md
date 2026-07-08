# Running FluorCast JSON jobs on NIBI

The JSON files are the job contract between FluorCast and the future portal.
Each command reads one input JSON and writes one success or failure JSON. These
scripts do not connect to Supabase or any other network service.

FluorCast currently predicts emission wavelength and quantum yield for the
portal workflow. Historical datasets and internal model code may still contain
absorption or lifetime fields, but the JSON job outputs do not expose them.

## Prediction test

Create a job directory and prediction input on NIBI:

```bash
mkdir -p /home/chrisl/scratch/fluorcast-jobs/job_test_001
cat > /home/chrisl/scratch/fluorcast-jobs/job_test_001/input.json <<'JSON'
{
  "job_id": "job_test_001",
  "user_id": "local-test-user",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "model_choice": "all",
  "requested_at": "2026-06-22T12:00:00Z"
}
JSON
```

From the repository, run it locally with the current Python environment:

```bash
python scripts/run_prediction_job.py \
  --input /home/chrisl/scratch/fluorcast-jobs/job_test_001/input.json \
  --output /home/chrisl/scratch/fluorcast-jobs/job_test_001/output.json
```

Submit the same prediction through Slurm:

```bash
mkdir -p /home/chrisl/scratch/fluorcast-jobs/job_test_001
export FLUORCAST_REPO=/home/chrisl/scratch/FluorCast
export FLUORCAST_INPUT_JSON=/home/chrisl/scratch/fluorcast-jobs/job_test_001/input.json
export FLUORCAST_OUTPUT_JSON=/home/chrisl/scratch/fluorcast-jobs/job_test_001/output.json
sbatch slurm/run_prediction_job.sbatch
```

Run `sbatch` from `$FLUORCAST_REPO`, as shown above, so the relative wrapper
path resolves. Slurm writes `slurm-fluorcast-predict-<jobid>.out` and `.err` in
the directory where `sbatch` is invoked.

## Duplicate-check test

Create a duplicate-check input JSON:

```bash
cat > /home/chrisl/scratch/fluorcast-jobs/job_test_001/duplicate-input.json <<'JSON'
{
  "submission_id": "submission_test_001",
  "user_id": "local-test-user",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "submitted_at": "2026-06-22T12:00:00Z"
}
JSON
```

Run it locally, optionally selecting a dataset:

```bash
python scripts/run_duplicate_check_job.py \
  --input /home/chrisl/scratch/fluorcast-jobs/job_test_001/duplicate-input.json \
  --output /home/chrisl/scratch/fluorcast-jobs/job_test_001/duplicate-output.json \
  --dataset data/processed/fluodb_lite/combined_deduplicated.csv
```

Submit it through Slurm:

```bash
export FLUORCAST_REPO=/home/chrisl/scratch/FluorCast
export FLUORCAST_INPUT_JSON=/home/chrisl/scratch/fluorcast-jobs/job_test_001/duplicate-input.json
export FLUORCAST_OUTPUT_JSON=/home/chrisl/scratch/fluorcast-jobs/job_test_001/duplicate-output.json
export FLUORCAST_DATASET=/home/chrisl/scratch/FluorCast/data/processed/fluodb_lite/combined_deduplicated.csv
sbatch slurm/run_duplicate_check_job.sbatch
```

`FLUORCAST_DATASET` is optional. When omitted, the duplicate runner uses its
configured default dataset if that file exists.

## Python environment

The wrappers change to `$FLUORCAST_REPO` and source `.venv/bin/activate` when
it exists. For a module, Conda, or differently named virtual environment,
either prepare the environment before submission or point
`FLUORCAST_ACTIVATE` at a shell activation script:

```bash
export FLUORCAST_ACTIVATE=/home/chrisl/scratch/venvs/fluorcast/bin/activate
```

If neither activation option is present, the wrappers use `python` from the
Slurm job's `PATH` and print a notice in the error log.

## Future portal integration

Portal integration will later create the input JSON in a per-job directory,
submit the appropriate `sbatch` wrapper, track the Slurm job, and read the
output JSON when it finishes. SSH submission, portal persistence, and
Supabase access remain outside the Python runners and these wrappers.

