# FluorCast JSON job contracts

These command-line entrypoints run one portal job from an input JSON file and
always attempt to write an output JSON file. They do not make network calls or
submit Slurm jobs.

## Prediction jobs

```powershell
python scripts/run_prediction_job.py --input job_input.json --output job_output.json
```

Input:

```json
{
  "job_id": "job-123",
  "user_id": "user-123",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "model_choice": "hybrid",
  "requested_at": "2026-06-22T12:00:00Z"
}
```

`model_choice` accepts `hybrid`, `all`, `rf`, `extratrees`, `gbdt`, `histgb`,
or `graph_model_later`. `hybrid` runs the full FluorCast workflow and returns
one combined absorption/emission/Stokes/QY prediction record. RF and
ExtraTrees are the supported base-model artifacts. GBDT, HistGB, and
the future graph model are experimental and are used only when their artifacts
load successfully in the current environment.

Successful output:

```json
{
  "status": "success",
  "job_id": "job-123",
  "canonical_molecule_smiles": "c1ccccc1",
  "canonical_solvent_smiles": "CCO",
  "predictions": [
    {
      "model_name": "hybrid",
      "predicted_absorption_nm": 390.0,
      "predicted_emission_nm": 450.0,
      "predicted_stokes_shift_nm": 60.0,
      "predicted_stokes_shift_cm^-1": 3418.8,
      "predicted_quantum_yield": 0.2,
      "brightness_class": "dim",
      "physically_valid_stokes": true,
      "prediction_intervals": {
        "absorption_nm": {"lower": 380.0, "upper": 400.0},
        "emission_nm": {"lower": 430.0, "upper": 470.0},
        "quantum_yield": {"lower": 0.1, "upper": 0.3}
      },
      "applicability_domain": {
        "outside_applicability_domain": false,
        "targets": {
          "absorption": {"outside_applicability_domain": false},
          "emission": {"outside_applicability_domain": false},
          "quantum_yield": {"outside_applicability_domain": false}
        }
      },
      "nearest_training_similarity": 0.8,
      "nearest_training_smiles": "c1ccccc1",
      "warnings": []
    }
  ],
  "warnings": []
}
```

The base-model runner adapts `scripts/predict_all_models.py`. The
hybrid runner reuses `scripts/predict_full_fluorcast.py`, writes temporary
workflow artifacts under `outputs/json_jobs/<job_id>/` unless `temp_dir` is
provided, and never invents values. If no requested model artifacts produce a
prediction, the job fails with `PREDICTION_BACKEND_NOT_CONNECTED`.

For `model_choice: "all"`, each model artifact is checked independently. Models
that cannot be loaded, including artifacts created by an incompatible
scikit-learn version, are skipped and described in the top-level `warnings`
array. Predictions from available models still produce a successful job.
Requesting an unavailable model explicitly, such as `histgb` or `gbdt`, fails
with `error_code: "MODEL_UNAVAILABLE"`; the error message explains that the
artifact could not be loaded in the current environment, and loader details
are retained in `warnings`.

## Duplicate-check jobs

```powershell
python scripts/run_duplicate_check_job.py --input duplicate_input.json --output duplicate_output.json --dataset data/processed/fluodb_lite/combined_deduplicated.csv --max-matches 5
```

The checked-in combined deduplicated CSV is used by default when present.
Custom datasets must contain a recognized molecule SMILES column and either
`canonical_solvent_smiles` or `solvent_smiles`.

Input:

```json
{
  "submission_id": "submission-123",
  "user_id": "user-123",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "submitted_at": "2026-06-22T12:00:00Z"
}
```

Successful output contains `exact_duplicate_found`, an optional exact record
ID, canonical molecule and solvent SMILES, and up to `max-matches` nearest
records. Each nearest record contains its molecule/solvent SMILES, Morgan
Tanimoto similarity, optional emission and quantum-yield values, and an
optional DOI.

```json
{
  "status": "success",
  "submission_id": "submission-123",
  "exact_duplicate_found": false,
  "exact_duplicate_record_id": null,
  "canonical_molecule_smiles": "c1ccccc1",
  "canonical_solvent_smiles": "CCO",
  "nearest_matches": [
    {
      "record_id": "record-123",
      "molecule_smiles": "c1ccccc1",
      "solvent_smiles": "CCO",
      "similarity": 1.0,
      "emission_nm": 450.0,
      "quantum_yield": 0.2,
      "source_doi": "10.1234/example"
    }
  ],
  "warnings": []
}
```

Duplicate-check output does not include absorption or lifetime fields, even
when those historical columns are present in the source dataset.

## Failure output

Both runners use this shape on validation, configuration, parsing, and runtime
errors:

```json
{
  "status": "failed",
  "job_id": "job-123",
  "error_code": "INVALID_INPUT",
  "error_message": "Missing required field(s): molecule_smiles",
  "traceback": "Traceback (most recent call last): ...",
  "warnings": []
}
```

Duplicate-check failures use `submission_id` in place of `job_id`. A missing
dataset produces `DATASET_NOT_CONFIGURED`.

## Portal and NIBI use

The future portal can write an input JSON file, submit a Slurm job that invokes
one of these commands on NIBI, and read the resulting JSON file after the job
finishes. Slurm submission, status polling, file transfer, authentication, and
portal persistence all happen outside these scripts.
