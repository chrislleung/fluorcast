# FluorCast desktop prediction job contract

This document defines the stable file boundary between the future FluorCast desktop application and the prediction engine running on NIBI. The desktop application writes `input.json`, submits a Slurm job, and reads `output.json`; it does not load models or run predictions locally.

JSON files must be UTF-8 encoded. Timestamps use ISO 8601 with a timezone. Producers may add fields in future versions, so consumers should ignore unknown fields. Fields documented as required must not be omitted.

## Prediction input (`input.json`)

The input is one JSON object with these required, non-empty string fields:

| Field | Meaning |
| --- | --- |
| `job_id` | Unique application job identifier; copied unchanged to output. |
| `user_id` | Application user identifier. |
| `molecule_smiles` | Molecule SMILES supplied by the user. |
| `solvent_smiles` | Solvent SMILES supplied by the user. |
| `model_choice` | Requested prediction model or supported model collection. |
| `requested_at` | ISO 8601 timestamp for job creation. |

The currently recognized `model_choice` values are `hybrid`, `all`, `rf`, `extratrees`, `gbdt`, `histgb`, and `graph_model_later`. Recognition does not guarantee that an artifact is installed on a particular NIBI deployment. `hybrid` is the preferred full FluorCast workflow.

```json
{
  "job_id": "job-example-001",
  "user_id": "user-example-001",
  "molecule_smiles": "c1ccccc1",
  "solvent_smiles": "CCO",
  "model_choice": "hybrid",
  "requested_at": "2026-07-03T14:30:00Z"
}
```

## Prediction output (`output.json`)

Every completed job attempt writes one JSON object. The following fields are required:

| Field | Meaning |
| --- | --- |
| `job_id` | Input job identifier, or `null` when input could not provide one. |
| `status` | `success` or `failed`. |
| `canonical_molecule_smiles` | Canonical molecule SMILES on success; otherwise `null`. |
| `canonical_solvent_smiles` | Canonical solvent SMILES on success; otherwise `null`. |
| `predictions` | Prediction records; an empty array on failure. |
| `applicability_domain` | Applicability assessment on success; `null` when unavailable or on failure. |
| `warnings` | Zero or more human-readable, non-fatal warning strings. |

`error` is optional. It is absent for success and required by this contract when `status` is `failed`. It contains stable `code` and human-readable `message` strings. Applications should branch on `code`, not parse `message`. Tracebacks and internal filesystem details are operational diagnostics and are not part of this app-facing contract.

Prediction records are model results, not a promise that every target is available. The hybrid engine may provide `model_name`, `predicted_absorption_nm`, `predicted_emission_nm`, `predicted_stokes_shift_nm`, `predicted_stokes_shift_cm^-1`, `predicted_quantum_yield`, `brightness_class`, `physically_valid_stokes`, `prediction_intervals`, `applicability_domain`, `nearest_training_similarity`, `nearest_training_smiles`, and per-model `warnings`. Legacy model choices may only provide emission, quantum yield, nearest-training fields, and warnings. Missing target values may be `null`.

When available, `applicability_domain` may contain `outside_applicability_domain`, `nearest_training_similarity`, `nearest_training_smiles`, and `confidence_label`. A successful result outside the domain remains `status: "success"` and must also carry a warning; it is not silently converted into a failure.

### Successful prediction

```json
{
  "job_id": "job-example-001",
  "status": "success",
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
  "applicability_domain": {
    "outside_applicability_domain": false,
    "nearest_training_similarity": 0.8,
    "nearest_training_smiles": "c1ccccc1",
    "confidence_label": "high"
  },
  "warnings": []
}
```

### Validation failure

Validation failures do not invoke a prediction backend.

```json
{
  "job_id": "job-example-002",
  "status": "failed",
  "canonical_molecule_smiles": null,
  "canonical_solvent_smiles": null,
  "predictions": [],
  "applicability_domain": null,
  "warnings": [],
  "error": {
    "code": "INVALID_INPUT",
    "message": "Missing required field(s): molecule_smiles"
  }
}
```

### Model failure

```json
{
  "job_id": "job-example-003",
  "status": "failed",
  "canonical_molecule_smiles": null,
  "canonical_solvent_smiles": null,
  "predictions": [],
  "applicability_domain": null,
  "warnings": [],
  "error": {
    "code": "MODEL_UNAVAILABLE",
    "message": "The requested model artifact could not be loaded in the current environment."
  }
}
```

### Outside applicability domain

```json
{
  "job_id": "job-example-004",
  "status": "success",
  "canonical_molecule_smiles": "c1ccccc1",
  "canonical_solvent_smiles": "CCO",
  "predictions": [
    {
      "model_name": "rf",
      "predicted_emission_nm": 450.0,
      "predicted_quantum_yield": 0.2,
      "nearest_training_similarity": 0.3,
      "nearest_training_smiles": "CC",
      "warnings": []
    }
  ],
  "applicability_domain": {
    "outside_applicability_domain": true,
    "nearest_training_similarity": 0.3,
    "nearest_training_smiles": "CC",
    "confidence_label": "low"
  },
  "warnings": [
    "Prediction is outside the applicability domain."
  ]
}
```

## Handoff behavior

The desktop application should treat a missing or unreadable `output.json` as an infrastructure/transfer condition, not a model result. Once valid output is available, `status` and `error.code` determine the application state. Warnings are displayable context and never replace status handling.
