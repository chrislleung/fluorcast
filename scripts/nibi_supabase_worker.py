"""Poll Supabase for FluorCast prediction jobs and submit/collect Slurm work."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_JOB_SCRIPT = Path("slurm") / "run_prediction_job.sbatch"
SLURM_JOB_RE = re.compile(r"Submitted batch job\s+(\S+)")


class WorkerError(Exception):
    """Expected worker configuration or runtime failure."""


@dataclass(frozen=True)
class WorkerConfig:
    supabase_url: str
    service_role_key: str
    fluorcast_repo: Path
    jobs_dir: Path
    poll_limit: int = 5


class SupabaseRestClient:
    """Small Supabase REST client using only the Python standard library."""

    def __init__(self, url: str, service_role_key: str) -> None:
        self.base_url = url.rstrip("/")
        self.headers = {
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str | int] | None = None,
        payload: Any | None = None,
        prefer: str | None = None,
    ) -> Any:
        url = f"{self.base_url}/rest/v1/{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        headers = dict(self.headers)
        if prefer:
            headers["Prefer"] = prefer
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=data, headers=headers, method=method)
        with urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
        return None if not body else json.loads(body)

    def select(
        self, table: str, query: dict[str, str | int] | None = None
    ) -> list[dict[str, Any]]:
        result = self._request("GET", table, query=query)
        if not isinstance(result, list):
            raise WorkerError(f"Expected list response from Supabase table {table}.")
        return result

    def insert(self, table: str, rows: list[dict[str, Any]]) -> Any:
        if not rows:
            return []
        return self._request(
            "POST",
            table,
            payload=rows,
            prefer="return=representation",
        )

    def patch(self, table: str, row_id: str, values: dict[str, Any]) -> Any:
        return self._request(
            "PATCH",
            table,
            query={"id": f"eq.{row_id}"},
            payload=values,
            prefer="return=representation",
        )


def log(message: str) -> None:
    print(f"[{datetime.now(UTC).isoformat()}] {message}", flush=True)


def read_config(environ: dict[str, str] | None = None) -> WorkerConfig:
    env = os.environ if environ is None else environ
    missing = [
        name
        for name in (
            "SUPABASE_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "FLUORCAST_REPO",
            "FLUORCAST_JOBS_DIR",
        )
        if not env.get(name)
    ]
    if missing:
        raise WorkerError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )
    try:
        poll_limit = int(env.get("FLUORCAST_POLL_LIMIT", "5"))
    except ValueError as exc:
        raise WorkerError("FLUORCAST_POLL_LIMIT must be an integer.") from exc
    if poll_limit < 1:
        raise WorkerError("FLUORCAST_POLL_LIMIT must be at least 1.")
    return WorkerConfig(
        supabase_url=env["SUPABASE_URL"],
        service_role_key=env["SUPABASE_SERVICE_ROLE_KEY"],
        fluorcast_repo=Path(env["FLUORCAST_REPO"]),
        jobs_dir=Path(env["FLUORCAST_JOBS_DIR"]),
        poll_limit=poll_limit,
    )


def prediction_job_dir(jobs_dir: Path, job_id: str) -> Path:
    return jobs_dir / f"prediction_{job_id}"


def create_prediction_input_payload(
    job: dict[str, Any], requested_at: str | None = None
) -> dict[str, Any]:
    return {
        "job_id": str(job["id"]),
        "user_id": str(job["user_id"]),
        "molecule_smiles": str(job["molecule_smiles"]),
        "solvent_smiles": str(job["solvent_smiles"]),
        "model_choice": str(job["model_choice"]),
        "requested_at": requested_at or datetime.now(UTC).isoformat(),
    }


def read_prediction_output(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise WorkerError(f"Prediction output JSON must be an object: {path}")
    if payload.get("status") not in {"success", "failed"}:
        raise WorkerError(f"Prediction output has invalid status: {path}")
    return payload


def prediction_results_rows(output: dict[str, Any]) -> list[dict[str, Any]]:
    job_id = output.get("job_id")
    rows = []
    for prediction in output.get("predictions", []):
        rows.append(
            {
                "prediction_job_id": job_id,
                "model_name": prediction.get("model_name"),
                "predicted_emission_nm": prediction.get("predicted_emission_nm"),
                "predicted_quantum_yield": prediction.get(
                    "predicted_quantum_yield"
                ),
                "nearest_training_similarity": prediction.get(
                    "nearest_training_similarity"
                ),
                "nearest_training_smiles": prediction.get(
                    "nearest_training_smiles"
                ),
                "warnings": prediction.get("warnings", []),
            }
        )
    return rows


def queued_prediction_jobs(
    client: SupabaseRestClient, poll_limit: int
) -> list[dict[str, Any]]:
    return client.select(
        "prediction_jobs",
        {
            "select": "id,user_id,molecule_smiles,solvent_smiles,model_choice,status",
            "status": "eq.queued",
            "order": "created_at.asc",
            "limit": poll_limit,
        },
    )


def update_prediction_job(
    client: SupabaseRestClient,
    job_id: str,
    values: dict[str, Any],
    *,
    optional_fields: set[str] | None = None,
) -> None:
    optional_fields = optional_fields or set()
    try:
        client.patch("prediction_jobs", job_id, values)
    except HTTPError as exc:
        removable = optional_fields.intersection(values)
        if not removable:
            raise
        for field in sorted(removable):
            log(f"Supabase rejected optional column {field}; retrying without it.")
        retry_values = {key: value for key, value in values.items() if key not in removable}
        client.patch("prediction_jobs", job_id, retry_values)


def capture_slurm_job_id(stdout: str) -> str | None:
    match = SLURM_JOB_RE.search(stdout)
    return None if match is None else match.group(1)


def submit_prediction_job(
    config: WorkerConfig, input_path: Path, output_path: Path
) -> str | None:
    script_path = config.fluorcast_repo / PREDICTION_JOB_SCRIPT
    env = os.environ.copy()
    env.update(
        {
            "FLUORCAST_REPO": str(config.fluorcast_repo),
            "FLUORCAST_INPUT_JSON": str(input_path),
            "FLUORCAST_OUTPUT_JSON": str(output_path),
        }
    )
    command = ["sbatch", str(script_path)]
    log("Submitting Slurm command: " + " ".join(command))
    completed = subprocess.run(
        command,
        cwd=config.fluorcast_repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stdout.strip():
        log("sbatch stdout: " + completed.stdout.strip())
    if completed.stderr.strip():
        log("sbatch stderr: " + completed.stderr.strip())
    return capture_slurm_job_id(completed.stdout)


def submit_queued_jobs(
    client: SupabaseRestClient, config: WorkerConfig, *, dry_run: bool = False
) -> int:
    jobs = queued_prediction_jobs(client, config.poll_limit)
    log(f"Queued prediction jobs found: {len(jobs)}")
    submitted = 0
    for job in jobs:
        job_id = str(job["id"])
        job_dir = prediction_job_dir(config.jobs_dir, job_id)
        input_path = job_dir / "input.json"
        output_path = job_dir / "output.json"
        payload = create_prediction_input_payload(job)
        if dry_run:
            log(f"Dry run: would write prediction input for job {job_id} to {input_path}")
            log(f"Dry run: would update job {job_id} to running and submit sbatch")
            continue
        job_dir.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        log(f"Input path written for job {job_id}: {input_path}")
        update_prediction_job(client, job_id, {"status": "running"})
        log(f"Supabase job {job_id} updated to running.")
        slurm_job_id = submit_prediction_job(config, input_path, output_path)
        update_values: dict[str, Any] = {"status": "running"}
        if slurm_job_id is not None:
            update_values["slurm_job_id"] = slurm_job_id
        update_prediction_job(
            client,
            job_id,
            update_values,
            optional_fields={"slurm_job_id"},
        )
        if slurm_job_id is None:
            log(f"Submitted job {job_id}; no Slurm job id was detected.")
        else:
            log(f"Submitted job {job_id} as Slurm job {slurm_job_id}.")
        submitted += 1
    return submitted


def get_prediction_job_status(client: SupabaseRestClient, job_id: str) -> str | None:
    rows = client.select(
        "prediction_jobs",
        {
            "select": "status",
            "id": f"eq.{job_id}",
            "limit": 1,
        },
    )
    if not rows:
        return None
    status = rows[0].get("status")
    return None if status is None else str(status)


def prediction_results_exist(client: SupabaseRestClient, job_id: str) -> bool:
    rows = client.select(
        "prediction_results",
        {
            "select": "id",
            "prediction_job_id": f"eq.{job_id}",
            "limit": 1,
        },
    )
    return bool(rows)


def iter_prediction_outputs(jobs_dir: Path) -> list[Path]:
    if not jobs_dir.exists():
        return []
    return sorted(jobs_dir.glob("prediction_*/output.json"))


def collect_completed_jobs(
    client: SupabaseRestClient, config: WorkerConfig, *, dry_run: bool = False
) -> int:
    collected = 0
    output_paths = iter_prediction_outputs(config.jobs_dir)
    log(f"Prediction output files found: {len(output_paths)}")
    for output_path in output_paths:
        output = read_prediction_output(output_path)
        job_id = str(output.get("job_id") or output_path.parent.name.removeprefix("prediction_"))
        current_status = get_prediction_job_status(client, job_id)
        if current_status == "completed":
            log(f"Skipping job {job_id}; Supabase status is already completed.")
            continue
        if dry_run:
            log(f"Dry run: would collect output for job {job_id}: {output_path}")
            continue
        if output["status"] == "success":
            rows = prediction_results_rows(output)
            if prediction_results_exist(client, job_id):
                log(f"Prediction results already exist for job {job_id}; not inserting duplicates.")
            elif rows:
                client.insert("prediction_results", rows)
                log(f"Inserted {len(rows)} prediction_results row(s) for job {job_id}.")
            else:
                log(f"Job {job_id} succeeded with no prediction rows.")
            update_prediction_job(client, job_id, {"status": "completed"})
            log(f"Supabase job {job_id} updated to completed.")
        else:
            error_message = output.get("error_message") or "Prediction job failed."
            update_prediction_job(
                client,
                job_id,
                {"status": "failed", "error_message": error_message},
                optional_fields={"error_message"},
            )
            log(f"Supabase job {job_id} updated to failed.")
        log(f"Output collected for job {job_id}: {output_path}")
        collected += 1
    return collected


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Run one submit and collect pass.")
    parser.add_argument("--submit-only", action="store_true", help="Only submit queued jobs.")
    parser.add_argument("--collect-only", action="store_true", help="Only collect completed outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Log planned work without writing or updating.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.submit_only and args.collect_only:
        print("--submit-only and --collect-only cannot be used together.", file=sys.stderr)
        return 2
    try:
        config = read_config()
        client = SupabaseRestClient(config.supabase_url, config.service_role_key)
        if not args.collect_only:
            submit_queued_jobs(client, config, dry_run=args.dry_run)
        if not args.submit_only:
            collect_completed_jobs(client, config, dry_run=args.dry_run)
    except Exception as exc:
        log(f"Worker failed: {exc}")
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
