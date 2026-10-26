
import json
import os
import time
import urllib.error
import urllib.request

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# 1. HOST RUNNER CONFIGURATION
# ============================================================

RUNNER_URL = os.environ.get(
    "THESIS_RUNNER_URL",
    "http://host.docker.internal:8765"
)

RUNNER_TOKEN = os.environ.get("THESIS_RUNNER_TOKEN")


# ============================================================
# 2. HTTP HELPER
# ============================================================

def runner_request(method, endpoint, payload=None):
    if not RUNNER_TOKEN:
        raise RuntimeError(
            "THESIS_RUNNER_TOKEN is not configured"
        )

    data = None

    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        f"{RUNNER_URL}{endpoint}",
        data=data,
        method=method,
        headers={
            "X-Runner-Token": RUNNER_TOKEN,
            "Content-Type": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(
            request, timeout=30
        ) as response:
            return json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8", errors="replace"
        )

        raise RuntimeError(
            f"Host runner HTTP {exc.code}: {body}"
        ) from exc


# ============================================================
# 3. EXECUTE SPARK JOB
# ============================================================

def execute_spark_job(job_name):
    print(f"Submitting Spark job: {job_name}")

    response = runner_request(
        "POST",
        "/run",
        {"job": job_name}
    )

    run_id = response["run_id"]

    print(f"Host runner ID: {run_id}")

    deadline = time.monotonic() + 7200

    while time.monotonic() < deadline:
        time.sleep(10)

        status = runner_request(
            "GET",
            f"/runs/{run_id}"
        )

        state = status["status"]

        print(f"{job_name}: {state}")

        if state == "SUCCESS":
            print(
                f"Spark job completed: {job_name}"
            )
            return

        if state == "FAILED":
            raise RuntimeError(
                f"Spark job failed: {job_name}. "
                f"Details: {status}"
            )

    raise TimeoutError(
        f"Spark job exceeded two hours: {job_name}. "
        f"Host run ID: {run_id}. "
        "Check the host runner before retrying."
    )


# ============================================================
# 4. PIPELINE CONFIGURATION
# ============================================================

PIPELINES = {
    "olist_pipeline": [
        "olist_bronze",
        "olist_silver",
        "olist_gold"
    ],
    "dataco_pipeline": [
        "dataco_bronze",
        "dataco_silver",
        "dataco_gold"
    ],
    "nyc_taxi_pipeline": [
        "nyc_bronze",
        "nyc_silver",
        "nyc_gold"
    ]
}


# ============================================================
# 5. CREATE AIRFLOW DAGS
# ============================================================

for dag_name, jobs in PIPELINES.items():

    with DAG(
        dag_id=dag_name,
        description=(
            "Bronze-Silver-Gold Spark pipeline "
            "executed through the Mac host runner"
        ),
        start_date=datetime(2026, 10, 1),
        schedule=None,
        catchup=False,
        max_active_runs=1,
        default_args={
            "owner": "thesis",
            "retries": 0,
            "execution_timeout": timedelta(
                hours=2, minutes=5
            )
        },
        tags=["master-thesis", "spark", "lakehouse"]
    ) as dag:

        previous_task = None

        for job_name in jobs:

            task = PythonOperator(
                task_id=job_name,
                python_callable=execute_spark_job,
                op_kwargs={
                    "job_name": job_name
                }
            )

            if previous_task is not None:
                previous_task >> task

            previous_task = task

    globals()[dag_name] = dag
