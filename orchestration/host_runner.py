
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"

TOKEN = os.environ.get("THESIS_RUNNER_TOKEN")

JOBS = {
    "olist_bronze": "pipelines/olist/bronze_ingestion.py",
    "olist_silver": "pipelines/olist/silver_transformation.py",
    "olist_gold": "pipelines/olist/gold_transformation.py",
    "dataco_bronze": "pipelines/dataco/bronze_ingestion.py",
    "dataco_silver": "pipelines/dataco/silver_transformation.py",
    "dataco_gold": "pipelines/dataco/gold_transformation.py",
    "nyc_bronze": "pipelines/nyc_taxi/bronze_ingestion.py",
    "nyc_silver": "pipelines/nyc_taxi/silver_transformation.py",
    "nyc_gold": "pipelines/nyc_taxi/gold_transformation.py",
}

RUNS = {}
LOCK = threading.Lock()

LOG_DIR = ROOT / "evidence" / "logs" / "orchestration"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def execute_job(run_id, job_name):
    script = ROOT / JOBS[job_name]
    log_path = LOG_DIR / f"{run_id}.log"

    with LOCK:
        RUNS[run_id]["status"] = "RUNNING"
        RUNS[run_id]["started_at"] = now()

    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [str(PYTHON), str(script)],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
            )

        status = (
            "SUCCESS" if result.returncode == 0
            else "FAILED"
        )

        with LOCK:
            RUNS[run_id].update({
                "status": status,
                "return_code": result.returncode,
                "finished_at": now(),
            })

    except Exception as exc:
        with LOCK:
            RUNS[run_id].update({
                "status": "FAILED",
                "error": str(exc),
                "finished_at": now(),
            })


class Handler(BaseHTTPRequestHandler):

    def respond(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        return (
            TOKEN is not None
            and self.headers.get("X-Runner-Token") == TOKEN
        )

    def do_GET(self):
        if not self.authorized():
            return self.respond(401, {"error": "Unauthorized"})

        path = urlparse(self.path).path

        if path == "/health":
            return self.respond(
                200,
                {"status": "OK", "jobs": len(JOBS)}
            )

        if path.startswith("/runs/"):
            run_id = path.split("/")[-1]

            with LOCK:
                run = RUNS.get(run_id)

            if run is None:
                return self.respond(
                    404, {"error": "Unknown run"}
                )

            return self.respond(200, run)

        return self.respond(404, {"error": "Not found"})

    def do_POST(self):
        if not self.authorized():
            return self.respond(401, {"error": "Unauthorized"})

        if urlparse(self.path).path != "/run":
            return self.respond(404, {"error": "Not found"})

        try:
            length = int(
                self.headers.get("Content-Length", "0")
            )

            if length <= 0 or length > 4096:
                return self.respond(
                    400, {"error": "Invalid request size"}
                )

            payload = json.loads(
                self.rfile.read(length)
            )

            job_name = payload.get("job")

            if job_name not in JOBS:
                return self.respond(
                    400, {"error": "Unknown job"}
                )

            # Prevent concurrent runs from overwriting
            # the same Delta tables.
            with LOCK:
                active = any(
                    run["status"] in ("QUEUED", "RUNNING")
                    for run in RUNS.values()
                )

                if active:
                    return self.respond(
                        409, {"error": "Another job is active"}
                    )

                run_id = str(uuid.uuid4())

                RUNS[run_id] = {
                    "run_id": run_id,
                    "job": job_name,
                    "status": "QUEUED",
                    "created_at": now(),
                }

            thread = threading.Thread(
                target=execute_job,
                args=(run_id, job_name),
                daemon=True,
            )
            thread.start()

            return self.respond(
                202, {"run_id": run_id, "status": "QUEUED"}
            )

        except (ValueError, TypeError, KeyError) as exc:
            return self.respond(
                400, {"error": str(exc)}
            )


if __name__ == "__main__":
    if not TOKEN or len(TOKEN) < 32:
        sys.exit(
            "Set THESIS_RUNNER_TOKEN to a random "
            "secret of at least 32 characters."
        )

    if not PYTHON.is_file():
        sys.exit(f"Python not found: {PYTHON}")

    print("Starting thesis host runner on port 8765")
    print("Press Ctrl+C to stop")

    server = ThreadingHTTPServer(
        ("0.0.0.0", 8765),
        Handler
    )

    server.serve_forever()
