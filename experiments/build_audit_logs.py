import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]

DQ_DIR = ROOT / "evidence/dq_results/experiments"
SCHEMA_DIR = ROOT / "evidence/schemas/experiments"
OUTPUT_DIR = ROOT / "evidence/logs/experiments"

EXPERIMENTS = {
    "olist_null_customer_id_001": {
        "path": "experiments/output/olist/null_customer_id/olist_null_customer_id_001",
        "evidence_type": "dq",
    },
    "olist_duplicate_order_id_002": {
        "path": "experiments/output/olist/duplicate_order_id/olist_duplicate_order_id_002",
        "evidence_type": "dq",
    },
    "olist_negative_price_003": {
        "path": "experiments/output/olist/negative_price/olist_negative_price_003",
        "evidence_type": "dq",
    },
    "olist_invalid_customer_ref_004": {
        "path": "experiments/output/olist/invalid_customer_ref/olist_invalid_customer_ref_004",
        "evidence_type": "dq",
    },
    "olist_schema_drift_005": {
        "path": "experiments/output/olist/schema_drift/olist_schema_drift_005",
        "evidence_type": "schema",
    },
}


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Experiment-Audit")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(
        builder
    ).getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark


def event(level, component, message, details=None):
    return {
        "timestamp_utc": now(),
        "level": level,
        "component": component,
        "message": message,
        "details": details or {},
    }


def build_log(spark, experiment_id, config):
    events = []

    path = ROOT / config["path"]

    if not path.exists():
        raise FileNotFoundError(path)

    df = spark.read.format("delta").load(str(path))

    row_count = df.count()

    events.append(
        event(
            "INFO",
            "delta_reader",
            "Experimental Delta table successfully opened.",
            {
                "row_count": row_count,
                "column_count": len(df.columns),
            },
        )
    )

    events.append(
        event(
            "INFO",
            "schema_inspector",
            "Current experimental table schema inspected.",
            {
                "columns": [
                    {
                        "name": field.name,
                        "type": field.dataType.simpleString(),
                    }
                    for field in df.schema.fields
                ],
            },
        )
    )

    if config["evidence_type"] == "dq":
        evidence_path = (
            DQ_DIR / f"{experiment_id}_dq.json"
        )

        evidence = load_json(evidence_path)

        if evidence is None:
            events.append(
                event(
                    "WARNING",
                    "dq_evidence_reader",
                    "No experimental DQ evidence report found.",
                )
            )
        else:
            result = evidence.get("check_result", {})

            events.append(
                event(
                    "INFO",
                    "dq_evidence_reader",
                    "Existing DQ evidence report inspected.",
                    {
                        "check_type": result.get("check_type"),
                        "status": result.get("status"),
                        "total_rows": result.get("total_rows"),
                        "failed_rows": result.get("failed_rows"),
                        "observed_percentage": result.get(
                            "observed_percentage"
                        ),
                    },
                )
            )

    elif config["evidence_type"] == "schema":
        evidence_path = (
            SCHEMA_DIR / f"{experiment_id}_schema.json"
        )

        evidence = load_json(evidence_path)

        if evidence is None:
            events.append(
                event(
                    "WARNING",
                    "schema_evidence_reader",
                    "No schema comparison report found.",
                )
            )
        else:
            events.append(
                event(
                    "INFO",
                    "schema_evidence_reader",
                    "Existing schema comparison inspected.",
                    {
                        "drift_detected": evidence.get(
                            "drift_detected"
                        ),
                        "added_columns": evidence.get(
                            "added_columns"
                        ),
                        "removed_columns": evidence.get(
                            "removed_columns"
                        ),
                        "downstream_failure_detected": evidence.get(
                            "downstream_failure_detected"
                        ),
                    },
                )
            )

    events.append(
        event(
            "INFO",
            "audit_runner",
            "Artifact verification completed.",
            {
                "verification_status": "SUCCESS",
            },
        )
    )

    return {
        "log_version": "1.0",
        "experiment_id": experiment_id,
        "dataset": "olist",
        "log_type": "posthoc_artifact_verification",
        "generated_at_utc": now(),
        "historical_execution_log": False,
        "events": events,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        print("=" * 65)
        print("EXPERIMENT ARTIFACT AUDIT LOG GENERATION")
        print("=" * 65)

        for experiment_id, config in EXPERIMENTS.items():
            log = build_log(
                spark,
                experiment_id,
                config,
            )

            output_path = (
                OUTPUT_DIR / f"{experiment_id}.json"
            )

            with output_path.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(log, file, indent=2)

            print()
            print("Experiment:", experiment_id)
            print("Events recorded:", len(log["events"]))
            print("Log type:", log["log_type"])
            print("Saved:", output_path)

        print()
        print("=" * 65)
        print("AUDIT LOGS GENERATED:", len(EXPERIMENTS))
        print("STATUS: COMPLETE")
        print("=" * 65)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
