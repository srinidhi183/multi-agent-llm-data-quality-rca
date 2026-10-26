
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.utils import AnalysisException


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "olist_schema_drift_005"

SOURCE_PATH = ROOT / "data/silver/olist/olist_orders_dataset"

EXPERIMENT_PATH = (
    ROOT / "experiments/output/olist/schema_drift"
    / EXPERIMENT_ID
)

EVIDENCE_PATH = (
    ROOT / "evidence/schemas/experiments"
    / f"{EXPERIMENT_ID}_schema.json"
)


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Schema-Drift-Detection")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def schema_dict(df):
    return {
        field.name: field.dataType.simpleString()
        for field in df.schema.fields
    }


def main():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    if not EXPERIMENT_PATH.exists():
        raise FileNotFoundError(EXPERIMENT_PATH)

    spark = create_spark()

    try:
        baseline = spark.read.format("delta").load(
            str(SOURCE_PATH)
        )

        experimental = spark.read.format("delta").load(
            str(EXPERIMENT_PATH)
        )

        baseline_schema = schema_dict(baseline)
        experimental_schema = schema_dict(experimental)

        added = sorted(
            set(experimental_schema) - set(baseline_schema)
        )

        removed = sorted(
            set(baseline_schema) - set(experimental_schema)
        )

        type_changes = [
            {
                "column": column,
                "original_type": baseline_schema[column],
                "modified_type": experimental_schema[column],
            }
            for column in sorted(
                set(baseline_schema) & set(experimental_schema)
            )
            if baseline_schema[column] != experimental_schema[column]
        ]

        drift_detected = bool(
            added or removed or type_changes
        )

        # Simulate a downstream consumer that still expects
        # the original customer_id column.
        downstream_failure_detected = False
        downstream_error = None

        try:
            experimental.select(
                "order_id",
                "customer_id",
            ).count()
        except AnalysisException as exc:
            downstream_failure_detected = True
            downstream_error = str(exc)[:1500]

        status = (
            "FAIL"
            if drift_detected and downstream_failure_detected
            else "PASS"
        )

        report = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "olist",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "baseline_path": str(
                SOURCE_PATH.relative_to(ROOT)
            ),
            "experimental_path": str(
                EXPERIMENT_PATH.relative_to(ROOT)
            ),
            "check_type": "schema_drift",
            "added_columns": added,
            "removed_columns": removed,
            "type_changes": type_changes,
            "drift_detected": drift_detected,
            "downstream_failure_detected": (
                downstream_failure_detected
            ),
            "downstream_error": downstream_error,
            "status": status,
        }

        EVIDENCE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with EVIDENCE_PATH.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(report, file, indent=2)

        print("=" * 55)
        print("EXPERIMENTAL SCHEMA DRIFT DETECTION")
        print("=" * 55)
        print("Experiment:", EXPERIMENT_ID)
        print("Added columns:", added)
        print("Removed columns:", removed)
        print("Type changes:", type_changes)
        print("Drift detected:", drift_detected)
        print(
            "Downstream failure detected:",
            downstream_failure_detected,
        )
        print("Detection status:", status)
        print("Evidence saved:", EVIDENCE_PATH)

        if not drift_detected:
            raise AssertionError("Expected schema drift missing")

        if not downstream_failure_detected:
            raise AssertionError(
                "Expected downstream failure missing"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

