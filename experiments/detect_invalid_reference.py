
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from dq_checks.monitor import check_referential_integrity


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "olist_invalid_customer_ref_004"

EXPERIMENT_PATH = (
    ROOT / "experiments/output/olist/invalid_customer_ref"
    / EXPERIMENT_ID
)

REFERENCE_PATH = (
    ROOT / "data/silver/olist/olist_customers_dataset"
)

EVIDENCE_PATH = (
    ROOT / "evidence/dq_results/experiments"
    / f"{EXPERIMENT_ID}_dq.json"
)


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Reference-Failure-Detection")
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


def main():
    spark = create_spark()

    try:
        df = spark.read.format("delta").load(
            str(EXPERIMENT_PATH)
        )

        result = check_referential_integrity(
            spark,
            df,
            column="customer_id",
            reference_path=str(REFERENCE_PATH),
            reference_column="customer_id",
            max_orphan_pct=0,
        )

        report = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "olist",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "table": "olist_orders_dataset",
            "data_path": str(
                EXPERIMENT_PATH.relative_to(ROOT)
            ),
            "check_type": "referential_integrity",
            "column": "customer_id",
            "reference_table": "olist_customers_dataset",
            "check_result": result,
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
        print("EXPERIMENTAL REFERENTIAL INTEGRITY DETECTION")
        print("=" * 55)
        print("Experiment:", EXPERIMENT_ID)
        print("DQ result:", result)
        print("Evidence saved:", EVIDENCE_PATH)

        if result["status"] != "FAIL":
            raise AssertionError(
                "Expected referential integrity failure "
                "was not detected"
            )

        if result["failed_rows"] != 100:
            raise AssertionError(
                "Expected exactly 100 orphan references"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

