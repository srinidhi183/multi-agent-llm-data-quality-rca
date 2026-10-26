
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

from dq_checks.monitor import (
    check_nulls,
    check_duplicates,
    check_range,
)


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "olist_null_customer_id_001": {
        "scenario": "null_customer_id",
        "check_type": "null_percentage",
        "column": "customer_id",
    },
    "olist_duplicate_order_id_002": {
        "scenario": "duplicate_order_id",
        "check_type": "duplicate_percentage",
        "column": "order_id",
    },
    "olist_negative_price_003": {
        "scenario": "negative_price",
        "check_type": "range",
        "column": "price",
        "minimum": 0,
    },
}

EVIDENCE_DIR = ROOT / "evidence/dq_results/experiments"


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Experimental-DQ-Detection")
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

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=list(EXPERIMENTS),
    )
    args = parser.parse_args()

    experiment_id = args.experiment
    config = EXPERIMENTS[experiment_id]

    experiment_path = (
        ROOT
        / "experiments/output/olist"
        / config["scenario"]
        / experiment_id
    )

    if not experiment_path.exists():
        raise FileNotFoundError(experiment_path)

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        df = spark.read.format("delta").load(
            str(experiment_path)
        )

        check_type = config["check_type"]

        if check_type == "null_percentage":
            result = check_nulls(
                df,
                config["column"],
                max_null_pct=0,
            )

        elif check_type == "duplicate_percentage":
            result = check_duplicates(
                df,
                [config["column"]],
                max_duplicate_pct=0,
            )

        elif check_type == "range":
            result = check_range(
                df,
                config["column"],
                minimum=config["minimum"],
            )

        else:
            raise ValueError(
                f"Unsupported check type: {check_type}"
            )

        report = {
            "experiment_id": experiment_id,
            "dataset": "olist",
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "table": (
                "olist_order_items_dataset"
                if check_type == "range"
                else "olist_orders_dataset"
            ),
            "data_path": str(
                experiment_path.relative_to(ROOT)
            ),
            "check_type": check_type,
            "column": config["column"],
            "check_result": result,
        }

        output = (
            EVIDENCE_DIR / f"{experiment_id}_dq.json"
        )

        with output.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("=" * 55)
        print("EXPERIMENTAL DQ DETECTION")
        print("=" * 55)
        print("Experiment:", experiment_id)
        print("DQ check result:", result)
        print("Evidence saved:", output)

        if result["status"] != "FAIL":
            raise AssertionError(
                "Expected DQ failure was not detected"
            )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

