
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    ROOT / "data/silver/olist/olist_order_items_dataset"
)

EXPERIMENT_ID = "olist_negative_price_003"

OUTPUT_PATH = (
    ROOT / "experiments/output/olist/negative_price"
    / EXPERIMENT_ID
)

GROUND_TRUTH_PATH = (
    ROOT / "experiments/ground_truth"
    / f"{EXPERIMENT_ID}.json"
)

INJECTION_ROWS = 100


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Negative-Price-Injection")
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
    if OUTPUT_PATH.exists() or GROUND_TRUTH_PATH.exists():
        raise FileExistsError(
            "Experiment 003 already exists. "
            "Existing results will not be overwritten."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    spark = create_spark()

    try:
        original = (
            spark.read.format("delta")
            .load(str(SOURCE_PATH))
        )

        original_rows = original.count()

        original_failures = original.filter(
            F.col("price") < 0
        ).count()

        # Select 100 valid rows deterministically.
        selected = (
            original.filter(F.col("price") >= 0)
            .withColumn(
                "_rank",
                F.row_number().over(
                    Window.orderBy("order_id", "order_item_id")
                ),
            )
            .filter(F.col("_rank") <= INJECTION_ROWS)
            .select("order_id", "order_item_id")
            .withColumn("_inject", F.lit(True))
        )

        modified = (
            original.join(
                selected,
                ["order_id", "order_item_id"],
                "left",
            )
            .withColumn(
                "price",
                F.when(
                    F.col("_inject") == True,
                    F.lit(-99.99),
                ).otherwise(F.col("price")),
            )
            .drop("_inject")
        )

        (
            modified.write.format("delta")
            .mode("errorifexists")
            .save(str(OUTPUT_PATH))
        )

        result = (
            spark.read.format("delta")
            .load(str(OUTPUT_PATH))
        )

        modified_rows = result.count()

        modified_failures = result.filter(
            F.col("price") < 0
        ).count()

        observed_increase = (
            modified_failures - original_failures
        )

        if modified_rows != original_rows:
            raise AssertionError("Row count changed")

        if observed_increase != INJECTION_ROWS:
            raise AssertionError(
                "Injected failure count mismatch"
            )

        ground_truth = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "olist",
            "scenario": "negative_price",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_table": "olist_order_items_dataset",
            "source_path": str(
                SOURCE_PATH.relative_to(ROOT)
            ),
            "output_path": str(
                OUTPUT_PATH.relative_to(ROOT)
            ),
            "affected_column": "price",
            "injected_rows": INJECTION_ROWS,
            "original_rows": original_rows,
            "modified_rows": modified_rows,
            "original_failure_count": original_failures,
            "modified_failure_count": modified_failures,
            "observed_failure_increase": observed_increase,
            "expected_dq_failure": {
                "type": "range",
                "column": "price",
                "minimum": 0,
            },
            "root_cause": (
                "Controlled injection replaced 100 "
                "valid product prices with -99.99."
            ),
        }

        with GROUND_TRUTH_PATH.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(ground_truth, file, indent=2)

        print("=" * 55)
        print("INVALID NUMERIC VALUE INJECTION")
        print("=" * 55)
        print("Experiment:", EXPERIMENT_ID)
        print("Original rows:", original_rows)
        print("Modified rows:", modified_rows)
        print("Original negative prices:", original_failures)
        print("Modified negative prices:", modified_failures)
        print("Injected negative prices:", observed_increase)
        print("Status: VERIFIED")
        print("Output:", OUTPUT_PATH)
        print("Ground truth:", GROUND_TRUTH_PATH)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

