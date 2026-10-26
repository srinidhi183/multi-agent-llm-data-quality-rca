
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "olist_invalid_customer_ref_004"
INJECTION_ROWS = 100

SOURCE_PATH = ROOT / "data/silver/olist/olist_orders_dataset"
REFERENCE_PATH = ROOT / "data/silver/olist/olist_customers_dataset"

OUTPUT_PATH = (
    ROOT / "experiments/output/olist/invalid_customer_ref"
    / EXPERIMENT_ID
)

GROUND_TRUTH_PATH = (
    ROOT / "experiments/ground_truth"
    / f"{EXPERIMENT_ID}.json"
)


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Invalid-Reference-Injection")
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


def count_orphans(source, reference):
    valid_keys = (
        reference.select("customer_id")
        .where(F.col("customer_id").isNotNull())
        .distinct()
    )

    return (
        source.where(F.col("customer_id").isNotNull())
        .join(valid_keys, "customer_id", "left_anti")
        .count()
    )


def main():
    if OUTPUT_PATH.exists() or GROUND_TRUTH_PATH.exists():
        raise FileExistsError(
            "Experiment 004 already exists. "
            "Existing results will not be overwritten."
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(REFERENCE_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        original = (
            spark.read.format("delta")
            .load(str(SOURCE_PATH))
        )

        reference = (
            spark.read.format("delta")
            .load(str(REFERENCE_PATH))
        )

        original_rows = original.count()
        original_orphans = count_orphans(original, reference)

        if original_orphans != 0:
            raise AssertionError(
                "Baseline contains orphan customer IDs. "
                "Review the source before injecting failures."
            )

        # Generate candidate IDs with a reserved experiment prefix.
        # Verify that none of them exists in the customer master.
        candidates = spark.range(INJECTION_ROWS).select(
            F.concat(
                F.lit("THESIS_INVALID_CUSTOMER_004_"),
                F.lpad(F.col("id").cast("string"), 6, "0"),
            ).alias("customer_id")
        )

        collisions = (
            candidates.join(
                reference.select("customer_id").distinct(),
                "customer_id",
                "inner",
            )
            .count()
        )

        if collisions:
            raise AssertionError(
                "Generated customer IDs already exist "
                "in the reference table."
            )

        # Select 100 existing orders deterministically.
        selected = (
            original.where(F.col("customer_id").isNotNull())
            .select("order_id")
            .withColumn(
                "_rank",
                F.row_number().over(
                    Window.orderBy("order_id")
                ),
            )
            .where(F.col("_rank") <= INJECTION_ROWS)
        )

        eligible_count = selected.count()

        if eligible_count != INJECTION_ROWS:
            raise AssertionError(
                "Not enough eligible orders for injection"
            )

        selected = selected.withColumn(
            "_new_customer_id",
            F.concat(
                F.lit("THESIS_INVALID_CUSTOMER_004_"),
                F.lpad(
                    (F.col("_rank") - 1).cast("string"),
                    6,
                    "0",
                ),
            ),
        ).select("order_id", "_new_customer_id")

        modified = (
            original.join(selected, "order_id", "left")
            .withColumn(
                "customer_id",
                F.coalesce(
                    F.col("_new_customer_id"),
                    F.col("customer_id"),
                ),
            )
            .drop("_new_customer_id")
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
        modified_orphans = count_orphans(result, reference)

        observed_increase = modified_orphans - original_orphans

        if modified_rows != original_rows:
            raise AssertionError(
                "Unexpected change in total row count"
            )

        if observed_increase != INJECTION_ROWS:
            raise AssertionError(
                "Orphan count does not match injection target"
            )

        ground_truth = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "olist",
            "scenario": "invalid_customer_ref",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_table": "olist_orders_dataset",
            "reference_table": "olist_customers_dataset",
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "reference_path": str(
                REFERENCE_PATH.relative_to(ROOT)
            ),
            "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
            "affected_column": "customer_id",
            "injected_rows": INJECTION_ROWS,
            "original_rows": original_rows,
            "modified_rows": modified_rows,
            "original_failure_count": original_orphans,
            "modified_failure_count": modified_orphans,
            "observed_failure_increase": observed_increase,
            "expected_dq_failure": {
                "type": "referential_integrity",
                "column": "customer_id",
                "reference_column": "customer_id",
                "reference_path": str(
                    REFERENCE_PATH.relative_to(ROOT)
                ),
                "max_orphan_pct": 0,
            },
            "root_cause": (
                "Controlled injection replaced 100 valid "
                "order customer IDs with values absent "
                "from the Olist customer master."
            ),
        }

        with GROUND_TRUTH_PATH.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(ground_truth, file, indent=2)

        print("=" * 55)
        print("REFERENTIAL INTEGRITY FAILURE INJECTION")
        print("=" * 55)
        print("Experiment:", EXPERIMENT_ID)
        print("Original rows:", original_rows)
        print("Modified rows:", modified_rows)
        print("Original orphan references:", original_orphans)
        print("Modified orphan references:", modified_orphans)
        print("Injected orphan references:", observed_increase)
        print("Status: VERIFIED")
        print("Output:", OUTPUT_PATH)
        print("Ground truth:", GROUND_TRUTH_PATH)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

