
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, functions as F


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "olist_schema_drift_005"

SOURCE_PATH = ROOT / "data/silver/olist/olist_orders_dataset"

OUTPUT_PATH = (
    ROOT / "experiments/output/olist/schema_drift"
    / EXPERIMENT_ID
)

GROUND_TRUTH_PATH = (
    ROOT / "experiments/ground_truth"
    / f"{EXPERIMENT_ID}.json"
)

ORIGINAL_COLUMN = "customer_id"
NEW_COLUMN = "customer_identifier"


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Schema-Drift-Injection")
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


def schema_dict(df):
    return {
        field.name: field.dataType.simpleString()
        for field in df.schema.fields
    }


def main():
    if OUTPUT_PATH.exists() or GROUND_TRUTH_PATH.exists():
        raise FileExistsError(
            "Experiment 005 already exists. "
            "Existing results will not be overwritten."
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        original = (
            spark.read.format("delta")
            .load(str(SOURCE_PATH))
        )

        if ORIGINAL_COLUMN not in original.columns:
            raise AssertionError(
                f"Missing source column: {ORIGINAL_COLUMN}"
            )

        if NEW_COLUMN in original.columns:
            raise AssertionError(
                f"Target column already exists: {NEW_COLUMN}"
            )

        original_rows = original.count()

        modified = original.withColumnRenamed(
            ORIGINAL_COLUMN,
            NEW_COLUMN,
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

        original_schema = schema_dict(original)
        modified_schema = schema_dict(result)

        removed_columns = sorted(
            set(original_schema) - set(modified_schema)
        )

        added_columns = sorted(
            set(modified_schema) - set(original_schema)
        )

        type_changes = [
            {
                "column": column,
                "original_type": original_schema[column],
                "modified_type": modified_schema[column],
            }
            for column in sorted(
                set(original_schema) & set(modified_schema)
            )
            if original_schema[column] != modified_schema[column]
        ]

        if original_rows != modified_rows:
            raise AssertionError("Row count changed unexpectedly")

        if removed_columns != [ORIGINAL_COLUMN]:
            raise AssertionError(
                f"Unexpected removed columns: {removed_columns}"
            )

        if added_columns != [NEW_COLUMN]:
            raise AssertionError(
                f"Unexpected added columns: {added_columns}"
            )

        if type_changes:
            raise AssertionError(
                f"Unexpected type changes: {type_changes}"
            )

        # Confirm that customer values were preserved.
        original_values = original.select(
            F.col("order_id"),
            F.col(ORIGINAL_COLUMN).alias("customer_key"),
        )

        modified_values = result.select(
            F.col("order_id"),
            F.col(NEW_COLUMN).alias("customer_key"),
        )

        changed_rows = (
            original_values.exceptAll(modified_values).count()
            + modified_values.exceptAll(original_values).count()
        )

        if changed_rows != 0:
            raise AssertionError(
                "Customer values changed during schema injection"
            )

        ground_truth = {
            "experiment_id": EXPERIMENT_ID,
            "dataset": "olist",
            "scenario": "breaking_column_rename",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_table": "olist_orders_dataset",
            "source_path": str(SOURCE_PATH.relative_to(ROOT)),
            "output_path": str(OUTPUT_PATH.relative_to(ROOT)),
            "original_rows": original_rows,
            "modified_rows": modified_rows,
            "original_schema": original_schema,
            "modified_schema": modified_schema,
            "injected_change": {
                "type": "column_rename",
                "old_column": ORIGINAL_COLUMN,
                "new_column": NEW_COLUMN,
            },
            "expected_schema_drift": {
                "added_columns": [NEW_COLUMN],
                "removed_columns": [ORIGINAL_COLUMN],
                "type_changes": [],
            },
            "changed_value_rows": changed_rows,
            "expected_downstream_failure": {
                "type": "missing_required_column",
                "column": ORIGINAL_COLUMN,
            },
            "root_cause": (
                "Controlled upstream schema change renamed "
                "customer_id to customer_identifier without "
                "updating downstream consumers."
            ),
        }

        with GROUND_TRUTH_PATH.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(ground_truth, file, indent=2)

        print("=" * 55)
        print("SCHEMA DRIFT INJECTION")
        print("=" * 55)
        print("Experiment:", EXPERIMENT_ID)
        print("Original rows:", original_rows)
        print("Modified rows:", modified_rows)
        print("Removed columns:", removed_columns)
        print("Added columns:", added_columns)
        print("Type changes:", type_changes)
        print("Changed value rows:", changed_rows)
        print("Status: VERIFIED")
        print("Output:", OUTPUT_PATH)
        print("Ground truth:", GROUND_TRUTH_PATH)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

