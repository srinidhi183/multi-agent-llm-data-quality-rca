import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "evidence/schemas/experiments"

EXPERIMENTS = {
    "olist_null_customer_id_001": {
        "source": "data/silver/olist/olist_orders_dataset",
        "experimental": (
            "experiments/output/olist/null_customer_id/"
            "olist_null_customer_id_001"
        ),
    },
    "olist_duplicate_order_id_002": {
        "source": "data/silver/olist/olist_orders_dataset",
        "experimental": (
            "experiments/output/olist/duplicate_order_id/"
            "olist_duplicate_order_id_002"
        ),
    },
    "olist_negative_price_003": {
        "source": "data/silver/olist/olist_order_items_dataset",
        "experimental": (
            "experiments/output/olist/negative_price/"
            "olist_negative_price_003"
        ),
    },
    "olist_invalid_customer_ref_004": {
        "source": "data/silver/olist/olist_orders_dataset",
        "experimental": (
            "experiments/output/olist/invalid_customer_ref/"
            "olist_invalid_customer_ref_004"
        ),
    },
    "olist_schema_drift_005": {
        "source": "data/silver/olist/olist_orders_dataset",
        "experimental": (
            "experiments/output/olist/schema_drift/"
            "olist_schema_drift_005"
        ),
    },
}


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Schema-Snapshot-Generation")
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


def extract_schema(df):
    return {
        field.name: field.dataType.simpleString()
        for field in df.schema.fields
    }


def compare_schemas(baseline, experimental):
    added = sorted(
        set(experimental) - set(baseline)
    )

    removed = sorted(
        set(baseline) - set(experimental)
    )

    type_changes = [
        {
            "column": column,
            "original_type": baseline[column],
            "modified_type": experimental[column],
        }
        for column in sorted(
            set(baseline) & set(experimental)
        )
        if baseline[column] != experimental[column]
    ]

    return added, removed, type_changes


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    spark = create_spark()

    try:
        print("=" * 65)
        print("EXPERIMENTAL SCHEMA SNAPSHOT GENERATION")
        print("=" * 65)

        for experiment_id, paths in EXPERIMENTS.items():
            source_path = ROOT / paths["source"]
            experimental_path = ROOT / paths["experimental"]

            if not source_path.exists():
                raise FileNotFoundError(source_path)

            if not experimental_path.exists():
                raise FileNotFoundError(experimental_path)

            baseline_df = (
                spark.read.format("delta")
                .load(str(source_path))
            )

            experimental_df = (
                spark.read.format("delta")
                .load(str(experimental_path))
            )

            baseline_schema = extract_schema(
                baseline_df
            )

            experimental_schema = extract_schema(
                experimental_df
            )

            added, removed, type_changes = compare_schemas(
                baseline_schema,
                experimental_schema,
            )

            drift_detected = bool(
                added or removed or type_changes
            )

            report = {
                "experiment_id": experiment_id,
                "dataset": "olist",
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "baseline_path": paths["source"],
                "experimental_path": paths["experimental"],
                "check_type": "schema_drift",
                "baseline_schema": baseline_schema,
                "experimental_schema": experimental_schema,
                "added_columns": added,
                "removed_columns": removed,
                "type_changes": type_changes,
                "drift_detected": drift_detected,
                "status": (
                    "FAIL"
                    if drift_detected
                    else "PASS"
                ),
            }

            output_path = (
                OUTPUT_DIR
                / f"{experiment_id}_schema.json"
            )

            # Preserve the existing Experiment 005 report,
            # which contains the actual downstream failure test.
            if output_path.exists():
                existing = json.loads(
                    output_path.read_text(
                        encoding="utf-8"
                    )
                )

                if (
                    existing.get("experiment_id")
                    != experiment_id
                ):
                    raise AssertionError(
                        "Existing report ID mismatch"
                    )

                # Retain previously observed downstream
                # compatibility test results.
                for key in (
                    "downstream_failure_detected",
                    "downstream_error",
                ):
                    if key in existing:
                        report[key] = existing[key]

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    report,
                    file,
                    indent=2,
                )

            print()
            print("Experiment:", experiment_id)
            print("Added:", added)
            print("Removed:", removed)
            print("Type changes:", type_changes)
            print("Drift detected:", drift_detected)
            print("Saved:", output_path)

        print()
        print("=" * 65)
        print("SCHEMA SNAPSHOTS GENERATED: 5")
        print("STATUS: COMPLETE")
        print("=" * 65)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
