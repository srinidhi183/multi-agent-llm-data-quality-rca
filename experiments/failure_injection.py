
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession, Window, functions as F


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data/silver/olist/olist_orders_dataset"
OUTPUT_DIR = ROOT / "experiments/output/olist"
GROUND_TRUTH_DIR = ROOT / "experiments/ground_truth"

SCENARIOS = {
    "null_customer_id": {
        "experiment_id": "olist_null_customer_id_001",
        "affected_column": "customer_id",
        "check_type": "null_percentage",
    },
    "duplicate_order_id": {
        "experiment_id": "olist_duplicate_order_id_002",
        "affected_column": "order_id",
        "check_type": "duplicate_percentage",
    },
}


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Failure-Injection")
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


def select_rows(df, rows):
    window = Window.orderBy("order_id")

    return (
        df.withColumn("_selection_rank", F.row_number().over(window))
        .filter(F.col("_selection_rank") <= rows)
        .drop("_selection_rank")
    )


def inject_null_customer_ids(df, rows):
    selected = (
        select_rows(
            df.filter(F.col("customer_id").isNotNull()),
            rows,
        )
        .select("order_id")
        .withColumn("_inject", F.lit(True))
    )

    return (
        df.join(selected, "order_id", "left")
        .withColumn(
            "customer_id",
            F.when(
                F.col("_inject") == True,
                F.lit(None).cast("string"),
            ).otherwise(F.col("customer_id")),
        )
        .drop("_inject")
    )


def inject_duplicate_order_ids(df, rows):
    selected = select_rows(df, rows)
    return df.unionByName(selected)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scenario",
        required=True,
        choices=list(SCENARIOS),
    )
    parser.add_argument("--rows", type=int, default=100)

    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than zero")

    config = SCENARIOS[args.scenario]
    experiment_id = config["experiment_id"]

    output_path = (
        OUTPUT_DIR / args.scenario / experiment_id
    )
    ground_truth_path = (
        GROUND_TRUTH_DIR / f"{experiment_id}.json"
    )

    if output_path.exists() or ground_truth_path.exists():
        raise FileExistsError(
            f"Experiment already exists: {experiment_id}. "
            "Existing results will not be overwritten."
        )

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    spark = create_spark()

    try:
        original = spark.read.format("delta").load(
            str(SOURCE_PATH)
        )

        original_rows = original.count()

        if args.rows > original_rows:
            raise ValueError(
                "Requested injection exceeds source row count"
            )

        if args.scenario == "null_customer_id":
            original_failures = original.filter(
                F.col("customer_id").isNull()
            ).count()

            eligible_rows = original.filter(
                F.col("customer_id").isNotNull()
            ).count()

            if args.rows > eligible_rows:
                raise ValueError(
                    "Not enough eligible customer IDs"
                )

            modified = inject_null_customer_ids(
                original, args.rows
            )

        else:
            original_failures = (
                original_rows
                - original.select("order_id").distinct().count()
            )

            modified = inject_duplicate_order_ids(
                original, args.rows
            )

        (
            modified.write.format("delta")
            .mode("errorifexists")
            .save(str(output_path))
        )

        result = spark.read.format("delta").load(
            str(output_path)
        )

        modified_rows = result.count()

        if args.scenario == "null_customer_id":
            observed_failures = result.filter(
                F.col("customer_id").isNull()
            ).count()

            expected_rows = original_rows

        else:
            observed_failures = (
                modified_rows
                - result.select("order_id").distinct().count()
            )

            expected_rows = original_rows + args.rows

        observed_increase = (
            observed_failures - original_failures
        )

        if modified_rows != expected_rows:
            raise AssertionError(
                "Unexpected experimental row count"
            )

        if observed_increase != args.rows:
            raise AssertionError(
                "Injected failure count does not match target"
            )

        ground_truth = {
            "experiment_id": experiment_id,
            "dataset": "olist",
            "scenario": args.scenario,
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "source_table": "olist_orders_dataset",
            "source_path": str(
                SOURCE_PATH.relative_to(ROOT)
            ),
            "output_path": str(
                output_path.relative_to(ROOT)
            ),
            "affected_column": config["affected_column"],
            "injected_rows": args.rows,
            "original_rows": original_rows,
            "modified_rows": modified_rows,
            "original_failure_count": original_failures,
            "modified_failure_count": observed_failures,
            "observed_failure_increase": observed_increase,
            "expected_dq_failure": {
                "type": config["check_type"],
                "column": config["affected_column"],
                "columns": [config["affected_column"]],
                "max_null_pct": 0,
                "max_duplicate_pct": 0,
            },
            "root_cause": (
                "Controlled experiment introduced "
                f"{args.rows} instances of "
                f"{args.scenario}."
            ),
        }

        with ground_truth_path.open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(ground_truth, file, indent=2)

        print("=" * 55)
        print("CONTROLLED FAILURE INJECTION")
        print("=" * 55)
        print("Experiment:", experiment_id)
        print("Original rows:", original_rows)
        print("Modified rows:", modified_rows)
        print("Original failures:", original_failures)
        print("Modified failures:", observed_failures)
        print("Injected failures:", observed_increase)
        print("Status: VERIFIED")
        print("Output:", output_path)
        print("Ground truth:", ground_truth_path)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

