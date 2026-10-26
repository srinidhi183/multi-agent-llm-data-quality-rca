
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BRONZE = ROOT / "data/bronze/nyc_taxi"
SILVER = ROOT / "data/silver/nyc_taxi"

LOGS = ROOT / "evidence/logs"
DQ = ROOT / "evidence/dq_results"
SCHEMAS = ROOT / "evidence/schemas"

for directory in [SILVER, LOGS, DQ, SCHEMAS]:
    directory.mkdir(parents=True, exist_ok=True)

SOURCE = BRONZE / "yellow_taxi_trips"
OUTPUT = SILVER / "yellow_taxi_trips"

MONTHS = [
    "01", "02", "03", "04",
    "05", "06", "07", "08"
]


# ============================================================
# 2. SPARK SESSION
# ============================================================

builder = (
    SparkSession.builder
    .appName("NYC-Taxi-Silver-Transformation")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config(
        "spark.sql.extensions",
        "io.delta.sql.DeltaSparkSessionExtension"
    )
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog"
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 3. CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "passenger_count",
    "_source_month"
]

results = []
dq_results = {}
schemas = {}

started_at = datetime.now(timezone.utc).isoformat()
pipeline_status = "SUCCESS"


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def check_columns(df):
    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def capture_schema(df):
    return [
        {
            "name": field.name,
            "type": field.dataType.simpleString(),
            "nullable": field.nullable
        }
        for field in df.schema.fields
    ]


# ============================================================
# 5. READ BRONZE
# ============================================================

try:

    bronze = spark.read.format("delta").load(
        str(SOURCE)
    )

    check_columns(bronze)

    # --------------------------------------------------------
    # PROCESS ONE MONTH AT A TIME
    # --------------------------------------------------------

    for index, month in enumerate(MONTHS):

        source_month = f"2026-{month}"

        print("\n" + "=" * 60)
        print(f"PROCESSING SILVER: {source_month}")
        print("=" * 60)

        try:

            df = bronze.filter(
                F.col("_source_month") == source_month
            )

            # -----------------------------------------------
            # A. DEFINE DATA QUALITY CONDITIONS
            # -----------------------------------------------

            invalid_pickup = (
                F.col("tpep_pickup_datetime").isNull()
            )

            invalid_dropoff = (
                F.col("tpep_dropoff_datetime").isNull()
            )

            invalid_time_order = (
                F.col("tpep_pickup_datetime").isNotNull()
                & F.col("tpep_dropoff_datetime").isNotNull()
                & (
                    F.col("tpep_dropoff_datetime")
                    < F.col("tpep_pickup_datetime")
                )
            )

            invalid_distance = (
                F.col("trip_distance").isNull()
                | F.isnan("trip_distance")
                | (F.col("trip_distance") < 0)
            )

            invalid_fare = (
                F.col("fare_amount").isNull()
                | F.isnan("fare_amount")
                | (F.col("fare_amount") < 0)
            )

            invalid_total = (
                F.col("total_amount").isNull()
                | F.isnan("total_amount")
                | (F.col("total_amount") < 0)
            )

            invalid_passenger = (
                F.col("passenger_count").isNull()
                | (F.col("passenger_count") <= 0)
            )

            # -----------------------------------------------
            # B. CREATE DQ ISSUE FLAGS
            # -----------------------------------------------

            # concat_ws skips NULL values, so a valid
            # record receives an empty issue string.

            df = df.withColumn(
                "_dq_issues",
                F.concat_ws(
                    "|",
                    F.when(
                        invalid_pickup,
                        F.lit("NULL_PICKUP_TIME")
                    ),
                    F.when(
                        invalid_dropoff,
                        F.lit("NULL_DROPOFF_TIME")
                    ),
                    F.when(
                        invalid_time_order,
                        F.lit("INVALID_TIME_ORDER")
                    ),
                    F.when(
                        invalid_distance,
                        F.lit("INVALID_DISTANCE")
                    ),
                    F.when(
                        invalid_fare,
                        F.lit("INVALID_FARE")
                    ),
                    F.when(
                        invalid_total,
                        F.lit("INVALID_TOTAL_AMOUNT")
                    ),
                    F.when(
                        invalid_passenger,
                        F.lit("INVALID_PASSENGER_COUNT")
                    )
                )
            )

            df = df.withColumn(
                "_dq_status",
                F.when(
                    F.length(F.col("_dq_issues")) == 0,
                    F.lit("PASS")
                ).otherwise(F.lit("FAIL"))
            )

            # -----------------------------------------------
            # C. ADD DERIVED FIELDS
            # -----------------------------------------------

            df = df.withColumn(
                "trip_duration_minutes",
                (
                    F.unix_timestamp(
                        "tpep_dropoff_datetime"
                    )
                    -
                    F.unix_timestamp(
                        "tpep_pickup_datetime"
                    )
                ) / F.lit(60.0)
            )

            df = df.withColumn(
                "pickup_date",
                F.to_date("tpep_pickup_datetime")
            )

            df = df.withColumn(
                "_silver_processed_at",
                F.current_timestamp()
            )

            # -----------------------------------------------
            # D. COLLECT MONTHLY DQ METRICS
            # -----------------------------------------------

            metrics = (
                df.agg(
                    F.count("*").alias("total_rows"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "PASS",
                            1
                        ).otherwise(0)
                    ).alias("passed_rows"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "FAIL",
                            1
                        ).otherwise(0)
                    ).alias("failed_rows"),

                    F.sum(
                        F.when(invalid_pickup, 1)
                        .otherwise(0)
                    ).alias("null_pickup_time"),

                    F.sum(
                        F.when(invalid_dropoff, 1)
                        .otherwise(0)
                    ).alias("null_dropoff_time"),

                    F.sum(
                        F.when(invalid_time_order, 1)
                        .otherwise(0)
                    ).alias("invalid_time_order"),

                    F.sum(
                        F.when(invalid_distance, 1)
                        .otherwise(0)
                    ).alias("invalid_distance"),

                    F.sum(
                        F.when(invalid_fare, 1)
                        .otherwise(0)
                    ).alias("invalid_fare"),

                    F.sum(
                        F.when(invalid_total, 1)
                        .otherwise(0)
                    ).alias("invalid_total_amount"),

                    F.sum(
                        F.when(invalid_passenger, 1)
                        .otherwise(0)
                    ).alias("invalid_passenger_count")
                )
                .first()
                .asDict()
            )

            dq_results[source_month] = metrics

            print(
                f"Source rows: {metrics['total_rows']}"
            )
            print(
                f"DQ passed: {metrics['passed_rows']}"
            )
            print(
                f"DQ failed: {metrics['failed_rows']}"
            )

            # -----------------------------------------------
            # E. SAVE SILVER DELTA
            # -----------------------------------------------

            writer = (
                df.write
                .format("delta")
                .mode(
                    "overwrite" if index == 0
                    else "append"
                )
            )

            if index == 0:
                writer = writer.option(
                    "overwriteSchema", "true"
                )

            writer.save(str(OUTPUT))

            # -----------------------------------------------
            # F. VERIFY ROW COUNTS
            # -----------------------------------------------

            saved = (
                spark.read
                .format("delta")
                .load(str(OUTPUT))
                .filter(
                    F.col("_source_month")
                    == source_month
                )
            )

            saved_count = saved.count()

            if saved_count != metrics["total_rows"]:
                raise RuntimeError(
                    f"Row mismatch for {source_month}: "
                    f"{metrics['total_rows']} != "
                    f"{saved_count}"
                )

            schemas[source_month] = {
                "columns": capture_schema(df)
            }

            results.append({
                "month": source_month,
                "status": "SUCCESS",
                "bronze_rows": metrics["total_rows"],
                "silver_rows": saved_count,
                "dq_passed": metrics["passed_rows"],
                "dq_failed": metrics["failed_rows"]
            })

            print(
                f"SUCCESS: {source_month} "
                f"({saved_count} rows)"
            )

        except Exception as exc:

            pipeline_status = "FAILED"

            results.append({
                "month": source_month,
                "status": "FAILED",
                "error": str(exc)
            })

            print(f"FAILED: {source_month}")
            print(f"ERROR: {exc}")

            raise

    # --------------------------------------------------------
    # G. FINAL VALIDATION
    # --------------------------------------------------------

    expected_total = sum(
        r["bronze_rows"] for r in results
    )

    final_count = (
        spark.read
        .format("delta")
        .load(str(OUTPUT))
        .count()
    )

    if final_count != expected_total:
        raise RuntimeError(
            f"Final row mismatch: "
            f"{expected_total} != {final_count}"
        )

    print("\n" + "=" * 60)
    print("NYC TAXI SILVER TRANSFORMATION COMPLETED")
    print("=" * 60)

    print(f"Total Silver rows: {final_count}")
    print(f"Months processed: {len(results)}")

except Exception:
    pipeline_status = "FAILED"
    raise

finally:

    save_json(
        LOGS / "nyc_taxi_silver_transformation.json",
        {
            "pipeline": "nyc_taxi_silver_transformation",
            "status": pipeline_status,
            "started_at": started_at,
            "finished_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "results": results
        }
    )

    save_json(
        DQ / "nyc_taxi_silver_dq.json",
        dq_results
    )

    save_json(
        SCHEMAS / "nyc_taxi_silver_schema.json",
        schemas
    )

    spark.stop()
