
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

SILVER = ROOT / "data/silver/nyc_taxi"
GOLD = ROOT / "data/gold/nyc_taxi"
LOGS = ROOT / "evidence/logs"

GOLD.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

SOURCE = SILVER / "yellow_taxi_trips"

MONTHS = [
    "01", "02", "03", "04",
    "05", "06", "07", "08"
]


# ============================================================
# 2. SPARK SESSION
# ============================================================

builder = (
    SparkSession.builder
    .appName("NYC-Taxi-Gold-Transformation")
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

results = []
started_at = datetime.now(timezone.utc).isoformat()
pipeline_status = "SUCCESS"

REQUIRED_COLUMNS = [
    "_source_month",
    "_dq_status",
    "pickup_date",
    "pulocationid",
    "trip_distance",
    "total_amount",
    "fare_amount",
    "tip_amount"
]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )


def save_gold(df, name, first_month):
    output = GOLD / name

    writer = (
        df.write
        .format("delta")
        .mode("overwrite" if first_month else "append")
    )

    if first_month:
        writer = writer.option(
            "overwriteSchema", "true"
        )

    writer.save(str(output))


def verify_columns(df):
    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing Silver columns: {missing}"
        )


# ============================================================
# 5. GOLD TRANSFORMATIONS
# ============================================================

try:

    silver = spark.read.format("delta").load(
        str(SOURCE)
    )

    verify_columns(silver)

    for index, month in enumerate(MONTHS):

        source_month = f"2026-{month}"
        first_month = index == 0

        print("\n" + "=" * 60)
        print(f"PROCESSING GOLD: {source_month}")
        print("=" * 60)

        try:

            monthly = silver.filter(
                F.col("_source_month") == source_month
            )

            # ------------------------------------------------
            # GOLD 1: MONTHLY SUMMARY
            # ------------------------------------------------

            # All records contribute to the DQ counts.
            # Only passing records contribute to financial
            # and distance measures.

            monthly_summary = (
                monthly
                .groupBy("_source_month")
                .agg(
                    F.count("*").alias("total_records"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "PASS", 1
                        ).otherwise(0)
                    ).alias("valid_trips"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "FAIL", 1
                        ).otherwise(0)
                    ).alias("flagged_trips"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "PASS",
                            F.col("total_amount")
                        ).otherwise(0.0)
                    ).alias("total_revenue_valid"),

                    F.sum(
                        F.when(
                            F.col("_dq_status") == "PASS",
                            F.col("trip_distance")
                        ).otherwise(0.0)
                    ).alias("total_distance_valid")
                )
                .withColumn(
                    "dq_pass_rate",
                    F.col("valid_trips")
                    / F.col("total_records")
                )
            )

            save_gold(
                monthly_summary,
                "gold_monthly_summary",
                first_month
            )

            # ------------------------------------------------
            # FILTER VALID TRIPS FOR BUSINESS ANALYTICS
            # ------------------------------------------------

            valid = monthly.filter(
                F.col("_dq_status") == "PASS"
            )

            # ------------------------------------------------
            # GOLD 2: DAILY TRIPS
            # ------------------------------------------------

            daily_trips = (
                valid
                .groupBy(
                    "_source_month",
                    "pickup_date"
                )
                .agg(
                    F.count("*").alias("trip_count"),

                    F.sum("total_amount").alias(
                        "total_revenue"
                    ),

                    F.sum("fare_amount").alias(
                        "total_fares"
                    ),

                    F.sum("tip_amount").alias(
                        "total_tips"
                    ),

                    F.avg("trip_distance").alias(
                        "avg_trip_distance"
                    ),

                    F.avg("total_amount").alias(
                        "avg_trip_revenue"
                    )
                )
            )

            save_gold(
                daily_trips,
                "gold_daily_trips",
                first_month
            )

            # ------------------------------------------------
            # GOLD 3: PICKUP LOCATION SUMMARY
            # ------------------------------------------------

            pickup_locations = (
                valid
                .groupBy(
                    "_source_month",
                    "pulocationid"
                )
                .agg(
                    F.count("*").alias("trip_count"),

                    F.sum("total_amount").alias(
                        "total_revenue"
                    ),

                    F.avg("trip_distance").alias(
                        "avg_trip_distance"
                    ),

                    F.avg("fare_amount").alias(
                        "avg_fare"
                    )
                )
            )

            save_gold(
                pickup_locations,
                "gold_pickup_locations",
                first_month
            )

            # ------------------------------------------------
            # VERIFY MONTHLY SUMMARY
            # ------------------------------------------------

            saved_summary = (
                spark.read
                .format("delta")
                .load(
                    str(GOLD / "gold_monthly_summary")
                )
                .filter(
                    F.col("_source_month") == source_month
                )
                .first()
            )

            if saved_summary is None:
                raise RuntimeError(
                    f"Missing Gold summary: {source_month}"
                )

            total = saved_summary["total_records"]
            passed = saved_summary["valid_trips"]
            flagged = saved_summary["flagged_trips"]

            if total != passed + flagged:
                raise RuntimeError(
                    f"DQ reconciliation failed: {source_month}"
                )

            results.append({
                "month": source_month,
                "status": "SUCCESS",
                "total_records": total,
                "valid_trips": passed,
                "flagged_trips": flagged
            })

            print(f"SUCCESS: {source_month}")
            print(f"Total records: {total}")
            print(f"Valid trips: {passed}")
            print(f"Flagged trips: {flagged}")

        except Exception as exc:

            pipeline_status = "FAILED"

            results.append({
                "month": source_month,
                "status": "FAILED",
                "error": str(exc)
            })

            raise

    # --------------------------------------------------------
    # FINAL GOLD VALIDATION
    # --------------------------------------------------------

    expected_total = sum(
        r["total_records"] for r in results
    )

    gold_total = (
        spark.read
        .format("delta")
        .load(str(GOLD / "gold_monthly_summary"))
        .agg(F.sum("total_records"))
        .first()[0]
    )

    if expected_total != gold_total:
        raise RuntimeError(
            f"Gold total mismatch: "
            f"{expected_total} != {gold_total}"
        )

    print("\n" + "=" * 60)
    print("NYC TAXI GOLD TRANSFORMATION COMPLETED")
    print("=" * 60)
    print(f"Total source records: {gold_total}")

except Exception:
    pipeline_status = "FAILED"
    raise

finally:

    save_json(
        LOGS / "nyc_taxi_gold_transformation.json",
        {
            "pipeline": "nyc_taxi_gold_transformation",
            "status": pipeline_status,
            "started_at": started_at,
            "finished_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "results": results
        }
    )

    spark.stop()
