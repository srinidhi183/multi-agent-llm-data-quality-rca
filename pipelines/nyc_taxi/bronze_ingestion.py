
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/nyc_taxi"
BRONZE = ROOT / "data/bronze/nyc_taxi"
LOGS = ROOT / "evidence/logs"
SCHEMAS = ROOT / "evidence/schemas"

for directory in [BRONZE, LOGS, SCHEMAS]:
    directory.mkdir(parents=True, exist_ok=True)

OUTPUT = BRONZE / "yellow_taxi_trips"


# ============================================================
# 2. SPARK SESSION
# ============================================================

builder = (
    SparkSession.builder
    .appName("NYC-Taxi-Bronze-Ingestion")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")


# ============================================================
# 3. CONFIGURATION
# ============================================================

MONTHS = [
    "01", "02", "03", "04",
    "05", "06", "07", "08"
]

results = []
schemas = {}
column_mappings = {}

started_at = datetime.now(timezone.utc).isoformat()
pipeline_status = "SUCCESS"


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def save_json(path, data):
    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def standardize_columns(df):
    original = df.columns
    cleaned = []
    mappings = []

    for name in original:
        new_name = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            name.strip().lower()
        ).strip("_")

        if not new_name:
            raise ValueError(
                f"Invalid source column: {name!r}"
            )

        cleaned.append(new_name)

        mappings.append({
            "original_name": name,
            "standardized_name": new_name
        })

    if len(cleaned) != len(set(cleaned)):
        raise ValueError(
            f"Duplicate standardized columns: {cleaned}"
        )

    return df.toDF(*cleaned), mappings


def get_schema(df):
    return [
        {
            "name": field.name,
            "type": field.dataType.simpleString(),
            "nullable": field.nullable
        }
        for field in df.schema.fields
    ]


# ============================================================
# 5. MONTHLY INGESTION
# ============================================================

try:

    # Validate all files before modifying the Bronze table.
    for month in MONTHS:
        filename = f"yellow_tripdata_2026-{month}.parquet"
        source = RAW / filename

        if not source.is_file():
            raise FileNotFoundError(
                f"Missing monthly file: {source}"
            )

    for index, month in enumerate(MONTHS):

        filename = f"yellow_tripdata_2026-{month}.parquet"
        source = RAW / filename

        print("\n" + "=" * 60)
        print(f"PROCESSING NYC TAXI: 2026-{month}")
        print("=" * 60)

        try:

            # ------------------------------------------------
            # A. READ PARQUET METADATA
            # ------------------------------------------------

            metadata = pq.read_metadata(source)
            expected_rows = metadata.num_rows

            print(f"Expected rows: {expected_rows}")

            # ------------------------------------------------
            # B. READ MONTHLY DATA
            # ------------------------------------------------

            df = spark.read.parquet(str(source))

            # ------------------------------------------------
            # C. STANDARDIZE COLUMN NAMES
            # ------------------------------------------------

            df, mappings = standardize_columns(df)

            column_mappings[month] = mappings

            # Record schema before adding ingestion metadata.
            schemas[month] = {
                "source_file": filename,
                "expected_rows": expected_rows,
                "columns": get_schema(df)
            }

            # ------------------------------------------------
            # D. ADD BRONZE METADATA
            # ------------------------------------------------

            df = (
                df
                .withColumn(
                    "_source_file",
                    F.lit(filename)
                )
                .withColumn(
                    "_ingested_at",
                    F.current_timestamp()
                )
                .withColumn(
                    "_dataset",
                    F.lit("nyc_yellow_taxi")
                )
                .withColumn(
                    "_source_month",
                    F.lit(f"2026-{month}")
                )
            )

            # ------------------------------------------------
            # E. WRITE DELTA TABLE
            # ------------------------------------------------

            # First month replaces an earlier run.
            # Subsequent months append to the same table.

            write_mode = (
                "overwrite" if index == 0 else "append"
            )

            writer = (
                df.write
                .format("delta")
                .mode(write_mode)
                .option("mergeSchema", "true")
            )

            if index == 0:
                writer = writer.option(
                    "overwriteSchema", "true"
                )

            writer.save(str(OUTPUT))

            # ------------------------------------------------
            # F. VERIFY MONTHLY ROW COUNT
            # ------------------------------------------------

            saved_df = (
                spark.read
                .format("delta")
                .load(str(OUTPUT))
                .filter(
                    F.col("_source_month")
                    == f"2026-{month}"
                )
            )

            saved_count = saved_df.count()

            if saved_count != expected_rows:
                raise RuntimeError(
                    f"Row count mismatch for {month}: "
                    f"expected={expected_rows}, "
                    f"saved={saved_count}"
                )

            results.append({
                "month": f"2026-{month}",
                "source_file": filename,
                "status": "SUCCESS",
                "source_rows": expected_rows,
                "bronze_rows": saved_count,
                "output_path": str(OUTPUT)
            })

            print(
                f"SUCCESS: 2026-{month} "
                f"({saved_count} rows)"
            )

        except Exception as exc:

            pipeline_status = "FAILED"

            results.append({
                "month": f"2026-{month}",
                "source_file": filename,
                "status": "FAILED",
                "error": str(exc)
            })

            print(f"FAILED: 2026-{month}")
            print(f"ERROR: {exc}")

            raise

    # --------------------------------------------------------
    # G. FINAL VALIDATION
    # --------------------------------------------------------

    expected_total = sum(
        r["source_rows"] for r in results
    )

    final_count = (
        spark.read
        .format("delta")
        .load(str(OUTPUT))
        .count()
    )

    if final_count != expected_total:
        pipeline_status = "FAILED"
        raise RuntimeError(
            f"Final count mismatch: "
            f"expected={expected_total}, "
            f"actual={final_count}"
        )

    print("\n" + "=" * 60)
    print("NYC TAXI BRONZE INGESTION COMPLETED")
    print("=" * 60)

    print(f"Months processed: {len(results)}")
    print(f"Total Bronze rows: {final_count}")

except Exception:
    pipeline_status = "FAILED"
    raise

finally:

    finished_at = datetime.now(
        timezone.utc
    ).isoformat()

    save_json(
        LOGS / "nyc_taxi_bronze_ingestion.json",
        {
            "pipeline": "nyc_taxi_bronze_ingestion",
            "status": pipeline_status,
            "started_at": started_at,
            "finished_at": finished_at,
            "results": results
        }
    )

    save_json(
        SCHEMAS / "nyc_taxi_bronze_schema.json",
        schemas
    )

    save_json(
        SCHEMAS / "nyc_taxi_column_mappings.json",
        column_mappings
    )

    spark.stop()
