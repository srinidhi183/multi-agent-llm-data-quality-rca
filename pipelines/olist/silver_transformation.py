
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

ROOT = Path(__file__).resolve().parents[2]

BRONZE_DIR = ROOT / "data/bronze/olist"
SILVER_DIR = ROOT / "data/silver/olist"
LOG_DIR = ROOT / "evidence/logs"
DQ_DIR = ROOT / "evidence/dq_results"
SCHEMA_DIR = ROOT / "evidence/schemas"

for directory in [SILVER_DIR, LOG_DIR, DQ_DIR, SCHEMA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

builder = (
    SparkSession.builder
    .appName("Olist-Silver-Transformation")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
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

# Columns that must not be NULL
REQUIRED_COLUMNS = {
    "olist_customers_dataset": ["customer_id"],
    "olist_orders_dataset": ["order_id", "customer_id"],
    "olist_order_items_dataset": ["order_id", "order_item_id"],
    "olist_order_payments_dataset": ["order_id", "payment_sequential"],
    "olist_order_reviews_dataset": ["review_id", "order_id"],
    "olist_products_dataset": ["product_id"],
    "olist_sellers_dataset": ["seller_id"],
    "olist_geolocation_dataset": ["geolocation_zip_code_prefix"],
    "product_category_name_translation": ["product_category_name"]
}

def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, default=str),
        encoding="utf-8"
    )

def standardize_columns(df):
    for old_name in df.columns:
        new_name = re.sub(
            r"[^a-zA-Z0-9_]+", "_", old_name.strip().lower()
        )
        if new_name != old_name:
            df = df.withColumnRenamed(old_name, new_name)
    return df

def clean_strings(df):
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            name = field.name

            # Trim whitespace without changing NULL values
            df = df.withColumn(
                name,
                F.trim(F.col(name))
            )
    return df

results = []
dq_results = {}
schema_snapshots = {}
started_at = datetime.now(timezone.utc).isoformat()

try:
    tables = sorted(
        p for p in BRONZE_DIR.iterdir()
        if p.is_dir() and (p / "_delta_log").exists()
    )

    if len(tables) != 9:
        raise RuntimeError(
            f"Expected 9 Bronze tables, found {len(tables)}"
        )

    for table_path in tables:
        name = table_path.name
        print(f"\nProcessing Silver table: {name}")

        try:
            df = spark.read.format("delta").load(str(table_path))
            bronze_count = df.count()

            df = standardize_columns(df)
            df = clean_strings(df)

            # Preserve all source records in Silver.
            # Record data-quality violations separately.
            df = df.withColumn(
                "_silver_processed_at",
                F.current_timestamp()
            )

            required = REQUIRED_COLUMNS[name]
            missing_columns = [
                c for c in required if c not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            null_exprs = [
                F.sum(
                    F.when(F.col(c).isNull(), 1).otherwise(0)
                ).alias(c)
                for c in required
            ]

            required_nulls = (
                df.select(*null_exprs).first().asDict()
            )

            output_path = SILVER_DIR / name

            (
                df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(str(output_path))
            )

            saved = spark.read.format("delta").load(
                str(output_path)
            )
            silver_count = saved.count()

            if bronze_count != silver_count:
                raise RuntimeError(
                    f"Row mismatch: Bronze={bronze_count}, "
                    f"Silver={silver_count}"
                )

            dq_results[name] = {
                "required_column_nulls": required_nulls,
                "row_count": silver_count,
                "row_count_match": True
            }

            schema_snapshots[name] = {
                "columns": [
                    {
                        "name": f.name,
                        "type": f.dataType.simpleString(),
                        "nullable": f.nullable
                    }
                    for f in saved.schema.fields
                ]
            }

            results.append({
                "table": name,
                "status": "SUCCESS",
                "bronze_rows": bronze_count,
                "silver_rows": silver_count
            })

            print(f"SUCCESS: {silver_count} rows")

        except Exception as exc:
            results.append({
                "table": name,
                "status": "FAILED",
                "error": str(exc)
            })
            print(f"FAILED: {name}: {exc}")
            raise

finally:
    save_json(
        LOG_DIR / "olist_silver_transformation.json",
        {
            "pipeline": "olist_silver_transformation",
            "started_at": started_at,
            "finished_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "results": results
        }
    )

    save_json(
        DQ_DIR / "olist_silver_dq.json",
        dq_results
    )

    save_json(
        SCHEMA_DIR / "olist_silver_schema.json",
        schema_snapshots
    )

    spark.stop()

print("\nSILVER TRANSFORMATION COMPLETED")
print(f"Tables processed: {len(results)}")
