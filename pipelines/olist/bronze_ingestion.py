
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, input_file_name, lit

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data/raw/olist"
BRONZE_DIR = ROOT / "data/bronze/olist"
LOG_DIR = ROOT / "evidence/logs"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

builder = (
    SparkSession.builder
    .appName("Olist-Bronze-Ingestion")
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

results = []
started_at = datetime.now(timezone.utc).isoformat()

try:
    files = sorted(RAW_DIR.glob("*.csv"))

    if len(files) != 9:
        raise RuntimeError(
            f"Expected 9 CSV files, found {len(files)}"
        )

    for file in files:
        table_name = file.stem
        output_path = BRONZE_DIR / table_name

        print(f"\nIngesting: {table_name}")

        try:
            df = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("multiLine", True)
                .option("quote", '"')
                .option("escape", '"')
                .option("mode", "FAILFAST")
                .csv(str(file))
            )

            # Preserve source and ingestion metadata
            df = (
                df.withColumn(
                    "_source_file",
                    input_file_name()
                )
                .withColumn(
                    "_ingested_at",
                    current_timestamp()
                )
                .withColumn(
                    "_dataset",
                    lit(table_name)
                )
            )

            source_count = df.count()

            (
                df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(str(output_path))
            )

            # Validate stored data
            saved_df = (
                spark.read
                .format("delta")
                .load(str(output_path))
            )

            saved_count = saved_df.count()

            if source_count != saved_count:
                raise RuntimeError(
                    f"Row count mismatch: "
                    f"{source_count} != {saved_count}"
                )

            results.append({
                "table": table_name,
                "status": "SUCCESS",
                "source_rows": source_count,
                "bronze_rows": saved_count,
                "path": str(output_path)
            })

            print(f"SUCCESS: {saved_count} rows")

        except Exception as exc:
            results.append({
                "table": table_name,
                "status": "FAILED",
                "error": str(exc)
            })
            print(f"FAILED: {table_name}: {exc}")
            raise

finally:
    log = {
        "pipeline": "olist_bronze_ingestion",
        "started_at": started_at,
        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results
    }

    log_file = (
        LOG_DIR / "olist_bronze_ingestion.json"
    )

    log_file.write_text(
        json.dumps(log, indent=2),
        encoding="utf-8"
    )

    spark.stop()

print("\nBRONZE INGESTION COMPLETED")
print(f"Tables created: {len(results)}")
