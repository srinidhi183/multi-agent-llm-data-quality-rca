
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

RAW = ROOT / "data/raw/dataco"
BRONZE = ROOT / "data/bronze/dataco"
LOGS = ROOT / "evidence/logs"
SCHEMAS = ROOT / "evidence/schemas"

for directory in [BRONZE, LOGS, SCHEMAS]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SPARK SESSION
# ============================================================

builder = (
    SparkSession.builder
    .appName("DataCo-Bronze-Ingestion")
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


# ============================================================
# 3. DATASET CONFIGURATION
# ============================================================

DATASETS = {
    "dataco_supply_chain": "DataCoSupplyChainDataset.csv",
    "dataco_access_logs": "tokenized_access_logs.csv"
}


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def save_json(path, content):
    path.write_text(
        json.dumps(
            content,
            indent=2,
            default=str,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def standardize_columns(df):
    """
    Convert column names to Delta-compatible snake_case.
    Return the DataFrame and original-to-new name mappings.
    """

    original_columns = df.columns
    renamed_columns = []
    mappings = []

    for old_name in original_columns:
        new_name = re.sub(
            r"[^a-zA-Z0-9_]+",
            "_",
            old_name.strip().lower()
        ).strip("_")

        if not new_name:
            raise ValueError(
                f"Invalid column name: {old_name!r}"
            )

        renamed_columns.append(new_name)

        mappings.append({
            "original_name": old_name,
            "standardized_name": new_name
        })

    # Detect collisions before renaming
    if len(renamed_columns) != len(set(renamed_columns)):
        duplicates = sorted({
            name
            for name in renamed_columns
            if renamed_columns.count(name) > 1
        })

        raise ValueError(
            "Duplicate column names after standardization: "
            f"{duplicates}"
        )

    # Rename all columns together
    df = df.toDF(*renamed_columns)

    return df, mappings


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
# 5. INGESTION EXECUTION
# ============================================================

results = []
schemas = {}
column_mappings = {}

started_at = datetime.now(timezone.utc).isoformat()

pipeline_status = "SUCCESS"

try:

    for table_name, filename in DATASETS.items():

        source = RAW / filename

        if not source.is_file():
            raise FileNotFoundError(
                f"Source file not found: {source}"
            )

        print("\n" + "=" * 60)
        print(f"INGESTING: {table_name}")
        print("=" * 60)

        try:

            # ------------------------------------------------
            # STEP A: READ CSV
            # ------------------------------------------------

            df = (
                spark.read
                .option("header", True)
                .option("inferSchema", True)
                .option("multiLine", True)
                .option("quote", '"')
                .option("escape", '"')
                .option("mode", "FAILFAST")
                .csv(str(source))
            )

            source_count = df.count()

            print(f"Source rows: {source_count}")
            print(f"Original columns: {len(df.columns)}")

            # ------------------------------------------------
            # STEP B: STANDARDIZE COLUMN NAMES
            # ------------------------------------------------

            df, mappings = standardize_columns(df)

            column_mappings[table_name] = mappings

            print("Column names standardized successfully")

            # ------------------------------------------------
            # STEP C: CAPTURE SCHEMA
            # ------------------------------------------------

            schemas[table_name] = {
                "source_file": filename,
                "row_count": source_count,
                "columns": get_schema(df)
            }

            # ------------------------------------------------
            # STEP D: ADD INGESTION METADATA
            # ------------------------------------------------

            df = (
                df.withColumn(
                    "_source_file",
                    F.lit(filename)
                )
                .withColumn(
                    "_ingested_at",
                    F.current_timestamp()
                )
                .withColumn(
                    "_dataset",
                    F.lit(table_name)
                )
            )

            # ------------------------------------------------
            # STEP E: WRITE DELTA TABLE
            # ------------------------------------------------

            output = BRONZE / table_name

            (
                df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(str(output))
            )

            print(f"Delta table saved: {output}")

            # ------------------------------------------------
            # STEP F: VERIFY SAVED DATA
            # ------------------------------------------------

            saved_df = (
                spark.read
                .format("delta")
                .load(str(output))
            )

            saved_count = saved_df.count()

            if source_count != saved_count:
                raise RuntimeError(
                    f"Row count mismatch: "
                    f"source={source_count}, "
                    f"bronze={saved_count}"
                )

            results.append({
                "table": table_name,
                "source_file": filename,
                "status": "SUCCESS",
                "source_rows": source_count,
                "bronze_rows": saved_count,
                "column_count": len(df.columns),
                "output_path": str(output)
            })

            print(
                f"SUCCESS: {table_name} "
                f"({saved_count} rows)"
            )

        except Exception as exc:

            pipeline_status = "FAILED"

            results.append({
                "table": table_name,
                "source_file": filename,
                "status": "FAILED",
                "error": str(exc)
            })

            print(f"FAILED: {table_name}")
            print(f"ERROR: {exc}")

            raise


# ============================================================
# 6. SAVE LOGS AND METADATA
# ============================================================

finally:

    finished_at = datetime.now(timezone.utc).isoformat()

    log = {
        "pipeline": "dataco_bronze_ingestion",
        "status": pipeline_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "results": results
    }

    save_json(
        LOGS / "dataco_bronze_ingestion.json",
        log
    )

    save_json(
        SCHEMAS / "dataco_bronze_schema.json",
        schemas
    )

    save_json(
        SCHEMAS / "dataco_column_mappings.json",
        column_mappings
    )

    spark.stop()


# ============================================================
# 7. COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("DATACO BRONZE INGESTION COMPLETED")
print("=" * 60)

print(f"Tables processed: {len(results)}")

for result in results:
    print(
        result["table"],
        result["status"],
        result.get("bronze_rows")
    )
