
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

ROOT = Path(__file__).resolve().parents[2]

BRONZE = ROOT / "data/bronze/dataco"
SILVER = ROOT / "data/silver/dataco"
LOGS = ROOT / "evidence/logs"
DQ = ROOT / "evidence/dq_results"
SCHEMAS = ROOT / "evidence/schemas"

for directory in [SILVER, LOGS, DQ, SCHEMAS]:
    directory.mkdir(parents=True, exist_ok=True)

builder = (
    SparkSession.builder
    .appName("DataCo-Silver-Transformation")
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

TABLES = [
    "dataco_supply_chain",
    "dataco_access_logs"
]

# Required fields for the supply-chain dataset.
# Access-log fields will be profiled without assuming
# that they have the same schema.
REQUIRED_COLUMNS = {
    "dataco_supply_chain": [
        "order_id",
        "order_item_id",
        "customer_id"
    ],
    "dataco_access_logs": []
}

results = []
dq_results = {}
schemas = {}

started_at = datetime.now(timezone.utc).isoformat()

def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8"
    )

try:
    for table in TABLES:
        print(f"\nProcessing: {table}")

        try:
            source = BRONZE / table

            df = spark.read.format("delta").load(
                str(source)
            )

            bronze_count = df.count()

            # Trim string columns without deleting records.
            for field in df.schema.fields:
                if isinstance(field.dataType, StringType):
                    df = df.withColumn(
                        field.name,
                        F.trim(F.col(field.name))
                    )

            required = REQUIRED_COLUMNS[table]

            missing = [
                c for c in required
                if c not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Missing required columns: {missing}"
                )

            # NULL counts for required columns.
            required_nulls = {}

            if required:
                required_nulls = (
                    df.select(*[
                        F.sum(
                            F.when(
                                F.col(c).isNull(), 1
                            ).otherwise(0)
                        ).alias(c)
                        for c in required
                    ])
                    .first()
                    .asDict()
                )

            df = df.withColumn(
                "_silver_processed_at",
                F.current_timestamp()
            )

            output = SILVER / table

            (
                df.write
                .format("delta")
                .mode("overwrite")
                .option("overwriteSchema", "true")
                .save(str(output))
            )

            saved = spark.read.format("delta").load(
                str(output)
            )

            silver_count = saved.count()

            if bronze_count != silver_count:
                raise RuntimeError(
                    f"Row count mismatch: "
                    f"{bronze_count} != {silver_count}"
                )

            dq_results[table] = {
                "row_count": silver_count,
                "row_count_match": True,
                "required_column_nulls": required_nulls
            }

            schemas[table] = {
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
                "table": table,
                "status": "SUCCESS",
                "bronze_rows": bronze_count,
                "silver_rows": silver_count
            })

            print(
                f"SUCCESS: {table} "
                f"({silver_count} rows)"
            )

        except Exception as exc:
            results.append({
                "table": table,
                "status": "FAILED",
                "error": str(exc)
            })
            raise

finally:
    save_json(
        LOGS / "dataco_silver_transformation.json",
        {
            "pipeline": "dataco_silver_transformation",
            "started_at": started_at,
            "finished_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "results": results
        }
    )

    save_json(
        DQ / "dataco_silver_dq.json",
        dq_results
    )

    save_json(
        SCHEMAS / "dataco_silver_schema.json",
        schemas
    )

    spark.stop()

print("\nDATACO SILVER TRANSFORMATION COMPLETED")
