
import json
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Locate the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "olist"
REPORT_PATH = PROJECT_ROOT / "docs" / "olist_profile.json"

# Start Spark locally
spark = (
    SparkSession.builder
    .appName("Olist-Data-Profiling")
    .master("local[2]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

profiles = {}

try:
    csv_files = sorted(RAW_PATH.glob("*.csv"))

    if len(csv_files) != 9:
        raise RuntimeError(
            f"Expected 9 CSV files, found {len(csv_files)}"
        )

    for file_path in csv_files:
        print(f"\nProfiling: {file_path.name}")

        # Read the CSV with a header and inferred types
        df = (
            spark.read
            .option("header", True)
            .option("inferSchema", True)
            .option("multiLine", True)
            .option("quote", '"')
            .option("escape", '"')
            .option("mode", "FAILFAST")
            .csv(str(file_path))
        )

        row_count = df.count()
        column_count = len(df.columns)

        # Count NULLs in every column
        null_counts = df.select([
            F.sum(
                F.when(F.col(f"`{c}`").isNull(), 1)
                .otherwise(0)
            ).alias(c)
            for c in df.columns
        ]).first().asDict()

        # Count fully identical rows
        distinct_count = df.distinct().count()
        duplicate_count = row_count - distinct_count

        schema = [
            {
                "column": field.name,
                "type": field.dataType.simpleString(),
                "nullable": field.nullable
            }
            for field in df.schema.fields
        ]

        profiles[file_path.name] = {
            "row_count": row_count,
            "column_count": column_count,
            "schema": schema,
            "null_counts": null_counts,
            "exact_duplicate_count": duplicate_count
        }

        print(f"Rows: {row_count}")
        print(f"Columns: {column_count}")
        print(f"Exact duplicates: {duplicate_count}")
        df.printSchema()

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)

    print("\nProfiling complete!")
    print(f"Report saved to: {REPORT_PATH}")

finally:
    spark.stop()
