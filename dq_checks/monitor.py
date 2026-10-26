
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta import configure_spark_with_delta_pip


ROOT = Path(__file__).resolve().parents[1]
RULES_DIR = ROOT / "configs" / "dq_rules"
OUTPUT_DIR = ROOT / "evidence" / "dq_results" / "monitoring"


def create_spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-DQ-Monitoring")
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
    return spark


def load_rules(dataset):
    path = RULES_DIR / f"{dataset}.json"

    if not path.exists():
        raise FileNotFoundError(f"Rules not found: {path}")

    with open(path, encoding="utf-8") as file:
        return json.load(file)


def check_nulls(df, column, max_null_pct):
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

    total = df.count()

    nulls = df.filter(
        F.col(column).isNull()
    ).count()

    percentage = (nulls / total * 100) if total else 0

    return {
        "check_type": "null_percentage",
        "column": column,
        "total_rows": total,
        "failed_rows": nulls,
        "observed_percentage": round(percentage, 4),
        "threshold_percentage": max_null_pct,
        "status": (
            "PASS" if percentage <= max_null_pct else "FAIL"
        ),
    }


def check_duplicates(df, columns, max_duplicate_pct):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    total = df.count()

    unique = df.select(*columns).distinct().count()
    duplicates = total - unique

    percentage = (
        duplicates / total * 100
        if total else 0
    )

    return {
        "check_type": "duplicate_percentage",
        "columns": columns,
        "total_rows": total,
        "failed_rows": duplicates,
        "observed_percentage": round(percentage, 4),
        "threshold_percentage": max_duplicate_pct,
        "status": (
            "PASS" if percentage <= max_duplicate_pct else "FAIL"
        ),
    }


def check_range(df, column, minimum=None, maximum=None):
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

    condition = F.col(column).isNull()

    if minimum is not None:
        condition = condition | (
            F.col(column) < minimum
        )

    if maximum is not None:
        condition = condition | (
            F.col(column) > maximum
        )

    total = df.count()
    failed = df.filter(condition).count()

    percentage = (
        failed / total * 100
        if total else 0
    )

    return {
        "check_type": "range",
        "column": column,
        "total_rows": total,
        "failed_rows": failed,
        "observed_percentage": round(percentage, 4),
        "minimum": minimum,
        "maximum": maximum,
        "status": (
            "PASS" if failed == 0 else "FAIL"
        ),
    }


def check_referential_integrity(
    spark,
    df,
    column,
    reference_path,
    reference_column,
    max_orphan_pct=0,
):
    if column not in df.columns:
        raise ValueError(
            f"Missing source column: {column}"
        )

    reference_df = (
        spark.read
        .format("delta")
        .load(str(ROOT / reference_path))
    )

    if reference_column not in reference_df.columns:
        raise ValueError(
            f"Missing reference column: {reference_column}"
        )

    source_keys = df.select(
        F.col(column).alias("_source_key")
    )

    reference_keys = (
        reference_df
        .select(
            F.col(reference_column).alias("_reference_key")
        )
        .filter(
            F.col("_reference_key").isNotNull()
        )
        .distinct()
    )

    total = source_keys.count()

    orphans = (
        source_keys
        .filter(
            F.col("_source_key").isNotNull()
        )
        .join(
            reference_keys,
            F.col("_source_key")
            == F.col("_reference_key"),
            "left_anti",
        )
        .count()
    )

    percentage = (
        orphans / total * 100
        if total else 0
    )

    return {
        "check_type": "referential_integrity",
        "column": column,
        "reference_path": reference_path,
        "reference_column": reference_column,
        "total_rows": total,
        "failed_rows": orphans,
        "observed_percentage": round(percentage, 4),
        "threshold_percentage": max_orphan_pct,
        "status": (
            "PASS"
            if percentage <= max_orphan_pct
            else "FAIL"
        ),
    }


def run_checks(spark, df, rules):
    results = []

    for rule in rules:
        check_type = rule["type"]

        if check_type == "null_percentage":
            result = check_nulls(
                df,
                rule["column"],
                rule.get("max_null_pct", 0),
            )

        elif check_type == "duplicate_percentage":
            result = check_duplicates(
                df,
                rule["columns"],
                rule.get("max_duplicate_pct", 0),
            )

        elif check_type == "range":
            result = check_range(
                df,
                rule["column"],
                rule.get("minimum"),
                rule.get("maximum"),
            )

        elif check_type == "referential_integrity":
            result = check_referential_integrity(
                spark,
                df,
                rule["column"],
                rule["reference_path"],
                rule["reference_column"],
                rule.get("max_orphan_pct", 0),
            )

        else:
            raise ValueError(
                f"Unsupported check type: {check_type}"
            )

        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "olist",
            "dataco",
            "nyc_taxi"
        ],
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config = load_rules(args.dataset)
    spark = create_spark()

    try:
        report = {
            "dataset": args.dataset,
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "tables": [],
        }

        for table in config["tables"]:
            path = ROOT / table["path"]

            if not path.exists():
                raise FileNotFoundError(
                    f"Delta table not found: {path}"
                )

            df = (
                spark.read
                .format("delta")
                .load(str(path))
            )

            results = run_checks(
                spark,
                df,
                table["checks"]
            )

            table_status = (
                "PASS"
                if all(
                    result["status"] == "PASS"
                    for result in results
                )
                else "FAIL"
            )

            report["tables"].append({
                "table": table["name"],
                "path": table["path"],
                "checks": results,
                "status": table_status,
            })

        report["status"] = (
            "PASS"
            if all(
                table["status"] == "PASS"
                for table in report["tables"]
            )
            else "FAIL"
        )

        output = (
            OUTPUT_DIR
            / f"{args.dataset}_monitoring.json"
        )

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=2
            )

        print(
            f"Monitoring completed: {args.dataset}"
        )

        print(
            f"Overall status: {report['status']}"
        )

        print(
            f"Evidence saved: {output}"
        )

        for table in report["tables"]:
            print(
                f"\nTable: {table['table']} "
                f"[{table['status']}]"
            )

            for result in table["checks"]:
                print(
                    f"  {result['check_type']}: "
                    f"{result['status']} | "
                    f"Failed rows: "
                    f"{result['failed_rows']}"
                )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()

