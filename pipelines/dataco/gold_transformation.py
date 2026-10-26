
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/silver/dataco"
GOLD = ROOT / "data/gold/dataco"
LOGS = ROOT / "evidence/logs"

GOLD.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

builder = (
    SparkSession.builder
    .appName("DataCo-Gold-Transformation")
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


def read_table(name):
    return spark.read.format("delta").load(
        str(SILVER / name)
    )


def save_table(df, name):
    path = GOLD / name

    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(str(path))
    )

    count = (
        spark.read
        .format("delta")
        .load(str(path))
        .count()
    )

    results.append({
        "table": name,
        "status": "SUCCESS",
        "rows": count,
        "path": str(path)
    })

    print(f"SUCCESS: {name} ({count} rows)")


def require_columns(df, columns, table_name):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            f"{table_name}: missing columns {missing}"
        )


try:
    supply = read_table("dataco_supply_chain")
    access = read_table("dataco_access_logs")

    # --------------------------------------------------------
    # GOLD 1: SALES BY CATEGORY
    # --------------------------------------------------------

    require_columns(
        supply,
        [
            "category_name",
            "order_id",
            "order_item_quantity",
            "sales",
            "order_profit_per_order"
        ],
        "dataco_supply_chain"
    )

    sales_by_category = (
        supply
        .groupBy("category_name")
        .agg(
            F.count("*").alias("order_item_rows"),
            F.countDistinct("order_id").alias(
                "distinct_orders"
            ),
            F.sum("order_item_quantity").alias(
                "total_quantity"
            ),
            F.sum("sales").alias("total_sales"),
            F.sum("order_profit_per_order").alias(
                "sum_recorded_profit"
            )
        )
    )

    save_table(
        sales_by_category,
        "gold_sales_by_category"
    )

    # --------------------------------------------------------
    # GOLD 2: SHIPPING PERFORMANCE
    # --------------------------------------------------------

    require_columns(
        supply,
        [
            "order_id",
            "shipping_mode",
            "days_for_shipping_real",
            "days_for_shipment_scheduled",
            "late_delivery_risk"
        ],
        "dataco_supply_chain"
    )

    # One row per order, using first non-null values.
    order_shipping = (
        supply
        .groupBy("order_id")
        .agg(
            F.first(
                "shipping_mode",
                ignorenulls=True
            ).alias("shipping_mode"),
            F.first(
                "days_for_shipping_real",
                ignorenulls=True
            ).alias("actual_days"),
            F.first(
                "days_for_shipment_scheduled",
                ignorenulls=True
            ).alias("scheduled_days"),
            F.max("late_delivery_risk").alias(
                "late_delivery_risk"
            )
        )
    )

    shipping_performance = (
        order_shipping
        .withColumn(
            "delay_days",
            F.col("actual_days") -
            F.col("scheduled_days")
        )
        .groupBy("shipping_mode")
        .agg(
            F.count("*").alias("total_orders"),
            F.avg("actual_days").alias(
                "avg_actual_shipping_days"
            ),
            F.avg("scheduled_days").alias(
                "avg_scheduled_shipping_days"
            ),
            F.avg("delay_days").alias(
                "avg_delay_days"
            ),
            F.avg("late_delivery_risk").alias(
                "late_delivery_risk_rate"
            )
        )
    )

    save_table(
        shipping_performance,
        "gold_shipping_performance"
    )

    # --------------------------------------------------------
    # GOLD 3: CUSTOMER SUMMARY
    # --------------------------------------------------------

    require_columns(
        supply,
        [
            "customer_id",
            "order_id",
            "sales"
        ],
        "dataco_supply_chain"
    )

    customer_summary = (
        supply
        .groupBy("customer_id")
        .agg(
            F.countDistinct("order_id").alias(
                "total_orders"
            ),
            F.count("*").alias(
                "order_item_rows"
            ),
            F.sum("sales").alias(
                "total_sales"
            )
        )
    )

    save_table(
        customer_summary,
        "gold_customer_summary"
    )

    # --------------------------------------------------------
    # GOLD 4: ACCESS ACTIVITY
    # --------------------------------------------------------

    # Keep access logs separate from transaction data.
    # Summarize by available categorical fields.

    access_columns = [
        c for c in access.columns
        if not c.startswith("_")
    ]

    if not access_columns:
        raise ValueError(
            "No source columns found in access logs"
        )

    # Use the first source column as a grouping dimension.
    # We'll refine this after inspecting its semantics.
    grouping_column = access_columns[0]

    access_activity = (
        access
        .groupBy(grouping_column)
        .agg(
            F.count("*").alias("event_count")
        )
    )

    save_table(
        access_activity,
        "gold_access_activity"
    )

    print("\nDATACO GOLD TRANSFORMATION COMPLETED")

except Exception as exc:
    results.append({
        "status": "FAILED",
        "error": str(exc)
    })
    raise

finally:
    log = {
        "pipeline": "dataco_gold_transformation",
        "started_at": started_at,
        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results
    }

    (LOGS / "dataco_gold_transformation.json").write_text(
        json.dumps(log, indent=2),
        encoding="utf-8"
    )

    spark.stop()
