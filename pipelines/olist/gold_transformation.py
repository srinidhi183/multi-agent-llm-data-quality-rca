
import json
from datetime import datetime, timezone
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

ROOT = Path(__file__).resolve().parents[2]
SILVER = ROOT / "data/silver/olist"
GOLD = ROOT / "data/gold/olist"
LOGS = ROOT / "evidence/logs"

GOLD.mkdir(parents=True, exist_ok=True)
LOGS.mkdir(parents=True, exist_ok=True)

builder = (
    SparkSession.builder
    .appName("Olist-Gold-Transformation")
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

    count = spark.read.format("delta").load(
        str(path)
    ).count()

    results.append({
        "table": name,
        "status": "SUCCESS",
        "rows": count,
        "path": str(path)
    })

    print(f"SUCCESS: {name} ({count} rows)")

try:
    orders = read_table("olist_orders_dataset")
    customers = read_table("olist_customers_dataset")
    items = read_table("olist_order_items_dataset")
    payments = read_table("olist_order_payments_dataset")
    products = read_table("olist_products_dataset")

    # Aggregate items by order before joining.
    item_totals = items.groupBy("order_id").agg(
        F.count("*").alias("item_count"),
        F.sum("price").alias("item_value"),
        F.sum("freight_value").alias("freight_value")
    )

    # Aggregate payments separately to avoid fan-out.
    payment_totals = payments.groupBy("order_id").agg(
        F.sum("payment_value").alias("payment_value"),
        F.count("*").alias("payment_count")
    )

    # Gold 1: One row per order.
    order_summary = (
        orders
        .join(
            customers.select(
                "customer_id",
                "customer_unique_id",
                "customer_state"
            ),
            "customer_id",
            "left"
        )
        .join(item_totals, "order_id", "left")
        .join(payment_totals, "order_id", "left")
        .select(
            "order_id",
            "customer_id",
            "customer_unique_id",
            "customer_state",
            "order_status",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "item_count",
            "item_value",
            "freight_value",
            "payment_value",
            "payment_count"
        )
    )

    save_table(order_summary, "gold_order_summary")

    # Gold 2: One row per unique customer.
    customer_summary = (
        order_summary
        .filter(F.col("customer_unique_id").isNotNull())
        .groupBy("customer_unique_id")
        .agg(
            F.countDistinct("order_id").alias(
                "total_orders"
            ),
            F.sum(
                F.coalesce(
                    F.col("payment_value"),
                    F.lit(0.0)
                )
            ).alias("total_payment_value"),
            F.sum(
                F.coalesce(
                    F.col("item_value"),
                    F.lit(0.0)
                )
            ).alias("total_item_value"),
            F.min("order_purchase_timestamp").alias(
                "first_order_date"
            ),
            F.max("order_purchase_timestamp").alias(
                "last_order_date"
            ),
            F.count(
                "order_delivered_customer_date"
            ).alias("delivered_orders")
        )
    )

    save_table(customer_summary, "gold_customer_summary")

    # Gold 3: Product sales summary.
    # Each item is counted once, without joining payments.
    product_sales = items.groupBy("product_id").agg(
        F.count("*").alias("units_sold"),
        F.sum("price").alias("total_item_revenue"),
        F.avg("price").alias("average_selling_price"),
        F.countDistinct("order_id").alias(
            "distinct_orders"
        )
    )

    product_summary = (
        products.select(
            "product_id",
            "product_category_name"
        )
        .join(product_sales, "product_id", "left")
    )

    save_table(product_summary, "gold_product_summary")

    print("\nGOLD TRANSFORMATION COMPLETED")

except Exception as exc:
    results.append({
        "status": "FAILED",
        "error": str(exc)
    })
    raise

finally:
    log = {
        "pipeline": "olist_gold_transformation",
        "started_at": started_at,
        "finished_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "results": results
    }

    (LOGS / "olist_gold_transformation.json").write_text(
        json.dumps(log, indent=2),
        encoding="utf-8"
    )

    spark.stop()
