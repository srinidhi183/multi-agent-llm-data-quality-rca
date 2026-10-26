
import json
from pathlib import Path

import pytest
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def spark():
    builder = (
        SparkSession.builder
        .appName("Thesis-Pipeline-Tests")
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

    session = configure_spark_with_delta_pip(
        builder
    ).getOrCreate()

    session.sparkContext.setLogLevel("ERROR")

    yield session

    session.stop()


def load_delta(spark, path):
    assert path.exists(), f"Missing Delta table: {path}"

    return (
        spark.read
        .format("delta")
        .load(str(path))
    )


def test_olist_gold_tables_exist():
    for table in [
        "gold_order_summary",
        "gold_customer_summary",
        "gold_product_summary"
    ]:
        path = ROOT / "data/gold/olist" / table

        assert path.exists(), f"Missing table: {table}"


def test_dataco_gold_tables_exist():
    for table in [
        "gold_sales_by_category",
        "gold_shipping_performance",
        "gold_customer_summary",
        "gold_access_activity"
    ]:
        path = ROOT / "data/gold/dataco" / table

        assert path.exists(), f"Missing table: {table}"


def test_nyc_gold_tables_exist():
    for table in [
        "gold_monthly_summary",
        "gold_daily_trips",
        "gold_pickup_locations"
    ]:
        path = ROOT / "data/gold/nyc_taxi" / table

        assert path.exists(), f"Missing table: {table}"


def test_nyc_bronze_ingestion_log():
    path = (
        ROOT
        / "evidence/logs/nyc_taxi_bronze_ingestion.json"
    )

    data = json.loads(path.read_text())

    assert data["status"] == "SUCCESS"
    assert len(data["results"]) == 8

    assert all(
        r["status"] == "SUCCESS"
        for r in data["results"]
    )

    assert sum(
        r["bronze_rows"] for r in data["results"]
    ) == 29703355


def test_nyc_silver_preserves_rows():
    path = (
        ROOT
        / "evidence/logs/nyc_taxi_silver_transformation.json"
    )

    data = json.loads(path.read_text())

    assert data["status"] == "SUCCESS"
    assert len(data["results"]) == 8

    for row in data["results"]:
        assert row["status"] == "SUCCESS"
        assert row["bronze_rows"] == row["silver_rows"]

        assert (
            row["dq_passed"] + row["dq_failed"]
            == row["silver_rows"]
        )


def test_nyc_gold_monthly_reconciliation(spark):
    df = load_delta(
        spark,
        ROOT / "data/gold/nyc_taxi/gold_monthly_summary"
    )

    rows = df.collect()

    assert len(rows) == 8

    total = sum(r["total_records"] for r in rows)

    assert total == 29703355

    for row in rows:
        assert (
            row["valid_trips"] + row["flagged_trips"]
            == row["total_records"]
        )

        assert 0 <= row["dq_pass_rate"] <= 1


def test_nyc_schema_change_recorded():
    path = (
        ROOT
        / "evidence/schemas/nyc_taxi_bronze_schema.json"
    )

    data = json.loads(path.read_text())

    january = {
        c["name"]
        for c in data["01"]["columns"]
    }

    june = {
        c["name"]
        for c in data["06"]["columns"]
    }

    assert "request_source" not in january
    assert "request_source" in june


def test_dataco_silver_log():
    path = (
        ROOT
        / "evidence/logs/dataco_silver_transformation.json"
    )

    data = json.loads(path.read_text())

    assert len(data["results"]) == 2

    for row in data["results"]:
        assert row["status"] == "SUCCESS"
        assert row["bronze_rows"] == row["silver_rows"]
