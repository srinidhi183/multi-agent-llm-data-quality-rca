
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "evidence" / "lineage"


def node(dataset, layer, table):
    return {
        "id": f"{dataset}.{layer}.{table}",
        "dataset": dataset,
        "layer": layer,
        "table": table,
        "path": f"data/{layer}/{dataset}/{table}",
    }


def edge(source, target, operation):
    return {
        "source": source,
        "target": target,
        "operation": operation,
    }


def build_olist():
    dataset = "olist"

    tables = [
        "olist_customers_dataset",
        "olist_geolocation_dataset",
        "olist_order_items_dataset",
        "olist_order_payments_dataset",
        "olist_order_reviews_dataset",
        "olist_orders_dataset",
        "olist_products_dataset",
        "olist_sellers_dataset",
        "product_category_name_translation",
    ]

    nodes = []
    edges = []

    for table in tables:
        raw_id = f"{dataset}.raw.{table}"
        bronze_id = f"{dataset}.bronze.{table}"
        silver_id = f"{dataset}.silver.{table}"

        nodes.append({
            "id": raw_id,
            "dataset": dataset,
            "layer": "raw",
            "table": table,
            "path": f"data/raw/olist/{table}.csv",
        })

        nodes.append(node(dataset, "bronze", table))
        nodes.append(node(dataset, "silver", table))

        edges.append(edge(raw_id, bronze_id, "csv_ingestion"))
        edges.append(edge(bronze_id, silver_id, "silver_transformation"))

    gold_tables = [
        "gold_order_summary",
        "gold_customer_summary",
        "gold_product_summary",
    ]

    for table in gold_tables:
        nodes.append(node(dataset, "gold", table))

    # Table-level dependencies for the Olist Gold transformations.
    gold_dependencies = {
        "gold_order_summary": [
            "olist_orders_dataset",
            "olist_customers_dataset",
            "olist_order_items_dataset",
            "olist_order_payments_dataset",
        ],
        "gold_customer_summary": [
            "olist_orders_dataset",
            "olist_customers_dataset",
            "olist_order_items_dataset",
            "olist_order_payments_dataset",
        ],
        "gold_product_summary": [
            "olist_order_items_dataset",
            "olist_products_dataset",
        ],
    }

    for gold_table, sources in gold_dependencies.items():
        for source_table in sources:
            edges.append(edge(
                f"{dataset}.silver.{source_table}",
                f"{dataset}.gold.{gold_table}",
                "gold_aggregation",
            ))

    return nodes, edges


def build_dataco():
    dataset = "dataco"

    tables = [
        "dataco_supply_chain",
        "dataco_access_logs",
    ]

    nodes = []
    edges = []

    raw_paths = {
        "dataco_supply_chain":
            "data/raw/dataco/DataCoSupplyChainDataset.csv",
        "dataco_access_logs":
            "data/raw/dataco/tokenized_access_logs.csv",
    }

    for table in tables:
        raw_id = f"{dataset}.raw.{table}"
        bronze_id = f"{dataset}.bronze.{table}"
        silver_id = f"{dataset}.silver.{table}"

        nodes.append({
            "id": raw_id,
            "dataset": dataset,
            "layer": "raw",
            "table": table,
            "path": raw_paths[table],
        })

        nodes.append(node(dataset, "bronze", table))
        nodes.append(node(dataset, "silver", table))

        edges.append(edge(raw_id, bronze_id, "csv_ingestion"))
        edges.append(edge(bronze_id, silver_id, "silver_transformation"))

    gold_dependencies = {
        "gold_sales_by_category": ["dataco_supply_chain"],
        "gold_shipping_performance": ["dataco_supply_chain"],
        "gold_customer_summary": ["dataco_supply_chain"],
        "gold_access_activity": ["dataco_access_logs"],
    }

    for gold_table, sources in gold_dependencies.items():
        nodes.append(node(dataset, "gold", gold_table))

        for source_table in sources:
            edges.append(edge(
                f"{dataset}.silver.{source_table}",
                f"{dataset}.gold.{gold_table}",
                "gold_aggregation",
            ))

    return nodes, edges


def build_nyc_taxi():
    dataset = "nyc_taxi"

    nodes = []
    edges = []

    raw_dir = ROOT / "data" / "raw" / dataset
    parquet_files = sorted(raw_dir.glob("*.parquet"))

    bronze_id = f"{dataset}.bronze.yellow_taxi_trips"
    silver_id = f"{dataset}.silver.yellow_taxi_trips"

    nodes.append(node(dataset, "bronze", "yellow_taxi_trips"))
    nodes.append(node(dataset, "silver", "yellow_taxi_trips"))

    for file_path in parquet_files:
        raw_id = f"{dataset}.raw.{file_path.stem}"

        nodes.append({
            "id": raw_id,
            "dataset": dataset,
            "layer": "raw",
            "table": file_path.stem,
            "path": str(file_path.relative_to(ROOT)),
        })

        edges.append(edge(
            raw_id,
            bronze_id,
            "parquet_ingestion",
        ))

    edges.append(edge(
        bronze_id,
        silver_id,
        "silver_transformation",
    ))

    gold_tables = [
        "gold_monthly_summary",
        "gold_daily_trips",
        "gold_pickup_locations",
    ]

    for table in gold_tables:
        nodes.append(node(dataset, "gold", table))

        edges.append(edge(
            silver_id,
            f"{dataset}.gold.{table}",
            "gold_aggregation",
        ))

    return nodes, edges


BUILDERS = {
    "olist": build_olist,
    "dataco": build_dataco,
    "nyc_taxi": build_nyc_taxi,
}


def validate_lineage(nodes, edges):
    node_ids = [item["id"] for item in nodes]

    if len(node_ids) != len(set(node_ids)):
        raise ValueError("Duplicate lineage node IDs detected")

    known_ids = set(node_ids)

    for item in nodes:
        path = ROOT / item["path"]

        if not path.exists():
            print(f"WARNING: Missing path: {item['path']}")

    for item in edges:
        if item["source"] not in known_ids:
            raise ValueError(
                f"Unknown source node: {item['source']}"
            )

        if item["target"] not in known_ids:
            raise ValueError(
                f"Unknown target node: {item['target']}"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["olist", "dataco", "nyc_taxi", "all"],
    )

    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = (
        list(BUILDERS)
        if args.dataset == "all"
        else [args.dataset]
    )

    for dataset in datasets:
        nodes, edges = BUILDERS[dataset]()
        validate_lineage(nodes, edges)

        report = {
            "dataset": dataset,
            "generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "lineage_type": "declared_table_level",
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
        }

        output = OUTPUT_DIR / f"{dataset}_lineage.json"

        with open(output, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print("=" * 55)
        print(f"DATASET: {dataset.upper()}")
        print(f"Nodes: {len(nodes)}")
        print(f"Edges: {len(edges)}")
        print(f"Evidence saved: {output}")

    print("\nLineage generation completed.")


if __name__ == "__main__":
    main()

