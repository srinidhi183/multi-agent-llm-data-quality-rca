
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nyc_taxi"
OUTPUT_DIR = ROOT / "evidence" / "schemas" / "drift"


def get_schema(file_path):
    schema = pq.read_schema(file_path)

    return {
        field.name: str(field.type)
        for field in schema
    }


def compare_schemas(previous, current):
    previous_columns = set(previous)
    current_columns = set(current)

    added = sorted(current_columns - previous_columns)
    removed = sorted(previous_columns - current_columns)

    changed_types = []

    for column in sorted(previous_columns & current_columns):
        if previous[column] != current[column]:
            changed_types.append({
                "column": column,
                "previous_type": previous[column],
                "current_type": current[column],
            })

    return {
        "added_columns": added,
        "removed_columns": removed,
        "type_changes": changed_types,
        "drift_detected": bool(
            added or removed or changed_types
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["nyc_taxi"],
        required=True,
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(RAW_DIR.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"No Parquet files found in {RAW_DIR}"
        )

    report = {
        "dataset": args.dataset,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "files_checked": len(files),
        "comparisons": [],
    }

    previous_schema = None
    previous_file = None

    for file_path in files:
        current_schema = get_schema(file_path)

        if previous_schema is not None:
            comparison = compare_schemas(
                previous_schema,
                current_schema,
            )

            report["comparisons"].append({
                "previous_file": previous_file,
                "current_file": file_path.name,
                **comparison,
            })

        previous_schema = current_schema
        previous_file = file_path.name

    report["drift_detected"] = any(
        item["drift_detected"]
        for item in report["comparisons"]
    )

    output = OUTPUT_DIR / "nyc_taxi_schema_drift.json"

    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("=" * 55)
    print("NYC TAXI SCHEMA DRIFT DETECTION")
    print("=" * 55)
    print(f"Files checked: {len(files)}")

    for comparison in report["comparisons"]:
        if comparison["drift_detected"]:
            print(
                f"\nDRIFT DETECTED: "
                f"{comparison['previous_file']} "
                f"-> {comparison['current_file']}"
            )
            print(
                "Added columns:",
                comparison["added_columns"],
            )
            print(
                "Removed columns:",
                comparison["removed_columns"],
            )
            print(
                "Type changes:",
                comparison["type_changes"],
            )

    print(
        "\nOverall drift detected:",
        report["drift_detected"],
    )
    print("Evidence saved:", output)


if __name__ == "__main__":
    main()

