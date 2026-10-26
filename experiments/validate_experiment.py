
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH_DIR = ROOT / "experiments/ground_truth"
DQ_EVIDENCE_DIR = ROOT / "evidence/dq_results/experiments"
VALIDATION_DIR = ROOT / "experiments/validation"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()

    experiment_id = args.experiment

    ground_truth_path = (
        GROUND_TRUTH_DIR / f"{experiment_id}.json"
    )
    evidence_path = (
        DQ_EVIDENCE_DIR / f"{experiment_id}_dq.json"
    )

    ground_truth = load_json(ground_truth_path)
    evidence = load_json(evidence_path)
    result = evidence["check_result"]

    expected = ground_truth["expected_dq_failure"]
    check_type = expected["type"]

    expected_failures = ground_truth.get(
        "modified_failure_count",
        ground_truth.get("modified_null_count"),
    )

    observed_increase = ground_truth.get(
        "observed_failure_increase",
        ground_truth.get("observed_null_increase"),
    )

    expected_column = ground_truth["affected_column"]

    if check_type == "duplicate_percentage":
        column_matches = (
            expected_column in result.get("columns", [])
        )
        threshold_matches = (
            result.get("threshold_percentage")
            == expected["max_duplicate_pct"]
        )

    elif check_type == "null_percentage":
        column_matches = (
            result.get("column") == expected_column
        )
        threshold_matches = (
            result.get("threshold_percentage")
            == expected["max_null_pct"]
        )

    elif check_type == "range":
        column_matches = (
            result.get("column") == expected_column
        )
        threshold_matches = (
            result.get("minimum") == expected["minimum"]
        )

    else:
        raise ValueError(
            f"Unsupported check type: {check_type}"
        )

    checks = {
        "experiment_id_matches": (
            ground_truth["experiment_id"]
            == evidence["experiment_id"]
            == experiment_id
        ),
        "dataset_matches": (
            ground_truth["dataset"] == evidence["dataset"]
        ),
        "check_type_matches": (
            result["check_type"] == check_type
        ),
        "column_matches": column_matches,
        "row_count_matches": (
            result["total_rows"]
            == ground_truth["modified_rows"]
        ),
        "failed_rows_match": (
            result["failed_rows"] == expected_failures
        ),
        "threshold_matches": threshold_matches,
        "failure_detected": result["status"] == "FAIL",
        "injection_count_verified": (
            observed_increase
            == ground_truth["injected_rows"]
        ),
    }

    overall_status = (
        "PASS" if all(checks.values()) else "FAIL"
    )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment_id": experiment_id,
        "validated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_status": overall_status,
        "checks": checks,
        "ground_truth_file": str(
            ground_truth_path.relative_to(ROOT)
        ),
        "dq_evidence_file": str(
            evidence_path.relative_to(ROOT)
        ),
    }

    output = (
        VALIDATION_DIR
        / f"{experiment_id}_validation.json"
    )

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("=" * 55)
    print("EXPERIMENT VALIDATION")
    print("=" * 55)
    print("Experiment:", experiment_id)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("-" * 55)
    print("Overall validation:", overall_status)
    print("Report saved:", output)

    if overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

