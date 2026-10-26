
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "olist_invalid_customer_ref_004"

GROUND_TRUTH_PATH = (
    ROOT / "experiments/ground_truth"
    / f"{EXPERIMENT_ID}.json"
)

EVIDENCE_PATH = (
    ROOT / "evidence/dq_results/experiments"
    / f"{EXPERIMENT_ID}_dq.json"
)

VALIDATION_PATH = (
    ROOT / "experiments/validation"
    / f"{EXPERIMENT_ID}_validation.json"
)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    truth = load_json(GROUND_TRUTH_PATH)
    evidence = load_json(EVIDENCE_PATH)

    result = evidence["check_result"]
    expected = truth["expected_dq_failure"]

    checks = {
        "experiment_id_matches": (
            truth["experiment_id"]
            == evidence["experiment_id"]
            == EXPERIMENT_ID
        ),
        "dataset_matches": (
            truth["dataset"] == evidence["dataset"]
        ),
        "check_type_matches": (
            result["check_type"] == expected["type"]
        ),
        "column_matches": (
            result["column"] == expected["column"]
        ),
        "row_count_matches": (
            result["total_rows"] == truth["modified_rows"]
        ),
        "failed_rows_match": (
            result["failed_rows"]
            == truth["modified_failure_count"]
        ),
        "failure_detected": result["status"] == "FAIL",
        "injection_count_verified": (
            truth["observed_failure_increase"]
            == truth["injected_rows"]
        ),
    }

    status = "PASS" if all(checks.values()) else "FAIL"

    report = {
        "experiment_id": EXPERIMENT_ID,
        "validated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_status": status,
        "checks": checks,
    }

    VALIDATION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with VALIDATION_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("=" * 55)
    print("EXPERIMENT 004 VALIDATION")
    print("=" * 55)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("-" * 55)
    print("Overall validation:", status)
    print("Report saved:", VALIDATION_PATH)

    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

