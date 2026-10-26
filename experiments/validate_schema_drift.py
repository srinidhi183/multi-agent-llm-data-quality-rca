
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_ID = "olist_schema_drift_005"

GROUND_TRUTH_PATH = (
    ROOT / "experiments/ground_truth"
    / f"{EXPERIMENT_ID}.json"
)

EVIDENCE_PATH = (
    ROOT / "evidence/schemas/experiments"
    / f"{EXPERIMENT_ID}_schema.json"
)

VALIDATION_PATH = (
    ROOT / "experiments/validation"
    / f"{EXPERIMENT_ID}_validation.json"
)


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main():
    truth = load_json(GROUND_TRUTH_PATH)
    evidence = load_json(EVIDENCE_PATH)

    expected = truth["expected_schema_drift"]

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
            evidence["check_type"] == "schema_drift"
        ),
        "added_columns_match": (
            evidence["added_columns"]
            == expected["added_columns"]
        ),
        "removed_columns_match": (
            evidence["removed_columns"]
            == expected["removed_columns"]
        ),
        "type_changes_match": (
            evidence["type_changes"]
            == expected["type_changes"]
        ),
        "row_count_preserved": (
            truth["original_rows"]
            == truth["modified_rows"]
        ),
        "values_preserved": (
            truth["changed_value_rows"] == 0
        ),
        "drift_detected": evidence["drift_detected"],
        "downstream_failure_detected": (
            evidence["downstream_failure_detected"]
        ),
        "failure_status_matches": (
            evidence["status"] == "FAIL"
        ),
    }

    overall_status = (
        "PASS" if all(checks.values()) else "FAIL"
    )

    report = {
        "experiment_id": EXPERIMENT_ID,
        "validated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "validation_status": overall_status,
        "checks": checks,
        "ground_truth_file": str(
            GROUND_TRUTH_PATH.relative_to(ROOT)
        ),
        "schema_evidence_file": str(
            EVIDENCE_PATH.relative_to(ROOT)
        ),
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
    print("EXPERIMENT 005 VALIDATION")
    print("=" * 55)

    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")

    print("-" * 55)
    print("Overall validation:", overall_status)
    print("Report saved:", VALIDATION_PATH)

    if overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

