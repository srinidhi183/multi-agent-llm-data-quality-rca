import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_DIR = ROOT / "experiments/ground_truth"
VALIDATION_DIR = ROOT / "experiments/validation"
BUNDLE_DIR = ROOT / "evidence/rca_bundles"
VIEW_DIR = ROOT / "evidence/agent_views"
LOG_DIR = ROOT / "evidence/logs/experiments"

MANIFEST_PATH = (
    ROOT / "experiments/evaluation/case_manifest.json"
)

REPORT_PATH = (
    ROOT / "experiments/evaluation/month1_final_report.json"
)

EXPERIMENTS = [
    "olist_null_customer_id_001",
    "olist_duplicate_order_id_002",
    "olist_negative_price_003",
    "olist_invalid_customer_ref_004",
    "olist_schema_drift_005",
]

AGENTS = [
    "dq_agent",
    "schema_agent",
    "lineage_agent",
    "logs_agent",
    "combined_agent",
]

EXPECTED_SECTIONS = {
    "dq_agent": {"data_quality"},
    "schema_agent": {"schema"},
    "lineage_agent": {"lineage"},
    "logs_agent": {"logs"},
    "combined_agent": {
        "data_quality",
        "schema",
        "lineage",
        "logs",
        "baseline_monitoring",
    },
}

PRIVATE_KEYS = {
    "root_cause",
    "ground_truth",
    "injected_rows",
    "injected_change",
    "expected_dq_failure",
    "expected_schema_drift",
    "expected_downstream_failure",
    "validation_status",
    "validation_checks",
}


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def contains_private_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                return True

            if contains_private_keys(item):
                return True

    elif isinstance(value, list):
        return any(
            contains_private_keys(item)
            for item in value
        )

    return False


def check_experiment(experiment_id, case_id):
    checks = {}

    truth_path = (
        GROUND_TRUTH_DIR / f"{experiment_id}.json"
    )

    validation_path = (
        VALIDATION_DIR / f"{experiment_id}_validation.json"
    )

    bundle_path = (
        BUNDLE_DIR / f"{experiment_id}.json"
    )

    log_path = (
        LOG_DIR / f"{experiment_id}.json"
    )

    checks["ground_truth_exists"] = truth_path.exists()
    checks["validation_exists"] = validation_path.exists()
    checks["bundle_exists"] = bundle_path.exists()
    checks["audit_log_exists"] = log_path.exists()

    if bundle_path.exists():
        bundle = load_json(bundle_path)

        checks["schema_available"] = (
            bundle["evidence_availability"].get("schema")
            is True
        )

        checks["lineage_available"] = (
            bundle["evidence_availability"].get("lineage")
            is True
        )

        checks["logs_available"] = (
            bundle["evidence_availability"].get("logs")
            is True
        )

        checks["bundle_no_private_keys"] = (
            not contains_private_keys(bundle)
        )

    if log_path.exists():
        log = load_json(log_path)

        checks["audit_log_type_correct"] = (
            log.get("log_type")
            == "posthoc_artifact_verification"
        )

        checks["not_historical_execution_log"] = (
            log.get("historical_execution_log")
            is False
        )

    for agent in AGENTS:
        view_path = (
            VIEW_DIR / case_id / f"{agent}.json"
        )

        checks[f"{agent}_exists"] = view_path.exists()

        if view_path.exists():
            view = load_json(view_path)

            checks[f"{agent}_sections_correct"] = (
                set(view.get("evidence", {}))
                == EXPECTED_SECTIONS[agent]
            )

            checks[f"{agent}_no_private_keys"] = (
                not contains_private_keys(view)
            )

            serialized = json.dumps(view)

            checks[f"{agent}_no_experiment_id"] = (
                experiment_id not in serialized
            )

    return {
        "experiment_id": experiment_id,
        "case_id": case_id,
        "status": (
            "PASS"
            if checks and all(checks.values())
            else "FAIL"
        ),
        "checks": checks,
    }


def main():
    manifest = load_json(MANIFEST_PATH)

    case_mapping = {
        item["experiment_id"]: item["case_id"]
        for item in manifest["cases"]
    }

    results = []

    print("=" * 65)
    print("MONTH 1 FINAL INTEGRATION REVIEW")
    print("=" * 65)

    for experiment_id in EXPERIMENTS:
        case_id = case_mapping.get(experiment_id)

        if case_id is None:
            raise ValueError(
                f"Missing case mapping: {experiment_id}"
            )

        result = check_experiment(
            experiment_id,
            case_id,
        )

        results.append(result)

        print()
        print(
            case_id,
            "->",
            result["status"],
        )

        for name, passed in result["checks"].items():
            if not passed:
                print("  FAILED:", name)

    passed = sum(
        result["status"] == "PASS"
        for result in results
    )

    report = {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_cases": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "overall_status": (
            "PASS"
            if passed == len(results)
            else "FAIL"
        ),
        "results": results,
        "limitations": [
            (
                "The audit logs are post-hoc artifact "
                "verification records, not historical "
                "pipeline execution logs."
            ),
            (
                "Declared lineage does not establish "
                "record-level data provenance."
            ),
            (
                "Ground-truth isolation checks detect "
                "known leakage patterns but do not "
                "guarantee complete semantic blinding."
            ),
            (
                "No LLM-based root-cause analysis "
                "accuracy has been evaluated yet."
            ),
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print()
    print("=" * 65)
    print(f"RESULT: {passed}/{len(results)} CASES PASSED")
    print("OVERALL:", report["overall_status"])
    print("REPORT:", REPORT_PATH)
    print("=" * 65)

    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
