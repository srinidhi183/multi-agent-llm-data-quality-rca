import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_DIR = ROOT / "evidence/rca_bundles"
VIEW_DIR = ROOT / "evidence/agent_views"
MANIFEST_PATH = ROOT / "experiments/evaluation/case_manifest.json"
REPORT_PATH = ROOT / "experiments/evaluation/evidence_integration_report.json"

AGENT_SECTIONS = {
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

FORBIDDEN_KEYS = {
    "root_cause",
    "ground_truth",
    "injected_rows",
    "injected_change",
    "expected_dq_failure",
    "expected_schema_drift",
    "expected_downstream_failure",
    "original_failure_count",
    "modified_failure_count",
    "observed_failure_increase",
    "original_null_count",
    "modified_null_count",
    "observed_null_increase",
    "validation_status",
    "validation_checks",
}

INJECTION_MARKERS = [
    "THESIS_INVALID_CUSTOMER_004_",
    "Controlled injection",
    "Controlled upstream schema change",
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_private_keys(value, location="$"):
    findings = []

    if isinstance(value, dict):
        for key, item in value.items():
            current = f"{location}.{key}"

            if key in FORBIDDEN_KEYS:
                findings.append(current)

            findings.extend(
                find_private_keys(item, current)
            )

    elif isinstance(value, list):
        for index, item in enumerate(value):
            findings.extend(
                find_private_keys(
                    item,
                    f"{location}[{index}]",
                )
            )

    return findings


def check_case(case):
    case_id = case["case_id"]
    experiment_id = case["experiment_id"]

    bundle_path = BUNDLE_DIR / f"{experiment_id}.json"
    bundle = load_json(bundle_path)

    checks = {}

    checks["bundle_id_matches"] = (
        bundle.get("experiment_id") == experiment_id
    )

    checks["bundle_has_five_sections"] = (
        set(bundle.get("evidence", {}))
        == AGENT_SECTIONS["combined_agent"]
    )

    checks["bundle_has_no_private_keys"] = (
        len(find_private_keys(bundle)) == 0
    )

    availability = bundle.get("evidence_availability", {})

    checks["availability_matches_sections"] = all(
        availability.get(section)
        == bundle["evidence"][section].get("available")
        for section in AGENT_SECTIONS["combined_agent"]
    )

    view_results = {}
    all_views_valid = True

    for agent, expected_sections in AGENT_SECTIONS.items():
        path = VIEW_DIR / case_id / f"{agent}.json"
        view = load_json(path)

        serialized = json.dumps(
            view,
            ensure_ascii=False,
        )

        expected_available = {
            section
            for section in expected_sections
            if availability.get(section) is True
        }

        actual_available = {
            section
            for section, content in view.get(
                "evidence", {}
            ).items()
            if content.get("available") is True
        }

        agent_checks = {
            "case_id_matches": (
                view.get("case_id") == case_id
            ),
            "agent_name_matches": (
                view.get("agent_name") == agent
            ),
            "correct_sections": (
                set(view.get("evidence", {}))
                == expected_sections
            ),
            "availability_consistent": (
                actual_available == expected_available
            ),
            "no_private_keys": (
                len(find_private_keys(view)) == 0
            ),
            "no_experiment_id": (
                experiment_id not in serialized
            ),
            "no_injection_markers": (
                not any(
                    marker in serialized
                    for marker in INJECTION_MARKERS
                )
            ),
            "no_ground_truth_path": (
                "experiments/ground_truth/"
                not in serialized
            ),
        }

        passed = all(agent_checks.values())

        view_results[agent] = {
            "status": "PASS" if passed else "FAIL",
            "checks": agent_checks,
            "available_sections": sorted(
                actual_available
            ),
        }

        if not passed:
            all_views_valid = False

    checks["all_agent_views_valid"] = all_views_valid

    return {
        "case_id": case_id,
        "experiment_id": experiment_id,
        "status": (
            "PASS" if all(checks.values()) else "FAIL"
        ),
        "checks": checks,
        "agent_views": view_results,
    }


def main():
    manifest = load_json(MANIFEST_PATH)
    cases = manifest["cases"]

    if len(cases) != 5:
        raise AssertionError(
            f"Expected 5 cases, found {len(cases)}"
        )

    case_ids = [case["case_id"] for case in cases]

    if len(set(case_ids)) != len(case_ids):
        raise AssertionError("Duplicate case IDs found")

    results = []

    print("=" * 65)
    print("RCA EVIDENCE INTEGRATION TESTING")
    print("=" * 65)

    for case in cases:
        result = check_case(case)
        results.append(result)

        print()
        print(
            result["case_id"],
            "->",
            result["status"],
        )

        for agent, details in result["agent_views"].items():
            print(
                f"  {agent:18} {details['status']}"
            )

            for check, passed in details["checks"].items():
                if not passed:
                    print(f"    FAILED: {check}")

        for check, passed in result["checks"].items():
            if not passed:
                print(f"  FAILED: {check}")

    passed_cases = sum(
        result["status"] == "PASS"
        for result in results
    )

    report = {
        "report_version": "1.0",
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_cases": len(results),
        "passed_cases": passed_cases,
        "failed_cases": len(results) - passed_cases,
        "overall_status": (
            "PASS"
            if passed_cases == len(results)
            else "FAIL"
        ),
        "results": results,
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
    print(
        f"RESULT: {passed_cases}/{len(results)} "
        "CASES PASSED"
    )
    print("OVERALL:", report["overall_status"])
    print("Report:", REPORT_PATH)
    print("=" * 65)

    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
