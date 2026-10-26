import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_DIR = ROOT / "experiments/ground_truth"
DQ_DIR = ROOT / "evidence/dq_results/experiments"
SCHEMA_DIR = ROOT / "evidence/schemas/experiments"
LINEAGE_DIR = ROOT / "evidence/lineage"
MONITORING_DIR = ROOT / "evidence/dq_results/monitoring"
LOG_DIR = ROOT / "evidence/logs/experiments"
OUTPUT_DIR = ROOT / "evidence/rca_bundles"

EXPERIMENTS = [
    "olist_null_customer_id_001",
    "olist_duplicate_order_id_002",
    "olist_negative_price_003",
    "olist_invalid_customer_ref_004",
    "olist_schema_drift_005",
]

# Ground-truth and validation fields must never be
# included in agent-visible evidence.

PRIVATE_KEYS = {
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


def load_json(path):
    """
    Load a JSON file.

    Return None when the file does not exist.
    """

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def remove_private_keys(value):
    """
    Recursively remove private experimental fields.

    This prevents known ground-truth fields from
    appearing in the generated evidence bundles.
    """

    if isinstance(value, dict):
        return {
            key: remove_private_keys(item)
            for key, item in value.items()
            if key not in PRIVATE_KEYS
        }

    if isinstance(value, list):
        return [
            remove_private_keys(item)
            for item in value
        ]

    return value


def get_dq_evidence(experiment_id):
    """
    Load experimental data quality results.

    Experiments 001-004 have DQ reports.

    Experiment 005 uses schema-drift detection,
    so a standard experimental DQ report may
    not be available.
    """

    path = DQ_DIR / f"{experiment_id}_dq.json"

    report = load_json(path)

    if report is None:
        return {
            "available": False,
            "reason": (
                "No experimental data quality "
                "report exists."
            ),
        }

    return {
        "available": True,
        "source_file": str(path.relative_to(ROOT)),
        "table": report.get("table"),
        "check_result": remove_private_keys(
            report.get("check_result")
        ),
    }


def get_schema_evidence(experiment_id, truth):
    """
    Load baseline and experimental schema snapshots.

    The schema report contains:
        - Baseline schema
        - Experimental schema
        - Added columns
        - Removed columns
        - Data type changes
        - Schema drift status

    For Experiment 005, downstream compatibility
    test results are also included when available.

    No schema-drift result is invented when the
    report is missing.
    """

    path = SCHEMA_DIR / f"{experiment_id}_schema.json"

    report = load_json(path)

    if report is None:
        return {
            "available": False,
            "reason": (
                "No experimental schema "
                "comparison report exists."
            ),
        }

    return {
        "available": True,

        "source_file": str(
            path.relative_to(ROOT)
        ),

        "check_type": report.get(
            "check_type"
        ),

        "baseline_schema": report.get(
            "baseline_schema"
        ),

        "experimental_schema": report.get(
            "experimental_schema"
        ),

        "added_columns": report.get(
            "added_columns"
        ),

        "removed_columns": report.get(
            "removed_columns"
        ),

        "type_changes": report.get(
            "type_changes"
        ),

        "drift_detected": report.get(
            "drift_detected"
        ),

        "downstream_failure_detected": report.get(
            "downstream_failure_detected"
        ),

        "downstream_error": report.get(
            "downstream_error"
        ),
    }


def get_lineage_evidence(dataset):
    """
    Load declared table-level lineage information.

    These lineage relationships are declared
    dependencies, not automatically captured
    record-level provenance.
    """

    path = LINEAGE_DIR / f"{dataset}_lineage.json"

    report = load_json(path)

    if report is None:
        return {
            "available": False,
            "reason": (
                "No lineage report exists."
            ),
        }

    return {
        "available": True,

        "source_file": str(
            path.relative_to(ROOT)
        ),

        "capture_method": (
            "declared_table_level_lineage"
        ),

        "graph": remove_private_keys(
            report
        ),
    }


def get_monitoring_evidence(dataset):
    """
    Load baseline data quality monitoring.

    Important:
    These results describe the original dataset,
    not the experimentally modified dataset.
    """

    path = (
        MONITORING_DIR
        / f"{dataset}_monitoring.json"
    )

    report = load_json(path)

    if report is None:
        return {
            "available": False,
            "reason": (
                "No baseline monitoring "
                "report exists."
            ),
        }

    return {
        "available": True,

        "source_file": str(
            path.relative_to(ROOT)
        ),

        "scope": (
            "baseline_dataset_not_injected_experiment"
        ),

        "report": remove_private_keys(
            report
        ),
    }


def get_log_evidence(experiment_id):
    """
    Load experiment-specific execution logs.

    Missing logs remain unavailable.

    Historical log events must not be invented.
    """

    path = LOG_DIR / f"{experiment_id}.json"

    report = load_json(path)

    if report is None:
        return {
            "available": False,

            "reason": (
                "No experiment-specific "
                "execution log has been captured."
            ),
        }

    return {
        "available": True,

        "source_file": str(
            path.relative_to(ROOT)
        ),

        "report": remove_private_keys(
            report
        ),
    }


def build_bundle(experiment_id):
    """
    Build a standardized RCA evidence bundle.

    Ground truth is used only to identify
    the dataset and relevant source tables.

    Root-cause answers are not copied into
    the evidence bundle.
    """

    truth_path = (
        GROUND_TRUTH_DIR
        / f"{experiment_id}.json"
    )

    truth = load_json(truth_path)

    if truth is None:
        raise FileNotFoundError(
            truth_path
        )

    dataset = truth["dataset"]

    dq = get_dq_evidence(
        experiment_id
    )

    schema = get_schema_evidence(
        experiment_id,
        truth
    )

    lineage = get_lineage_evidence(
        dataset
    )

    monitoring = get_monitoring_evidence(
        dataset
    )

    logs = get_log_evidence(
        experiment_id
    )

    bundle = {
        "bundle_version": "1.1",

        "experiment_id": experiment_id,

        "dataset": dataset,

        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "context": {
            "source_table": truth.get(
                "source_table"
            ),

            "reference_table": truth.get(
                "reference_table"
            ),
        },

        "evidence": {
            "data_quality": dq,
            "schema": schema,
            "lineage": lineage,
            "logs": logs,
            "baseline_monitoring": monitoring,
        },

        "evidence_availability": {
            "data_quality": dq["available"],
            "schema": schema["available"],
            "lineage": lineage["available"],
            "logs": logs["available"],
            "baseline_monitoring": monitoring["available"],
        },

        "limitations": [
            (
                "Declared table-level lineage does "
                "not establish record-level provenance."
            ),

            (
                "Baseline monitoring results describe "
                "the original dataset, not the "
                "experimentally modified dataset."
            ),

            (
                "Unavailable evidence is explicitly "
                "marked and must not be invented."
            ),

            (
                "Schema comparisons identify structural "
                "differences but do not independently "
                "establish the root cause of every "
                "data quality failure."
            ),
        ],
    }

    return bundle


def validate_bundle(bundle):
    """
    Verify the evidence bundle structure.

    Check that:
        1. All required evidence sections exist.
        2. Availability flags are consistent.
        3. Known private ground-truth keys are absent.
    """

    required_sections = {
        "data_quality",
        "schema",
        "lineage",
        "logs",
        "baseline_monitoring",
    }

    actual_sections = set(
        bundle["evidence"]
    )

    if actual_sections != required_sections:
        raise AssertionError(
            "Missing or unexpected evidence sections"
        )

    for section in required_sections:
        evidence = bundle["evidence"][section]

        declared_available = (
            bundle["evidence_availability"][section]
        )

        actual_available = evidence.get(
            "available"
        )

        if declared_available != actual_available:
            raise AssertionError(
                f"Evidence availability mismatch: {section}"
            )

    def scan(value):
        if isinstance(value, dict):
            for key, item in value.items():

                if key in PRIVATE_KEYS:
                    raise AssertionError(
                        f"Private field leaked: {key}"
                    )

                scan(item)

        elif isinstance(value, list):
            for item in value:
                scan(item)

    scan(bundle)


def main():
    """
    Generate RCA evidence bundles for all
    five controlled failure experiments.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 65)
    print("STRUCTURED RCA EVIDENCE BUNDLE GENERATION")
    print("=" * 65)

    generated = 0

    for experiment_id in EXPERIMENTS:

        bundle = build_bundle(
            experiment_id
        )

        validate_bundle(
            bundle
        )

        output_path = (
            OUTPUT_DIR
            / f"{experiment_id}.json"
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                bundle,
                file,
                indent=2
            )

        available = [
            name
            for name, status
            in bundle["evidence_availability"].items()
            if status
        ]

        missing = [
            name
            for name, status
            in bundle["evidence_availability"].items()
            if not status
        ]

        print()
        print("Experiment:", experiment_id)

        print(
            "Available evidence:",
            available
        )

        print(
            "Missing evidence:",
            missing
        )

        print(
            "Bundle validation: PASS"
        )

        print(
            "Saved:",
            output_path
        )

        generated += 1

    print()
    print("=" * 65)

    print(
        "TOTAL EVIDENCE BUNDLES:",
        generated
    )

    print(
        "STATUS: COMPLETE"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()
