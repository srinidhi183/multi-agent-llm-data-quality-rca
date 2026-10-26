import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BUNDLE_DIR = ROOT / "evidence/rca_bundles"
OUTPUT_DIR = ROOT / "evidence/agent_views"
MANIFEST_DIR = ROOT / "experiments/evaluation"

EXPERIMENTS = [
    "olist_null_customer_id_001",
    "olist_duplicate_order_id_002",
    "olist_negative_price_003",
    "olist_invalid_customer_ref_004",
    "olist_schema_drift_005",
]

AGENT_EVIDENCE = {
    "dq_agent": ["data_quality"],
    "schema_agent": ["schema"],
    "lineage_agent": ["lineage"],
    "logs_agent": ["logs"],
    "combined_agent": [
        "data_quality",
        "schema",
        "lineage",
        "logs",
        "baseline_monitoring",
    ],
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

# Prevent descriptive experiment IDs and injection markers
# from appearing in agent-visible data.
FORBIDDEN_TEXT = [
    "olist_null_customer_id_001",
    "olist_duplicate_order_id_002",
    "olist_negative_price_003",
    "olist_invalid_customer_ref_004",
    "olist_schema_drift_005",
    "THESIS_INVALID_CUSTOMER_004_",
]


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def remove_private_keys(value):
    if isinstance(value, dict):
        return {
            key: remove_private_keys(item)
            for key, item in value.items()
            if key not in FORBIDDEN_KEYS
        }

    if isinstance(value, list):
        return [remove_private_keys(item) for item in value]

    return value


def sanitize_text(value):
    """
    Remove descriptive experiment identifiers from strings.

    Source paths containing experiment IDs are replaced with
    neutral references, while retaining other diagnostic content.
    """

    if isinstance(value, dict):
        return {
            key: sanitize_text(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [sanitize_text(item) for item in value]

    if isinstance(value, str):
        cleaned = value

        for experiment_id in FORBIDDEN_TEXT:
            cleaned = cleaned.replace(
                experiment_id,
                "[REDACTED_EXPERIMENT_IDENTIFIER]",
            )

        return cleaned

    return value


def validate_no_leakage(value):
    serialized = json.dumps(value, ensure_ascii=False)

    for term in FORBIDDEN_TEXT:
        if term in serialized:
            raise AssertionError(
                f"Experiment identifier leaked: {term}"
            )

    def scan(item):
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in FORBIDDEN_KEYS:
                    raise AssertionError(
                        f"Private field leaked: {key}"
                    )
                scan(nested)

        elif isinstance(item, list):
            for nested in item:
                scan(nested)

    scan(value)


def make_agent_view(bundle, case_id, agent_name):
    selected_sections = AGENT_EVIDENCE[agent_name]

    evidence = {}

    for section in selected_sections:
        if section not in bundle["evidence"]:
            raise AssertionError(
                f"Missing evidence section: {section}"
            )

        cleaned = remove_private_keys(
            bundle["evidence"][section]
        )

        evidence[section] = sanitize_text(cleaned)

    view = {
        "view_version": "1.0",
        "case_id": case_id,
        "agent_name": agent_name,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evidence": evidence,
        "instructions": (
            "Analyze only the supplied evidence. "
            "Identify plausible failure causes, "
            "distinguish observations from hypotheses, "
            "and state when evidence is insufficient. "
            "Do not invent missing logs, schema changes, "
            "or lineage relationships."
        ),
    }

    validate_no_leakage(view)

    return view


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "manifest_version": "1.0",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "cases": [],
    }

    total_views = 0

    print("=" * 65)
    print("AGENT-SPECIFIC EVIDENCE VIEW GENERATION")
    print("=" * 65)

    for index, experiment_id in enumerate(
        EXPERIMENTS, start=1
    ):
        case_id = f"case_{index:03d}"

        bundle_path = BUNDLE_DIR / f"{experiment_id}.json"
        bundle = load_json(bundle_path)

        if bundle["experiment_id"] != experiment_id:
            raise AssertionError(
                f"Bundle ID mismatch: {experiment_id}"
            )

        case_dir = OUTPUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        manifest["cases"].append({
            "case_id": case_id,
            "experiment_id": experiment_id,
            "ground_truth_file": (
                f"experiments/ground_truth/"
                f"{experiment_id}.json"
            ),
        })

        print()
        print("Case:", case_id)

        for agent_name in AGENT_EVIDENCE:
            view = make_agent_view(
                bundle,
                case_id,
                agent_name,
            )

            output_path = case_dir / f"{agent_name}.json"

            with output_path.open(
                "w", encoding="utf-8"
            ) as file:
                json.dump(view, file, indent=2)

            available = [
                section
                for section, content
                in view["evidence"].items()
                if content.get("available", False)
            ]

            print(
                f"  {agent_name:18} "
                f"AVAILABLE: {available}"
            )

            total_views += 1

    manifest_path = (
        MANIFEST_DIR / "case_manifest.json"
    )

    with manifest_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print()
    print("=" * 65)
    print("TOTAL CASES:", len(EXPERIMENTS))
    print("TOTAL AGENT VIEWS:", total_views)
    print("STATUS: COMPLETE")
    print("=" * 65)

    print("Manifest saved:", manifest_path)


if __name__ == "__main__":
    main()
