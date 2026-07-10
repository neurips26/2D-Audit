#!/usr/bin/env python3
"""
Apply explicit legacy metadata to the strict source-audit ledger and rebuild
authoritative accepted-only tables.

This script does not infer or guess metadata. It only applies values from
legacy_result_metadata_manifest.json when:
- the source basename matches exactly;
- the dataset is allowed by the manifest entry;
- the metric family is allowed by the manifest entry.

Inputs
------
outputs/revision/cross_dataset_source_audit/source_audit_ledger.csv
legacy_result_metadata_manifest.json

Outputs
-------
outputs/revision/cross_dataset_source_audit_v2/
    metadata_application_ledger.csv
    accepted_metrics.csv
    review_metrics.csv
    rejected_metrics.csv
    authoritative_long.csv
    authoritative_wide.csv
    audit_report.json
    audit_report.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from exp_config import ROOT
except Exception:
    ROOT = SCRIPT_DIR

DEFAULT_INPUT = (
    ROOT / "outputs" / "revision" / "cross_dataset_source_audit"
    / "source_audit_ledger.csv"
)
DEFAULT_MANIFEST = ROOT / "legacy_result_metadata_manifest.json"
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "revision" / "cross_dataset_source_audit_v2"
)

CORE_METHODS = ["ga", "npo", "mmunlearner", "cagul"]
TARGET_DATASETS = ["mllmu_real", "mllmu_bench", "unlok_vqa"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: "" if row.get(field) is None else row.get(field)
                for field in fields
            })


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ValueError("Manifest must contain a top-level 'files' object.")
    return data


def apply_manifest(
    row: dict[str, str],
    manifest_files: dict[str, Any],
) -> tuple[dict[str, str], bool, str]:
    updated = dict(row)
    basename = Path(row["source_file"]).name
    entry = manifest_files.get(basename)

    if not entry:
        return updated, False, "No exact manifest entry for this source basename."

    allowed_datasets = set(entry.get("allowed_datasets", []))
    if allowed_datasets and row.get("dataset") not in allowed_datasets:
        return updated, False, (
            f"Manifest entry does not allow dataset {row.get('dataset')}."
        )

    allowed_families = set(entry.get("allowed_metric_families", []))
    if allowed_families and row.get("metric_family") not in allowed_families:
        return updated, False, (
            f"Manifest entry does not allow metric family "
            f"{row.get('metric_family')}."
        )

    applied = []

    if not updated.get("model"):
        updated["model"] = str(entry["model"])
        applied.append("model")

    if not updated.get("seed"):
        updated["seed"] = str(entry["seed"])
        applied.append("seed")

    if not updated.get("checkpoint"):
        template = str(entry["checkpoint_template"])
        updated["checkpoint"] = template.format(
            dataset=updated.get("dataset", ""),
            method=updated.get("method", ""),
            model=updated.get("model", ""),
            seed=updated.get("seed", ""),
        )
        applied.append("checkpoint")

    if applied:
        updated["metadata_source"] = basename
        updated["metadata_fields_applied"] = ",".join(applied)
        return updated, True, "Applied explicit legacy manifest metadata."

    return updated, False, "Manifest matched, but no fields required filling."


def reclassify(row: dict[str, str]) -> tuple[str, str]:
    # Preserve hard rejects from the strict audit.
    if row.get("status") == "REJECT":
        return "REJECT", row.get("reason", "Rejected by strict audit.")

    required = ["dataset", "model", "method", "seed", "checkpoint"]
    missing = [field for field in required if not row.get(field)]

    if missing:
        return "REVIEW", "Missing required metadata: " + ", ".join(missing)

    if row.get("metric_family") not in {"behavioural", "crp", "semantic"}:
        return "REJECT", "Unsupported metric family."

    if not row.get("metric_name") or row.get("metric_value") == "":
        return "REJECT", "Metric name or value missing."

    return "ACCEPT", "Complete explicit experiment identity."


def experiment_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["dataset"],
        row["model"],
        row["method"],
        row["seed"],
        row["checkpoint"],
        row["metric_family"],
        row["metric_name"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    manifest_path = Path(args.manifest).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing strict audit ledger: {input_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing metadata manifest: {manifest_path}")

    rows = read_csv(input_path)
    manifest = load_manifest(manifest_path)
    manifest_files = manifest["files"]

    processed = []
    applied_count = 0

    for row in rows:
        updated, applied, note = apply_manifest(row, manifest_files)
        if applied:
            applied_count += 1

        status, reason = reclassify(updated)
        updated["status_v2"] = status
        updated["reason_v2"] = reason
        updated["manifest_note"] = note
        processed.append(updated)

    # Resolve exact duplicate accepted metrics deterministically.
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in processed:
        if row["status_v2"] == "ACCEPT":
            grouped[experiment_key(row)].append(row)

    authoritative = []
    duplicate_conflicts = 0
    duplicate_redundant = 0

    for key, candidates in grouped.items():
        if len(candidates) == 1:
            authoritative.append(candidates[0])
            continue

        values = {
            round(float(candidate["metric_value"]), 12)
            for candidate in candidates
        }

        ordered = sorted(
            candidates,
            key=lambda row: (
                0 if row.get("metadata_source") else 1,
                len(row["source_file"]),
                row["source_file"],
                row.get("source_container", ""),
            ),
        )

        if len(values) == 1:
            canonical = ordered[0]
            authoritative.append(canonical)
            for row in ordered[1:]:
                row["status_v2"] = "REJECT"
                row["reason_v2"] = (
                    "Redundant duplicate of accepted authoritative metric."
                )
                duplicate_redundant += 1
        else:
            for row in ordered:
                row["status_v2"] = "REVIEW"
                row["reason_v2"] = (
                    "Conflicting values for identical experiment identity."
                )
            duplicate_conflicts += 1

    accepted = [row for row in processed if row["status_v2"] == "ACCEPT"]
    review = [row for row in processed if row["status_v2"] == "REVIEW"]
    rejected = [row for row in processed if row["status_v2"] == "REJECT"]
    authoritative = [
        row for row in authoritative
        if row["status_v2"] == "ACCEPT"
    ]

    fields = list(dict.fromkeys(
        list(rows[0].keys())
        + [
            "metadata_source",
            "metadata_fields_applied",
            "manifest_note",
            "status_v2",
            "reason_v2",
        ]
    )) if rows else []

    write_csv(output / "metadata_application_ledger.csv", processed, fields)
    write_csv(output / "accepted_metrics.csv", accepted, fields)
    write_csv(output / "review_metrics.csv", review, fields)
    write_csv(output / "rejected_metrics.csv", rejected, fields)
    write_csv(output / "authoritative_long.csv", authoritative, fields)

    # Wide table with no averaging.
    by_experiment: dict[tuple[str, ...], dict[str, Any]] = {}
    metric_columns = sorted({
        f"{row['metric_family']}__{row['metric_name']}"
        for row in authoritative
    })

    for row in authoritative:
        key = (
            row["dataset"],
            row["model"],
            row["method"],
            row["seed"],
            row["checkpoint"],
        )
        record = by_experiment.setdefault(key, {
            "dataset": row["dataset"],
            "model": row["model"],
            "method": row["method"],
            "seed": row["seed"],
            "checkpoint": row["checkpoint"],
            "source_files": set(),
        })
        column = f"{row['metric_family']}__{row['metric_name']}"
        record[column] = row["metric_value"]
        record["source_files"].add(row["source_file"])

    wide = []
    for record in by_experiment.values():
        item = dict(record)
        item["source_files"] = " | ".join(sorted(item["source_files"]))
        wide.append(item)

    wide.sort(key=lambda row: (
        row["dataset"],
        row["model"],
        row["method"],
        row["seed"],
        row["checkpoint"],
    ))

    write_csv(
        output / "authoritative_wide.csv",
        wide,
        [
            "dataset", "model", "method", "seed", "checkpoint",
            *metric_columns, "source_files",
        ],
    )

    coverage = {}
    for dataset in TARGET_DATASETS:
        coverage[dataset] = {}
        for method in CORE_METHODS:
            subset = [
                row for row in authoritative
                if row["dataset"] == dataset and row["method"] == method
            ]
            families = sorted({row["metric_family"] for row in subset})
            coverage[dataset][method] = {
                "rows": len(subset),
                "families": families,
                "has_behavioural": "behavioural" in families,
                "has_crp": "crp" in families,
                "has_semantic": "semantic" in families,
            }

    complete_behavioural = [
        method for method in CORE_METHODS
        if all(
            coverage[dataset][method]["has_behavioural"]
            for dataset in TARGET_DATASETS
        )
    ]

    report = {
        "input_ledger": str(input_path),
        "manifest": str(manifest_path),
        "rows_input": len(rows),
        "rows_with_manifest_metadata_applied": applied_count,
        "accepted_rows": len(accepted),
        "review_rows": len(review),
        "rejected_rows": len(rejected),
        "authoritative_rows": len(authoritative),
        "redundant_duplicates_removed": duplicate_redundant,
        "conflicting_duplicate_groups": duplicate_conflicts,
        "coverage": coverage,
        "core_methods_with_behavioural_results_on_all_three_datasets": (
            complete_behavioural
        ),
        "files": {
            "ledger": str(output / "metadata_application_ledger.csv"),
            "accepted": str(output / "accepted_metrics.csv"),
            "review": str(output / "review_metrics.csv"),
            "rejected": str(output / "rejected_metrics.csv"),
            "authoritative_long": str(output / "authoritative_long.csv"),
            "authoritative_wide": str(output / "authoritative_wide.csv"),
        },
    }

    if not authoritative:
        verdict = "FAIL"
        reason = "No authoritative rows remained after manifest application."
        exit_code = 1
    elif review:
        verdict = "WARN"
        reason = (
            "Authoritative rows were created, but unresolved review rows remain."
        )
        exit_code = 2
    elif not complete_behavioural:
        verdict = "WARN"
        reason = (
            "Metadata repair succeeded, but behavioural coverage is still "
            "incomplete across all three datasets."
        )
        exit_code = 2
    else:
        verdict = "PASS"
        reason = (
            "Explicit legacy metadata produced a complete authoritative audit."
        )
        exit_code = 0

    report["overall_verdict"] = verdict
    report["reason"] = reason

    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "LEGACY METADATA AUDIT V2",
        "=" * 92,
        f"Input rows: {len(rows)}",
        f"Rows updated from explicit manifest: {applied_count}",
        f"Accepted rows: {len(accepted)}",
        f"Review rows: {len(review)}",
        f"Rejected rows: {len(rejected)}",
        f"Authoritative rows: {len(authoritative)}",
        f"Redundant duplicates removed: {duplicate_redundant}",
        f"Conflicting duplicate groups: {duplicate_conflicts}",
        "",
        "CORE COVERAGE",
        "-" * 92,
    ]

    for dataset in TARGET_DATASETS:
        for method in CORE_METHODS:
            c = coverage[dataset][method]
            lines.append(
                f"{dataset} / {method}: "
                f"behavioural={'YES' if c['has_behavioural'] else 'NO'}, "
                f"crp={'YES' if c['has_crp'] else 'NO'}, "
                f"semantic={'YES' if c['has_semantic'] else 'NO'}, "
                f"rows={c['rows']}"
            )

    lines.extend([
        "",
        "CORE METHODS WITH BEHAVIOURAL COVERAGE ON ALL THREE DATASETS",
        "-" * 92,
        ", ".join(complete_behavioural) or "none",
        "",
        f"LEDGER: {output / 'metadata_application_ledger.csv'}",
        f"AUTHORITATIVE LONG: {output / 'authoritative_long.csv'}",
        f"AUTHORITATIVE WIDE: {output / 'authoritative_wide.csv'}",
        f"REVIEW: {output / 'review_metrics.csv'}",
        f"REASON: {reason}",
        f"OVERALL VERDICT: {verdict}",
    ])

    (output / "audit_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
