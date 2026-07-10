#!/usr/bin/env python3
"""
Strict source audit for cross-dataset unlearning results.

Goal
----
Turn the broad discovery inventory into an auditable source-level ledger.

The script:
- reads matched_results_long.csv and source_inventory.csv;
- re-opens every matched JSON source;
- classifies each source as behavioural, CRP/CKA, CNIS, mixed, prediction dump,
  duplicate, or unknown;
- extracts dataset, model, method, seed, checkpoint, metric family, metric value;
- detects exact and near duplicates;
- marks each extracted metric row ACCEPT / REVIEW / REJECT;
- builds an authoritative accepted-only table without cross-checkpoint averaging.

No values are invented and no heterogeneous files are averaged.

Outputs
-------
outputs/revision/cross_dataset_source_audit/
    source_audit_ledger.csv
    accepted_metrics.csv
    review_metrics.csv
    rejected_metrics.csv
    duplicate_groups.csv
    authoritative_long.csv
    authoritative_wide.csv
    audit_report.json
    audit_report.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    from exp_config import ROOT
except Exception:
    ROOT = SCRIPT_DIR

DEFAULT_DISCOVERY = ROOT / "outputs" / "revision" / "matched_cross_dataset"
DEFAULT_OUTPUT = ROOT / "outputs" / "revision" / "cross_dataset_source_audit"

TARGET_DATASETS = ["mllmu_real", "mllmu_bench", "unlok_vqa"]
CORE_METHODS = ["ga", "npo", "mmunlearner", "cagul"]
OPTIONAL_METHODS = ["sineproject", "manu", "graddiff"]

DATASET_ALIASES = {
    "mllmu_real": ["mllmu_real", "mllmu-real", "mllmureal", "real_mllmu"],
    "mllmu_bench": ["mllmu_bench", "mllmu-bench", "mllmubench", "mllmu"],
    "unlok_vqa": ["unlok_vqa", "unlok-vqa", "unlokvqa", "unlok"],
}

METHOD_ALIASES = {
    "ga": ["gradient_ascent", "gradient-ascent", "ga"],
    "npo": ["npo"],
    "mmunlearner": ["mmunlearner", "mm_unlearner", "mm-unlearner"],
    "cagul": ["cagul"],
    "sineproject": ["sineproject", "sine_project", "sine-project"],
    "manu": ["manu"],
    "graddiff": ["graddiff", "grad_diff", "grad-diff"],
    "original": ["original", "orig", "base", "no_unlearn", "no-unlearn"],
    "retrain": ["retrain", "full_retrain", "full-retrain"],
}

MODEL_ALIASES = {
    "llava-1.5-7b": [
        "llava-1.5-7b", "llava_1_5_7b", "llava15_7b",
        "llava-hf/llava-1.5-7b-hf", "llava",
    ],
    "blip2-opt-2.7b": [
        "blip-2-opt-2.7b", "blip2-opt-2.7b", "blip2_opt_2_7b",
        "blip2", "blip-2",
    ],
}

METRICS = {
    "behavioural": {
        "forget_accuracy": [
            "forget_accuracy", "forget_acc", "forgetacc", "forget_score",
        ],
        "retain_accuracy": [
            "retain_accuracy", "retain_acc", "retainacc", "retain_score",
        ],
        "mia": [
            "mia", "mia_accuracy", "mia_score",
            "membership_inference", "membership_inference_accuracy",
        ],
        "ad": ["ad", "answer_distance", "answer_dist"],
        "js": ["js", "js_divergence", "jensen_shannon"],
    },
    "crp": {
        "cka_vision": [
            "ve_cka", "vision_cka", "cka_vision", "vision_encoder_cka",
        ],
        "cka_bridge": [
            "bridge_cka", "br_cka", "cka_bridge", "projector_cka",
        ],
        "cka_language": [
            "lb_cka", "language_cka", "cka_language", "decoder_cka",
        ],
    },
    "semantic": {
        "cnis": [
            "cnis", "semantic_neighborhood", "semantic_neighbourhood",
        ],
    },
}

GENERIC_RESULT_CONTAINERS = {
    "results", "rows", "methods", "experiments", "runs", "summary", "metrics",
}

PREDICTION_HINTS = {
    "response", "prediction", "generated_text", "expected_answer",
    "question", "image_used", "correct",
}

REJECT_PATH_HINTS = [
    "_runner", "crash_diagnostics", "fiubench", "prediction",
    "outputs.json", "malformed", "inventory", "matched_cross_dataset",
    "cross_dataset_source_audit",
]

SEED_KEYS = ["seed", "random_seed", "rng_seed"]
CHECKPOINT_KEYS = [
    "checkpoint", "checkpoint_path", "adapter_path", "model_path",
    "output_checkpoint", "ckpt",
]
MODEL_KEYS = ["model", "model_name", "base_model", "architecture"]
DATASET_KEYS = ["dataset", "dataset_name", "benchmark", "data_name"]
METHOD_KEYS = ["method", "method_name", "algorithm", "unlearning_method"]


def normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def flatten(obj: Any, prefix: str = "", out: dict[str, Any] | None = None) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            flatten(value, next_key, out)
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(
            isinstance(x, (str, int, float, bool, type(None))) for x in obj
        ):
            out[prefix] = obj
    else:
        out[prefix] = obj

    return out


def iter_candidates(data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        yield "root", data
        for key, value in data.items():
            if normalize(key) not in GENERIC_RESULT_CONTAINERS:
                continue
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        yield f"{key}[{index}]", item
            elif isinstance(value, dict):
                for child_key, item in value.items():
                    if isinstance(item, dict):
                        candidate = dict(item)
                        candidate.setdefault("_container_key", child_key)
                        yield f"{key}.{child_key}", candidate
    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                yield f"[{index}]", item


def canonical_from_aliases(
    values: Iterable[Any],
    aliases: dict[str, list[str]],
    ordered: list[str] | None = None,
) -> str | None:
    joined = normalize(" ".join(str(v or "") for v in values))
    keys = ordered or list(aliases)

    candidates = []
    for canonical in keys:
        for alias in aliases[canonical]:
            alias_n = normalize(alias)
            candidates.append((len(alias_n), canonical, alias_n))

    for _, canonical, alias in sorted(candidates, reverse=True):
        if re.search(rf"(^|_){re.escape(alias)}($|_)", joined):
            return canonical
    return None


def find_leaf(flattened: dict[str, Any], keys: Iterable[str]) -> Any:
    wanted = {normalize(key) for key in keys}
    for full_key, value in flattened.items():
        if normalize(full_key.split(".")[-1]) in wanted:
            return value
    return None


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    text = str(value).strip()
    is_percent = text.endswith("%")
    text = text.rstrip("%").strip()
    try:
        result = float(text)
    except ValueError:
        return None

    if is_percent:
        result /= 100.0
    return result if math.isfinite(result) else None


def infer_seed(flattened: dict[str, Any], path: Path) -> int | None:
    value = find_leaf(flattened, SEED_KEYS)
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass

    match = re.search(r"(?:seed|s)[_\-]?(\d{1,6})", str(path), re.I)
    return int(match.group(1)) if match else None


def infer_checkpoint(flattened: dict[str, Any], path: Path) -> str:
    value = find_leaf(flattened, CHECKPOINT_KEYS)
    if value:
        return str(value)

    parts = [part for part in path.parts if "checkpoint" in part.casefold()]
    if parts:
        return parts[-1]
    return ""


def classify_prediction_dump(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False
    sample = [item for item in data[:5] if isinstance(item, dict)]
    if not sample:
        return False
    hits = 0
    for item in sample:
        keys = {normalize(key) for key in item}
        if len(keys & PREDICTION_HINTS) >= 3:
            hits += 1
    return hits >= max(1, len(sample) // 2)


def metric_candidates(
    flattened: dict[str, Any],
) -> list[tuple[str, str, float, str]]:
    output = []
    for family, metric_map in METRICS.items():
        for canonical_metric, aliases in metric_map.items():
            accepted = {normalize(alias) for alias in aliases}
            choices = []
            for full_key, value in flattened.items():
                leaf = normalize(full_key.split(".")[-1])
                full_n = normalize(full_key)
                score = 0
                if leaf in accepted:
                    score = 100
                else:
                    for alias in accepted:
                        if full_n.endswith("_" + alias):
                            score = max(score, 90)
                        elif re.search(rf"(^|_){re.escape(alias)}($|_)", full_n):
                            score = max(score, 70)
                if score:
                    number = numeric(value)
                    if number is not None:
                        choices.append((score, len(full_n), number, full_key))
            if choices:
                choices.sort(key=lambda x: (-x[0], x[1], x[3]))
                _, _, number, source_key = choices[0]
                output.append((family, canonical_metric, number, source_key))
    return output


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def metric_signature(row: dict[str, Any]) -> str:
    raw = "|".join([
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("method", "")),
        str(row.get("seed", "")),
        str(row.get("checkpoint", "")),
        str(row.get("metric_family", "")),
        str(row.get("metric_name", "")),
        f"{float(row.get('metric_value', 0.0)):.12g}",
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def determine_status(row: dict[str, Any]) -> tuple[str, str]:
    path_n = normalize(row["source_file"])

    if any(normalize(hint) in path_n for hint in REJECT_PATH_HINTS):
        return "REJECT", "Path indicates runner, diagnostic, prediction, FIUBench, or derived audit output."

    if row["dataset"] not in TARGET_DATASETS:
        return "REJECT", "Not one of the three target datasets."

    if not row["method"]:
        return "REJECT", "Method could not be inferred."

    if row["metric_family"] not in {"behavioural", "crp", "semantic"}:
        return "REJECT", "Unsupported metric family."

    value = row["metric_value"]
    if row["metric_name"] in {
        "forget_accuracy", "retain_accuracy", "mia",
        "cka_vision", "cka_bridge", "cka_language", "cnis",
    } and not (0.0 <= value <= 1.0):
        return "REJECT", "Metric expected in [0,1] but value is outside range."

    if not row["model"]:
        return "REVIEW", "Model is missing or ambiguous."

    if row["seed"] is None:
        return "REVIEW", "Seed is missing."

    if not row["checkpoint"]:
        return "REVIEW", "Checkpoint identity is missing."

    return "ACCEPT", "Dataset, model, method, seed, checkpoint, and metric are explicit."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", default=str(DEFAULT_DISCOVERY))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    discovery = Path(args.discovery).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    inventory_file = discovery / "source_inventory.csv"
    if not inventory_file.exists():
        raise FileNotFoundError(
            f"Missing discovery inventory: {inventory_file}. "
            "Run the matched cross-dataset collector first."
        )

    inventory = read_csv(inventory_file)
    matched_sources = sorted({
        Path(row["source_file"]).resolve()
        for row in inventory
        if row.get("status") == "matched" and row.get("source_file")
    })

    ledger: list[dict[str, Any]] = []
    file_hashes: dict[str, list[str]] = defaultdict(list)
    parse_errors = []

    for path in matched_sources:
        if not path.exists():
            parse_errors.append({
                "source_file": str(path),
                "error": "Source file no longer exists.",
            })
            continue

        sha256 = hash_file(path)
        file_hashes[sha256].append(str(path))

        try:
            data = read_json(path)
        except Exception as exc:
            parse_errors.append({
                "source_file": str(path),
                "error": repr(exc),
            })
            continue

        if classify_prediction_dump(data):
            ledger.append({
                "dataset": canonical_from_aliases(
                    [path], DATASET_ALIASES,
                    ["mllmu_real", "mllmu_bench", "unlok_vqa"],
                ) or "",
                "model": canonical_from_aliases([path], MODEL_ALIASES) or "",
                "method": canonical_from_aliases([path], METHOD_ALIASES) or "",
                "seed": "",
                "checkpoint": "",
                "metric_family": "prediction_dump",
                "metric_name": "",
                "metric_value": "",
                "source_key": "",
                "source_container": "root",
                "source_file": str(path),
                "file_sha256": sha256,
                "metric_signature": "",
                "duplicate_group": "",
                "status": "REJECT",
                "reason": "Prediction-level output file, not aggregate metrics.",
            })
            continue

        found_any = False

        for container, candidate in iter_candidates(data):
            flat = flatten(candidate)
            dataset = canonical_from_aliases(
                [
                    find_leaf(flat, DATASET_KEYS),
                    candidate.get("_container_key", ""),
                    path,
                ],
                DATASET_ALIASES,
                ["mllmu_real", "mllmu_bench", "unlok_vqa"],
            )
            method = canonical_from_aliases(
                [
                    find_leaf(flat, METHOD_KEYS),
                    candidate.get("_container_key", ""),
                    path,
                ],
                METHOD_ALIASES,
            )
            model = canonical_from_aliases(
                [
                    find_leaf(flat, MODEL_KEYS),
                    find_leaf(flat, CHECKPOINT_KEYS),
                    path,
                ],
                MODEL_ALIASES,
            )
            seed = infer_seed(flat, path)
            checkpoint = infer_checkpoint(flat, path)

            for family, metric_name, value, source_key in metric_candidates(flat):
                found_any = True
                row = {
                    "dataset": dataset or "",
                    "model": model or "",
                    "method": method or "",
                    "seed": seed,
                    "checkpoint": checkpoint,
                    "metric_family": family,
                    "metric_name": metric_name,
                    "metric_value": value,
                    "source_key": source_key,
                    "source_container": container,
                    "source_file": str(path),
                    "file_sha256": sha256,
                }
                row["metric_signature"] = metric_signature(row)
                row["duplicate_group"] = ""
                status, reason = determine_status(row)
                row["status"] = status
                row["reason"] = reason
                ledger.append(row)

        if not found_any:
            ledger.append({
                "dataset": canonical_from_aliases(
                    [path], DATASET_ALIASES,
                    ["mllmu_real", "mllmu_bench", "unlok_vqa"],
                ) or "",
                "model": canonical_from_aliases([path], MODEL_ALIASES) or "",
                "method": canonical_from_aliases([path], METHOD_ALIASES) or "",
                "seed": "",
                "checkpoint": "",
                "metric_family": "unknown",
                "metric_name": "",
                "metric_value": "",
                "source_key": "",
                "source_container": "root",
                "source_file": str(path),
                "file_sha256": sha256,
                "metric_signature": "",
                "duplicate_group": "",
                "status": "REJECT",
                "reason": "No supported aggregate metrics found.",
            })

    # Mark exact file duplicates.
    duplicate_groups = []
    file_duplicate_id = {}
    group_counter = 1
    for sha256, files in sorted(file_hashes.items()):
        if len(files) <= 1:
            continue
        group_id = f"FILE_DUP_{group_counter:03d}"
        group_counter += 1
        for file in files:
            file_duplicate_id[file] = group_id
        duplicate_groups.append({
            "duplicate_group": group_id,
            "duplicate_type": "exact_file",
            "sha256_or_signature": sha256,
            "count": len(files),
            "files": " | ".join(sorted(files)),
        })

    # Mark metric duplicates.
    by_signature: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ledger):
        if row.get("metric_signature"):
            by_signature[row["metric_signature"]].append(index)

    metric_group_counter = 1
    for signature, indexes in sorted(by_signature.items()):
        if len(indexes) <= 1:
            continue
        group_id = f"METRIC_DUP_{metric_group_counter:03d}"
        metric_group_counter += 1

        files = sorted({ledger[index]["source_file"] for index in indexes})
        duplicate_groups.append({
            "duplicate_group": group_id,
            "duplicate_type": "same_metric_identity_and_value",
            "sha256_or_signature": signature,
            "count": len(indexes),
            "files": " | ".join(files),
        })

        # Keep one canonical row, reject the rest.
        indexes_sorted = sorted(
            indexes,
            key=lambda i: (
                0 if ledger[i]["status"] == "ACCEPT" else 1,
                len(ledger[i]["source_file"]),
                ledger[i]["source_file"],
            ),
        )
        canonical_index = indexes_sorted[0]

        for index in indexes_sorted:
            ledger[index]["duplicate_group"] = group_id
            if index != canonical_index:
                ledger[index]["status"] = "REJECT"
                ledger[index]["reason"] = (
                    f"Duplicate of canonical metric row in {group_id}."
                )

    # Exact duplicate files should not both contribute.
    for row in ledger:
        file_group = file_duplicate_id.get(row["source_file"])
        if not file_group:
            continue
        if row["duplicate_group"]:
            row["duplicate_group"] += f";{file_group}"
        else:
            row["duplicate_group"] = file_group

    accepted = [row for row in ledger if row["status"] == "ACCEPT"]
    review = [row for row in ledger if row["status"] == "REVIEW"]
    rejected = [row for row in ledger if row["status"] == "REJECT"]

    # Enforce one authoritative value per dataset/model/method/seed/checkpoint/metric.
    authoritative_index: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        key = (
            row["dataset"],
            row["model"],
            row["method"],
            row["seed"],
            row["checkpoint"],
            row["metric_family"],
            row["metric_name"],
        )
        authoritative_index[key].append(row)

    authoritative = []
    for key, rows in authoritative_index.items():
        if len(rows) == 1:
            authoritative.append(rows[0])
            continue

        values = {round(float(row["metric_value"]), 12) for row in rows}
        if len(values) == 1:
            canonical = sorted(rows, key=lambda r: r["source_file"])[0]
            authoritative.append(canonical)
            for row in rows:
                if row is not canonical:
                    row["status"] = "REJECT"
                    row["reason"] = "Redundant authoritative duplicate."
        else:
            for row in rows:
                row["status"] = "REVIEW"
                row["reason"] = (
                    "Conflicting values for the same dataset/model/method/"
                    "seed/checkpoint/metric identity."
                )

    # Recompute final status lists after conflict handling.
    accepted = [row for row in ledger if row["status"] == "ACCEPT"]
    review = [row for row in ledger if row["status"] == "REVIEW"]
    rejected = [row for row in ledger if row["status"] == "REJECT"]

    authoritative = [
        row for row in authoritative
        if row["status"] == "ACCEPT"
    ]

    ledger_fields = [
        "dataset", "model", "method", "seed", "checkpoint",
        "metric_family", "metric_name", "metric_value",
        "source_key", "source_container", "source_file",
        "file_sha256", "metric_signature", "duplicate_group",
        "status", "reason",
    ]

    write_csv(output / "source_audit_ledger.csv", ledger, ledger_fields)
    write_csv(output / "accepted_metrics.csv", accepted, ledger_fields)
    write_csv(output / "review_metrics.csv", review, ledger_fields)
    write_csv(output / "rejected_metrics.csv", rejected, ledger_fields)
    write_csv(
        output / "duplicate_groups.csv",
        duplicate_groups,
        [
            "duplicate_group", "duplicate_type",
            "sha256_or_signature", "count", "files",
        ],
    )

    # Long authoritative table, no averaging.
    authoritative_rows = sorted(
        authoritative,
        key=lambda row: (
            row["dataset"], row["model"], row["method"],
            str(row["seed"]), row["checkpoint"],
            row["metric_family"], row["metric_name"],
        ),
    )
    write_csv(
        output / "authoritative_long.csv",
        authoritative_rows,
        ledger_fields,
    )

    # Wide table keyed by exact experimental identity.
    by_experiment: dict[tuple[Any, ...], dict[str, Any]] = {}
    all_metric_columns = sorted({
        f"{row['metric_family']}__{row['metric_name']}"
        for row in authoritative_rows
    })

    for row in authoritative_rows:
        key = (
            row["dataset"], row["model"], row["method"],
            row["seed"], row["checkpoint"],
        )
        record = by_experiment.setdefault(key, {
            "dataset": row["dataset"],
            "model": row["model"],
            "method": row["method"],
            "seed": row["seed"],
            "checkpoint": row["checkpoint"],
            "source_files": set(),
        })
        metric_column = f"{row['metric_family']}__{row['metric_name']}"
        record[metric_column] = row["metric_value"]
        record["source_files"].add(row["source_file"])

    authoritative_wide = []
    for record in by_experiment.values():
        record = dict(record)
        record["source_files"] = " | ".join(sorted(record["source_files"]))
        authoritative_wide.append(record)

    authoritative_wide.sort(
        key=lambda row: (
            row["dataset"], row["model"], row["method"],
            str(row["seed"]), row["checkpoint"],
        )
    )
    write_csv(
        output / "authoritative_wide.csv",
        authoritative_wide,
        [
            "dataset", "model", "method", "seed", "checkpoint",
            *all_metric_columns, "source_files",
        ],
    )

    core_coverage = {}
    for dataset in TARGET_DATASETS:
        core_coverage[dataset] = {}
        for method in CORE_METHODS:
            rows = [
                row for row in authoritative_rows
                if row["dataset"] == dataset and row["method"] == method
            ]
            families = sorted({row["metric_family"] for row in rows})
            core_coverage[dataset][method] = {
                "rows": len(rows),
                "families": families,
                "has_behavioural": "behavioural" in families,
                "has_crp": "crp" in families,
                "has_semantic": "semantic" in families,
            }

    complete_behavioural_core = [
        method for method in CORE_METHODS
        if all(
            core_coverage[dataset][method]["has_behavioural"]
            for dataset in TARGET_DATASETS
        )
    ]

    report = {
        "matched_source_files_input": len(matched_sources),
        "ledger_rows": len(ledger),
        "accepted_rows": len(accepted),
        "review_rows": len(review),
        "rejected_rows": len(rejected),
        "authoritative_rows": len(authoritative_rows),
        "duplicate_groups": len(duplicate_groups),
        "parse_errors": parse_errors,
        "core_coverage": core_coverage,
        "core_methods_with_behavioural_results_on_all_three_datasets": (
            complete_behavioural_core
        ),
        "files": {
            "ledger": str(output / "source_audit_ledger.csv"),
            "accepted": str(output / "accepted_metrics.csv"),
            "review": str(output / "review_metrics.csv"),
            "rejected": str(output / "rejected_metrics.csv"),
            "duplicates": str(output / "duplicate_groups.csv"),
            "authoritative_long": str(output / "authoritative_long.csv"),
            "authoritative_wide": str(output / "authoritative_wide.csv"),
        },
    }

    if not authoritative_rows:
        verdict = "FAIL"
        reason = "No authoritative metric rows passed the strict audit."
        exit_code = 1
    elif review:
        verdict = "WARN"
        reason = (
            "Authoritative rows were produced, but some sources still require review."
        )
        exit_code = 2
    elif not complete_behavioural_core:
        verdict = "WARN"
        reason = (
            "Audit completed, but no core method has behavioural coverage "
            "on all three datasets."
        )
        exit_code = 2
    else:
        verdict = "PASS"
        reason = (
            "Strict source audit completed with behavioural core coverage "
            "on all three datasets."
        )
        exit_code = 0

    report["overall_verdict"] = verdict
    report["reason"] = reason

    (output / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "CROSS-DATASET STRICT SOURCE AUDIT",
        "=" * 92,
        f"Matched source files audited: {len(matched_sources)}",
        f"Ledger rows: {len(ledger)}",
        f"Accepted rows: {len(accepted)}",
        f"Review rows: {len(review)}",
        f"Rejected rows: {len(rejected)}",
        f"Authoritative rows: {len(authoritative_rows)}",
        f"Duplicate groups: {len(duplicate_groups)}",
        "",
        "CORE BEHAVIOURAL COVERAGE",
        "-" * 92,
    ]

    for dataset in TARGET_DATASETS:
        for method in CORE_METHODS:
            coverage = core_coverage[dataset][method]
            lines.append(
                f"{dataset} / {method}: "
                f"behavioural={'YES' if coverage['has_behavioural'] else 'NO'}, "
                f"crp={'YES' if coverage['has_crp'] else 'NO'}, "
                f"semantic={'YES' if coverage['has_semantic'] else 'NO'}, "
                f"rows={coverage['rows']}"
            )

    lines.extend([
        "",
        "CORE METHODS WITH BEHAVIOURAL COVERAGE ON ALL THREE DATASETS",
        "-" * 92,
        ", ".join(complete_behavioural_core) or "none",
        "",
        f"LEDGER: {output / 'source_audit_ledger.csv'}",
        f"REVIEW: {output / 'review_metrics.csv'}",
        f"AUTHORITATIVE LONG: {output / 'authoritative_long.csv'}",
        f"AUTHORITATIVE WIDE: {output / 'authoritative_wide.csv'}",
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
