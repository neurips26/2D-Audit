#!/usr/bin/env python3
"""
Targeted locator for missing cross-dataset result sources.

Searches the project tree specifically for:
1. mllmu_real behavioural metrics:
   forget accuracy, retain accuracy, MIA, AD, JS
2. mllmu_bench behavioural metrics:
   forget accuracy, retain accuracy, MIA, AD, JS
3. mllmu_bench CRP metrics:
   vision CKA, bridge CKA, language CKA

The script does not decide which result is correct. It inventories exact files,
schemas, values, methods, seeds, checkpoints, and ambiguity indicators so the
authoritative sources can be mapped explicitly.

Outputs
-------
outputs/revision/missing_result_locator/
    candidate_files.csv
    candidate_metrics.csv
    candidate_schemas.json
    exact_target_coverage.csv
    locator_report.json
    locator_report.txt
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_OUTPUT = ROOT / "outputs" / "revision" / "missing_result_locator"
DEFAULT_ROOTS = [
    ROOT / "outputs",
    ROOT / "results",
    ROOT / "checkpoints",
]

TARGETS = {
    "mllmu_real": {
        "behavioural": [
            "forget_accuracy",
            "retain_accuracy",
            "mia",
            "ad",
            "js",
        ],
    },
    "mllmu_bench": {
        "behavioural": [
            "forget_accuracy",
            "retain_accuracy",
            "mia",
            "ad",
            "js",
        ],
        "crp": [
            "cka_vision",
            "cka_bridge",
            "cka_language",
        ],
    },
}

DATASET_ALIASES = {
    "mllmu_real": [
        "mllmu_real", "mllmu-real", "mllmureal", "real_mllmu",
    ],
    "mllmu_bench": [
        "mllmu_bench", "mllmu-bench", "mllmubench", "mllmu",
    ],
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
    "random_label": ["random_label", "random-label", "randomlabel"],
}

MODEL_ALIASES = {
    "llava-1.5-7b": [
        "llava-1.5-7b", "llava_1_5_7b", "llava15_7b",
        "llava-hf/llava-1.5-7b-hf", "llava",
    ],
    "blip2-opt-2.7b": [
        "blip-2-opt-2.7b", "blip2-opt-2.7b",
        "blip2_opt_2_7b", "blip2", "blip-2",
    ],
}

METRIC_ALIASES = {
    "behavioural": {
        "forget_accuracy": [
            "forget_accuracy", "forget_acc", "forgetacc",
            "forget_score", "forget_accuracy_mean",
        ],
        "retain_accuracy": [
            "retain_accuracy", "retain_acc", "retainacc",
            "retain_score", "retain_accuracy_mean",
        ],
        "mia": [
            "mia", "mia_accuracy", "mia_score",
            "membership_inference", "membership_inference_accuracy",
        ],
        "ad": [
            "ad", "answer_distance", "answer_dist",
        ],
        "js": [
            "js", "js_divergence", "jensen_shannon",
        ],
    },
    "crp": {
        "cka_vision": [
            "ve_cka", "vision_cka", "cka_vision",
            "vision_encoder_cka",
        ],
        "cka_bridge": [
            "bridge_cka", "br_cka", "cka_bridge",
            "projector_cka",
        ],
        "cka_language": [
            "lb_cka", "language_cka", "cka_language",
            "decoder_cka",
        ],
    },
}

DATASET_KEYS = ["dataset", "dataset_name", "benchmark", "data_name"]
METHOD_KEYS = ["method", "method_name", "algorithm", "unlearning_method"]
MODEL_KEYS = ["model", "model_name", "base_model", "architecture"]
SEED_KEYS = ["seed", "random_seed", "rng_seed"]
CHECKPOINT_KEYS = [
    "checkpoint", "checkpoint_path", "adapter_path",
    "model_path", "ckpt",
]

SKIP_HINTS = [
    "fiubench",
    "cross_dataset_source_audit",
    "matched_cross_dataset",
    "missing_result_locator",
    "_runner",
    "crash_diagnostics",
    "prediction",
    "malformed",
]

PREDICTION_KEYS = {
    "response", "prediction", "generated_text",
    "expected_answer", "question", "image_used",
}


def normalize(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        result = float(value)
        return result if math.isfinite(result) else None

    text = str(value).strip()
    percent = text.endswith("%")
    text = text.rstrip("%").strip()

    try:
        result = float(text)
    except ValueError:
        return None

    if percent:
        result /= 100.0

    return result if math.isfinite(result) else None


def flatten(
    obj: Any,
    prefix: str = "",
    out: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            next_key = f"{prefix}.{key}" if prefix else str(key)
            flatten(value, next_key, out)
    elif isinstance(obj, list):
        if len(obj) <= 20 and all(
            isinstance(x, (str, int, float, bool, type(None)))
            for x in obj
        ):
            out[prefix] = obj
    else:
        out[prefix] = obj

    return out


def iter_objects(data: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(data, dict):
        yield "root", data
        for key, value in data.items():
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, dict):
                        yield f"{key}[{index}]", item
            elif isinstance(value, dict):
                for child_key, item in value.items():
                    if isinstance(item, dict):
                        augmented = dict(item)
                        augmented.setdefault("_container_key", child_key)
                        yield f"{key}.{child_key}", augmented

    elif isinstance(data, list):
        for index, item in enumerate(data):
            if isinstance(item, dict):
                yield f"[{index}]", item


def canonical(
    values: Iterable[Any],
    aliases: dict[str, list[str]],
    ordered: list[str] | None = None,
) -> str | None:
    joined = normalize(" ".join(str(v or "") for v in values))
    keys = ordered or list(aliases)

    candidates = []
    for canonical_name in keys:
        for alias in aliases[canonical_name]:
            alias_n = normalize(alias)
            candidates.append((len(alias_n), canonical_name, alias_n))

    for _, canonical_name, alias in sorted(candidates, reverse=True):
        if re.search(rf"(^|_){re.escape(alias)}($|_)", joined):
            return canonical_name

    return None


def find_leaf(flat: dict[str, Any], keys: Iterable[str]) -> Any:
    wanted = {normalize(key) for key in keys}

    for full_key, value in flat.items():
        if normalize(full_key.split(".")[-1]) in wanted:
            return value

    return None


def infer_seed(flat: dict[str, Any], path: Path) -> str:
    value = find_leaf(flat, SEED_KEYS)
    if value not in (None, ""):
        return str(value)

    match = re.search(r"(?:seed|s)[_\-]?(\d{1,6})", str(path), re.I)
    return match.group(1) if match else ""


def infer_checkpoint(flat: dict[str, Any], path: Path) -> str:
    value = find_leaf(flat, CHECKPOINT_KEYS)
    if value:
        return str(value)

    for part in reversed(path.parts):
        if (
            "checkpoint" in part.casefold()
            or "adapter" in part.casefold()
            or "seed" in part.casefold()
        ):
            return part

    return ""


def is_prediction_dump(data: Any) -> bool:
    if not isinstance(data, list) or not data:
        return False

    sample = [item for item in data[:5] if isinstance(item, dict)]
    if not sample:
        return False

    hits = 0
    for item in sample:
        keys = {normalize(key) for key in item}
        if len(keys & PREDICTION_KEYS) >= 3:
            hits += 1

    return hits >= max(1, len(sample) // 2)


def extract_metric_matches(
    flat: dict[str, Any],
    dataset: str,
) -> list[tuple[str, str, float, str, int]]:
    matches = []

    for family, metric_map in METRIC_ALIASES.items():
        if family not in TARGETS.get(dataset, {}):
            continue

        for metric_name, aliases in metric_map.items():
            if metric_name not in TARGETS[dataset][family]:
                continue

            accepted = {normalize(alias) for alias in aliases}
            choices = []

            for full_key, value in flat.items():
                leaf = normalize(full_key.split(".")[-1])
                full_n = normalize(full_key)
                score = 0

                if leaf in accepted:
                    score = 100
                else:
                    for alias in accepted:
                        if full_n.endswith("_" + alias):
                            score = max(score, 90)
                        elif re.search(
                            rf"(^|_){re.escape(alias)}($|_)",
                            full_n,
                        ):
                            score = max(score, 70)

                if not score:
                    continue

                number = numeric(value)
                if number is not None:
                    choices.append(
                        (score, len(full_n), number, full_key)
                    )

            if choices:
                choices.sort(key=lambda x: (-x[0], x[1], x[3]))
                score, _, number, source_key = choices[0]
                matches.append(
                    (family, metric_name, number, source_key, score)
                )

    return matches


def schema_summary(data: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "root_type": type(data).__name__,
    }

    if isinstance(data, dict):
        result["root_keys"] = sorted(str(key) for key in data.keys())
        result["nested_dict_keys"] = {
            str(key): sorted(str(k) for k in value.keys())
            for key, value in data.items()
            if isinstance(value, dict)
        }
        result["list_lengths"] = {
            str(key): len(value)
            for key, value in data.items()
            if isinstance(value, list)
        }

    elif isinstance(data, list):
        result["length"] = len(data)
        if data and isinstance(data[0], dict):
            result["first_item_keys"] = sorted(
                str(key) for key in data[0].keys()
            )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="May be supplied more than once.",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    roots = (
        [Path(value).resolve() for value in args.search_root]
        if args.search_root
        else [path.resolve() for path in DEFAULT_ROOTS]
    )

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    candidate_files = []
    candidate_metrics = []
    schemas = {}
    seen = set()

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.json"):
            resolved = path.resolve()

            if resolved in seen:
                continue
            seen.add(resolved)

            path_n = normalize(str(path))
            if any(normalize(hint) in path_n for hint in SKIP_HINTS):
                continue

            try:
                data = json.loads(
                    path.read_text(encoding="utf-8-sig")
                )
            except Exception as exc:
                candidate_files.append({
                    "source_file": str(resolved),
                    "dataset": "",
                    "model": "",
                    "method": "",
                    "seed": "",
                    "checkpoint": "",
                    "metric_families": "",
                    "metric_names": "",
                    "candidate_metric_count": 0,
                    "prediction_dump": False,
                    "status": "JSON_ERROR",
                    "reason": repr(exc),
                })
                continue

            if is_prediction_dump(data):
                continue

            file_matches = []
            file_datasets = set()
            file_models = set()
            file_methods = set()
            file_seeds = set()
            file_checkpoints = set()

            for container, obj in iter_objects(data):
                flat = flatten(obj)

                dataset = canonical(
                    [
                        find_leaf(flat, DATASET_KEYS),
                        obj.get("_container_key", ""),
                        path,
                    ],
                    DATASET_ALIASES,
                    ["mllmu_real", "mllmu_bench"],
                )

                if dataset not in TARGETS:
                    continue

                method = canonical(
                    [
                        find_leaf(flat, METHOD_KEYS),
                        obj.get("_container_key", ""),
                        path,
                    ],
                    METHOD_ALIASES,
                ) or ""

                model = canonical(
                    [
                        find_leaf(flat, MODEL_KEYS),
                        find_leaf(flat, CHECKPOINT_KEYS),
                        path,
                    ],
                    MODEL_ALIASES,
                ) or ""

                seed = infer_seed(flat, path)
                checkpoint = infer_checkpoint(flat, path)

                matches = extract_metric_matches(flat, dataset)

                for (
                    family,
                    metric_name,
                    value,
                    source_key,
                    confidence,
                ) in matches:
                    row = {
                        "dataset": dataset,
                        "model": model,
                        "method": method,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "metric_family": family,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "source_key": source_key,
                        "source_container": container,
                        "match_confidence": confidence,
                        "source_file": str(resolved),
                    }
                    candidate_metrics.append(row)
                    file_matches.append(row)
                    file_datasets.add(dataset)
                    if model:
                        file_models.add(model)
                    if method:
                        file_methods.add(method)
                    if seed:
                        file_seeds.add(seed)
                    if checkpoint:
                        file_checkpoints.add(checkpoint)

            if not file_matches:
                continue

            families = sorted({
                row["metric_family"] for row in file_matches
            })
            names = sorted({
                row["metric_name"] for row in file_matches
            })

            status = "CANDIDATE"
            reasons = []

            if not file_methods:
                status = "REVIEW"
                reasons.append("method missing")

            if not file_models:
                status = "REVIEW"
                reasons.append("model missing")

            if not file_seeds:
                reasons.append("seed missing")

            if not file_checkpoints:
                reasons.append("checkpoint missing")

            candidate_files.append({
                "source_file": str(resolved),
                "dataset": ",".join(sorted(file_datasets)),
                "model": ",".join(sorted(file_models)),
                "method": ",".join(sorted(file_methods)),
                "seed": ",".join(sorted(file_seeds)),
                "checkpoint": ",".join(sorted(file_checkpoints)),
                "metric_families": ",".join(families),
                "metric_names": ",".join(names),
                "candidate_metric_count": len(file_matches),
                "prediction_dump": False,
                "status": status,
                "reason": "; ".join(reasons) or "target metrics found",
            })

            schemas[str(resolved)] = schema_summary(data)

    metric_fields = [
        "dataset", "model", "method", "seed", "checkpoint",
        "metric_family", "metric_name", "metric_value",
        "source_key", "source_container", "match_confidence",
        "source_file",
    ]
    file_fields = [
        "source_file", "dataset", "model", "method",
        "seed", "checkpoint", "metric_families",
        "metric_names", "candidate_metric_count",
        "prediction_dump", "status", "reason",
    ]

    with (output / "candidate_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(candidate_metrics)

    with (output / "candidate_files.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=file_fields)
        writer.writeheader()
        writer.writerows(candidate_files)

    (output / "candidate_schemas.json").write_text(
        json.dumps(schemas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    coverage_rows = []
    for dataset, families in TARGETS.items():
        for family, metric_names in families.items():
            for metric_name in metric_names:
                matches = [
                    row for row in candidate_metrics
                    if row["dataset"] == dataset
                    and row["metric_family"] == family
                    and row["metric_name"] == metric_name
                ]

                methods = sorted({
                    row["method"] for row in matches if row["method"]
                })
                files = sorted({
                    row["source_file"] for row in matches
                })

                coverage_rows.append({
                    "dataset": dataset,
                    "metric_family": family,
                    "metric_name": metric_name,
                    "candidate_rows": len(matches),
                    "methods_found": ",".join(methods),
                    "source_file_count": len(files),
                    "source_files": " | ".join(files),
                    "coverage_status": (
                        "FOUND" if matches else "MISSING"
                    ),
                })

    coverage_fields = [
        "dataset", "metric_family", "metric_name",
        "candidate_rows", "methods_found",
        "source_file_count", "source_files",
        "coverage_status",
    ]

    with (output / "exact_target_coverage.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=coverage_fields)
        writer.writeheader()
        writer.writerows(coverage_rows)

    missing_targets = [
        row for row in coverage_rows
        if row["coverage_status"] == "MISSING"
    ]

    report = {
        "search_roots": [str(root) for root in roots],
        "json_files_scanned": len(seen),
        "candidate_files": len(candidate_files),
        "candidate_metric_rows": len(candidate_metrics),
        "target_metric_slots": len(coverage_rows),
        "missing_target_slots": len(missing_targets),
        "missing_targets": missing_targets,
        "files": {
            "candidate_files": str(output / "candidate_files.csv"),
            "candidate_metrics": str(output / "candidate_metrics.csv"),
            "schemas": str(output / "candidate_schemas.json"),
            "coverage": str(output / "exact_target_coverage.csv"),
        },
    }

    if not candidate_metrics:
        verdict = "FAIL"
        reason = "No candidate target metrics were found."
        exit_code = 1
    elif missing_targets:
        verdict = "WARN"
        reason = (
            "Some target metrics were found, but one or more target slots "
            "remain missing."
        )
        exit_code = 2
    else:
        verdict = "PASS"
        reason = "Candidate sources were found for every target metric slot."
        exit_code = 0

    report["overall_verdict"] = verdict
    report["reason"] = reason

    (output / "locator_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "TARGETED MISSING-RESULT LOCATOR",
        "=" * 92,
        f"JSON files scanned: {len(seen)}",
        f"Candidate files: {len(candidate_files)}",
        f"Candidate metric rows: {len(candidate_metrics)}",
        f"Target metric slots: {len(coverage_rows)}",
        f"Missing target slots: {len(missing_targets)}",
        "",
        "TARGET COVERAGE",
        "-" * 92,
    ]

    for row in coverage_rows:
        lines.append(
            f"{row['dataset']} / {row['metric_family']} / "
            f"{row['metric_name']}: {row['coverage_status']} "
            f"(rows={row['candidate_rows']}, "
            f"files={row['source_file_count']}, "
            f"methods={row['methods_found'] or 'none'})"
        )

    lines.extend([
        "",
        f"CANDIDATE FILES: {output / 'candidate_files.csv'}",
        f"CANDIDATE METRICS: {output / 'candidate_metrics.csv'}",
        f"SCHEMAS: {output / 'candidate_schemas.json'}",
        f"COVERAGE: {output / 'exact_target_coverage.csv'}",
        f"REASON: {reason}",
        f"OVERALL VERDICT: {verdict}",
    ])

    (output / "locator_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
